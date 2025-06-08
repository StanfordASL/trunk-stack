import os
import csv
import jax
import jax.numpy as jnp
import logging
import numpy as np

logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import rclpy                                             # type: ignore
from rclpy.node import Node                              # type: ignore
from controller.mpc_solver_node import jnp2arr, arr2jnp  # type: ignore
from interfaces.srv import ControlSolver
from .utils.models import control_SSMR

run_on_pauls_computer = True

@jax.jit
def check_control_inputs(u_opt, u_opt_previous):
    """
    Check control inputs for safety constraints, rejecting vector norms that are too large.
    """
    tip_range, mid_range, base_range = 81, 51, 31

    u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

    # First we clip to max and min values
    u1 = jnp.clip(u1, -tip_range, tip_range)
    u6 = jnp.clip(u6, -tip_range, tip_range)
    u2 = jnp.clip(u2, -mid_range, mid_range)
    u4 = jnp.clip(u5, -mid_range, mid_range)
    u3 = jnp.clip(u3, -base_range, base_range)
    u5 = jnp.clip(u4, -base_range, base_range)
    u_opt = jnp.array([u1, u2, u3, u4, u5, u6])
    
    return u_opt


class TestMPCNode(Node):
    """
    This node is responsible for testing the MPC loop.
    """
    def __init__(self):
        super().__init__('run_experiment_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('results_name', 'pauls_first_test')            # name of the results file
        ])

        config = {
            "delay_embedding": {
                "perf_var_dim": 3,
                "also_embedd_u": True
            },
            "model": "first_mpc_model_real_trunk.pkl"
        }

        self.delay_config = config["delay_embedding"]

        self.debug = self.get_parameter('debug').value
        self.model_name = config["model"]

        self.results_name = self.get_parameter('results_name').value
        # self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        if run_on_pauls_computer:
            self.data_dir = os.getenv('TRUNK_DATA', '/Users/paulleonardwolff/Desktop/trunk-stack/stack/main/data')
        else:
            self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Load the model
        self._load_model()
        self.n_delay = self.model.ssm.specified_params["embedding_up_to"]  # self.model.n_y // self.model.n_z - 1

        # Initialize the CSV file
        self.results_file = os.path.join(self.data_dir, f"trajectories/test_mpc/{self.results_name}.csv")
        self.initialize_csv()

        # Key to control randomness in added noise
        self.rnd_key = jax.random.key(seed=0)

        # Create MPC solver service client
        self.mpc_client = self.create_client(
            ControlSolver,
            'mpc_solver'
        )
        self.get_logger().info('MPC client created.')
        while not self.mpc_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('MPC solver not available, waiting...')
        
        # Request message definition
        self.req = ControlSolver.Request()

        # Maintain current observations because of the delay embedding
        self.latest_y = None

        # Maintain previous control inputs
        self.uopt_previous = jnp.zeros(self.model.n_u)
        
        self.clock = self.get_clock()

        # Need some initialization
        self.initialized = False

        # Initialize by calling mpc callback function
        self.mpc_executor_callback()

        embedding_up_to = self.model.ssm.specified_params["embedding_up_to"]
        shift = self.model.ssm.specified_params["shift_steps"]  # Is 0 if there is no subsampling
        pad_length = self.model.n_u * ((1 + shift) * embedding_up_to - shift)
        self.u_ref_init = jnp.zeros((pad_length,))

        x0_red_u_init = jnp.concatenate([jnp.zeros(self.model.n_x), self.u_ref_init], axis=0)
        # JIT compile couple of functions
        check_control_inputs(jnp.zeros(self.model.n_u), self.uopt_previous)
        
        self.model.rollout(x0_red_u_init, jnp.zeros((1, self.model.n_u)))

        self.model.decode(jnp.zeros(self.model.n_x))

        # Create timer to execute MPC at fixed frequency
        self.controller_period = 0.04
        self.mpc_exec_timer = self.create_timer(
            self.controller_period,
            self.mpc_executor_callback
        )

        self.get_logger().info(f'MPC test node has been started with controller frequency: {1/self.controller_period:.2f} [Hz].')

        # Define reference time
        self.start_time = self.clock.now().nanoseconds / 1e9

    def _load_model(self):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """
        model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}')

        # Load the model
        self.model = control_SSMR(self.delay_config, model_path)
        print(f'---- Model loaded: {self.model_name}')
        print('Dimensions:')
        print('     n_x:', self.model.n_x)
        print('     n_u:', self.model.n_u)
        print('     n_z:', self.model.n_z)
        print('     n_y:', self.model.n_y)

    def mpc_executor_callback(self):
        """
        Execute MPC at a fixed rate.
        """
        if not self.initialized:
            self.send_request(0.0, jnp.zeros(self.model.n_y), self.uopt_previous, wait=True)
            self.future.add_done_callback(self.service_callback)
            self.initialized = True
        else:
            self.t0 = self.clock.now().nanoseconds / 1e9 - self.start_time
            self.update_observations(eps_noise=0)
            # self.latest_y should be the delay embedded state vector
            print(f"(DEBUG) latest_y shape = {self.latest_y.shape}")
            self.send_request(self.t0, self.latest_y, self.uopt_previous, wait=False)
            self.future.add_done_callback(self.service_callback)

    def send_request(self, t0, y0, u0, wait=False):
        """
        Send request to MPC solver.
        """
        self.req.t0 = t0
        print("Sending request with y0 shape:", y0.shape)  # Debugging line
        self.req.y0 = jnp2arr(y0)
        print("Shape of u0:", u0.shape)  # Debugging line
        self.req.u0 = jnp2arr(u0)
        self.future = self.mpc_client.call_async(self.req)

        if wait:
            # Synchronous call, not compatible for real-time applications
            rclpy.spin_until_future_complete(self, self.future)

    def service_callback(self, async_response):
        """
        Callback that defines what happens when the MPC solver node returns a result.
        """
        try:
            response = async_response.result()
            if response.done:
                self.get_logger().info('Trajectory is finished!')
                self.destroy_node()
                rclpy.shutdown()
            else:
                topt, xopt, uopt, zopt = response.t, response.xopt, response.uopt, response.zopt
                
                # We do not execute the control inputs here but it's still being checked
                safe_control_inputs = check_control_inputs(jnp.array(uopt[:self.model.n_u]), self.uopt_previous)
                print("Shape of safe_control_inputs:", safe_control_inputs.shape)  # Debugging line
                self.uopt_previous = safe_control_inputs[np.array([2, 4])]  # HERE
                print("Shape of self.uopt_previous:", self.uopt_previous.shape)

                # Save the predicted observations and control inputs
                if self.latest_y is not None:
                    self.save_to_csv(topt, xopt, uopt, zopt)
                self.topt = arr2jnp(topt, 1, squeeze=True)
                self.x0 = jnp.array(xopt[:self.model.n_x])
                self.uopt = arr2jnp(uopt, self.model.n_u)

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def update_observations(self, eps_noise=1e-4):
        """
        Update the latest observations using predicted observations from MPC plus added noise.
        """
        # Figure out what predictions to use for observations update
        idx0 = jnp.searchsorted(self.topt, self.t0, side='right')

        if self.u_ref_init.shape[0] >= self.model.n_u:
            self.u_ref_init = jnp.concatenate([self.uopt[0], self.u_ref_init[:-self.model.n_u]], axis=0)

        print("Shape of uopt:", self.uopt.shape)  # Shape is correct
        print("Shape of x0:", self.x0.shape)
        x0_aug = jnp.concatenate([self.x0, self.u_ref_init], axis=0)
        # x_predicted = self.model.rollout(self.x0, self.uopt)
        x_predicted = self.model.rollout(x0_aug, self.uopt)
        y_predicted = self.model.decode(x_predicted.T).T

        print(f"(DEBUG) model.n_z = {self.model.n_z}, model.n_y = {self.model.n_y}")
        print(f"(DEBUG) y_predicted.shape = {y_predicted.shape}")
        print(f"(DEBUG) idx0 = {idx0}")

        y_centered_tip = y_predicted[:idx0+1, :8]  # 8 is hardcoded for the measured state dimension
        N_new_obs = y_centered_tip.shape[0]
        print(f"(DEBUG) N_new_obs = {N_new_obs}")

        # Add noise to simulate real experiment
        y_tip_noisy = y_centered_tip + eps_noise * jax.random.normal(key=self.rnd_key, shape=y_centered_tip.shape)

        # Update tracked observation
        if self.latest_y is None:
            # At initialization use current obs. as delay embedding
            self.latest_y = jnp.tile(y_tip_noisy[-1:].squeeze(), (self.n_delay+1))
            print(f"(DEBUG) First‐time latest_y length = {self.latest_y.shape[0]}")
            self.start_time = self.clock.now().nanoseconds / 1e9
        else:
            # Note the different ordering of MPC horizon and delay embeddings which requires the flipping
            if N_new_obs > self.n_delay + 1:
                # If we have more than self.n_delay + 1 new observations, we only keep the last self.n_delay + 1
                self.latest_y = jnp.flip(y_tip_noisy[-(self.n_delay+1):].T, 1).T.flatten()
                print(f"(DEBUG) N_new_obs >= num_blocks, latest_y length = {self.latest_y.shape[0]}")
            else:
                print("shape of y_tip_noisy:", y_tip_noisy.shape)
                num_blocks = self.n_delay + 1
                block_size = self.model.n_y // num_blocks
                # Otherwise we concatenate the new observations with the old ones
                new_block = y_tip_noisy[-N_new_obs:, :].reshape(-1)
                old_needed = num_blocks - N_new_obs
                old_part = self.latest_y[: old_needed * block_size]
                self.latest_y= jnp.concatenate([new_block, old_part])
                #self.latest_y = jnp.concatenate([jnp.flip(y_tip_noisy.T, 1).T.flatten(),
                #                                 self.latest_y[:(self.n_delay + 1 - N_new_obs) * self.model.n_z]])
                print(f"(DEBUG) N_new_obs < num_blocks, latest_y length = {self.latest_y.shape}")

    def initialize_csv(self):
        """
        Initialize the CSV file with headers.
        """
        with open(self.results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['topt', 'xopt', 'uopt', 'zopt'])

    def save_to_csv(self, topt, xopt, uopt, zopt):
        """
        Save optimized quantities by MPC to CSV file.
        """
        with open(self.results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([list(topt), list(xopt), list(uopt), list(zopt)])


def main(args=None):
    """
    Run single-threaded ROS2 node. 
    """
    rclpy.init(args=args)
    test_mpc_node = TestMPCNode()
    rclpy.spin(test_mpc_node)
    test_mpc_node.destroy_node()
    # rclpy.shutdown() <- shutdown is called in service_callback when the trajectory is finished


if __name__ == '__main__':
    main()
