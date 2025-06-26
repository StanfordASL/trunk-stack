import os
import csv
import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import rclpy                                             # type: ignore
from rclpy.node import Node                              # type: ignore
from controller.mpc_solver_node import jnp2arr, arr2jnp  # type: ignore
from interfaces.srv import ControlSolver
from .utils.models import SSMR, KoopmanSSMR
import numpy as np


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
            ('model_name', 'koopman_real_trunk_perf3'),     # 'koopman_real_trunk_perf3' ,  'origin_ssm_baseline(1)', 'ssmr_200g' (what model to use)
            ('results_name', 'test_experiment_koopman')             # name of the results file
        ])

        model_type = "koopman"

        self.debug = self.get_parameter('debug').value
        self.model_name = self.get_parameter('model_name').value
        self.results_name = self.get_parameter('results_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Load the model

        if model_type == "koopman":
            self.koopman = True
        else:
            self.koopman = False

        self._load_model(model_type)
        num_measurements = 6
        self.n_delay = int(self.model.n_y // num_measurements - 1)

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
        self.latest_y_koopman = None

        # Maintain previous control inputs
        self.uopt_previous = jnp.zeros(self.model.n_u)
        
        self.clock = self.get_clock()

        # Need some initialization
        self.initialized = False

        # Initialize by calling mpc callback function
        self.mpc_executor_callback()

        # JIT compile couple of functions
        check_control_inputs(jnp.zeros(self.model.n_u), self.uopt_previous)
        self.model.rollout(jnp.zeros(self.model.n_x), jnp.zeros((1, self.model.n_u)))
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

    def _load_model(self, model_type):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """
        # Load the model
        if model_type == "ssm":
            model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}.npz')
            self.model = SSMR(model_path=model_path)
        elif model_type == "koopman":
            model_path = os.path.join(self.data_dir, f'models/koopman/{self.model_name}.npz')
            self.model = KoopmanSSMR.from_file(model_path)
        else:
            KeyError(f"The requested model type {model_type} was not recognized.")

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
            if self.koopman:
                send_y = self.latest_y_koopman
            else:
                send_y = self.latest_y
            self.send_request(self.t0, send_y, self.uopt_previous, wait=False)
            self.future.add_done_callback(self.service_callback)

    def send_request(self, t0, y0, u0, wait=False):
        """
        Send request to MPC solver.
        """
        self.req.t0 = t0
        self.req.y0 = jnp2arr(y0)
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
                
                # uopt_np = np.array(uopt)
                # xopt_np = np.array(xopt)
                # zopt_np = np.array(zopt)

                # print("Converted u: ", type(uopt_np), "— Shape:", uopt_np.shape)
                # print("Converted x: ", type(xopt_np), "— Shape:", xopt_np.shape)
                # print("Converted z: ", type(zopt_np), "— Shape:", zopt_np.shape)

                """
                self.uopt_previous:  [-0.00055862 -0.00055862]
                Converted u:  <class 'numpy.ndarray'> — Shape: (20,)
                Converted x:  <class 'numpy.ndarray'> — Shape: (1309,)
                Converted z:  <class 'numpy.ndarray'> — Shape: (33,)
                """

                # We do not execute the control inputs here but it's still being checked
                safe_control_inputs = check_control_inputs(jnp.array(uopt[:self.model.n_u]), self.uopt_previous)
                self.uopt_previous = safe_control_inputs[np.array([2, 4])]
                print("self.uopt_previous: ", self.uopt_previous)

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

        # Get index based on current time
        idx0 = jnp.searchsorted(self.topt, self.t0, side='right')

        # Rollout predicted states and decode to preself.latest_y_koopmandicted observations
        x_predicted = self.model.rollout(self.x0, self.uopt)
        y_predicted = self.model.decode(x_predicted.T).T

        # Slice predicted tip observations up to idx0
        y_centered_tip = y_predicted[:idx0+1, :6]

        # Number of new observations available
        N_new_obs = y_centered_tip.shape[0]

        # Add noise to simulate measurement noise
        noise = eps_noise * jax.random.normal(key=self.rnd_key, shape=y_centered_tip.shape)
        y_tip_noisy = y_centered_tip + noise

        # Update tracked observation (delay embedding)
        if self.latest_y is None:
            init_obs = y_tip_noisy[-1:].squeeze()
            self.latest_y = jnp.tile(init_obs, (self.n_delay + 1,))
            self.start_time = self.clock.now().nanoseconds / 1e9
        else:
            if N_new_obs > self.n_delay + 1:
                flipped = jnp.flip(y_tip_noisy[-(self.n_delay + 1):].T, 1).T
                self.latest_y = flipped.flatten()
            else:
                num_blocks = self.n_delay + 1
                block_size = self.model.n_y // num_blocks
                new_block = y_tip_noisy[-N_new_obs:, :].reshape(-1)
                old_needed = num_blocks - N_new_obs
                old_part = self.latest_y[: old_needed * block_size]
                self.latest_y = jnp.concatenate([new_block, old_part])

        # Normalize shape
        self.latest_y = self.latest_y.reshape(-1)

        # === NEW: Augment observations with previous inputs for Koopman MPC ===
        if self.koopman:
            # Assuming self.u_previous contains the previous control input(s), shape (n_u,) or (N, n_u)
            self.latest_y_koopman = jnp.concatenate([self.latest_y, self.uopt[0].reshape(-1)])  # self.uopt_previous

        # Update current time
        self.t0 = self.clock.now().nanoseconds / 1e9 - self.start_time

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
    rclpy.shutdown()


if __name__ == '__main__':
    main()
