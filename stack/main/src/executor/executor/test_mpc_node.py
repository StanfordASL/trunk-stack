import os
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


@jax.jit
def check_control_inputs(u_opt, u_opt_previous):
    """
    Check control inputs for safety constraints, rejecting vector norms that are too large.
    """
    tip_range, mid_range, base_range = 0.45, 0.35, 0.3

    u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

    # First we clip to max and min values
    u1 = jnp.clip(u1, -tip_range, tip_range)
    u6 = jnp.clip(u6, -tip_range, tip_range)
    u2 = jnp.clip(u2, -mid_range, mid_range)
    u4 = jnp.clip(u5, -mid_range, mid_range)
    u3 = jnp.clip(u3, -base_range, base_range)
    u5 = jnp.clip(u4, -base_range, base_range)

    # Compute control input vectors
    u1_vec = u1 * jnp.array([-jnp.cos(15 * jnp.pi/180), jnp.sin(15 * jnp.pi/180)])
    u2_vec = u2 * jnp.array([jnp.cos(45 * jnp.pi/180), jnp.sin(45 * jnp.pi/180)])
    u3_vec = u3 * jnp.array([-jnp.cos(15 * jnp.pi/180), -jnp.sin(15 * jnp.pi/180)])
    u4_vec = u4 * jnp.array([-jnp.cos(45 * jnp.pi/180), jnp.sin(45 * jnp.pi/180)])
    u5_vec = u5 * jnp.array([jnp.cos(75 * jnp.pi/180), -jnp.sin(75 * jnp.pi/180)])
    u6_vec = u6 * jnp.array([-jnp.cos(75 * jnp.pi/180), -jnp.sin(75 * jnp.pi/180)])

    # Calculate the norm based on the constraint
    vector_sum = (
        0.75 * (u3_vec + u5_vec) +
        1.0 * (u2_vec + u4_vec) +
        1.4 * (u1_vec + u6_vec)
    )
    norm_value = jnp.linalg.norm(vector_sum)

    # Check the constraint: if the constraint is met, then keep previous control command
    u_opt = jnp.where(norm_value > 0.8, u_opt_previous, jnp.array([u1, u2, u3, u4, u5, u6]))

    return u_opt


class TestMPCNode(Node):
    """
    This node is responsible for testing the MPC loop.
    """
    def __init__(self):
        super().__init__('run_experiment_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('results_name', 'test_experiment')             # name of the results file
        ])

        self.debug = self.get_parameter('debug').value
        self.results_name = self.get_parameter('results_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

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
        self.uopt_previous = jnp.zeros(6)
        
        self.clock = self.get_clock()

        # Need some initialization
        self.initialized = False

        # Initialize by calling mpc callback function
        self.mpc_executor_callback()

        # JIT compile this function
        check_control_inputs(jnp.zeros(6), self.uopt_previous)

        # Create timer to execute MPC at fixed frequency
        self.controller_period = 0.03
        self.mpc_exec_timer = self.create_timer(
            self.controller_period,
            self.mpc_executor_callback
        )

        self.get_logger().info(f'MPC test node has been started with controller frequency: {1/self.controller_period:.2f} [Hz].')

        # Define reference time
        self.start_time = self.clock.now().nanoseconds / 1e9

    def mpc_executor_callback(self):
        """
        Execute MPC at a fixed rate.
        """
        if not self.initialized:
            self.send_request(0.0, jnp.zeros(12), wait=True)
            self.future.add_done_callback(self.service_callback)
            self.initialized = True
        else:
            t0 = self.clock.now().nanoseconds / 1e9 - self.start_time
            self.update_observations(t0)
            self.send_request(t0, self.latest_y, wait=False)
            self.future.add_done_callback(self.service_callback)

    def send_request(self, t0, y0, wait=False):
        """
        Send request to MPC solver.
        """
        self.req.t0 = t0
        self.req.y0 = jnp2arr(y0)
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
                # We do not execute the control inputs here but it's still being checked
                safe_control_inputs = check_control_inputs(jnp.array(response.uopt[:6]), self.uopt_previous)
                self.uopt_previous = safe_control_inputs

                # Save the predicted observations
                self.topt, self.zopt = arr2jnp(response.t, 1, squeeze=True), arr2jnp(response.zopt, 3)

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def update_observations(self, t0, eps=1e-4):
        """
        Update the latest observations using predicted observations from MPC plus added noise.
        """
        # Figure out what predictions to use for observations update
        idx0 = jnp.searchsorted(self.topt, t0, side='right')
        y_predicted = self.zopt[:idx0+1]

        # Add noise to simulate real experiment
        y_centered_tip = y_predicted + eps * jax.random.normal(key=self.rnd_key, shape=y_predicted.shape)
        N_new_obs = y_centered_tip.shape[0]

        # Update tracked observation
        if self.latest_y is None:
            # At initialization use current obs. as delay embedding
            self.latest_y = jnp.tile(y_centered_tip[-1:].squeeze(), 4)
            self.start_time = self.clock.now().nanoseconds / 1e9
        else:
            # Note the different ordering of MPC horizon and delay embeddings which requires the flipping
            if N_new_obs > 4:
                # If we have more than 4 new observations, we only keep the last 4
                self.latest_y = jnp.flip(y_centered_tip[-4:].T, 1).T.flatten()
            else:
                # Otherwise we concatenate the new observations with the old ones
                self.latest_y = jnp.concatenate([jnp.flip(y_centered_tip.T, 1).T.flatten(), self.latest_y[:(4-N_new_obs)*3]])


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
