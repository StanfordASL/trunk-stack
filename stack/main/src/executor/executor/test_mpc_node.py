import os
import time
import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from rclpy.qos import QoSProfile    # type: ignore
from controller.mpc_solver_node import jnp2arr  # type: ignore
from interfaces.msg import TrunkRigidBodies
from interfaces.srv import ControlSolver


def check_control_inputs(u_opt, u_opt_previous):
    # reject vector norms of u that are too large
    tip_range, mid_range, base_range = 0.45, 0.35, 0.3

    u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

    # First we clip to max and min values
    u1 = jnp.clip(u1, -tip_range, tip_range)
    u6 = jnp.clip(u6, -tip_range, tip_range)
    u2 = jnp.clip(u2, -mid_range, mid_range)
    u5 = jnp.clip(u5, -mid_range, mid_range)
    u3 = jnp.clip(u3, -base_range, base_range)
    u4 = jnp.clip(u4, -base_range, base_range)

    # Compute control input vectors
    u1_vec = u1 * jnp.array([-jnp.cos(15 * jnp.pi/180), jnp.sin(15 * jnp.pi/180)])
    u2_vec = u2 * jnp.array([jnp.cos(45 * jnp.pi/180), jnp.sin(45 * jnp.pi/180)])
    u3_vec = u3 * jnp.array([-jnp.cos(15 * jnp.pi/180), -jnp.sin(15 * jnp.pi/180)])
    u4_vec = u4 * jnp.array([-jnp.cos(75 * jnp.pi/180), jnp.sin(75 * jnp.pi/180)])
    u5_vec = u5 * jnp.array([jnp.cos(45 * jnp.pi/180), -jnp.sin(45 * jnp.pi/180)])
    u6_vec = u6 * jnp.array([-jnp.cos(75 * jnp.pi/180), -jnp.sin(75 * jnp.pi/180)])

    # Calculate the norm based on the constraint
    vector_sum = (
        0.75 * (u3_vec + u4_vec) +
        1.0 * (u2_vec + u5_vec) +
        1.4 * (u1_vec + u6_vec)
    )
    norm_value = jnp.linalg.norm(vector_sum)

    # Check the constraint: if the constraint is met, then keep previous control command
    if norm_value > 0.8:
        print(f'Sample {u_opt} got rejected')
        u_opt = u_opt_previous
    else:
        # Else the clipped command is published
        u_opt = jnp.array([u1, u2, u3, u4, u5, u6])

    return u_opt


class TestMPCNode(Node):
    """
    This node is responsible for testing MPC.
    """
    def __init__(self):
        super().__init__('run_experiment_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('controller_type', 'mpc'),                     # 'ik' or 'mpc' (what controller to use)
            ('results_name', 'test_experiment')             # name of the results file
        ])

        self.debug = self.get_parameter('debug').value
        self.controller_type = self.get_parameter('controller_type').value
        self.results_name = self.get_parameter('results_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Settled positions of the rigid bodies
        self.rest_position = jnp.array([0.10056, -0.10541, 0.10350, 0.09808, -0.20127, 0.10645, 0.09242, -0.31915, 0.09713])

        if self.controller_type == 'mpc':
            # Subscribe to current positions
            self.mocap_subscription = self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.mocap_listener_callback,
                QoSProfile(depth=10)
            )

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

            # Send over an example request
            self.send_request(0.0, jnp.zeros(12), wait=False)
            
            # Sleep a bit right after as that was found to help
            self.get_logger().info('Waiting for a sec...')
            time.sleep(0.5)

        # Maintain current observations because of the delay embedding
        self.y = None

        # Maintain previous control inputs
        self.uopt_previous = jnp.zeros(6)

        # Keep a clock for timing
        self.clock = self.get_clock()

        controller_freq = 30  # [Hz]
        self.mpc_exec_timer = self.create_timer(1 / controller_freq, self.mpc_executor_callback, clock=self.clock)

        self.get_logger().info('Run experiment node has been started.')

    def mpc_executor_callback(self):
        self.get_logger().info(f'Sent the request at {self.clock.now().nanoseconds / 1e9 - self.start_time}')
        self.get_logger().info(f'   during which t0 was {self.t0}')
        self.send_request(self.t0, self.y, wait=False)
        self.future.add_done_callback(self.service_callback)

    def service_callback(self, async_response):
        try:
            response = async_response.result()
            self.get_logger().info(f'Received the uopt at {self.clock.now().nanoseconds / 1e9 - self.start_time}')
            self.get_logger().info(f'   which was for t0: {response.t[0]}')
            if response.done:
                self.get_logger().info('Trajectory is finished!')
                self.destroy_node()
                rclpy.shutdown()
            else:
                safe_control_inputs = check_control_inputs(response.uopt[:6], self.uopt_previous)
                # self.get_logger().info(f'We command the control inputs: {safe_control_inputs.tolist()}.')
                self.get_logger().info(f'We would command the control inputs: {response.uopt[:6]}.')
                self.uopt_previous = safe_control_inputs
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

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


def main(args=None):
    rclpy.init(args=args)
    test_mpc_node = TestMPCNode()
    rclpy.spin(test_mpc_node)
    test_mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
