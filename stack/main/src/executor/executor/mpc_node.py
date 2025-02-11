import os

import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import rclpy                                                # type: ignore
from rclpy.node import Node                                 # type: ignore
from rclpy.callback_groups import ReentrantCallbackGroup    # type: ignore
from rclpy.executors import MultiThreadedExecutor           # type: ignore
from rclpy.qos import QoSProfile                            # type: ignore

from controller.mpc_solver_node import jnp2arr              # type: ignore
from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies
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


class MPCNode(Node):
    """
    This node is responsible for running the main experiment.
    """
    def __init__(self):
        super().__init__('mpc_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('results_name', 'mpc_experiment')             # name of the results file
        ])

        self.debug = self.get_parameter('debug').value
        self.results_name = self.get_parameter('results_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        self.alpha_patrick = 0.2
        self.safe_control_input = jnp.zeros(6)

        # Settled positions of the rigid bodies
        self.rest_position = jnp.array([0.10056, -0.10541, 0.10350,
                                        0.09808, -0.20127, 0.10645,
                                        0.09242, -0.31915, 0.09713])
        
        self.callback_group = ReentrantCallbackGroup()

        # Subscribe to current positions
        self.mocap_subscription = self.create_subscription(
            TrunkRigidBodies,
            '/trunk_rigid_bodies',
            self.mocap_listener_callback,
            QoSProfile(depth=10),
            callback_group=self.callback_group
        )

        # Create MPC solver service client
        self.mpc_client = self.create_client(
            ControlSolver,
            'mpc_solver',
            callback_group=self.callback_group
        )
        self.get_logger().info('MPC client created.')
        while not self.mpc_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('MPC solver not available, waiting...')
        
        # Request message definition
        self.req = ControlSolver.Request()

        # Create publisher to execute found control inputs
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )

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

        self.controller_period = 0.04
        
        self.mpc_exec_timer = self.create_timer(
                    self.controller_period,
                    self.mpc_executor_callback,
                    callback_group=self.callback_group)

        self.get_logger().info(f'MPC node has been started with controller frequency: {1/self.controller_period:.2f} [Hz].')

        self.start_time = self.clock.now().nanoseconds / 1e9

    def mocap_listener_callback(self, msg):
        """
        Callback to process mocap data. It updates the latest observation.
        """
        if self.debug:
            self.get_logger().info(f'Received mocap data: {msg.positions}.')

        # Unpack the message into simple list of positions, eg [x1, y1, z1, x2, y2, z2, ...]
        y_new = jnp.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])
        y_centered = y_new - self.rest_position

        # Subselect tip
        y_centered_tip = y_centered[-3:]

        # Update the current observations, including 3 delay embeddings
        if self.latest_y is None:
            # At initialization use current obs. as delay embedding
            self.latest_y = jnp.tile(y_centered_tip, 4)
            self.start_time = self.clock.now().nanoseconds / 1e9
        else:
            self.latest_y = jnp.concatenate([y_centered_tip, self.latest_y[:-3]])
        
        self.t0 = self.clock.now().nanoseconds / 1e9 - self.start_time

    def mpc_executor_callback(self):
        if not self.initialized:
            self.send_request(0.0, jnp.zeros(12), wait=True)
            self.future.add_done_callback(self.service_callback)
            self.initialized = True
        elif self.latest_y is not None:
            # self.get_logger().info(f'Sent the request at {(self.clock.now().nanoseconds / 1e9 - self.start_time):.3f}')
            # self.get_logger().info(f'Sent the request at {(time.time() - self.start_time):.3f}')
            self.send_request(self.t0, self.latest_y, wait=False)
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
        try:
            response = async_response.result()
            # self.get_logger().info(f'Received uopt at: {(self.clock.now().nanoseconds / 1e9 - self.start_time):.3f} for t0: {response.t[0]:.3f}')

            if response.done:
                self.get_logger().info(f'Trajectory is finished! At {(self.clock.now().nanoseconds / 1e9 - self.start_time):.3f}')
                self.destroy_node()
                rclpy.shutdown()
            else:
                safe_control_inputs = check_control_inputs(jnp.array(response.uopt[:6]), self.uopt_previous)
                # self.publish_control_inputs(safe_control_inputs.tolist())

                # TODO: John Edits
                self.safe_control_input = (1 - self.alpha_patrick) * self.safe_control_input + self.alpha_patrick * safe_control_inputs
                self.publish_control_inputs(self.safe_control_input.tolist())

                # self.get_logger().info(f'We command the control inputs: {safe_control_inputs.tolist()}.')
                # self.get_logger().info(f'We would command the control inputs: {response.uopt[:6]}.')

                self.uopt_previous = safe_control_inputs
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def publish_control_inputs(self, control_inputs):
        """
        Publish the control inputs.
        """
        control_message = AllMotorsControl()
        control_message.motors_control = [
            SingleMotorControl(mode=0, value=value) for value in control_inputs
        ]
        self.controls_publisher.publish(control_message)
        if self.debug:
            self.get_logger().info(f'Published new motor control setting: {control_inputs}.')


def main(args=None):
    rclpy.init(args=args)
    mpc_node = MPCNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(mpc_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        mpc_node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        mpc_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
