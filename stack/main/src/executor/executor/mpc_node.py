import os
import time
import threading

import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import rclpy                                        # type: ignore
from rclpy.node import Node                         # type: ignore
from rclpy.executors import MultiThreadedExecutor   # type: ignore
from rclpy.qos import QoSProfile                    # type: ignore

from controller.mpc_solver_node import jnp2arr      # type: ignore
from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies
from interfaces.srv import ControlSolver


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
    if norm_value > 0.8:
        print(f'Sample {u_opt} got rejected')
        u_opt = u_opt_previous
    else:
        # Else the clipped command is published
        u_opt = jnp.array([u1, u2, u3, u4, u5, u6])

    return u_opt


class MPCNode(Node):
    """
    This node is responsible for running the main experiment.
    """
    def __init__(self):
        super().__init__('mpc_node')
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
        self.rest_position = jnp.array([0.10056, -0.10541, 0.10350,
                                        0.09808, -0.20127, 0.10645,
                                        0.09242, -0.31915, 0.09713])

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

        # Create publisher to execute found control inputs
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )

        # Maintain current observations because of the delay embedding
        self.lastest_y = None

        # Maintain previous control inputs
        self.uopt_previous = jnp.zeros(6)

        self.get_logger().info('MPC node has been started.')

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
            self.latest_y = jnp.concatenate([y_centered_tip, self.y[:-3]])

    def control_loop(self):
        """
        Control loop running in its own thread. It periodically sends a request
        to the MPC solver and publishes control commands when the response arrives.
        """
        while rclpy.ok():
            if self.latest_y is None:
                time.sleep(0.1)
                continue

            # Compute the current time t0 relative to start
            t0 = self.get_clock().now().nanoseconds / 1e9 - self.start_time

            # Prepare the service request
            self.req.t0 = t0
            self.req.y0 = jnp2arr(self.latest_y)

            # Call the MPC service synchronously (since the solver is fast)
            future = self.mpc_client.call_async(self.req)

            # Wait for the future to complete without blocking other callbacks due to multithreading
            rclpy.spin_until_future_complete(self, future)
            if future.done():
                try:
                    response = future.result()
                    
                    # Check if the trajectory is done
                    if response.done:
                        self.get_logger().info(f'Trajectory finished at {t0:.3f} seconds. Shutting down.')
                        rclpy.shutdown()

                    # Apply safety checks on the control inputs
                    safe_control = check_control_inputs(response.uopt[:6], self.uopt_previous)
                    self.uopt_previous = safe_control

                    # Publish the safe control inputs
                    self.publish_control_inputs(safe_control.tolist())

                    if self.debug:
                        self.get_logger().info(f'Commanded control inputs: {safe_control.tolist()}')

                except Exception as e:
                    self.get_logger().error(f'Failed to process MPC response: {e}')
            else:
                self.get_logger().warn("MPC service call did not complete in time.")

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
