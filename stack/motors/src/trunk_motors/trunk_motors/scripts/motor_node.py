#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
# import time
# from std_msgs.msg import Float64MultiArray
from motor_utils.dynamixel_client_motor import DynamixelClient
from interfaces.msg import AllMotorsControl, TrunkRigidBodies
from interfaces.msg import AllMotorsStatus


class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')
        self.MPC_SECURITY_MODE = True  # True if MPC is running

        # Execution occurs in multiple threads
        self.callback_group = ReentrantCallbackGroup()

        # CHANGE THIS WHENEVER TENDONS ARE RE-TENSIONED
        self.rest_positions = np.array([193.0, 189.0, 186.0, 183.0, 187.0, 204])
        self.motor_ids = [1, 2, 3, 4, 5, 6]  # all 6 trunk motors

        # Define a safe region to operate the motors in (position and velocity):
        self.limits_safe = np.array([51, 81, 31, 81, 31, 51])
        self.delta_limits_safe = np.array([0, 0, 0, 0, 0, 0])  # TODO: discuss values with Mark

        self.last_motor_positions = None

        # initialize motors client
        self.dxl_client = DynamixelClient(motor_ids=self.motor_ids, port='/dev/ttyUSB0')

        # connect to motors
        self.dxl_client.connect()
        self.get_logger().info('Connected to motors!')

        # enable torque
        self.dxl_client.set_torque_enabled(self.motor_ids, True)
        self.get_logger().info('Torque enabled!')

        # subscriber to receive control commands
        self.controls_subscriber = self.create_subscription(
            AllMotorsControl,
            '/all_motors_control',
            self.command_positions,  # callback function
            10,
            callback_group=self.callback_group
        )

        if self.MPC_security_mode:
            # Subscribe to current positions
            self.mocap_subscription = self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.mocap_listener_callback,
                QoSProfile(depth=3),
                callback_group=self.callback_group
            )

        # create status publisher
        self.status_publisher = self.create_publisher(
            AllMotorsStatus, 
            '/all_motors_status', 
            10
        )

        self.rest_position_trunk = np.array([0.1018, -0.1075, 0.1062,
                                            0.1037, -0.2055, 0.1148,
                                            0.1025, -0.3254, 0.1129])

        self.timer = self.create_timer(1.0/100, self.read_status)  # publish at 100Hz

        # read out initial positions
        self.get_logger().info('Initial motor status: ')
        positions = self.read_status()
        positions_raw = positions + self.rest_positions

        for idx, id in enumerate(self.motor_ids):
            self.get_logger().info(f"Motor {id} position: {positions[idx]:.2f} degrees")  # display in degrees
        for idx, id in enumerate(self.motor_ids):
            self.get_logger().info(f"Motor {id} raw position: {positions_raw[idx]:.2f} degrees")  # display in degrees

        self.get_logger().info('Motor control node initialized!')

    def command_positions(self, msg):
        # commands new positions to the motors
        positions = msg.motors_control
        positions = np.array(positions)

        if self.last_motor_positions is None:
            delta_positions = np.zeros_like(positions)
        else:
            delta_positions = positions - self.last_motor_positions

        # calculate relative position change
        mask_delta_low = delta_positions < -self.delta_limits_safe
        mask_delta_high = delta_positions > self.delta_limits_safe

        # inputs from ROS message are zero centered, need to center them about rest positions before sending to motor
        mask_low = positions < -self.limits_safe
        mask_high = positions > self.limits_safe

        if np.any(mask_low | mask_high) or self.MPC_SECURITY_MODE and np.any(mask_delta_low | mask_delta_high):
            bad_idxs = np.where(mask_low | mask_high)[0]
            bad_vals = positions[bad_idxs]
            self.get_logger().error(
                f"Unsafe motor commands at indices {bad_idxs.tolist()}: {bad_vals.tolist()}. "
                "Shutting down without sending to motors."
            )

            # clean up torque + disconnect
            self.shutdown()

            # signal ROS to exit
            rclpy.shutdown()
            return

        self.last_motor_positions = positions
        # inputs from ROS message are zero centered, need to center them about rest positions before sending to motor
        positions += self.rest_positions

        positions *= np.pi/180  # receives a position in degrees, convert to radians for dynamixel sdk

        self.dxl_client.write_desired_pos(self.motor_ids, positions)

    def mocap_listener_callback(self, msg):
        """
        Callback to process mocap data, updating the latest observation.
        """
        if self.debug:
            self.get_logger().info(f'Received mocap data: {msg.positions}.')

        # Unpack the message into simple list of positions, eg [x1, y1, z1, x2, y2, z2, ...]
        y_new = np.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])
        y_centered = y_new - self.rest_position_trunk

        # Subselect tip
        y_centered_tip = y_centered[-3:]

        # TODO: discuss value with Mark

        if np.linalg.norm(y_centered_tip) > 0.0:
            self.get_logger().error(
                f"Unsafe trunk position at value commands at indices {np.linalg.norm(y_centered_tip)}. "
                "Shutting down the motor node."
            )

            # clean up torque + disconnect
            self.shutdown()

            # signal ROS to exit
            rclpy.shutdown()
            return

        return

    def read_status(self):
        # reads and publishes the motor position, velocity, and current
        positions, velocities, currents = self.dxl_client.read_pos_vel_cur()

        positions *= 180/np.pi  # dynamixel position in rad, convert to degrees
        positions -= self.rest_positions  # motor sends real positions, we want to read zero centered positions

        msg = AllMotorsStatus()
        msg.positions = positions.tolist()
        msg.velocities = velocities.tolist()  # TODO: determine if in rpm (should be)
        msg.currents = currents.tolist()  # TODO: determine if in mA (should be)

        self.status_publisher.publish(msg)
        return positions

    def shutdown(self):
        # cleanup before shutdown
        self.get_logger().info("Disabling torque and disconnecting motors")
        self.dxl_client.set_torque_enabled(self.motor_ids, False)
        self.dxl_client.disconnect()


def main():
    rclpy.init()
    node = MotorNode()
    try:
        # checks to see maximum read and write rate of the dynamixels ~20Hz read, ~124Hz write
        # # READ rate check
        # dxl = node.dxl_client
        # dxl.motor_ids = [1, 2, 3, 4, 5, 6]
        # N = 100
        # start = time.time()
        # for _ in range(N):
        #     dxl.read_pos_vel_cur()
        # end = time.time()
        # node.get_logger().info(f"READ avg rate: {N / (end - start):.2f} Hz")

        # # WRITE rate check
        # positions = [198.0, 204.0, 189.0, 211.0, 200.0, 192.0]
        # positions = [np.pi/180 * pos for pos in positions] # receives a position in degrees, convert to radians for dynamixel sdk
        # start = time.time()
        # for _ in range(N):
        #     dxl.write_desired_pos(dxl.motor_ids, np.array(positions))

        # end = time.time()
        # node.get_logger().info(f"WRITE avg rate: {N / (end - start):.2f} Hz")

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
