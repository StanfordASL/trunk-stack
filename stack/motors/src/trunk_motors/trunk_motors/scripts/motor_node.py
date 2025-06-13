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


class DummyMotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')
        # match original parameter name
        self.declare_parameter('secure_mode', True)
        self.MPC_SECURITY_MODE = self.get_parameter('secure_mode').value

        # allow callbacks in parallel
        self.callback_group = ReentrantCallbackGroup()

        # subscribe to control commands
        self.create_subscription(
            AllMotorsControl,
            '/all_motors_control',
            self.command_positions,
            10,
            callback_group=self.callback_group
        )

        # subscribe to mocap if in secure mode
        if self.MPC_SECURITY_MODE:
            self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.mocap_listener_callback,
                QoSProfile(depth=3),
                callback_group=self.callback_group
            )

        # publisher for motor status
        self.status_publisher = self.create_publisher(
            AllMotorsStatus,
            '/all_motors_status',
            10
        )

        # keep track of last commanded positions
        self.last_motor_positions = np.zeros(6)

        # publish dummy status at 100 Hz
        self.timer = self.create_timer(1.0/100.0, self.read_status)

        self.get_logger().info(
            'Dummy motor node initialized: publishing zeroed states at 100 Hz'
        )

    def command_positions(self, msg: AllMotorsControl):
        # record last commanded positions, but do not send to hardware
        positions = np.array(msg.motors_control, dtype=float)
        if positions.shape[0] == 6:
            self.last_motor_positions = positions
        else:
            self.get_logger().warn(
                f"Expected 6 motor commands, got {positions.shape[0]}; ignoring extras."
            )

    def mocap_listener_callback(self, msg: TrunkRigidBodies):
        # mimic original interface; no-op for dummy
        # msg.positions is a list of geometry_msgs/Point
        # you could log or perform checks here if desired
        return

    def read_status(self):
        # always publish zeros for 6 motors
        zeros = np.zeros(6, dtype=float)
        status = AllMotorsStatus()
        status.positions = zeros.tolist()
        status.velocities = zeros.tolist()
        status.currents = zeros.tolist()
        self.status_publisher.publish(status)
        return zeros

    def shutdown(self):
        # no hardware to clean up in dummy
        self.get_logger().info('Shutting down dummy motor node.')


class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')
        self.declare_parameters(namespace='', parameters=[
            ('secure_mode', True)
        ])

        self.MPC_SECURITY_MODE = self.get_parameter('secure_mode').value  # True if MPC is running

        # Execution occurs in multiple threads
        self.callback_group = ReentrantCallbackGroup()

        # CHANGE THIS WHENEVER TENDONS ARE RE-TENSIONED
        self.rest_positions = np.array([193.0, 189.0, 186.0, 183.0, 187.0, 204.0])
        self.motor_ids = [1, 2, 3, 4, 5, 6]  # all 6 trunk motors

        # Define a safe region to operate the motors in (position and velocity):
        self.limits_safe = np.array([51, 81, 31, 81, 31, 51])
        self.delta_limits_safe = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

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

        if self.MPC_SECURITY_MODE:
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

        self.rest_position_trunk = np.array([0.10753094404935837, -0.11212190985679626, 0.10474388301372528,
                                            0.10156622529029846, -0.20444495975971222, 0.11144950985908508,
                                            0.10224875807762146, -0.3151078522205353, 0.10935673117637634])

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

        if self.MPC_SECURITY_MODE:
            print(f"Positions: {positions}, Delta Positions: {delta_positions}")
        
        if np.any(mask_low | mask_high) or self.MPC_SECURITY_MODE and np.any(mask_delta_low | mask_delta_high):
            bad_idxs = np.where(mask_low | mask_high)[0]
            bad_idxs_delta = np.where(mask_delta_low | mask_delta_high)[0]
            bad_vals = positions[bad_idxs]
            bad_vals_delta = delta_positions[bad_idxs_delta]

            self.get_logger().error(
                f"Unsafe motor commands at indices {bad_idxs.tolist()}: {bad_vals.tolist()}. "
                f"Unsafe delta commands at indices {bad_idxs_delta.tolist()}: {bad_vals_delta.tolist()}. "
                "Shutting down without sending to motors."
            )

            # clean up torque + disconnect
            self.shutdown()

            # signal ROS to exit
            rclpy.shutdown()
            return

        self.last_motor_positions = positions.copy()
        # inputs from ROS message are zero centered, need to center them about rest positions before sending to motor
        positions += self.rest_positions

        positions *= np.pi/180  # receives a position in degrees, convert to radians for dynamixel sdk

        self.dxl_client.write_desired_pos(self.motor_ids, positions)

    def mocap_listener_callback(self, msg):
        """
        Callback to process mocap data, updating the latest observation.
        """
        # Unpack the message into simple list of positions, eg [x1, y1, z1, x2, y2, z2, ...]
        y_new = np.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])
        y_centered = y_new - self.rest_position_trunk

        # Subselect tip
        y_centered_tip = y_centered[-3:]

        if np.linalg.norm(y_centered_tip) > 0.1:
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
    # node = DummyMotorNode()
    node = MotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
