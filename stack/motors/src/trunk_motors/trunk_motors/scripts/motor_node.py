#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float64MultiArray
from motor_utils.dynamixel_client_motor import DynamixelClient
from interfaces.msg import AllMotorsControl
from interfaces.msg import AllMotorsStatus


class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')
        self.control_augmented = False # True if running Paul's control augmented data collected, False else

        if self.control_augmented:
            # self.rest_positions = np.array([196.0, 201.0, 193.0, 210.0, 202.0, 197]) update these
            self.motor_ids = [1, 2, 3, 4, 5, 6]
        else:
            self.rest_positions = np.array([193.0, 189.0, 186.0, 183.0, 187.0, 204]) # CHANGE THIS WHENEVER TENDONS ARE RE-TENSIONED
            self.motor_ids = [1, 2, 3, 4, 5, 6] # all 6 trunk motors

        # Define a safe region to operate the motors in:
        self.limits_safe = np.array([51, 81, 31, 81, 31, 51])

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
            self.command_positions, # callback function
            10
        )

        # create status publisher
        self.status_publisher = self.create_publisher(
            AllMotorsStatus, 
            '/all_motors_status', 
            10
        )
        self.timer = self.create_timer(1.0/100, self.read_status) # publish at 100Hz 

        # read out initial positions
        self.get_logger().info('Initial motor status: ')
        positions = self.read_status()
        positions_raw = positions + self.rest_positions

        for idx, id in enumerate(self.motor_ids):
            self.get_logger().info(f"Motor {id} position: {positions[idx]:.2f} degrees") # display in degrees
        for idx, id in enumerate(self.motor_ids):
            self.get_logger().info(f"Motor {id} raw position: {positions_raw[idx]:.2f} degrees") # display in degrees

        self.get_logger().info('Motor control node initialized!')
        

    def command_positions(self, msg):
        # commands new positions to the motors
        positions = msg.motors_control
        positions = np.array(positions)

        # inputs from ROS message are zero centered, need to center them about rest positions before sending to motor
        mask_low = positions < -self.limits_safe
        mask_high = positions > self.limits_safe
        if np.any(mask_low | mask_high):
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
            
        positions += self.rest_positions # inputs from ROS message are zero centered, need to center them about rest positions before sending to motor

        positions *= np.pi/180 # receives a position in degrees, convert to radians for dynamixel sdk
        self.dxl_client.write_desired_pos(self.motor_ids, positions)

        # for idx, id in enumerate(self.motor_ids):
        #     self.get_logger().info(f"commanded motor {id} to {positions[idx]*180/np.pi:.2f} degrees")
        

    def read_status(self):
        # reads and publishes the motor position, velocity, and current
        positions, velocities, currents = self.dxl_client.read_pos_vel_cur()

        positions *= 180/np.pi # dynamixel position in rad, convert to degrees
        positions -= self.rest_positions # motor sends real positions, we want to read zero centered positions

        msg = AllMotorsStatus()
        msg.positions = positions.tolist()
        msg.velocities = velocities.tolist() # TODO: determine if in rpm (should be)
        msg.currents = currents.tolist() # TODO: determine if in mA (should be)

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