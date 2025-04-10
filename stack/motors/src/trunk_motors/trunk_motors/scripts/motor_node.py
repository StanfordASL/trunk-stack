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

        # initialize motors client
        self.motor_ids = [1, 2, 3, 4, 5, 6] # all 6 trunk motors
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

        # create status publisher and publish at 10Hz
        self.status_publisher = self.create_publisher(
            AllMotorsStatus, 
            '/all_motors_status', 
            10
        )
        self.timer = self.create_timer(1.0/10, self.read_status) # publish at 10Hz TODO: can we increase this (perhaps baud rate)? and what is our write frequency

        # read out initial positions
        self.get_logger().info('Initial motor status: ')
        positions, velocities, currents = self.read_status()
        for idx, id in enumerate(self.motor_ids):
            self.get_logger().info(f"Motor {id} position: {positions[idx]*180/np.pi:.2f} degrees") # display in degrees
            # self.get_logger().info(f"Motor {id} velocity: {velocities[idx]} ")
            # self.get_logger().info(f'Motor {id} current: {currents[idx]} ')

        self.get_logger().info('Motor control node initialized!')
        

    def command_positions(self, msg):
        # commands new positions to the motors
        positions = msg.motors_control
        positions = [np.pi/180 * pos for pos in positions] # receives a position in degrees, convert to radians for dynamixel sdk
        self.dxl_client.write_desired_pos(self.motor_ids, np.array(positions))

        for idx, id in enumerate(self.motor_ids):
            self.get_logger().info(f"commanded motor {id} to {positions[idx]*180/np.pi:.2f} degrees")
        

    def read_status(self):
        # reads and publishes the motor position, velocity, and current
        positions = self.dxl_client.read_pos() # returned in radians 
        positions = positions.tolist()
        
        velocities = self.dxl_client.read_vel() # returned in __ 
        velocities = velocities.tolist()

        currents = self.dxl_client.read_cur() # returned in __
        currents = currents.tolist()

        msg = AllMotorsStatus()
        msg.positions = [180/np.pi * pos for pos in positions] # dynamixel position in rad, publish in degrees
        msg.velocities = [1 * vel for vel in velocities] # TODO: determine if in rpm (should be)
        msg.currents = [1 * cur for cur in currents] # TODO: determine if in mA (should be)

        self.status_publisher.publish(msg)
        return (positions, velocities, currents)
        

    def shutdown(self):
        # cleanup before shutdown
        self.get_logger().info("Disabling torque and disconnecting motors")
        self.dxl_client.set_torque_enabled(self.motor_ids, False)
        self.dxl_client.disconnect()

def main():
    rclpy.init()
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