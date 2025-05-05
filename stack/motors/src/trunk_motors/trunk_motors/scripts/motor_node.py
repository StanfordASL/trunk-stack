#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class ReceiverNode(Node):
    def __init__(self):
        super().__init__('receiver_node')
        self.subscription = self.create_subscription(
            Float64,  # Subscribe to the Bool message type
            '/ping',  # Topic from sender
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(Float64, '/ack', 10)  # Acknowledgment topic
        self.get_logger().info("Receiver Node started, waiting for messages...")

    def listener_callback(self, msg):
        # Upon receiving the message, send an acknowledgment back to sender
        self.get_logger().info(f"Received ping message, sending acknowledgment.")
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReceiverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
