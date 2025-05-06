#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time

class SenderNode(Node):
    def __init__(self):
        super().__init__('sender_node')
        self.publisher = self.create_publisher(Float64, '/ping', 10)
        self.subscription = self.create_subscription(
            Float64,  # Acknowledge message (Float64)
            '/ack',  # Acknowledgment topic
            self.acknowledgment_callback,
            10
        )
        self.timer = self.create_timer(0.1, self.publish_message)  # Publish at 10 Hz
        self.get_logger().info("Sender Node started, sending messages...")

        # To track latency
        self.send_time = None
        self.average_latency = 0
        self.counter= 0

    def publish_message(self):
        msg = Float64()
        msg.data = time.time()

        # Publish the message with header
        self.publisher.publish(msg)

    def acknowledgment_callback(self, msg):
        # Calculate round-trip latency
        latency = time.time() - msg.data
        self.average_latency += latency
        self.counter +=1
        self.get_logger().info(f"Received acknowledgment. Round-trip latency: {self.average_latency/self.counter} seconds")

def main(args=None):
    rclpy.init(args=args)
    node = SenderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
