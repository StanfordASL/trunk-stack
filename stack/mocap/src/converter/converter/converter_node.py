import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from interfaces.msg import TrunkMarkers, TrunkRigidBodies
from mocap4r2_msgs.msg import Markers, RigidBodies  # type: ignore

class ConverterNode(Node):
    def __init__(self):
        super().__init__('converter_node')

        self.declare_parameter('type', 'rigid_bodies')
        self.type = self.get_parameter('type').get_parameter_value().string_value

        if self.type == 'rigid_bodies':
            self.publisher = self.create_publisher(TrunkRigidBodies, '/trunk_rigid_bodies', 10)
            self.subscription = self.create_subscription(
                RigidBodies,
                '/rigid_bodies',
                self.rigid_body_callback,
                10)
        elif self.type == 'markers':
            self.publisher = self.create_publisher(TrunkMarkers, '/trunk_markers', 10)
            self.subscription = self.create_subscription(
                Markers,
                '/markers',
                self.marker_callback,
                10)
        else:
            self.get_logger().error('Invalid type parameter. Choose from {markers, rigid_bodies}. Exiting...')
            self.destroy_node()
            rclpy.shutdown()
            return
        self.get_logger().info('Mocap converter node has started.')

    def marker_callback(self, msg):
        trunk_msg = TrunkMarkers()
        trunk_msg.header = msg.header
        trunk_msg.frame_number = msg.frame_number
        trunk_msg.translations = [marker.translation for marker in msg.markers]
        self.publisher.publish(trunk_msg)
        # self.get_logger().info('Published TrunkMarkers message.')

    def rigid_body_callback(self, msg):
        trunk_msg = TrunkRigidBodies()
        trunk_msg.header = msg.header
        trunk_msg.frame_number = msg.frame_number
        trunk_msg.rigid_body_names = [rigid_body.rigid_body_name for rigid_body in msg.rigidbodies]
        trunk_msg.positions = [rigid_body.pose.position for rigid_body in msg.rigidbodies]
        self.publisher.publish(trunk_msg)
        # self.get_logger().info('Published TrunkRigidBodies message.')


def main(args=None):
    rclpy.init(args=args)
    node = ConverterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
