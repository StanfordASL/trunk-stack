import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from interfaces.msg import TrunkMarkers, TrunkRigidBodies
from mocap4r2_msgs.msg import Markers, RigidBodies  # type: ignore
from geometry_msgs.msg import Point, Quaternion, Vector3


class DummyConverterNode(Node):
    def __init__(self):
        super().__init__('dummy_converter_node')

        self.declare_parameter('type', 'rigid_bodies')
        self.type = self.get_parameter('type').get_parameter_value().string_value

        if self.type == 'rigid_bodies':
            self.publisher = self.create_publisher(TrunkRigidBodies, '/trunk_rigid_bodies', 10)
        elif self.type == 'markers':
            self.publisher = self.create_publisher(TrunkMarkers, '/trunk_markers', 10)
        else:
            self.get_logger().error(
                f'Invalid type parameter "{self.type}". Choose "markers" or "rigid_bodies".'
            )
            rclpy.shutdown()
            return

        self.frame_number = 0
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100 Hz
        self.get_logger().info(f'Dummy converter publishing "{self.type}" at 100 Hz')

    def timer_callback(self):
        t = self.get_clock().now().to_msg()

        if self.type == 'rigid_bodies':
            msg = TrunkRigidBodies()
            msg.header.stamp = t
            msg.frame_number = self.frame_number

            # three dummy rigid bodies
            msg.rigid_body_names = ['body_1', 'body_2', 'body_3']
            msg.positions = [
                Point(x=0.0, y=0.0, z=0.0),
                Point(x=0.0, y=0.0, z=0.0),
                Point(x=0.0, y=0.0, z=0.0),
            ]
            msg.orientations = [
                Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ]

        else:  # markers
            msg = TrunkMarkers()
            msg.header.stamp = t
            msg.frame_number = self.frame_number

            # three dummy markers
            msg.translations = [
                Vector3(x=0.0, y=0.0, z=0.0),
                Vector3(x=0.0, y=0.0, z=0.0),
                Vector3(x=0.0, y=0.0, z=0.0),
            ]

        self.publisher.publish(msg)
        self.frame_number += 1


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

    def rigid_body_callback(self, msg):
        trunk_msg = TrunkRigidBodies()
        trunk_msg.header = msg.header
        trunk_msg.frame_number = msg.frame_number
        trunk_msg.rigid_body_names = [rigid_body.rigid_body_name for rigid_body in msg.rigidbodies]
        trunk_msg.positions = [rigid_body.pose.position for rigid_body in msg.rigidbodies]
        trunk_msg.orientations = [rigid_body.pose.orientation for rigid_body in msg.rigidbodies]
        self.publisher.publish(trunk_msg)


def main(args=None):
    rclpy.init(args=args)
    # TODO: Change that back when using the real MOCAB system again
    # node = ConverterNode()
    node = DummyConverterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
