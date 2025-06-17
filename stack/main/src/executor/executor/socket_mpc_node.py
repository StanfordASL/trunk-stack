import jax.numpy as jnp
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from interfaces.msg import AllMotorsControl, TrunkRigidBodies, AllMotorsStatus

from .utils.socket_utils import send_state, recv_control, setup_socket_client

MPC_HOST = 'localhost'
MPC_PORT = 12345


class SocketMPCNode(Node):
    def __init__(self):
        super().__init__('socket_mpc_node')
        self.rate = 0.02  # Control rate in seconds (50 Hz)
        self.avoid_pid = True  # Set to True to mitigate P effects in PID motor control
        self.limits = jnp.array([51, 81, 31, 81, 31, 51])  # Safe limits for motor positions
        
        self.current_state = None
        self.current_time = None
        self.last_motor_angles = None
        
        self.socket = setup_socket_client(MPC_HOST, MPC_PORT)

        self.publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=3)
        )
        self.create_subscription(
            TrunkRigidBodies,
            '/trunk_rigid_bodies',
            self.mocap_callback,
            QoSProfile(depth=3)
        )
        self.subscription_angles = self.create_subscription(
            AllMotorsStatus,
            '/all_motors_status',
            self.motor_angles_callback,
            QoSProfile(depth=10)
        )

        self.create_timer(self.rate, self.control_callback)
        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def motor_angles_callback(self, msg): 
        self.last_motor_angles = msg.positions

    def mocap_callback(self, msg):
        self.current_state = jnp.array([[pos.x, pos.y, pos.z] for pos in msg.positions])
        self.current_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time

    def publish_control(self, u_opt):
        if self.avoid_pid:
            u_opt = u_opt + self.last_motor_angles

        msg = AllMotorsControl()
        msg.motors_control = tuple(u_opt.tolist())
        self.publisher.publish(msg)

    def control_callback(self):
        if self.current_state is None or self.current_time is None:
            self.get_logger().warn('Current state or time not set, skipping control callback.')
            return
        
        if self.last_motor_angles is None:
            self.get_logger().warn('Last motor angles not set, skipping control callback.')
            return
        
        now = self.get_clock().now().nanoseconds / 1e9
        send_state(self.socket, self.current_time, self.current_state, self.last_motor_angles)
        
        u_opt = recv_control(self.socket)
        delta_time = self.get_clock().now().nanoseconds / 1e9 - now

        if delta_time > self.rate:
            self.get_logger().warn(f'Control callback took too long: {delta_time:.4f} seconds, expected ~0.01 seconds.')

        assert jnp.all(jnp.abs(u_opt) <= self.limits), "Control exceeds limits"
        
        self.publish_control(u_opt)


def main(args=None):
    while True:
        rclpy.init(args=args)
        node = SocketMPCNode()
        try:
            rclpy.spin(node)
        except ConnectionError:
            node.get_logger().warn('Socket disconnected, restarting node...')
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard interrupt, shutting down.')
            break
        except Exception as e:
            node.get_logger().error(f'An error occurred: {e}')
            break
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
