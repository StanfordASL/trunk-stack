import numpy as np
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
        self.rate = 0.01  # Control rate in seconds (100 Hz)
        self.limits = np.array([51, 81, 31, 81, 31, 51])  # Safe limits for motor positions
        
        self.current_state = None
        self.current_velocity = None
        self.current_time = None
        self.last_state = None
        self.last_position = None

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
        self.last_motor_state = np.stack([
            np.array(msg.positions),
            np.array(msg.velocities),
            np.array(msg.currents)
        ], axis=1)

    def mocap_callback(self, msg):
        prev_state = self.current_state
        self.current_state = np.array([[pos.x, pos.y, pos.z] for pos in msg.positions])

        old_time = self.current_time
        self.current_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time

        if old_time is not None:
            dt = self.current_time - old_time
            if prev_state is not None and dt > 0:
                self.current_velocity = (self.current_state - prev_state) / dt

            if self.current_time - old_time > 0.02:  # If the time difference is greater than expected
                self.get_logger().warn(f'Mocap callback took too long: {self.current_time - old_time:.4f} seconds, expected ~{0.01} seconds.')
            
    def publish_control(self, u_opt):
        assert np.all(np.abs(u_opt) <= self.limits), "Control exceeds limits"

        msg = AllMotorsControl()
        msg.motors_control = tuple(u_opt.tolist())
        self.publisher.publish(msg)

    def control_callback(self):
        if self.current_state is None or self.current_time is None or self.current_velocity is None:
            self.get_logger().warn('Current state or time not set, skipping control callback.')
            return
        
        if self.last_motor_state is None:
            self.get_logger().warn('Last motor angles not set, skipping control callback.')
            return
        
        full_state = np.concatenate((self.current_state, self.current_velocity), axis=1)

        now = self.get_clock().now().nanoseconds / 1e9
        send_state(self.socket, self.current_time, full_state, self.last_motor_state)
        
        u_opt = recv_control(self.socket)
        delta_time = self.get_clock().now().nanoseconds / 1e9 - now

        if delta_time > self.rate:
            self.get_logger().warn(f'Control callback took too long: {delta_time:.4f} seconds, expected ~{self.rate} seconds.')
        
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
