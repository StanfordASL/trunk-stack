import socket
import pickle
import jax
import jax.numpy as jnp
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile
from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies

MPC_HOST = 'localhost'
MPC_PORT = 12345

def transform_states(y):
    # TODO: Translation and axis swapping
    return y

def transform_controls(u):
    # TODO: Reorder correctly for trun
    return u


class SocketMPCNode(Node):
    def __init__(self):
        super().__init__('socket_mpc_node')

        self.current_state = None
        self.current_time = None

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((MPC_HOST, MPC_PORT))

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

        self.create_timer(0.01, self.control_callback) # 100 Hz control rate
        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def socket_send(self, t, x):
        data = {'state': x, 'time': t}
        self.socket.sendall(pickle.dumps(data))
        
    def socket_receive(self):
        response = pickle.loads(self.socket.recv(1024))
        return jnp.array(response['control'])
        
    def mocap_callback(self, msg):
        y_new = jnp.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])
        self.current_state = transform_states(y_new)
        self.current_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time

    def publish_control(self, u_opt):
        msg = AllMotorsControl()
        msg.motors_control = [SingleMotorControl(mode=0, value=float(v)) for v in u_opt]
        self.publisher.publish(msg)

    def control_callback(self):
        if self.current_state is None or self.current_time is None:
            self.get_logger().warn('Current state or time not set, skipping control callback.')
            return
        try:
            self.socket_send(self.current_time, self.current_state)
            u_opt = self.socket_receive()
            self.publish_control(u_opt)
            
        except Exception as e:
            self.get_logger().error(f'Socket communication failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SocketMPCNode()

    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
