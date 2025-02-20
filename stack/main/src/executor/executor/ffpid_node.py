import os
from scipy.interpolate import interp1d

import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import rclpy                                                # type: ignore
from rclpy.node import Node                                 # type: ignore
from rclpy.qos import QoSProfile                            # type: ignore

from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies
from interfaces.srv import ControlSolver


@jax.jit
def check_control_inputs(u_opt, u_opt_previous):
    """
    Check control inputs for safety constraints, rejecting vector norms that are too large.
    """
    tip_range, mid_range, base_range = 0.45, 0.35, 0.3

    u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

    # First we clip to max and min values
    u1 = jnp.clip(u1, -tip_range, tip_range)
    u6 = jnp.clip(u6, -tip_range, tip_range)
    u2 = jnp.clip(u2, -mid_range, mid_range)
    u4 = jnp.clip(u5, -mid_range, mid_range)
    u3 = jnp.clip(u3, -base_range, base_range)
    u5 = jnp.clip(u4, -base_range, base_range)

    # Compute control input vectors
    u1_vec = u1 * jnp.array([-jnp.cos(15 * jnp.pi/180), jnp.sin(15 * jnp.pi/180)])
    u2_vec = u2 * jnp.array([jnp.cos(45 * jnp.pi/180), jnp.sin(45 * jnp.pi/180)])
    u3_vec = u3 * jnp.array([-jnp.cos(15 * jnp.pi/180), -jnp.sin(15 * jnp.pi/180)])
    u4_vec = u4 * jnp.array([-jnp.cos(45 * jnp.pi/180), jnp.sin(45 * jnp.pi/180)])
    u5_vec = u5 * jnp.array([jnp.cos(75 * jnp.pi/180), -jnp.sin(75 * jnp.pi/180)])
    u6_vec = u6 * jnp.array([-jnp.cos(75 * jnp.pi/180), -jnp.sin(75 * jnp.pi/180)])

    # Calculate the norm based on the constraint
    vector_sum = (
        0.75 * (u3_vec + u5_vec) +
        1.0 * (u2_vec + u4_vec) +
        1.4 * (u1_vec + u6_vec)
    )
    norm_value = jnp.linalg.norm(vector_sum)

    # Check the constraint: if the constraint is met, then keep previous control command
    u_opt = jnp.where(norm_value > 0.8, u_opt_previous, jnp.array([u1, u2, u3, u4, u5, u6]))

    return u_opt


class FFPIDNode(Node):
    """
    This node is responsible for running MPC.
    """
    def __init__(self):
        super().__init__('mpc_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
        ])

        self.debug = self.get_parameter('debug').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        
        self.n_u = 6

        # Settled positions of the rigid bodies
        self.rest_position = jnp.array([0.10056, -0.10541, 0.10350,
                                        0.09808, -0.20127, 0.10645,
                                        0.09242, -0.31915, 0.09713])

        # Subscribe to current positions
        self.mocap_subscription = self.create_subscription(
            TrunkRigidBodies,
            '/trunk_rigid_bodies',
            self.mocap_listener_callback,
            QoSProfile(depth=10)
        )

        # Create FFPID solver service client
        self.ffpid_client = self.create_client(
            ControlSolver,
            'ffpid_solver'
        )
        self.get_logger().info('FFPID client created.')
        while not self.ffpid_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('FFPID solver not available, waiting...')
        
        # Request message definition
        self.req = ControlSolver.Request()

        # Create publisher to execute found control inputs
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )

        # Maintain previous control inputs
        self.uopt_previous = jnp.zeros(self.n_u)

        # JIT compile this function
        check_control_inputs(jnp.zeros(self.n_u), self.uopt_previous)

        # Generate reference trajectory
        z_ref, t_ref = self.generate_ref_trajectory(10, 0.01, 'circle', 0.1)
        self.z_interp = interp1d(t_ref, z_ref, axis=0,
                                 bounds_error=False, fill_value=(z_ref[0, :], z_ref[-1, :]))
        self.T = t_ref[-1]
        self.clock = self.get_clock()
        self.start_time = self.clock.now().nanoseconds / 1e9

        self.get_logger().info(f'FFPID node has been started.')

    def mocap_listener_callback(self, msg):
        """
        Callback to process mocap data, updating the latest observation.
        """
        if self.debug:
            self.get_logger().info(f'Received mocap data: {msg.positions}.')

        # Unpack the message into simple list of positions, eg [x1, y1, z1, x2, y2, z2, ...]
        y_new = jnp.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])
        y_centered = y_new - self.rest_position
        
        # The tip x and z positions is what we want to track
        y_tip_xz = y_centered[jnp.array([-3, -1])]
        self.req.y0 = y_tip_xz.flatten().tolist()

        t0 = self.clock.now().nanoseconds / 1e9 - self.start_time

        if t0 > self.T:
            self.get_logger().info(f'Trajectory is finished! At {(self.clock.now().nanoseconds / 1e9 - self.start_time):.3f}')
            self.destroy_node()
            rclpy.shutdown()
        else:
            z = self.z_interp(t0)
            self.req.z = z.flatten().tolist()

            # Send the request
            self.future = self.ffpid_client.call_async(self.req)
            self.future.add_done_callback(self.service_callback)

    def service_callback(self, async_response):
        """
        Callback that defines what happens when the MPC solver node returns a result.
        """
        try:
            response = async_response.result()

            safe_control_inputs = check_control_inputs(jnp.array(response.uopt[:self.n_u]), self.uopt_previous)
            self.publish_control_inputs(safe_control_inputs.tolist())

            self.uopt_previous = safe_control_inputs
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def publish_control_inputs(self, control_inputs):
        """
        Publish the control inputs.
        """
        control_message = AllMotorsControl()
        control_message.motors_control = [
            SingleMotorControl(mode=0, value=value) for value in control_inputs
        ]
        self.controls_publisher.publish(control_message)
        if self.debug:
            self.get_logger().info(f'Published new motor control setting: {control_inputs}.')

    def generate_ref_trajectory(self, T, dt, traj_type, size):
        """
        Generate a reference trajectory for the system to track.
        """
        t = jnp.linspace(0, T, int(T/dt))
        z_ref = jnp.zeros((len(t), 2))

        if traj_type == 'circle':
            z_ref = z_ref.at[:, 0].set(size * (jnp.cos(2 * jnp.pi / T * t)))
            z_ref = z_ref.at[:, 1].set(size * jnp.sin(2 * jnp.pi / T * t))
        elif traj_type == 'figure_eight':
            z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(size * jnp.sin(4 * jnp.pi / T * t))
        elif traj_type == 'periodic_line':
            m = -1
            z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(m * size * jnp.sin(2 * jnp.pi / T * t))
        elif traj_type == 'arc':
            m = -1
            l_trunk = 0.35
            z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(m * size * jnp.sin(2 * jnp.pi / T * t))
        else:
            raise ValueError('Invalid trajectory type: ' + traj_type + '. Valid options are: "circle" or "figure_eight".')
        return z_ref, t


def main(args=None):
    """
    Run the ROS2 node with single-threaded executor. 
    """
    rclpy.init(args=args)
    ffpid_node = FFPIDNode()
    rclpy.spin(ffpid_node)
    ffpid_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
