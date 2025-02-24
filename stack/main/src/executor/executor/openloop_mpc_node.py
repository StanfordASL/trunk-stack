import os
import ast
import csv
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import rclpy                                                # type: ignore
from rclpy.node import Node                                 # type: ignore
from rclpy.callback_groups import ReentrantCallbackGroup    # type: ignore
from rclpy.executors import MultiThreadedExecutor           # type: ignore
from rclpy.qos import QoSProfile                            # type: ignore

from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies


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


class MPCNode(Node):
    """
    This node is responsible for running MPC.
    """
    def __init__(self):
        super().__init__('mpc_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('n_x', 5),                                     # 5 (number of latent states)
            ('n_z', 3),                                     # 2 (number of performance vars)
            ('n_u', 6),                                     # 6 (number of control inputs)
            ('n_obs', 3),                                   # 2 (2D, 3D or 6D observations)
            ('n_delay', 3),                                 # 4 (number of delays applied to observations)
            ('N', 6),                                       # 6 (MPC horizon)
            ('experiment_name', 'mpc_Rdu005'),
        ])

        self.debug = self.get_parameter('debug').value
        self.n_x = self.get_parameter('n_x').value
        self.n_z = self.get_parameter('n_z').value
        self.n_u = self.get_parameter('n_u').value
        self.n_obs = self.get_parameter('n_obs').value
        self.n_delay = self.get_parameter('n_delay').value
        self.N = self.get_parameter('N').value
        self.experiment_name = self.get_parameter('experiment_name').value

        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        ol_mpc_file = os.path.join(self.data_dir, f'trajectories/test_mpc/{self.experiment_name}.csv')
        self.load_control_inputs(ol_mpc_file)
        self.traj_length = self.uopt.shape[0]
        self.current_control_id = -1

        # Initializations
        self.safe_control_input = jnp.zeros(6)
        self.stored_positions = []

        # Size of observations vector
        self.n_y = self.n_obs * (self.n_delay + 1)

        # Settled positions of the rigid bodies
        self.rest_position = jnp.array([0.1018, -0.1075, 0.1062,
                                        0.1037, -0.2055, 0.1148,
                                        0.1025, -0.3254, 0.1129])
        
        self.callback_group = ReentrantCallbackGroup()

        # Subscribe to current positions
        self.mocap_subscription = self.create_subscription(
            TrunkRigidBodies,
            '/trunk_rigid_bodies',
            self.mocap_listener_callback,
            QoSProfile(depth=10),
            callback_group=self.callback_group
        )

        # Create publisher to execute found control inputs
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )

        # Maintain previous control inputs
        self.uopt_previous = jnp.zeros(self.n_u)

        self.clock = self.get_clock()

        # JIT compile this function
        check_control_inputs(jnp.zeros(self.n_u), self.uopt_previous)

        # Create timer to execute MPC at fixed frequency
        self.controller_period = 0.04
        self.mpc_exec_timer = self.create_timer(
                    self.controller_period,
                    self.mpc_executor_callback,
                    callback_group=self.callback_group)

        self.get_logger().info(f'Open-loop MPC node has been started with control execution frequency: {1/self.controller_period:.2f} [Hz].')

    def load_control_inputs(self, filename):
        """
        Load control inputs found by MPC.
        """
        mpc_test_data = pd.read_csv(filename)
        topt_pd, xopt_pd, uopt_pd, zopt_pd = mpc_test_data['topt'], mpc_test_data['xopt'], mpc_test_data['uopt'], mpc_test_data['zopt']
        num_samples = len(topt_pd)
        topt_np = np.zeros((num_samples, self.N+1))
        xopt_np = np.zeros((num_samples, self.N+1, self.n_x))
        uopt_np = np.zeros((num_samples, self.N, self.n_u))
        zopt_np = np.zeros((num_samples, self.N+1, self.n_z))
        for sample_idx in range(num_samples):
            topt_list = ast.literal_eval(topt_pd[sample_idx].strip('"'))
            topt_np[sample_idx, :] = topt_list
            xopt_list = ast.literal_eval(xopt_pd[sample_idx].strip('"'))
            xopt_np[sample_idx, :] = np.array(xopt_list).reshape(-1, self.n_x)
            uopt_list = ast.literal_eval(uopt_pd[sample_idx].strip('"'))
            uopt_np[sample_idx, :] = np.array(uopt_list).reshape(-1, self.n_u)
            zopt_list = ast.literal_eval(zopt_pd[sample_idx].strip('"'))
            zopt_np[sample_idx, :] = np.array(zopt_list).reshape(-1, self.n_z)
        self.uopt = jnp.asarray(uopt_np)

    def mocap_listener_callback(self, msg):
        """
        Callback to process mocap data, updating the latest observation.
        """
        if self.debug:
            self.get_logger().info(f'Received mocap data: {msg.positions}.')

        if self.current_control_id == 0:
            self.obs_names = msg.rigid_body_names

        # Store current positions
        self.store_positions(msg)

    def mpc_executor_callback(self):
        """
        Execute control inputs found by MPC at fixed rate (open-loop).
        """
        self.current_control_id += 1
        if self.current_control_id >= self.traj_length:
            self.get_logger().info('Trajectory is finished!')
            self.process_data(self.obs_names)

            self.destroy_node()
            rclpy.shutdown()
        else:
            u = self.uopt[self.current_control_id, 0, :self.n_u].squeeze()
            safe_control_inputs = check_control_inputs(u, self.uopt_previous)
            self.publish_control_inputs(safe_control_inputs.tolist())

            self.uopt_previous = safe_control_inputs

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

    def store_positions(self, msg):
        self.stored_positions.append(msg.positions)

    def process_data(self, names):
        # Populate the header row of the CSV file with states if it does not exist
        recording_csv_file = os.path.join(self.data_dir, f'trajectories/open_loop/{self.experiment_name}.csv')
        if not os.path.exists(recording_csv_file):
            header = [f'{axis}{name}' for name in names for axis in ['x', 'y', 'z']]
            with open(recording_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
    
        # Store all positions in a CSV file
        with open(recording_csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for pos_list in self.stored_positions:
                row = [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]]
                writer.writerow(row)


def main(args=None):
    """
    Run the ROS2 node with multi-threaded executor. 
    """
    rclpy.init(args=args)
    mpc_node = MPCNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(mpc_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        mpc_node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        mpc_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
