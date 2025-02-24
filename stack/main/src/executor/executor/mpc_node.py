import os
import csv
import time
from threading import Lock

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

from controller.mpc_solver_node import jnp2arr              # type: ignore
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


class MPCNode(Node):
    """
    This node is responsible for running MPC.
    """
    def __init__(self):
        super().__init__('mpc_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # print debug messages
            ('n_z', 3),                                     # number of performance vars
            ('n_u', 6),                                     # number of control inputs
            ('n_obs', 3),                                   # 2D, 3D or 6D observations
            ('n_delay', 3),                                 # number of delays applied to observations
            ('n_exec', 2),                                  # number of control inputs to execute from MPC solution
            ('results_name', 'test_experiment')             # name of the results file
        ])

        self.debug = self.get_parameter('debug').value
        self.n_z = self.get_parameter('n_z').value
        self.n_u = self.get_parameter('n_u').value
        self.n_obs = self.get_parameter('n_obs').value
        self.n_delay = self.get_parameter('n_delay').value
        self.n_exec = self.get_parameter('n_exec').value
        self.results_name = self.get_parameter('results_name').value

        # Initialize the CSV file
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        self.results_file = os.path.join(self.data_dir, f"trajectories/closed_loop/{self.results_name}.csv")
        self.initialize_csv()

        # Collect buffer of control inputs for multiple executions
        self.control_buffer = []
        self.buffer_index = 0
        self.buffer_lock = Lock()
        
        # We perform smoothing to handle initial transients
        self.alpha_smooth = 0.3
        self.smooth_control_inputs = jnp.zeros(self.n_u)

        # Size of observations vector
        self.n_y = self.n_obs * (self.n_delay + 1)

        # Settled positions of the rigid bodies
        self.rest_position = jnp.array([0.1018, -0.1075, 0.1062,
                                        0.1037, -0.2055, 0.1148,
                                        0.1025, -0.3254, 0.1129])
        
        # Execution occurs in multiple threads
        self.callback_group = ReentrantCallbackGroup()

        # Subscribe to current positions
        self.mocap_subscription = self.create_subscription(
            TrunkRigidBodies,
            '/trunk_rigid_bodies',
            self.mocap_listener_callback,
            QoSProfile(depth=3),
            callback_group=self.callback_group
        )

        # Create MPC solver service client
        self.mpc_client = self.create_client(
            ControlSolver,
            'mpc_solver',
            callback_group=self.callback_group
        )
        self.get_logger().info('MPC client created.')
        while not self.mpc_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('MPC solver not available, waiting...')
        
        # Request message definition
        self.req = ControlSolver.Request()

        # Create publisher to execute found control inputs
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=3)
        )

        # Maintain current observations because of the delay embedding
        self.latest_y = None

        # Maintain previous control inputs
        self.u_previous = jnp.zeros(self.n_u)

        self.clock = self.get_clock()

        # Need some initialization
        self.initialized = False

        # Initialize by calling mpc callback function
        self.mpc_callback()

        # JIT compile this function
        check_control_inputs(jnp.zeros(self.n_u), self.u_previous)

        # Create timer to receive MPC results at fixed frequency
        self.controller_period = 0.025
        self.mpc_exec_timer = self.create_timer(
                    self.controller_period,
                    self.mpc_callback,
                    callback_group=self.callback_group)
        
        # Timer for executing buffered controls
        self.buffer_execution_period = 0.01  # 100 Hz, same as dt in MPC
        self.buffer_timer = self.create_timer(
            self.buffer_execution_period,
            self.execute_buffer_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info(f'MPC node has been started with controller frequency: {1/self.controller_period:.2f} [Hz].')

        # Define reference time
        self.start_time = self.clock.now().nanoseconds / 1e9

    def mocap_listener_callback(self, msg):
        """
        Callback to process mocap data, updating the latest observation.
        """
        if self.debug:
            self.get_logger().info(f'Received mocap data: {msg.positions}.')

        # Unpack the message into simple list of positions, eg [x1, y1, z1, x2, y2, z2, ...]
        y_new = jnp.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])
        y_centered = y_new - self.rest_position

        # Subselect tip
        y_centered_tip = y_centered[-3:]

        # Update the current observations, including delay embeddings
        if self.latest_y is None:
            # At initialization use current obs. as delay embedding
            self.latest_y = jnp.tile(y_centered_tip, self.n_delay + 1)
            self.start_time = self.clock.now().nanoseconds / 1e9
        else:
            self.latest_y = jnp.concatenate([y_centered_tip, self.latest_y[:-self.n_z]])
        
        self.t0 = self.clock.now().nanoseconds / 1e9 - self.start_time

    def execute_buffer_callback(self):
        """
        Execute the next control input from the buffer.
        """
        with self.buffer_lock:
            if not self.control_buffer or self.buffer_index >= len(self.control_buffer):
                return
                            
            control_inputs = self.control_buffer[self.buffer_index]
            safe_control_inputs = check_control_inputs(control_inputs, self.u_previous)
            self.smooth_control_inputs = (1 - self.alpha_smooth) * safe_control_inputs + self.alpha_smooth * self.smooth_control_inputs

            if self.debug:
                self.get_logger().info(f'Executing buffer index {self.buffer_index} of {len(self.control_buffer)}')
            
            self.buffer_index += 1
        
        self.publish_control_inputs(self.smooth_control_inputs.tolist())
        self.u_previous = self.smooth_control_inputs
        

    def mpc_callback(self):
        """
        Receive MPC results at a fixed rate.
        """
        if not self.initialized:
            self.y0 = jnp.zeros(self.n_y)
            self.send_request(0.0, self.y0, self.u_previous, wait=True)
            self.future.add_done_callback(self.service_callback)
            self.initialized = True
        elif self.latest_y is not None:
            self.y0 = self.latest_y
            self.send_request(self.t0, self.y0, self.u_previous, wait=False)
            self.future.add_done_callback(self.service_callback)

    def send_request(self, t0, y0, u0, wait=False):
        """
        Send request to MPC solver.
        """
        self.req.t0 = t0
        self.req.y0 = jnp2arr(y0)
        self.req.u0 = jnp2arr(u0)
        self.future = self.mpc_client.call_async(self.req)

        if wait:
            # Synchronous call, not compatible for real-time applications
            rclpy.spin_until_future_complete(self, self.future)

    def service_callback(self, async_response):
        """
        Callback that defines what happens when the MPC solver node returns a result.
        """
        try:
            response = async_response.result()

            if response.done:
                self.get_logger().info(f'Trajectory is finished! At {(self.clock.now().nanoseconds / 1e9 - self.start_time):.3f}')
                self.destroy_node()
                rclpy.shutdown()
            else:
                # Store the optimized control inputs in the buffer for execution
                new_buffer = []
                for i in range(self.n_exec):
                    new_buffer.append(jnp.array(response.uopt[i*self.n_u:(i+1)*self.n_u]))
                with self.buffer_lock:
                    self.control_buffer = new_buffer
                    self.buffer_index = 0

                # Save to csv file
                self.save_to_csv(response.t, response.xopt, response.uopt, response.zopt, self.y0[:self.n_y])
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

    def initialize_csv(self):
        """
        Initialize the CSV file with headers.
        """
        with open(self.results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['topt', 'xopt', 'uopt', 'zopt', 'y'])

    def save_to_csv(self, topt, xopt, uopt, zopt, y):
        """
        Save optimized quantities by MPC and observations to CSV file.
        """
        with open(self.results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([list(topt), list(xopt), list(uopt), list(zopt), y.tolist()])


def main(args=None):
    """
    Run the ROS2 node with multi-threaded executor. 
    """
    rclpy.init(args=args)
    mpc_node = MPCNode()

    executor = MultiThreadedExecutor(num_threads=6)
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
