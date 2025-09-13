import os
import csv
from threading import Lock
import time
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
from interfaces.msg import AllMotorsControl, TrunkRigidBodies
from interfaces.srv import ControlSolver

from .actuator import Actuator


config = {
    "actuator_lambda": [[-5.0, 0.0], [0.0, -5.5]]
}

@jax.jit
def _check_control_inputs_jit(u_opt):
    """
    JIT-safe core function to clip control inputs.
    
    scale = 1.0
    tip_range = 80 * scale
    mid_range = 50 * scale
    base_range = 30 * scale
    """

    tip_range = 80
    mid_range = 50
    base_range = 30

    u1, u2, u3, u4, u5, u6 = u_opt

    u1 = jnp.clip(u1, -mid_range, mid_range)
    u2 = jnp.clip(u2, -tip_range, tip_range)
    u3 = jnp.clip(u3, -base_range, base_range)
    u4 = jnp.clip(u4, -tip_range, tip_range)
    u5 = jnp.clip(u5, -base_range, base_range)
    u6 = jnp.clip(u6, -mid_range, mid_range)

    return jnp.array([u1, u2, u3, u4, u5, u6])


def check_control_inputs(u_opt, u_previous=None):
    """
    Wrapper that prints when clipping happens, while calling JIT-safe core logic.
    """
    u_opt_clipped = _check_control_inputs_jit(u_opt)

    # Compare and print if clipping happened
    diffs = jnp.abs(u_opt - u_opt_clipped)
    for i, diff in enumerate(diffs):
        if diff > 1e-6:
            print(f"[WARNING] u{i+1} was clipped from {float(u_opt[i]):.3f} to {float(u_opt_clipped[i]):.3f}")

    return u_opt_clipped


@jax.jit
def u2_to6u_mapping(u2, u4):
    # angle and radius
    teta = jnp.arctan2(u4, u2)
    r_scaling = jnp.hypot(u2, u4)

    # compute the six raw legs
    u3 = r_scaling * jnp.cos(teta - jnp.pi / 3)
    u5 = r_scaling * jnp.sin(teta - jnp.pi / 3)
    u1 = -r_scaling * jnp.sin(teta - jnp.pi / 6)
    u6 = -r_scaling * jnp.cos(teta - jnp.pi / 6)

    # stack into a vector and apply your per‐leg weights
    raw = jnp.stack([u1, u2, u3, u4, u5, u6])
    weights = jnp.array([50, 80, 30, 80, 30, 50], dtype=raw.dtype)
    scaled = raw * weights

    return scaled

@jax.jit
def u6_to6u_mapping(u1, u2, u3, u4, u5, u6):
    # stack into a vector and apply your per‐leg weights
    raw = jnp.array([u1, u2, u3, u4, u5, u6])
    weights = jnp.array([50, 80, 30, 80, 30, 50], dtype=raw.dtype)
    scaled = raw * weights

    return scaled


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
            ('n_obs', 6),                                   # 2D, 3D or 6D observations
            ('n_delay', 1),     # ssm: 3                             # number of delays applied to observations
            ('n_exec', 2),                                  # number of control inputs to execute from MPC solution
            ('results_name', 'test_experiment_pat')             # name of the results file
        ])

        self.koopman_mpc = True

        self.debug = self.get_parameter('debug').value
        self.n_z = self.get_parameter('n_z').value
        self.n_u = self.get_parameter('n_u').value
        self.n_obs = self.get_parameter('n_obs').value
        self.n_delay = self.get_parameter('n_delay').value
        self.n_exec = self.get_parameter('n_exec').value
        self.results_name = self.get_parameter('results_name').value

        # Initialize the CSV file
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack-ssmr/stack/main/data')
        self.results_file = os.path.join(self.data_dir, f"trajectories/closed_loop/{self.results_name}.csv")
        self.initialize_csv()

        # Collect buffer of control inputs for multiple executions
        self.control_buffer = []
        self.buffer_index = 0
        self.buffer_lock = Lock()
        
        # We perform smoothing to handle initial transients
        self.alpha_smooth = 0.97
        self.smooth_control_inputs = jnp.zeros(self.n_u)

        # Size of observations vector
        self.n_y = self.n_obs * (self.n_delay + 1)

        # Settled positions of the rigid bodies
        self.rest_position = None
        self.actuator_dynamics = None
        self.clock = self.get_clock()

        # Create publisher to execute found control inputs
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=3)
        )


        self.publish_control_inputs(jnp.zeros(self.n_u))
        self.get_logger().info('Published zero control inputs to all motors.')
        time.sleep(5)
        self.get_logger().info('Published zero control inputs and waited for 10 seconds.')
        
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

        # Maintain current observations because of the delay embedding
        self.latest_y = None
        if self.koopman_mpc:
            self.latest_y_koopman = None


        # Maintain previous control inputs
        self.u_previous = jnp.zeros(self.n_u)

        # track whether we’re finished so mpc_callback stops sending new requests
        self.finished = False

        # Need some initialization
        self.initialized = False

        # Initialize by calling mpc callback function
        self.mpc_callback()

        # JIT compile this functions
        u6_init = u6_to6u_mapping(*self.u_previous)
        check_control_inputs(u6_init, self.u_previous)

        # Create timer to receive MPC results at fixed frequency
        self.controller_period = 0.02
        self.mpc_exec_timer = self.create_timer(
                    self.controller_period,
                    self.mpc_callback,
                    callback_group=self.callback_group)
        
        # Timer for executing buffered controls
        self.buffer_execution_period = 0.02  # same as dt in MPC
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

        # 1) flatten and center, into simple list of positions, eg [x1, y1, z1, x2, y2, z2, ...]
        y_new = jnp.array([coord for pos in msg.positions for coord in (pos.x, pos.y, pos.z)])

        if self.rest_position is None:
            self.rest_position = y_new

        y_centered = y_new - self.rest_position

        perm_idx = jnp.array([6, 8, 7, 3, 5, 4, 0, 2, 1])
        y_reordered = y_centered[perm_idx]
        # then take only the first 6 entries (body 3 then 2)
        y_observables = y_reordered[:6]

        # 4) form your block the same way
        block = y_observables

        # Update the current observations, including delay embeddings
        if self.latest_y is None:
            # At initialization use current obs. as delay embedding
            self.latest_y = jnp.tile(block, (self.n_delay + 1,))
            self.start_time = self.clock.now().nanoseconds / 1e9
        else:
            self.latest_y = jnp.concatenate([block, self.latest_y[:-self.n_obs]])

        if self.koopman_mpc:
            self.latest_y_koopman = jnp.concatenate([self.latest_y, self.u_previous])  # augment the last applied input 

        self.t0 = self.clock.now().nanoseconds / 1e9 - self.start_time

    def execute_buffer_callback(self):
        """
        Execute the next control input from the buffer.
        """
        with self.buffer_lock:
            if not self.control_buffer or self.buffer_index >= len(self.control_buffer):
                return
    
            control_inputs = self.control_buffer[self.buffer_index]
            # safe_control_inputs = check_control_inputs(control_inputs, self.u_previous)
            self.smooth_control_inputs = (1 - self.alpha_smooth) * control_inputs + self.alpha_smooth * self.smooth_control_inputs

            if self.debug:
                self.get_logger().info(f'Executing buffer index {self.buffer_index} of {len(self.control_buffer)}')

            self.buffer_index += 1

        
        self.publish_control_inputs(self.smooth_control_inputs.tolist())
        self.u_previous = self.smooth_control_inputs

    def mpc_callback(self):
        """
        Receive MPC results at a fixed rate.
        """
        if self.finished:
            return
        
        if self.koopman_mpc:
            y_latest = self.latest_y_koopman
        else:
            y_latest = self.latest_y

        if not self.initialized:
            if self.koopman_mpc:
                self.y0 = jnp.zeros(self.n_y + self.n_u)
            else:
                self.y0 = jnp.zeros(self.n_y)
            self.send_request(0.0, self.y0, self.u_previous, wait=True)
            self.future.add_done_callback(self.service_callback)
            self.initialized = True
        elif y_latest is not None:
            self.y0 = y_latest
            print(f"y0: {self.y0}")
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
                # mark finished and cancel our periodic timers
                self.finished = True
                self.mpc_exec_timer.cancel()
                self.buffer_timer.cancel()

                self.get_logger().info(f'Trajectory is finished! At {(self.clock.now().nanoseconds / 1e9 - self.start_time):.3f}')
                self.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                return
            else:
                # Store the optimized control inputs in the buffer for execution
                new_buffer = []
                for i in range(self.n_exec):
                    new_buffer.append(jnp.array(response.uopt[i*self.n_u:(i+1)*self.n_u]))
                with self.buffer_lock:
                    self.control_buffer = new_buffer
                    self.buffer_index = 0

                # Save to csv file
                self.save_to_csv(response.t, response.xopt, response.uopt, response.zopt, self.y0[:self.n_y],
                                 response.solve_time)
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def publish_control_inputs(self, control_inputs):
        """
        Publish the control inputs.
        """


        control_inputs_6 = u6_to6u_mapping(*control_inputs)
        safe_control_inputs_6 = check_control_inputs(control_inputs_6)

        control_message = AllMotorsControl()
        control_message.motors_control = tuple(safe_control_inputs_6.tolist())
        self.controls_publisher.publish(control_message)
        if self.debug:
            self.get_logger().info('Published new motor control setting: ' + str(safe_control_inputs_6))

    def initialize_csv(self):
        """
        Initialize the CSV file with headers.
        """
        with open(self.results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['topt', 'xopt', 'uopt', 'zopt', 'y', 'solve_time'])

    def save_to_csv(self, topt, xopt, uopt, zopt, y, solve_time):
        """
        Save optimized quantities by MPC and observations to CSV file.
        """
        with open(self.results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([list(topt), list(xopt), list(uopt), list(zopt), y.tolist(), solve_time])


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
