import os
import csv
from threading import Lock

import jax
import jax.numpy as jnp
import logging

import rclpy                                                # type: ignore
from rclpy.node import Node                                 # type: ignore
from rclpy.callback_groups import ReentrantCallbackGroup    # type: ignore
from rclpy.executors import MultiThreadedExecutor           # type: ignore
from rclpy.qos import QoSProfile                            # type: ignore

from controller.mpc_solver_node import jnp2arr              # type: ignore
from interfaces.msg import AllMotorsControl, TrunkRigidBodies, AllMotorsStatus
from interfaces.srv import ControlSolver

logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)



class MPCNode(Node):
    """
    This node is responsible for running MPC.
    """
    def __init__(self):
        super().__init__('mpc_sanity_check')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # print debug messages
            ('n_z', 3),                                     # number of performance vars
            ('n_u', 6),                                     # number of control inputs
            ('n_obs', 6),                                   # 2D, 3D or 6D observations
            ('n_delay', 3),                               # number of delays applied to observations
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
        self.results_file = os.path.join(self.data_dir, f"trajectories/closed_loop/adiabatic/{self.results_name}.csv")
        self.initialize_csv()

        # Collect buffer of control inputs for multiple executions
        self.control_buffer = []
        self.buffer_index = 0
        self.buffer_lock = Lock()
        
        # We perform smoothing to handle initial transients
        self.alpha_smooth = 0.0  # TODO: Change
        self.smooth_control_inputs = jnp.zeros(self.n_u)
        self.collect_angles = True
        self.last_motor_angles = None

        # track whether we’re finished so mpc_callback stops sending new requests
        self.finished = False

        # store solve times
        self.solve_times = []

        # Size of observations vector
        self.block_size = self.n_obs 
        self.n_y = self.block_size * (self.n_delay + 1)

        print(f"n_y: {self.n_y}, n_obs: {self.n_obs}, n_delay: {self.n_delay}, block_size: {self.block_size}")
        assert self.n_y == 24, "wrong n_y calculated"

        # TODO: give this to roshan
        # # Settled positions of the rigid bodies    
        self.rest_positions = jnp.array([0.10368,-0.14215,0.11343,0.10405,-0.27971,0.11191,0.10730,-0.40798,0.12463]) #updated 10/28/25
        self.perm_idx = jnp.array([6, 7, 8, 3, 4, 5])
        self.rest_y = self.rest_positions[self.perm_idx]

        # Execution occurs in multiple threads
        self.callback_group = ReentrantCallbackGroup()

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

        # Maintain previous control inputs
        self.u_previous = jnp.zeros(self.n_u)
        self.angle_update_count = 0
        self.angle_callback_received = False  # flag

        self.clock = self.get_clock()

        # Need some initialization
        self.initialized = False

        # Initialize by calling mpc callback function
        self.mpc_callback()

        # Create timer to receive MPC results at fixed frequency
        self.controller_period = 0.02 # TODO: we need to reason about this
        self.mpc_exec_timer = self.create_timer(
                    self.controller_period,
                    self.mpc_callback,
                    callback_group=self.callback_group)
        
        # Timer for executing buffered controls
        self.buffer_execution_period = 0.02  # same as dt in MPC TODO: we need to reason about this
        self.buffer_timer = self.create_timer(
            self.buffer_execution_period,
            self.execute_buffer_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info(f'MPC node has been started with controller frequency: {1/self.controller_period:.2f} [Hz].')

        # Define reference time
        self.start_time = self.clock.now().nanoseconds / 1e9

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

        self.u_previous = self.smooth_control_inputs

    def mpc_callback(self):
        """
        Receive MPC results at a fixed rate.
        """
        if self.finished:
            return

        if not self.initialized:
            self.y0 = jnp.tile(self.rest_y, (self.n_delay + 1,))
            print(f'Initial y0: {self.y0}')
            self.send_request(0.0, self.y0, self.u_previous, wait=True)
            self.future.add_done_callback(self.service_callback)
            self.initialized = True
            self.latest_y = self.y0
        elif self.latest_y is not None:
            self.t0 = self.clock.now().nanoseconds / 1e9 - self.start_time
            self.y0 = jnp.tile(self.rest_y, (self.n_delay + 1,))
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
                    # rclpy.shutdown()
                    pass
                return
            else:
                # Store the optimized control inputs in the buffer for execution
                new_buffer = []
                for i in range(self.n_exec):
                    new_buffer.append(jnp.array(response.uopt[i*self.n_u:(i+1)*self.n_u]))
                with self.buffer_lock:
                    self.control_buffer = new_buffer
                    self.buffer_index = 0

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def initialize_csv(self):
        """
        Initialize the CSV file with headers.
        """
        with open(self.results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['topt', 'xopt', 'uopt', 'zopt', 'y', 'solve_time'])


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
