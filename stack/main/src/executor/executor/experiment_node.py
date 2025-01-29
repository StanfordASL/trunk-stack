import os
import jax
import jax.numpy as jnp
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from rclpy.qos import QoSProfile    # type: ignore
from controller.mpc_solver_node import arr2jnp, jnp2arr  # type: ignore
from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies
from interfaces.srv import ControlSolver


class RunExperimentNode(Node):
    """
    This node is responsible for running the main experiment.
    """
    def __init__(self):
        super().__init__('run_experiment_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('controller_type', 'mpc'),                     # 'ik' or 'mpc' (what controller to use)
            ('results_name', 'test_experiment')             # name of the results file
        ])

        self.debug = self.get_parameter('debug').value
        self.controller_type = self.get_parameter('controller_type').value
        self.results_name = self.get_parameter('results_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Settled positions of the rigid bodies
        self.rest_position = jnp.array([0.1005, -0.10698, 0.10445, -0.10302, -0.20407, 0.10933, 0.10581, -0.32308, 0.10566])

        if self.controller_type == 'mpc':
            # Subscribe to current positions
            self.mocap_subscription = self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.mocap_listener_callback,
                QoSProfile(depth=10)
            )

            # Create MPC solver service client
            self.mpc_client = self.create_client(
                ControlSolver,
                'mpc_solver'
            )
            self.get_logger().info('MPC client created.')
            while not self.mpc_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('MPC solver not available, waiting...')
            # Request message definition
            self.req = ControlSolver.Request()

        elif self.controller_type == 'ik':
            # Create control solver service client
            self.ik_client = self.create_client(
                ControlSolver,
                'ik_solver'
            )
            self.get_logger().info('IK client created.')
            while not self.ik_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('IK solver not available, waiting...')
        else:
            raise ValueError('Invalid controller type: ' + self.controller_type + '. Valid options are: "ik" or "mpc".')

        # Create publisher to execute found control inputs
        # NOTE: still disabled later in code for now
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )
        
        # Maintain current observations because of the delay embedding
        # self.y = jnp.zeros(self.model.n_y)

        self.get_logger().info('Run experiment node has been started.')

        self.clock = self.get_clock()
        # self.start_time = self.clock.now().nanoseconds / 1e9

    def mocap_listener_callback(self, msg):
        if self.debug:
            self.get_logger().info(f'Received mocap data: {msg.positions}.')

        # Unpack the message into simple list of positions, eg [x1, y1, z1, x2, y2, z2, ...]
        y_new = jnp.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])

        # Center the data around settled positions
        y_centered = y_new - self.settled_position

        # Subselect bottom two segments
        y_centered_midtip = y_centered[3:]

        # Update the current observations, including *single* delay embedding
        self.y = jnp.concatenate([y_centered_midtip, self.y[:6]])

        t0 = self.clock.now().nanoseconds / 1e9
        x0 = self.model.encode(self.y)

        # Call the service
        self.mpc_client.send_request(t0, x0, wait=False)
        self.mpc_client.future.add_done_callback(self.service_callback)

    def service_callback(self, async_response):
        try:
            response = async_response.result()
            # TODO: enable control execution
            # self.publish_control_inputs(response.uopt)
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}.')

    def publish_control_inputs(self, control_inputs):
        control_message = AllMotorsControl()
        control_message.motors_control = [
            SingleMotorControl(mode=0, value=value) for value in control_inputs
        ]
        self.controls_publisher.publish(control_message)
        if self.debug:
            self.get_logger().info(f'Published new motor control setting: {control_inputs}.')

    def send_request(self, t0, x0, wait=False):
        """
        Send request to MPC solver.
        """
        self.req.t0 = t0
        self.req.x0 = jnp2arr(x0)
        self.future = self.mpc_client.call_async(self.req)

        if wait:
            # Synchronous call, not compatible for real-time applications
            rclpy.spin_until_future_complete(self, self.future)

    def get_solution(self, n_x, n_u):
        """
        Obtain result from MPC solver.
        """
        res = self.future.result()
        t = arr2jnp(res.t, 1, squeeze=True)
        xopt = arr2jnp(res.xopt, n_x)
        uopt = arr2jnp(res.uopt, n_u)
        t_solve = res.solve_time

        return t, uopt, xopt, t_solve

    def force_spin(self):
        if not self.check_if_done():
            rclpy.spin_once(self, timeout_sec=0)

    def check_if_done(self):
        return self.future.done()

    def force_wait(self):
        self.get_logger().warning('Overrides realtime compatibility, solve is too slow. Consider modifying problem')
        rclpy.spin_until_future_complete(self, self.future)


def main(args=None):
    rclpy.init(args=args)
    run_experiment_node = RunExperimentNode()
    rclpy.spin(run_experiment_node)
    run_experiment_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
