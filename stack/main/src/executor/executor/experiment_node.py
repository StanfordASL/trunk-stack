import os
import jax
import jax.numpy as jnp
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from rclpy.qos import QoSProfile    # type: ignore
from controller.mpc.gusto import GuSTOConfig  # type: ignore
from controller.mpc_solver_node import run_mpc_solver_node, MPCClientNode  # type: ignore
from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies
from interfaces.srv import ControlSolver
from .utils.models import SSMR


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
        self.experiment_type = self.get_parameter('experiment_type').value
        self.model_name = self.get_parameter('model_name').value
        self.controller_type = self.get_parameter('controller_type').value
        self.results_name = self.get_parameter('results_name').value

        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Settled positions of the rigid bodies
        self.rest_position = jnp.array([0.1005, -0.10698, 0.10445, -0.10302, -0.20407, 0.10933, 0.10581, -0.32308, 0.10566])
        self.avp_offset = jnp.array([0, -0.10698, 0, 0, -0.20407, 0, 0, -0.32308, 0])

        # Get desired states
        if self.experiment_type == 'traj':
            # Generate reference trajectory
            z_ref, t = self._generate_ref_trajectory(4, 0.01, 'circle', 0.15)
        elif self.experiment_type == 'user':
            # We track positions as defined by the user (through the AVP)
            self.avp_subscription = self.create_subscription(
                TrunkRigidBodies,
                '/avp_des_positions',
                self.teleop_listener_callback,
                QoSProfile(depth=10)
            )
        else:
            raise ValueError('Invalid experiment type: ' + self.experiment_type + '. Valid options are: "trajectory" or "user".')

        if self.controller_type == 'mpc':
            # Subscribe to current positions
            self.mocap_subscription = self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.mocap_listener_callback,
                QoSProfile(depth=10)
            )
            # Load the model
            self._load_model()
            # Start MPC solver node
            gusto_config = GuSTOConfig(
                Qz=jnp.eye(self.model.n_z),
                Qzf=10*jnp.eye(self.model.n_z),
                R=0.0001*jnp.eye(self.model.n_u),
                x_char=0.05*jnp.ones(self.model.n_x),
                f_char=0.5*jnp.ones(self.model.n_x),
                N=7
            )
            x0 = jnp.zeros(self.model.n_x)
            self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0, t=t, z=z_ref)
            self.get_logger().info('mpc solver node created')
            # Create MPC solver service client
            self.mpc_client = MPCClientNode()
            self.get_logger().info('mpc client node created')

        elif self.controller_type == 'ik':
            # Create control solver service client
            self.ik_client = self.create_client(
                ControlSolver,
                'ik_solver'
            )
            # Wait for service to become available
            while not self.ik_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Service not available, waiting...')
        else:
            raise ValueError('Invalid controller type: ' + self.controller_type + '. Valid options are: "ik" or "mpc".')

        # Create publisher to execute found control inputs
        """ TODO: enable control execution
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )
        """
        # Maintain current observations because of the delay embedding
        self.y = jnp.zeros(self.model.n_y)

        self.get_logger().info('Run experiment node has been started.')

        self.clock = self.get_clock()
        # self.start_time = self.clock.now().nanoseconds / 1e9

    def _load_model(self):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """
        model_path = os.path.join(self.data_dir, f'models/ssmr/{self.model_name}.npz')

        # Load the model
        self.model = SSMR(model_path=model_path)
        print('---- Model loaded. Dimensions:')
        print('     n_x:', self.model.n_x)
        print('     n_u:', self.model.n_u)
        print('     n_z:', self.model.n_z)
        print('     n_y:', self.model.n_y)

    def _generate_ref_trajectory(self, T, dt, traj_type, size):
        """
        Generate a 3D reference trajectory for the system to track.
        """
        t = jnp.linspace(0, T, int(T/dt))
        z_ref = jnp.zeros((len(t), 3))

        # Note that y is up
        if traj_type == 'circle':
            z_ref = z_ref.at[:, 0].set(size * jnp.cos(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(size / 2 * jnp.ones_like(t))
            z_ref = z_ref.at[:, 2].set(size * jnp.sin(2 * jnp.pi / T * t))
        elif traj_type == 'figure_eight':
            z_ref = z_ref.at[:, 0].set(size * jnp.sin(jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(size / 2 * jnp.ones_like(t))
            z_ref = z_ref.at[:, 2].set(size * jnp.sin(2 * jnp.pi / T * t))
        else:
            raise ValueError('Invalid trajectory type: ' + traj_type + '. Valid options are: "circle" or "figure_eight".')
        return z_ref, t

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

    def teleop_listener_callback(self, msg):
        if self.debug:
            self.get_logger().info(f'Received teleop desired positions: {msg.positions}.')
        
        # Create request
        request = ControlSolver.Request()

        # Populate request with desired positions from teleop
        zf = jnp.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])

        # Center data around zero
        zf_centered = zf - self.settled_positions
        request.zf = zf_centered.tolist()

        # Call the service
        self.async_response = self.ik_client.call_async(request)
        self.async_response.add_done_callback(self.service_callback)

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


def main(args=None):
    rclpy.init(args=args)
    run_experiment_node = RunExperimentNode()
    rclpy.spin(run_experiment_node)
    run_experiment_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
