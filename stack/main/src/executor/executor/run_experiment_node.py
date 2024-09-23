import os
import dill
import jax
import jax.numpy as jnp
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from rclpy.qos import QoSProfile    # type: ignore
from controller.controller.mpc_solver_node import run_mpc_solver_node, MPCClientNode
from controller.controller.mpc import GuSTOConfig
from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies
from interfaces.srv import ControlSolver
from .models.ssm import DelaySSM
from .models.models import SSMR
from .models.residual import ResidualBr, NeuralBr


class RunExperimentNode(Node):
    """
    This node is responsible for running the experiment, which can be either a trajectory tracking or user-defined position tracking.
    """
    def __init__(self):
        super().__init__('run_experiment_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('experiment_type', 'traj'),                    # 'traj' or 'user' (what input is being tracked)
            ('model', 'poly'),                              # 'nn' or 'poly' (what model to use)
            ('controller_type', 'mpc'),                     # 'ik' or 'mpc' (what controller to use)
            ('output_name', 'base_experiment')              # name of the output file
        ])

        self.debug = self.get_parameter('debug').value
        self.experiment_type = self.get_parameter('experiment_type').value
        self.model_type = self.get_parameter('model').value
        self.controller_type = self.get_parameter('controller_type').value
        self.output_name = self.get_parameter('output_name').value

        self.data_dir = os.getenv('TRUNK_DATA', '/home/asl/Documents/asl_trunk_ws/data')

        # Settled positions of the rigid bodies
        self.settled_positions = jnp.array([0, -0.10665, 0, 0, -0.20432, 0, 0, -0.320682, 0])

        # Get desired states
        if self.experiment_type == 'trajectory':
            # Generate reference trajectory
            z_ref, t = self._generate_ref_trajectory(3, 0.01, 'circle', 0.1)
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
                x_char=jnp.ones(self.model.n_x),
                f_char=jnp.ones(self.model.n_x),
                N=6
            )
            x0 = jnp.zeros(self.model.n_x)
            self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0, t=t, z=z_ref)
            # Create MPC solver service client
            self.mpc_client = MPCClientNode()
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
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )
        self.clock = self.get_clock()
        self.start_time = self.clock.now().nanoseconds / 1e9

        # Maintain current observations because of the delay embedding
        self.y = jnp.zeros(6)

        self.get_logger().info('Run experiment node has been started.')

    def _load_model(self):
        """
        Load the learned dynamics model of the system used for control.
        """
        # Get location of model file
        if self.model_type == 'nn':
            model_file = os.path.join(self.data_dir, 'models/nn_ssmr.pkl')
        elif self.model_type == 'poly':
            model_file = os.path.join(self.data_dir, 'models/poly_ssmr.pkl')
        else:
            raise ValueError('Invalid model type: ' + self.model_type + '. Valid options are: "nn" or "poly".')
        
        # Load the model
        with open(model_file, 'rb') as f:
            self.model = dill.load(f)
        print('Model loaded. Dimensions:')
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
        y_centered = y_new - self.settled_positions

        # Currently we only use the tip positions
        y_centered_tip = y_centered[-3:]

        # Update the current observations, including *single* delay embedding
        self.y = jnp.concatenate([y_centered_tip, self.y[:3]])

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
            self.publish_control_inputs(response.uopt)
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
