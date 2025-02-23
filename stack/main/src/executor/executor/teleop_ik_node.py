import os
import jax
import jax.numpy as jnp
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from rclpy.qos import QoSProfile    # type: ignore
from interfaces.msg import SingleMotorControl, AllMotorsControl, TrunkRigidBodies
from interfaces.srv import ControlSolver


class RunExperimentNode(Node):
    """
    This node is responsible for running the experiment, which can be either a trajectory tracking or user-defined position tracking.
    """
    def __init__(self):
        super().__init__('run_experiment_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('experiment_type', 'user'),                    # 'traj' or 'user' (what input is being tracked)
            ('model', 'poly'),                              # 'nn' or 'poly' (what model to use)
            ('controller_type', 'ik'),                     # 'ik' or 'mpc' (what controller to use)
            ('output_name', 'base_experiment')              # name of the output file
        ])

        self.debug = self.get_parameter('debug').value
        self.experiment_type = self.get_parameter('experiment_type').value
        self.model_type = self.get_parameter('model').value
        self.controller_type = self.get_parameter('controller_type').value
        self.output_name = self.get_parameter('output_name').value

        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Offset of avp with respect to Trunk mocap coordinate system
        self.avp_offset = jnp.array([0, -0.10698, 0, 0, -0.20407, 0, 0, -0.32308, 0])

        # Get desired states
        if self.experiment_type == 'user':
            # We track positions as defined by the user (through the AVP)
            self.avp_subscription = self.create_subscription(
                TrunkRigidBodies,
                '/avp_des_positions',
                self.teleop_listener_callback,
                QoSProfile(depth=10)
            )
        else:
            raise ValueError('Invalid experiment type: ' + self.experiment_type + '. Valid options are: "trajectory" or "user".')

        if self.controller_type == 'ik':
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

        self.get_logger().info('Run experiment node has been started.')

    def teleop_listener_callback(self, msg):
        if self.debug:
            self.get_logger().info(f'Received teleop desired positions: {msg.positions}.')
        
        # Create request
        request = ControlSolver.Request()

        # Populate request with desired positions from teleop
        zf = jnp.array([coord for pos in msg.positions for coord in [pos.x, pos.y, pos.z]])

        # Center data around zero
        zf_centered = zf - self.avp_offset
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

    def publish_control_inputs(self, control_inputs=None):
        if control_inputs is None:
            control_inputs = 0
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
