from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import rclpy
from rclpy.node import Node
from rclpy.service import Service
from interfaces.srv import GripperAction


class ServoControlNode(Node):
    def __init__(self):
        super().__init__('servo_control_node')

        # Initialize the parameter for the initial condition
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),            # True or False
            ('gripper_ic', 'opened'),    # 'opened' or 'closed'
            ('servo_pin', 21)            # any valid int
        ])
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        gripper_ic = self.get_parameter('gripper_ic').get_parameter_value().string_value
        servo_pin = self.get_parameter('servo_pin').get_parameter_value().integer_value

        self.gripper_state = gripper_ic.lower() == 'opened'

        # Set the pin factory
        factory = PiGPIOFactory()

        # Use the pin factory in the Servo object
        self.servo = Servo(servo_pin, pin_factory=factory)
        
        # Create the service
        self.srv = self.create_service(GripperAction, 'move_gripper', self.gripper_action_callback)

        self.get_logger().info('Gripper servo control service has been created.')
        if self.debug:
            self.get_logger().info(f'Gripper initialized as {self.gripper_state}.')

    def gripper_action_callback(self, request, response):
        """
        Open or close the gripper.
        """
        action = request.action
        if self.gripper_state == (action + 'ed'):
            # If current gripper state is equal to desired state, no motion
            pass
        else:
            if action == 'open':
                # Open the gripper
                self.servo.min()
                if self.debug:
                    self.get_logger().info('Opened the gripper')
            elif action == 'close':
                # Close the gripper
                self.servo.max()
                if self.debug:
                    self.get_logger().info('Closed the gripper')
            else:
                raise ValueError(f"The provided action '{action}' is invalid. It should be equal to 'open' or 'close'.")
        return response

def main(args=None):
    rclpy.init(args=args)
    servo_control_node = ServoControlNode()
    rclpy.spin(servo_control_node)
    servo_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
