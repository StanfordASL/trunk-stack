import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from ros_phoenix.msg import MotorControl, MotorStatus
from interfaces.msg import SingleMotorControl, AllMotorsControl, SingleMotorStatus, AllMotorsStatus

class MotorControlConverter(Node):
    def __init__(self):
        super().__init__('converter_node')
        
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value

        # Subscribe using the AllMotorsControl msg type from the interfaces package
        self.subscription = self.create_subscription(
            AllMotorsControl,
            '/all_motors_control',
            self.execute_controls_callback,
            10)

        self.publisher = {}
        for i in range(1, 6 + 1):  # 6 motors
            topic_name = f'/talon{i}/set'
            self.publisher[i] = self.create_publisher(MotorControl, topic_name, 10)
        
        # Store the latest motor status and initialize subscribers
        self.motor_status = {}
        self.status_subscriber = {}
        for i in range(1, 6 + 1):
            status_topic = f'/talon{i}/status'
            self.motor_status[i] = None
            self.status_subscriber[i] = self.create_subscription(
                MotorStatus,
                status_topic,
                lambda msg, i=i: self.status_callback(msg, i),
                10)

        # Publisher for AllMotorsStatus
        self.all_status_publisher = self.create_publisher(AllMotorsStatus, '/all_motors_status', 10)
        
        self.get_logger().info('Motor control converter node has started.')

    def execute_controls_callback(self, msg):
        # Iterate through each motor's mode and value and publish them individually
        for i, single_motor_control in enumerate(msg.motors_control, start=1):
            new_msg = MotorControl()
            new_msg.mode = single_motor_control.mode
            new_msg.value = single_motor_control.value
            self.publisher[i].publish(new_msg)
            if self.debug: self.get_logger().info(f'Publishing to /talon{i}/set: mode={new_msg.mode}, value={new_msg.value}')

    def status_callback(self, msg, motor_id):
            self.motor_status[motor_id] = msg
            self.publish_all_motors_status()

    def publish_all_motors_status(self):
        all_status_msg = AllMotorsStatus()
        all_status_msg.motors_status = [self.convert_motor_status(self.motor_status[i]) for i in sorted(self.motor_status) if self.motor_status[i] is not None]
        if len(all_status_msg.motors_status) == 6:
            self.all_status_publisher.publish(all_status_msg)
            if self.debug: self.get_logger().info('Publishing aggregated motor statuses')

    def convert_motor_status(self, motor_status):
        single_motor_status = SingleMotorStatus()
        
        single_motor_status.temperature = motor_status.temperature
        single_motor_status.bus_voltage = motor_status.bus_voltage
        single_motor_status.output_percent = motor_status.output_percent
        single_motor_status.output_voltage = motor_status.output_voltage
        single_motor_status.output_current = motor_status.output_current
        single_motor_status.position = motor_status.position
        single_motor_status.velocity = motor_status.velocity
        single_motor_status.fwd_limit = motor_status.fwd_limit
        single_motor_status.rev_limit = motor_status.rev_limit

        return single_motor_status


def main(args=None):
    rclpy.init(args=args)
    converter = MotorControlConverter()
    rclpy.spin(converter)
    converter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
