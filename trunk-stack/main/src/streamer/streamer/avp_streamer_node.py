import os
import csv
import time
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore
from interfaces.msg import TrunkRigidBodies
from geometry_msgs.msg import Point
from .avp_subscriber import AVPSubscriber
from interfaces.srv import TriggerImageSaving


class AVPStreamerNode(Node):
    def __init__(self):
        super().__init__('avp_streamer_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True
            ('recording_name', 'test_task_recording'),
            ('mocap_type', 'rigid_bodies')                  # 'rigid_bodies' or 'markers' - always going to be 'rigid_bodies' for now
        ])

        self.debug = self.get_parameter('debug').value
        self.mocap_type = self.get_parameter('mocap_type').value
        self.recording_name = self.get_parameter('recording_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/asl/Documents/asl_trunk_ws/data')
        self.recording_file = os.path.join(self.data_dir, f'trajectories/teleop/mocap_rb/{self.recording_name}.csv')
        self.img_filename = None
        self.client = self.create_client(TriggerImageSaving, 'trigger_image_saving')
        # while not self.client.wait_for_service(timeout_sec=1.0):
            # self.get_logger().info('Image saving service not yet available, waiting...')
        self.latest_mocap_positions = None 
        self.rigid_body_names = ["1", "2", "3"]

        # Initialize stored positions and gripper states
        self.stored_positions = []
        self.stored_gripper_states = []

        # Keep track of trajectory ID
        self.recording_id = -1

        self.avp_publisher = self.create_publisher(
            TrunkRigidBodies,
            '/avp_des_positions',
            QoSProfile(depth=10)
        )
        
        # subscribe to mocap positions
        self.subscription_rigid_bodies = self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.mocap_listener_callback,
                QoSProfile(depth=10)
            )

        self.streamer = AVPSubscriber(ip='10.93.181.122')
        self._timer = self.create_timer(1.0 / 10.0, self.streamer_data_sampling_callback) # runs at 10Hz
        self.get_logger().info('AVP streaming node has been started.')

    def mocap_listener_callback(self, msg):
        self.latest_mocap_positions = msg


    def streamer_data_sampling_callback(self):
        # Determine trajectory ID - when recording starts
        if self.streamer.isRecording and not self.streamer.previousRecordingState:
            self.get_logger().info('New traj.')
            self.stored_positions = []
            self.stored_gripper_states = []
            self.recording_id += 1
            self.trigger_image_saving() #save an image
            self.get_logger().info('saved image')

        # self.get_logger().info(f'Current: {self.streamer.isRecording}, previous: {self.streamer.previousRecordingState}')

        avp_positions = self.streamer.get_latest()
        avp_rigid_bodies_msg = self.convert_avp_positions_to_trunk_rigid_bodies(avp_positions)
        
        # save real rigid body positions
        trunk_rigid_bodies_msg = self.latest_mocap_positions
        print(self.latest_mocap_positions)
        self.stored_positions.append(trunk_rigid_bodies_msg.positions)

        self.stored_gripper_states.append(self.streamer.isGripperOpen)
        self.avp_publisher.publish(avp_rigid_bodies_msg)
        if self.debug:
            self.get_logger().info(f'Published AVP desired positions {avp_positions}.')

        # Store trajectory - when recording ends
        if not self.streamer.isRecording and self.streamer.previousRecordingState:
            # if self.debug:
            self.get_logger().info('Processing data.')
            self.process_data()

    def convert_avp_positions_to_trunk_rigid_bodies(self, avp_positions):
        """
        Converts the given AVP positions into a TrunkRigidBodies message.
        """
        msg = TrunkRigidBodies()

        # Extract rigid body names and positions
        positions_list = [
            avp_positions['disk_positions'].get(f'disk{name}', [0, 0, 0]) for name in self.rigid_body_names
        ]

        msg.frame_number = 0  # TODO: replace?
        msg.rigid_body_names = self.rigid_body_names
        points = []
        for position in positions_list:
            point = Point()
            point.x = position[0]
            point.y = position[1]
            point.z = position[2]
            points.append(point)
        msg.positions = points

        return msg

    def process_data(self):
        # Populate the header row of the CSV file with states if it does not exist
        if not os.path.exists(self.recording_file):
            header = ['ID'] + [f'{axis}{name}' for name in self.rigid_body_names for axis in ['x', 'y', 'z']] + ['isGripperOpen'] + ['img_filename']
            with open(self.recording_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
        
        # Store all positions and gripper state in a CSV file
        with open(self.recording_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for i, pos_list in enumerate(self.stored_positions):
                row = [self.recording_id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]] + [self.stored_gripper_states[i]] + [self.img_filename]
                writer.writerow(row)
        self.get_logger().info(f'Stored the data corresponding to the {self.recording_id}th trajectory.')

    def trigger_image_saving(self):
        req = TriggerImageSaving.Request()

        self.async_response = self.client.call_async(req)
        self.async_response.add_done_callback(self.service_callback)

    def service_callback(self, async_response):
        try:
            response = self.async_response.result()
            self.img_filename = response.img_filename

            if response.success:
                self.get_logger().info('Image saving triggered successfully')
            else:
                self.get_logger().error('Image saving failed')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
        


def main(args=None):
    rclpy.init(args=args)
    avp_streamer_node = AVPStreamerNode()
    rclpy.spin(avp_streamer_node)
    avp_streamer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
