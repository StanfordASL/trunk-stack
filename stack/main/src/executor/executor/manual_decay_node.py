import os
import sys
import csv
import time
import select
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from std_msgs.msg import String  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore
from interfaces.msg import TrunkMarkers, TrunkRigidBodies


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('manual_decay_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                   # False or True
            ('sample_size', 10),                # for checking settling condition and averaging (steady state)
            ('min_collection_time', 2),         # minimum time it takes to collect data, avoiding immediate settling condition
            ('max_traj_length', 750),           # maximum number of samples in a dynamic trajectory
            ('mocap_type', 'rigid_bodies'),     # 'rigid_bodies' or 'markers'
            ('control_type', 'output'),         # 'output' or 'position'
            ('results_name', 'observations')
        ])

        self.debug = self.get_parameter('debug').value
        self.sample_size = self.get_parameter('sample_size').value
        self.min_collection_time = self.get_parameter('min_collection_time').value
        self.max_traj_length = self.get_parameter('max_traj_length').value
        self.mocap_type = self.get_parameter('mocap_type').value
        self.control_type = self.get_parameter('control_type').value
        self.results_name = self.get_parameter('results_name').value
        
        # Initializations
        self.is_collecting = False
        self.trajectory_count = -1
        self.stored_positions = []
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')        
        
        if self.mocap_type == 'markers':
            self.subscription_markers = self.create_subscription(
                TrunkMarkers,
                '/trunk_markers',
                self.mocap_callback, 
                QoSProfile(depth=10)
            )
        elif self.mocap_type == 'rigid_bodies':
            self.subscription_rigid_bodies = self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.mocap_callback,
                QoSProfile(depth=10)
            )
        else:
            raise ValueError('Invalid mocap type: ' + self.mocap_type + '. Valid options are: "rigid_bodies" or "markers".')

        # Timer to periodically check for user input at 10Hz (adjustable frequency)
        self.input_timer = self.create_timer(0.1, self.listen_for_input)

        # State machine to manage user input and data collection
        self.state = 'idle'  # 'idle' for waiting for user input, 'collecting' for collecting data

        self.get_logger().info('Manual decay data collection node has been started.')

    def listen_for_input(self):
        """Check for user input to start/stop data collection."""
        if self.state == 'idle':
            if self._kbhit():
                char = sys.stdin.read(1)
                if char == 's':
                    # Start data collection
                    self.is_collecting = True
                    self.trajectory_count += 1
                    self.start_time = time.time()
                    self.stored_positions.clear()
                    self.check_settled_positions = []
                    self.state = 'collecting'
                    self.get_logger().info("Started collecting data...")
                elif char == 'f':
                    # Finish data collection and shut down
                    self.is_collecting = False
                    self.get_logger().info("Finished data collection!")
                    rclpy.shutdown()

    def mocap_callback(self, msg):
        if self.is_collecting:
            # Collect data
            self.store_positions(msg)

            # Check if the trajectory has settled
            if (self.check_settled(window=30) or len(self.stored_positions) >= self.max_traj_length) and \
            (time.time() - self.start_time) >= self.min_collection_time:
                # Finished collecting trajectory
                self.is_collecting = False
                names = self.extract_names(msg)
                self.process_data(names)
                self.state = 'idle'
                self.get_logger().info(f'Stored trajectory number {self.trajectory_count}')
            else:
                self.check_settled_positions.append(self.extract_positions(msg))

    def _kbhit(self):
        """Check if a key has been pressed (non-blocking)."""
        return select.select([sys.stdin], [], [], 0.0)[0] != []
    
    def extract_positions(self, msg):
        if self.mocap_type == 'markers':
            return msg.translations
        elif self.mocap_type == 'rigid_bodies':
            return msg.positions
        
    def extract_names(self, msg):
        if self.mocap_type == 'markers':
            raise NotImplementedError('Extracting names from markers is not implemented.')
        elif self.mocap_type == 'rigid_bodies': 
            return msg.rigid_body_names

    def store_positions(self, msg):
        self.stored_positions.append(self.extract_positions(msg))

    def check_settled(self, tolerance=0.00025, window=5):
        if len(self.check_settled_positions) < window:
            # Not enough positions to determine if settled
            return False

        num_positions = len(self.check_settled_positions[0])  # usually 3 (rigid bodies) for the trunk robot
        
        min_positions = [{'x': float('inf'), 'y': float('inf'), 'z': float('inf')} for _ in range(num_positions)]
        max_positions = [{'x': float('-inf'), 'y': float('-inf'), 'z': float('-inf')} for _ in range(num_positions)]
        
        recent_positions = self.check_settled_positions[-window:]
        
        for pos_list in recent_positions:
            for idx, pos in enumerate(pos_list):
                min_positions[idx]['x'] = min(min_positions[idx]['x'], pos.x)
                max_positions[idx]['x'] = max(max_positions[idx]['x'], pos.x)
                min_positions[idx]['y'] = min(min_positions[idx]['y'], pos.y)
                max_positions[idx]['y'] = max(max_positions[idx]['y'], pos.y)
                min_positions[idx]['z'] = min(min_positions[idx]['z'], pos.z)
                max_positions[idx]['z'] = max(max_positions[idx]['z'], pos.z)

        for idx in range(num_positions):
            range_x = max_positions[idx]['x'] - min_positions[idx]['x']
            range_y = max_positions[idx]['y'] - min_positions[idx]['y']
            range_z = max_positions[idx]['z'] - min_positions[idx]['z']
            
            if range_x > tolerance or range_y > tolerance or range_z > tolerance:
                return False

        return True

    def process_data(self, names):
        # Populate the header row of the CSV file with states if it does not exist
        trajectory_csv_file = os.path.join(self.data_dir, f'trajectories/dynamic/{self.results_name}.csv')
        if not os.path.exists(trajectory_csv_file):
            header = ['ID'] + [f'{axis}{name}' for name in names for axis in ['x', 'y', 'z']]
            with open(trajectory_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        # Store all positions in a CSV file
        with open(trajectory_csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for pos_list in self.stored_positions:
                row = [self.trajectory_count] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]]
                writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    manual_decay_node = DataCollectorNode()
    rclpy.spin(manual_decay_node)
    manual_decay_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
