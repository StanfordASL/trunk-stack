import os
import csv
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore
from interfaces.msg import TrunkMarkers, TrunkRigidBodies


class DataCollectionNode(Node):
    """
    This node stores the observations such that trajectories can be recorded in real-time.
    """
    def __init__(self):
        super().__init__('data_collection_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                   # False or True
            ('update_period', 0.1),             # for steady state and avoiding dynamic trajectories to interrupt each other, in [s]
            ('traj_time', 10),                  # trajectory time in [s]
            ('mocap_type', 'rigid_bodies'),     # 'rigid_bodies' or 'markers'
            ('control_type', 'output'),         # 'output' or 'position'
            ('results_name', 'observations'),
        ])

        self.debug = self.get_parameter('debug').value
        self.update_period = self.get_parameter('update_period').value
        self.traj_time = self.get_parameter('traj_time').value
        self.mocap_type = self.get_parameter('mocap_type').value
        self.control_type = self.get_parameter('control_type').value
        self.results_name = self.get_parameter('results_name').value

        self.is_collecting = True
        self.stored_positions = []
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        if self.mocap_type == 'markers':
            self.subscription_markers = self.create_subscription(
                TrunkMarkers,
                '/trunk_markers',
                self.listener_callback, 
                QoSProfile(depth=10)
            )
        elif self.mocap_type == 'rigid_bodies':
            self.subscription_rigid_bodies = self.create_subscription(
                TrunkRigidBodies,
                '/trunk_rigid_bodies',
                self.listener_callback,
                QoSProfile(depth=10)
            )
        else:
            raise ValueError('Invalid mocap type: ' + self.mocap_type + '. Valid options are: "rigid_bodies" or "markers".')
        
        self.clock = self.get_clock()
        self.start_time = self.clock.now().nanoseconds / 1e9

        self.get_logger().info('Store observations node has been started.')

    def listener_callback(self, msg):
        if self.is_collecting:
            # Store current positions
            self.store_positions(msg)

            if (self.clock.now().nanoseconds / 1e9 - self.start_time) > self.traj_time :
                # Data collection is complete and ready to be processed
                self.is_collecting = False
                names = self.extract_names(msg)
                self.process_data(names)

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

    def process_data(self, names):
        # Populate the header row of the CSV file with states if it does not exist
        recording_csv_file = os.path.join(self.data_dir, f'trajectories/closed_loop/{self.results_name}.csv')
        if not os.path.exists(recording_csv_file):
            header = [f'{axis}{name}' for name in names for axis in ['x', 'y', 'z']]
            with open(recording_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
    
        # Store all positions in a CSV file
        with open(recording_csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for pos_list in self.stored_positions:
                row = [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]]
                writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    data_collection_node = DataCollectionNode()
    rclpy.spin(data_collection_node)
    data_collection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
