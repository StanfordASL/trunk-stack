import os
import csv
import time
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore
from interfaces.msg import AllMotorsControl, TrunkMarkers, TrunkRigidBodies, AllMotorsStatus


def load_control_inputs(control_input_csv_file):
    control_inputs_dict = {}
    
    # Read the control inputs first
    with open(control_input_csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # Skip the header row
        rows = [row for row in csv_reader]
    
    # Extract the control ids and find the minimum control id
    control_ids = [int(row[0]) for row in rows]
    min_control_id = min(control_ids)
    
    # Shift the control ids if the minimum control id is not zero
    if min_control_id != 0:
        shift = -min_control_id
    else:
        shift = 0
    
    # Now process the rows and update control_ids
    for row in rows:
        control_id = int(row[0]) + shift
        control_inputs = [float(u) for u in row[1:]]
        control_inputs_dict[control_id] = control_inputs
    
    return control_inputs_dict


class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                   # False or True
            ('sample_size', 10),                # for checking settling condition and averaging (steady state)
            ('update_period', 0.1),             # for steady state and avoiding dynamic trajectories to interrupt each other, in [s]
            ('max_traj_length', 600),           # maximum number of samples in a dynamic trajectory
            ('data_type', 'dynamic'),           # 'steady_state' or 'dynamic'
            ('data_subtype', 'decay'),          # 'decay' or 'controlled' or 'adiabatic_manual' or 'adiabatic_step' or 'adiabatic_jolt' for dynamic and e.g. 'circle' or 'beta' or 'uniform' for steady_state
            ('mocap_type', 'rigid_bodies'),     # 'rigid_bodies' or 'markers'
            ('control_type', 'output'),         # 'output' or 'position'
            ('results_name', 'observations'),
            ('input_num', 1),                    # number of the input file type i.e. control_inputs_controlled_1
            ('collect_angles', True),            # to collect motor angle measurements
            ('collect_orientations', True)       # to collect mocap rigid body orientation data
        ])

        self.debug = self.get_parameter('debug').value
        self.sample_size = self.get_parameter('sample_size').value
        self.update_period = self.get_parameter('update_period').value
        self.max_traj_length = self.get_parameter('max_traj_length').value
        self.data_type = self.get_parameter('data_type').value
        self.data_subtype = self.get_parameter('data_subtype').value
        self.mocap_type = self.get_parameter('mocap_type').value
        self.control_type = self.get_parameter('control_type').value
        self.results_name = self.get_parameter('results_name').value
        self.input_num = str(self.get_parameter('input_num').value)
        self.collect_angles = self.get_parameter('collect_angles').value
        self.collect_orientations = self.get_parameter('collect_orientations').value

        self.angle_callback_received = False #flag
        self.angle_update_count = 0
        self.is_collecting = False
        self.ic_settled = False
        self.previous_time = time.time()
        self.current_control_id = -1
        self.stored_positions = []
        self.stored_orientations = []
        self.stored_angles = []
        self.last_motor_angles = None
        self.control_inputs = None
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        if self.data_type == 'steady_state':
            control_input_csv_file = os.path.join(self.data_dir, f'trajectories/steady_state/control_inputs_{self.data_subtype}.csv')
        elif self.data_type == 'dynamic':
            control_input_csv_file = os.path.join(self.data_dir, f'trajectories/dynamic/control_inputs_{self.data_subtype}_{self.input_num}.csv')
        else:
            raise ValueError('Invalid data type: ' + self.data_type + '. Valid options are: "steady_state" or "dynamic".')
        self.control_inputs_dict = load_control_inputs(control_input_csv_file)
        self.num_control_inputs = len(self.control_inputs_dict)

        if self.collect_angles:
            self.subscription_angles = self.create_subscription(
                AllMotorsStatus,
                '/all_motors_status',
                self.motor_angles_callback,
                QoSProfile(depth=10)
            )

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

        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )
        self.get_logger().info('Data collection node has been started.')

    def motor_angles_callback(self, msg):
        if self.data_type == 'dynamic' and (self.data_subtype == 'controlled' or self.data_subtype == 'adiabatic_global'):   
            self.last_motor_angles = self.extract_angles(msg)

            if not self.angle_callback_received:
                self.get_logger().info('Motor angles callback received first message')
                self.angle_callback_received = True
        else: # allows you to get around angle callback if you are not doing a control trajectory (only one to record angles)
            self.angle_callback_received = True

    def listener_callback(self, msg):
        if not self.angle_callback_received:
            self.get_logger().info('Waiting for first motor angle message...')
            return
        
        if self.data_type == 'dynamic' and self.data_subtype == 'controlled':
            # Store current positions + orientations
            self.store_positions(msg)
            
            # Publish new motor control inputs
            self.current_control_id += 1
            self.control_inputs = self.control_inputs_dict.get(self.current_control_id)
            if self.control_inputs is None:
                # Process data
                names = self.extract_names(msg)
                self.process_data(names)

                # Finish
                self.get_logger().info('Controlled data collection has finished.')
                self.destroy_node()
                rclpy.shutdown()
            else:
                self.publish_control_inputs()

        else:
            if not self.is_collecting:
                # Reset and start collecting new mocap data
                self.stored_positions = []
                if self.collect_orientations:
                    self.stored_orientations = []
                self.check_settled_positions = []
                self.is_collecting = True

                # Print and publish new motor control inputs
                self.current_control_id += 1
                self.get_logger().info(f'Publishing motor command {self.current_control_id} / {self.num_control_inputs}.')
                self.control_inputs = self.control_inputs_dict.get(self.current_control_id)
                if self.control_inputs is None:
                    self.get_logger().info('Data collection has finished.')
                    self.destroy_node()
                    rclpy.shutdown()
                else:
                    self.publish_control_inputs()

        if self.data_type == 'steady_state':
            if self.is_collecting and (time.time() - self.previous_time) >= self.update_period:
                self.previous_time = time.time()
                if self.check_settled():
                    # Store positions + orientations
                    self.store_positions(msg)

                    if len(self.stored_positions) >= self.sample_size:
                        # Data collection is complete and ready to be processed
                        self.is_collecting = False
                        names = self.extract_names(msg)
                        self.process_data(names)
                else:
                    self.check_settled_positions.append(self.extract_positions(msg))
        
        elif self.data_type == 'dynamic' and self.data_subtype == 'decay':
            if self.is_collecting:
                if not self.ic_settled:
                    # If it has not settled yet we do not want to start measuring the decay yet
                    self.ic_settled = self.check_settled(window=20)
                    if self.ic_settled:
                        # Remove control inputs
                        self.publish_control_inputs(control_inputs=[0.0]*6)
                        self.check_settled_positions = []
                    else:
                        self.check_settled_positions.append(self.extract_positions(msg))
                else:
                    self.store_positions(msg)

                    # Check settled because then the dynamic trajectory is done and we can continue
                    if (self.check_settled(window=30) or len(self.stored_positions) >= self.max_traj_length) and \
                    (time.time() - self.previous_time) >= self.update_period:
                        self.previous_time = time.time()
                        self.is_collecting = False
                        self.ic_settled = False
                        names = self.extract_names(msg)
                        self.process_data(names)
                    else:
                        self.check_settled_positions.append(self.extract_positions(msg))

        elif self.data_type == 'dynamic' and self.data_subtype == 'adiabatic_manual':
            # Store current positions + orientations
            self.store_positions(msg)
            
            # Publish new motor control inputs
            self.current_control_id += 1
            self.control_inputs = self.control_inputs_dict.get(self.current_control_id)
            if self.control_inputs is None:
                # Process data
                names = self.extract_names(msg)
                self.process_data(names)

                # Finish
                self.get_logger().info('Adiabatic manual data collection has finished.')
                self.destroy_node()
                rclpy.shutdown()
            else:
                self.publish_control_inputs()

        # TODO: finish this code block
        elif self.data_type == 'dynamic' and self.data_subtype == 'adiabatic_automatic':
            # Store current positions + orientations
            self.store_positions(msg)
            
            # Publish new motor control inputs
            self.current_control_id += 1
            self.control_inputs = self.control_inputs_dict.get(self.current_control_id)
            if self.control_inputs is None:
                # Process data
                names = self.extract_names(msg)
                self.process_data(names)

                # Finish
                self.get_logger().info('Adiabatic manual data collection has finished.')
                self.destroy_node()
                rclpy.shutdown()
            else:
                self.publish_control_inputs()
        
        elif self.data_type == 'dynamic' and self.data_subtype == 'adiabatic_global':
            # always store the message
            # send a new control input once settled
            # have IDs correspond
            if self.is_collecting: 
                self.store_positions(msg)

                if (self.check_settled(window=30) or len(self.stored_positions) >= self.max_traj_length) and \
                    (time.time() - self.previous_time) >= self.update_period: # if dynamic traj is done or we've exceeded max traj length
                    self.previous_time = time.time()
                    self.is_collecting = False
                    self.ic_settled = False
                    names = self.extract_names(msg)
                    self.process_data(names)

                    # send new control inputs
                    self.control_inputs = self.control_inputs_dict.get(self.current_control_id)
                    self.publish_control_inputs()
                    self.check_settled_positions = []
                else:
                    self.check_settled_positions.append(self.extract_positions(msg))


    def publish_control_inputs(self, control_inputs=None):
        if control_inputs is None:
            control_inputs = self.control_inputs
        control_message = AllMotorsControl()
        control_message.motors_control = control_inputs
        self.controls_publisher.publish(control_message)
        if self.debug:
            self.get_logger().info('Published new motor control setting: ' + str(control_inputs))

    def extract_angles(self, msg):
        angles = msg.positions
        self.angle_update_count += 1
        if self.debug:
            self.get_logger().info("Received new angle status update, number " + str(self.angle_update_count))
        return angles

    def extract_positions(self, msg):
        if self.mocap_type == 'markers':
            return msg.translations
        elif self.mocap_type == 'rigid_bodies':
            return msg.positions
        
    def extract_orientations(self, msg):
        if self.mocap_type == 'rigid_bodies':
            return msg.orientations
        elif self.mocap_type == 'markers':
            raise ValueError('Invalid request: orientations cannot be extracted with ' + self.mocap_type + ' mocap type')

        
    def extract_names(self, msg):
        if self.mocap_type == 'markers':
            raise NotImplementedError('Extracting names from markers is not implemented.')
        elif self.mocap_type == 'rigid_bodies': 
            return msg.rigid_body_names

    def store_positions(self, msg):
        self.stored_positions.append(self.extract_positions(msg))
        if self.collect_orientations:
            self.stored_orientations.append(self.extract_orientations(msg))
        self.stored_angles.append(self.last_motor_angles) #store last motor angles when position is available
        if self.debug:
            self.get_logger().info("Stored angles: "+ str(self.last_motor_angles))


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
        trajectory_csv_file = os.path.join(self.data_dir, f'trajectories/{self.data_type}/{self.results_name}.csv')
        if not os.path.exists(trajectory_csv_file):
            header = ['ID'] + [f'{axis}{name}' for name in names for axis in ['x', 'y', 'z']] + [f'{axis}{name}' for name in names for axis in ['qx', 'qy', 'qz', 'w']] + [f'phi{num+1}' for num in range(6)] 
            with open(trajectory_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
        
        if self.data_type == 'steady_state': # TODO add angle and orientation recording
            # Take average positions over all stored samples
            average_positions = [
                sum(coords) / len(self.stored_positions)
                for pos_list in zip(*self.stored_positions)
                for coords in zip(*[(pos.x, pos.y, pos.z) for pos in pos_list])
            ]
            # Save data to CSV
            with open(trajectory_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)            
                writer.writerow([self.current_control_id] + average_positions)
            if self.debug:
                self.get_logger().info('Stored new sample with positions: ' + str(average_positions) + ' [m].')
        
        elif self.data_type == 'dynamic' and self.data_subtype == 'decay': # TODO add angle and orientation recording
            # Store all positions in a CSV file
            with open(trajectory_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                for id, pos_list in enumerate(self.stored_positions):
                    row = [self.current_control_id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]]
                    writer.writerow(row)

        elif self.data_type == 'dynamic' and self.data_subtype == 'controlled':
            # Store all positions in a CSV file
            with open(trajectory_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                for id, pos_list in enumerate(self.stored_positions):
                    angle_list = self.stored_angles[id]
                    ornt_list = self.stored_orientations[id]
                    row = [id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]] + [coord for ornt in ornt_list for coord in [ornt.x, ornt.y, ornt.z, ornt.w]] + [angle for angle in angle_list]
                    writer.writerow(row)
                    
            if self.debug:
                self.get_logger().info(f'Stored the data corresponding to the {self.current_control_id}th trajectory.')

        elif self.data_type == 'dynamic' and self.data_subtype == 'adiabatic_manual': # TODO add angle and orientation recording
            # Store all positions in a CSV file
            with open(trajectory_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                for id, pos_list in enumerate(self.stored_positions):
                    row = [id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]]
                    writer.writerow(row)
            if self.debug:
                self.get_logger().info(f'Stored the data corresponding to the {self.current_control_id}th trajectory.')

        elif self.data_type == 'dynamic' and self.data_subtype == 'adiabatic_global': 
            # Store all positions in a CSV file
            with open(trajectory_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                for id, pos_list in enumerate(self.stored_positions):
                    angle_list = self.stored_angles[id]
                    ornt_list = self.stored_orientations[id]
                    row = [self.current_control_id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]] + [coord for ornt in ornt_list for coord in [ornt.x, ornt.y, ornt.z, ornt.w]] + [angle for angle in angle_list]
                    writer.writerow(row)
            if self.debug:
                self.get_logger().info(f'Stored the data corresponding to the {self.current_control_id}th trajectory.')


def main(args=None):
    rclpy.init(args=args)
    data_collection_node = DataCollectionNode()
    rclpy.spin(data_collection_node)
    data_collection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
