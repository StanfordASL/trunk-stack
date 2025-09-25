import os
import csv
import time
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore
import numpy as np
from interfaces.msg import AllMotorsControl, TrunkMarkers, TrunkRigidBodies, AllMotorsStatus


def u2_to6u_mapping(u2, u4):
    # angle
    teta = np.arctan2(u4, u2)
    # radius scaling
    r_scaling = np.hypot(u2, u4)

    # reconstruct to check
    u2_calc = r_scaling * np.cos(teta)
    u4_calc = r_scaling * np.sin(teta)

    # floating‐point tolerant checks
    if not np.isclose(u2, u2_calc, atol=1e-6):
        raise AssertionError(f"u2 mismatch: {u2_calc:.6f} != {u2:.6f}")
    if not np.isclose(u4, u4_calc, atol=1e-6):
        raise AssertionError(f"u4 mismatch: {u4_calc:.6f} != {u4:.6f}")

    # the other four
    u3 = r_scaling * np.cos(teta - np.pi/3)
    u5 = r_scaling * np.sin(teta - np.pi/3)

    u1 = -r_scaling * np.sin(teta - np.pi/6)
    u6 = -r_scaling * np.cos(teta - np.pi/6)

    return u1, u2, u3, u4, u5, u6


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

        self.angle_callback_received = False  # flag
        self.angle_update_count = 0
        self.is_collecting = False
        self.ic_settled = False
        self.previous_time = time.time()
        self.current_control_id = -1
        self.stored_positions = []
        self.stored_orientations = []
        self.stored_angles = []
        self.stored_currents = []
        self.last_motor_angles = None
        self.last_motor_currents = None
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
            self.last_motor_currents = self.extract_currents(msg)
            if not self.angle_callback_received:
                self.get_logger().info('Motor angles callback received first message')
                self.angle_callback_received = True
        else:  # allows you to get around angle callback if you are not doing a control trajectory
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
                        (time.time() - self.previous_time) >= self.update_period:
                    # if dynamic traj is done or we've exceeded max traj length
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

    def extract_currents(self, msg):
        currents = msg.currents
        self.last_motor_currents = currents
        return currents

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
        self.stored_angles.append(self.last_motor_angles)  # store last motor angles when position is available
        self.stored_currents.append(self.last_motor_currents)  # store last motor currents when position is available
        if self.debug:
            self.get_logger().info("Stored angles: " + str(self.last_motor_angles))

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
            header = ['ID'] + [f'{axis}{name}' for name in names for axis in ['x', 'y', 'z']] + [f'{axis}{name}' for name in names for axis in ['qx', 'qy', 'qz', 'w']] + [f'phi{num+1}' for num in range(6)] + [f'current{num+1}' for num in range(6)]
            with open(trajectory_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
        
        if self.data_type == 'steady_state':  # TODO add angle and orientation recording
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
        
        elif self.data_type == 'dynamic' and self.data_subtype == 'decay':  # TODO add angle and orientation recording
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
                    current_list = self.stored_currents[id]
                    ornt_list = self.stored_orientations[id]
                    row = [id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]] + [coord for ornt in ornt_list for coord in [ornt.x, ornt.y, ornt.z, ornt.w]] + [angle for angle in angle_list] + [current for current in current_list]
                    writer.writerow(row)
                    
            if self.debug:
                self.get_logger().info(f'Stored the data corresponding to the {self.current_control_id}th trajectory.')

        # TODO add angle and orientation recording
        elif self.data_type == 'dynamic' and self.data_subtype == 'adiabatic_manual':
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
                    current_list = self.stored_currents[id]
                    ornt_list = self.stored_orientations[id]
                    row = [self.current_control_id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]] + [coord for ornt in ornt_list for coord in [ornt.x, ornt.y, ornt.z, ornt.w]] + [angle for angle in angle_list] + [current for current in current_list]
                    writer.writerow(row)
            if self.debug:
                self.get_logger().info(f'Stored the data corresponding to the {self.current_control_id}th trajectory.')


class DataCollectionNode_feedback(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),  # False or True
            ('sample_size', 10),  # for checking settling condition and averaging (steady state)
            ('update_period', 0.1),
            # for steady state and avoiding dynamic trajectories to interrupt each other, in [s]
            ('max_traj_length', 600),  # maximum number of samples in a dynamic trajectory
            ('data_type', 'dynamic'),  # 'steady_state' or 'dynamic'
            ('data_subtype', 'decay'),
            # 'decay' or 'controlled' or 'adiabatic_manual' or 'adiabatic_step' or 'adiabatic_jolt' for dynamic and e.g. 'circle' or 'beta' or 'uniform' for steady_state
            ('mocap_type', 'rigid_bodies'),  # 'rigid_bodies' or 'markers'
            ('control_type', 'output'),  # 'output' or 'position'
            ('results_name', 'observations'),
            ('input_num', 1),  # number of the input file type i.e. control_inputs_controlled_1
            ('collect_angles', True),  # to collect motor angle measurements
            ('collect_orientations', True)  # to collect mocap rigid body orientation data
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

        self.angle_callback_received = False  # flag
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

        self.use_feedback = True
        self.Lambda = np.diag([-5.0, -5.5])
        #self.G = np.array([[-500.0, 0, 0, 0, 0, 0, 0, 0, 0], 
        #                   [0, 0, 500.0, 0, 0, 0, 0, 0, 0]])
        
        self.G = np.array([[5000, 0, 0, 0, 0, 0, 0, 0, 0], 
                           [0, 0, -5000, 0, 0, 0, 0, 0, 0]])
        
        self.last_fb_time = None
        self.u = None

        self.fb_time_sum = 0.0  # seconds
        self.fb_time_count = 0
        self.fb_time_max = 0.0

        # --- zero-calibration (average pos over ticks 100..120) ---
        self.calib_start_tick = 100
        self.calib_end_tick = 120
        self.tick = 0
        self.calib_accum = None
        self.calib_count = 0
        self.x_offset = None  # np.array, shape = n_state
        self.calibrated = False

        if self.data_type == 'steady_state':
            control_input_csv_file = os.path.join(self.data_dir,
                                                  f'trajectories/steady_state/control_inputs_{self.data_subtype}.csv')
        elif self.data_type == 'dynamic':
            control_input_csv_file = os.path.join(self.data_dir,
                                                  f'trajectories/dynamic/control_inputs_{self.data_subtype}_{self.input_num}.csv')
        else:
            raise ValueError(
                'Invalid data type: ' + self.data_type + '. Valid options are: "steady_state" or "dynamic".')
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
            raise ValueError(
                'Invalid mocap type: ' + self.mocap_type + '. Valid options are: "rigid_bodies" or "markers".')

        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=10)
        )
        self.get_logger().info('Data collection node has been started.')
    


    def motor_angles_callback(self, msg):
        if self.data_type == 'dynamic' and (
                self.data_subtype == 'controlled' or self.data_subtype == 'adiabatic_global'):
            self.last_motor_angles = self.extract_angles(msg)

            if self.use_feedback and self.u is None and self.last_motor_angles is not None:
                self.u = list(self.last_motor_angles)

            if not self.angle_callback_received:
                self.get_logger().info('Motor angles callback received first message')
                self.angle_callback_received = True
        else:  # allows you to get around angle callback if you are not doing a control trajectory
            self.angle_callback_received = True

    def _get_state_vector(self, msg):
        # x := [x1,y1,z1, x2,y2,z2, ...] using current mocap positions
        pos_list = self.extract_positions(msg)
        return np.array([c for p in pos_list for c in (p.x, p.y, p.z)], dtype=float)

    def _get_state_vector_calibrated(self, msg):
        # returns calibrated x for control (subtract offset once available)
        x = self._get_state_vector(msg)
        if self.calibrated and self.x_offset is not None and self.x_offset.shape == x.shape:
            x = x - self.x_offset
        else:
            return np.zeros_like(x)
        return x

    def _feedback_u(self, msg, u_ref):
        now = time.time()
        dt = (now - self.last_fb_time) if self.last_fb_time is not None else 0.0
        self.last_fb_time = now
        if dt <= 0.0:
            dt = self.update_period
        dt = min(dt, 5.0 * self.update_period)

        # start high-resolution timer for the integration work
        t0 = time.perf_counter()

        x = self._get_state_vector_calibrated(msg)

        # --- pull current 6D motor angles, then extract motors 2 & 4 (indices 1 and 3) ---
        num_actuators = 6  # adjust if needed
        if self.last_motor_angles is not None:
            u_k_full = np.array(self.last_motor_angles, dtype=float)
        elif isinstance(self.u, (list, np.ndarray)) and len(self.u) == num_actuators:
            u_k_full = np.array(self.u, dtype=float)
        else:
            u_k_full = np.zeros(num_actuators, dtype=float)

        # extract reduced 2D state [u2, u4]
        u_k = np.array([u_k_full[1], u_k_full[3]], dtype=float)

        # --- reduce u_ref to 2D if it comes in as 6D ---
        u_ref_arr = np.array(u_ref, dtype=float).ravel()
        if u_ref_arr.size == 2:
            u_ref_red = u_ref_arr
        elif u_ref_arr.size == num_actuators:
            u_ref_red = np.array([u_ref_arr[1], u_ref_arr[3]], dtype=float)
        else:
            raise ValueError(f"u_ref must be length 2 or {num_actuators}, got {u_ref_arr.size}")

        # --- Crank–Nicolson step in 2D: u' = G x + Λ (u - u_ref_red) ---
        m = self.Lambda.shape[0]
        if m != 2:
            raise ValueError(f"Lambda must be 2x2 for this reduced solve, got {self.Lambda.shape}")
        if self.G.shape[0] != 2 or self.G.shape[1] != x.size:
            raise ValueError(f"G must be (2, {x.size}), got {self.G.shape}")

        I = np.eye(2)
        A = I - 0.5 * dt * self.Lambda
        B = (I + 0.5 * dt * self.Lambda) @ u_k + dt * (self.G @ x - self.Lambda @ u_ref_red)
        u_next_red = np.linalg.solve(A, B)

        # --- map 2D -> 6D (your mapping) ---
        u1, u2, u3, u4, u5, u6 = u2_to6u_mapping(u_next_red[0], u_next_red[1])
        u_next_6 = [u1, u2, u3, u4, u5, u6]

        # stop timer & record
        elapsed = time.perf_counter() - t0
        self.fb_time_sum += elapsed
        self.fb_time_count += 1
        if elapsed > self.fb_time_max:
            self.fb_time_max = elapsed
        if self.debug:
            self.get_logger().info(f"[feedback] solve took {elapsed * 1e6:.2f} µs (dt={dt * 1e3:.2f} ms)")

        # update states for next step
        self.u_red = u_next_red.tolist()  # keep 2D reduced internal state
        self.u = u_next_6  # 6D (for logging/inspection)

        return u_next_6  # publish this

    def listener_callback(self, msg):

        # --- tick & calibration window ---
        self.tick += 1

        # initialize accumulator when we first enter the window
        if self.tick == self.calib_start_tick:
            x0 = self._get_state_vector(msg)
            self.calib_accum = np.zeros_like(x0)
            self.calib_count = 0

        # accumulate raw states in [start..end]
        if self.calib_start_tick <= self.tick <= self.calib_end_tick:
            x_raw = self._get_state_vector(msg)
            # (re)shape-safety in case n_state changes unexpectedly
            if self.calib_accum is None or self.calib_accum.shape != x_raw.shape:
                self.calib_accum = np.zeros_like(x_raw)
                self.calib_count = 0
            self.calib_accum += x_raw
            self.calib_count += 1
            # finalize at end tick
            if self.tick == self.calib_end_tick:
                self.x_offset = self.calib_accum / max(self.calib_count, 1)
                self.calibrated = True
                self.get_logger().info(
                    f"[calib] x_offset set using ticks {self.calib_start_tick}-{self.calib_end_tick} "
                    f"({self.calib_count} samples)"
                )

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

                if self.fb_time_count > 0:
                    avg_us = (self.fb_time_sum / self.fb_time_count) * 1e6
                    max_us = self.fb_time_max * 1e6
                    self.get_logger().info(
                        f"[feedback] timing summary — calls: {self.fb_time_count}, "
                        f"avg: {avg_us:.2f} µs, max: {max_us:.2f} µs"
                    )

                # Process data
                names = self.extract_names(msg)
                self.process_data(names)

                # Finish
                self.get_logger().info('Controlled data collection has finished.')
                self.destroy_node()
                rclpy.shutdown()
            else:
                u_cmd = self._feedback_u(msg, self.control_inputs)
                self.publish_control_inputs(control_inputs=u_cmd)

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
                self.get_logger().info(
                    f'Publishing motor command {self.current_control_id} / {self.num_control_inputs}.')
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
                        self.publish_control_inputs(control_inputs=[0.0] * 6)
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
                        (time.time() - self.previous_time) >= self.update_period:
                    # if dynamic traj is done or we've exceeded max traj length
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
            raise ValueError(
                'Invalid request: orientations cannot be extracted with ' + self.mocap_type + ' mocap type')

    def extract_names(self, msg):
        if self.mocap_type == 'markers':
            raise NotImplementedError('Extracting names from markers is not implemented.')
        elif self.mocap_type == 'rigid_bodies':
            return msg.rigid_body_names

    def store_positions(self, msg):
        self.stored_positions.append(self.extract_positions(msg))
        if self.collect_orientations:
            self.stored_orientations.append(self.extract_orientations(msg))
        self.stored_angles.append(self.last_motor_angles)  # store last motor angles when position is available
        self.stored_currents.append(self.last_motor_currents)  # store last motor currents when position is available

        if self.debug:
            self.get_logger().info("Stored angles: " + str(self.last_motor_angles))
            self.get_logger().info("Stored currents: " + str(self.last_motor_currents))

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
            header = ['ID'] + [f'{axis}{name}' for name in names for axis in ['x', 'y', 'z']] + [f'{axis}{name}' for
                                                                                                 name in names for axis
                                                                                                 in ['qx', 'qy', 'qz',
                                                                                                     'w']] + [
                         f'phi{num + 1}' for num in range(6)]
            with open(trajectory_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        if self.data_type == 'steady_state':  # TODO add angle and orientation recording
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

        elif self.data_type == 'dynamic' and self.data_subtype == 'decay':  # TODO add angle and orientation recording
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
                    current_list = self.stored_currents[id]
                    ornt_list = self.stored_orientations[id]
                    row = [id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]] + [coord for ornt in
                                                                                                   ornt_list for coord
                                                                                                   in [ornt.x, ornt.y,
                                                                                                       ornt.z,
                                                                                                       ornt.w]] + [angle
                                                                                                                   for
                                                                                                                   angle
                                                                                                                   in
                                                                                                                   angle_list] + [current for current in current_list]
                    writer.writerow(row)

            if self.debug:
                self.get_logger().info(f'Stored the data corresponding to the {self.current_control_id}th trajectory.')

        # TODO add angle and orientation recording
        elif self.data_type == 'dynamic' and self.data_subtype == 'adiabatic_manual':
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
                    current_list = self.stored_currents[id]
                    ornt_list = self.stored_orientations[id]
                    row = [self.current_control_id] + [coord for pos in pos_list for coord in [pos.x, pos.y, pos.z]] + [
                        coord for ornt in ornt_list for coord in [ornt.x, ornt.y, ornt.z, ornt.w]] + [angle for angle in
                                                                                                      angle_list] + [current for current in current_list]
                    writer.writerow(row)
            if self.debug:
                self.get_logger().info(f'Stored the data corresponding to the {self.current_control_id}th trajectory.')


def main(args=None):
    rclpy.init(args=args)
    data_collection_node = DataCollectionNode()
    # data_collection_node = DataCollectionNode_feedback()
    rclpy.spin(data_collection_node)
    data_collection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
