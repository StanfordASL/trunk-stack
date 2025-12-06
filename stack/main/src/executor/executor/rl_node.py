#!/usr/bin/env python3
"""
ROS2 Node for Closed-Loop RL Control of Trunk Robot

This node follows the same structure as mpc_node.py but uses a trained
RL policy instead of MPC for control.

Author: Roberto
Date: 2025-01-20
"""

import os
import csv
from threading import Lock

import numpy as np
import logging

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile

from stable_baselines3 import PPO
import torch

from interfaces.msg import AllMotorsControl, TrunkRigidBodies, AllMotorsStatus


def check_control_inputs(u_opt):
    """
    Check control inputs for safety constraints, clipping to max ranges.

    Control limits match aSSM training data (in degrees):
    - Base (u1, u6): ±30°
    - Mid (u3, u5): ±50°
    - Tip (u2, u4): ±90°
    """
    tip_range, mid_range, base_range = 90, 50, 30

    u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

    # Clip to safety limits
    u2 = np.clip(u2, -tip_range, tip_range)
    u4 = np.clip(u4, -tip_range, tip_range)

    u3 = np.clip(u3, -mid_range, mid_range)
    u5 = np.clip(u5, -mid_range, mid_range)

    u1 = np.clip(u1, -base_range, base_range)
    u6 = np.clip(u6, -base_range, base_range)

    u_opt = np.array([u1, u2, u3, u4, u5, u6])

    return u_opt


def get_circle_target(t, radius=0.15, period=8.0, vel_coeff=0.5, hover_time=3.0, center=None, ramp_time=2.0):
    """
    Generate circular trajectory target in XZ plane

    SAFETY FEATURE:
    1. Stays at center for hover_time seconds before starting circle
    2. Gradually ramps up radius over ramp_time seconds to avoid discontinuous jumps

    Args:
        t: current time [s]
        radius: circle radius [m]
        period: time to complete one circle [s]
        vel_coeff: velocity coefficient (0.5 = half speed)
        hover_time: time to hover at center before starting [s]
        center: [x_c, z_c] center of circle [m], defaults to [0, 0]
        ramp_time: time to ramp from 0 to full radius [s], defaults to 2.0

    Returns:
        target: [x, z] position [m]
    """
    if center is None:
        center = np.array([0.0, 0.0], dtype=np.float32)
    else:
        center = np.array(center, dtype=np.float32)

    # SAFETY: Hover at center for first hover_time seconds
    if t < hover_time:
        return center.copy()

    # After hover, start circle centered at center
    t_circle = t - hover_time
    omega = (2 * np.pi / period) * vel_coeff

    # SAFETY: Gradually ramp up radius to avoid discontinuous jump
    # Use smooth sigmoid-like ramp: r(t) = radius * (1 - exp(-3*t/ramp_time))
    if t_circle < ramp_time:
        # Exponential ramp: starts slow, reaches ~95% at ramp_time
        ramp_factor = 1.0 - np.exp(-3.0 * t_circle / ramp_time)
        effective_radius = radius * ramp_factor
    else:
        effective_radius = radius

    x = center[0] + effective_radius * np.cos(omega * t_circle)
    z = center[1] + effective_radius * np.sin(omega * t_circle)
    return np.array([x, z], dtype=np.float32)


class RLNode(Node):
    """
    This node is responsible for running RL policy-based control.
    Structure follows mpc_node.py for easy integration.
    """

    def __init__(self, checkpoint_path):
        super().__init__('rl_node')

        self.declare_parameters(namespace='', parameters=[
            ('debug', False),
            ('control_rate', 30.0),
            ('vel_coeff', 0.5),
            ('radius', 0.15),
            ('period', 8.0),
            ('hover_time', 3.0),  # SAFETY: hover at origin before starting
            ('num_laps', 0),  # Number of laps to complete, 0 = unlimited
            ('results_name', 'rl_experiment'),
            ('alpha_smooth', 0.3),  # Low-pass filter coefficient
            ('max_rate_limit', 10.0)  # Max degrees change per timestep
        ])

        # Timing log
        self.timing_log = {
            'node_init': None,
            'first_mocap': None,
            'first_control': None,
        }

        # Parameters
        self.debug = self.get_parameter('debug').value
        self.control_rate = self.get_parameter('control_rate').value
        self.dt = 1.0 / self.control_rate
        self.vel_coeff = self.get_parameter('vel_coeff').value
        self.radius = self.get_parameter('radius').value
        self.period = self.get_parameter('period').value
        self.hover_time = self.get_parameter('hover_time').value
        self.num_laps = self.get_parameter('num_laps').value
        self.results_name = self.get_parameter('results_name').value

        self.get_logger().info(f'Control rate: {self.control_rate} Hz')
        self.get_logger().info(f'Circle params: radius={self.radius}m, period={self.period}s, vel_coeff={self.vel_coeff}')
        self.get_logger().info(f'SAFETY: Hovering at origin for {self.hover_time}s before starting trajectory')
        if self.num_laps > 0:
            self.get_logger().info(f'Will stop after {self.num_laps} lap(s)')
        else:
            self.get_logger().info('Running indefinitely (num_laps=0)')

        # Initialize CSV file
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        self.results_file = os.path.join(self.data_dir, f"trajectories/closed_loop/rl/{self.results_name}.csv")
        self.initialize_csv()

        # Control smoothing
        self.alpha_smooth = self.get_parameter('alpha_smooth').value
        self.max_rate_limit = self.get_parameter('max_rate_limit').value
        self.smooth_control_inputs = np.zeros(6, dtype=np.float32)
        self.prev_published_u = np.zeros(6, dtype=np.float32)  # For rate limiting

        # Track completion
        self.finished = False

        # RL Policy observation dimension (14D for XZ-only)
        self.n_obs = 14  # [xz_prev(2), xz_curr(2), xz_target(2), xz_target_next(2), u_prev(6)]

        # ========================================
        # Load RL Policy
        # ========================================
        self.get_logger().info(f'Loading policy from: {checkpoint_path}')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {device}')

        try:
            self.policy = PPO.load(checkpoint_path, device=device)
            self.get_logger().info('Policy loaded successfully!')
        except Exception as e:
            self.get_logger().error(f'Failed to load policy: {e}')
            raise

        # ========================================
        # State Variables
        # ========================================
        self.pos_prev = None  # Previous XZ position [m]
        self.pos_curr = None  # Current XZ position [m]
        self.u_prev = np.zeros(6, dtype=np.float32)  # Previous control (SCALED, in degrees)

        # Full mocap data for CSV logging
        self.rigid_bodies_data = None  # Store all 3 rigid bodies (positions + orientations)

        # Motor status data for CSV logging
        self.motor_positions = np.zeros(6, dtype=np.float32)  # Actual motor positions [deg]
        self.motor_currents = np.zeros(6, dtype=np.float32)   # Motor currents [mA]

        # CSV row counter
        self.csv_row_id = 0

        # Rest position (measured during hover_time)
        self.rest_position = None  # [x, z] rest position measured at start
        self.rest_samples = []  # Accumulate samples during hover to compute average

        # Lap tracking
        self.laps_completed = 0
        self.prev_angle = None  # Previous angle for lap detection
        self.cumulative_angle = 0.0  # Total angle rotated (accumulated, not wrapped)

        # Execution
        self.callback_group = ReentrantCallbackGroup()

        # ========================================
        # Subscribe to OptiTrack
        # ========================================
        self.mocap_subscription = self.create_subscription(
            TrunkRigidBodies,
            '/trunk_rigid_bodies',
            self.mocap_listener_callback,
            QoSProfile(depth=3),
            callback_group=self.callback_group
        )
        self.get_logger().info('Subscribed to /trunk_rigid_bodies (using rigid body #3 for tip position)')

        # ========================================
        # Subscribe to Motor Status
        # ========================================
        self.motor_status_subscription = self.create_subscription(
            AllMotorsStatus,
            '/all_motors_status',
            self.motor_status_callback,
            QoSProfile(depth=3),
            callback_group=self.callback_group
        )
        self.get_logger().info('Subscribed to /all_motors_status for motor positions and currents')

        # ========================================
        # Publisher for Control Commands
        # ========================================
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            '/all_motors_control',
            QoSProfile(depth=3)
        )

        # ========================================
        # Timer for Control Loop
        # ========================================
        self.controller_period = self.dt
        self.control_timer = self.create_timer(
            self.controller_period,
            self.control_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info(f'RL node started with control frequency: {self.control_rate:.2f} Hz')

        # Initialize timing
        self.clock = self.get_clock()
        self.timing_log['node_init'] = self.clock.now().nanoseconds / 1e9
        self.start_time = None

    def mocap_listener_callback(self, msg):
        """
        Callback to process mocap data, extracting tip position (rigid body #3)
        and storing all rigid bodies data for CSV logging
        """
        # Timing
        if self.timing_log['first_mocap'] is None:
            current_time = self.clock.now().nanoseconds / 1e9
            self.timing_log['first_mocap'] = current_time
            delay = current_time - self.timing_log['node_init']
            self.get_logger().info(f'First mocap received {delay:.3f}s after node init')

        if self.debug:
            self.get_logger().info(f'Received mocap data: {msg.positions}')

        # Extract rigid body #3 (tip) position - index 2
        if len(msg.positions) < 3:
            self.get_logger().warn(
                f'Expected at least 3 rigid bodies, got {len(msg.positions)}. Skipping...',
                throttle_duration_sec=1.0
            )
            return

        # Store all rigid bodies data for CSV logging
        self.rigid_bodies_data = {
            'positions': msg.positions,
            'orientations': msg.orientations
        }

        rigid_body_3 = msg.positions[2]  # Third rigid body (index 2)
        self.pos_curr = np.array([
            rigid_body_3.x,
            rigid_body_3.z
        ], dtype=np.float32)

        # Initialize start time on first valid position
        if self.start_time is None:
            self.start_time = self.clock.now().nanoseconds / 1e9
            self.pos_prev = self.pos_curr.copy()
            self.get_logger().info(f'Initial position: x={self.pos_curr[0]:.3f}, z={self.pos_curr[1]:.3f}')

    def motor_status_callback(self, msg):
        """
        Callback to process motor status data (positions and currents)
        """
        if len(msg.positions) >= 6:
            # Convert from radians to degrees for consistency with control commands
            self.motor_positions = np.rad2deg(np.array(msg.positions[:6], dtype=np.float32))

        if len(msg.currents) >= 6:
            self.motor_currents = np.array(msg.currents[:6], dtype=np.float32)

    def control_callback(self):
        """
        Main control loop - called at fixed frequency
        """
        if self.finished:
            return

        if self.pos_curr is None or self.start_time is None:
            # Waiting for first OptiTrack data
            return

        # Timing
        if self.timing_log['first_control'] is None:
            current_time = self.clock.now().nanoseconds / 1e9
            self.timing_log['first_control'] = current_time
            delay = current_time - self.timing_log['first_mocap']
            self.get_logger().info(f'First control executed {delay:.3f}s after first mocap')

        # ========================================
        # 1. Compute elapsed time
        # ========================================
        current_time = self.clock.now().nanoseconds / 1e9
        elapsed = current_time - self.start_time

        # ========================================
        # 1b. Measure rest position during hover (with u=0)
        # ========================================
        if elapsed < self.hover_time:
            # During hover time: FORCE u=[0,0,0,0,0,0] and measure rest position
            self.rest_samples.append(self.pos_curr.copy())

            # Publish zero commands to establish rest configuration
            u_safe = np.zeros(6, dtype=np.float32)
            self.publish_control_inputs(u_safe)

            # Save to CSV (during hover, motor positions are being measured)
            self.save_to_csv()

            # Update state
            self.pos_prev = self.pos_curr.copy()
            self.u_prev = np.zeros(6, dtype=np.float32)  # Keep u_prev = 0 during hover

            # Logging (every 1 second during hover)
            if int(elapsed * self.control_rate) % int(self.control_rate) == 0:
                self.get_logger().info(
                    f't={elapsed:.2f}s | HOVER MODE | '
                    f'pos=[{self.pos_curr[0]:.3f}, {self.pos_curr[1]:.3f}] | '
                    f'u=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | '
                    f'samples={len(self.rest_samples)}'
                )

            return  # Skip normal control loop during hover

        elif self.rest_position is None:
            # At end of hover, compute average rest position
            self.rest_position = np.mean(self.rest_samples, axis=0).astype(np.float32)
            self.get_logger().info('=' * 60)
            self.get_logger().info(
                f'REST POSITION MEASURED: x={self.rest_position[0]:.4f}m, z={self.rest_position[1]:.4f}m '
                f'(averaged over {len(self.rest_samples)} samples)'
            )
            self.get_logger().info('Starting circle trajectory centered at rest position...')
            self.get_logger().info('=' * 60)

        # ========================================
        # 2. Get target positions from circle trajectory
        # ========================================
        target_curr = get_circle_target(elapsed, self.radius, self.period, self.vel_coeff, self.hover_time, self.rest_position)
        # CRITICAL: Use fixed 30Hz lookahead (0.0333s) regardless of control_rate
        # The policy was trained with 30Hz observations, so target_next must maintain that temporal spacing
        lookahead_dt = 1.0 / 30.0  # Always use 30Hz lookahead (0.0333s)
        target_next = get_circle_target(elapsed + lookahead_dt, self.radius, self.period, self.vel_coeff, self.hover_time, self.rest_position)

        # ========================================
        # 3. Construct RL observation (14D)
        # ========================================
        # Observation: [xz_prev(2), xz_curr(2), xz_target(2), xz_target_next(2), u_prev(6)]
        # IMPORTANT: Normalize u_prev back to [-1, 1] range for observation!
        # u_prev is stored as scaled values, but policy expects normalized inputs
        control_scale = np.array([30.0, 90.0, 50.0, 90.0, 50.0, 30.0], dtype=np.float32)
        u_prev_normalized = self.u_prev / control_scale

        obs = np.hstack([
            self.pos_prev,      # XZ previous [2]
            self.pos_curr,      # XZ current [2]
            target_curr,        # XZ target current [2]
            target_next,        # XZ target next [2]
            u_prev_normalized   # Previous control NORMALIZED to [-1,1] [6]
        ]).astype(np.float32)

        assert obs.shape == (14,), f"Observation shape mismatch: {obs.shape} != (14,)"

        # ========================================
        # 4. Policy prediction
        # ========================================
        action, _ = self.policy.predict(obs, deterministic=True)

        # ========================================
        # 5. Scale action to control limits
        # ========================================
        control_scale = np.array([30.0, 90.0, 50.0, 90.0, 50.0, 30.0], dtype=np.float32)
        u_new = action * control_scale

        # ========================================
        # 6. Low-pass filter (smooth control) - MUST MATCH TRAINING!
        # ========================================
        # CRITICAL: The training environment applies smoothing with alpha=0.3
        # We MUST apply the same smoothing here to match training conditions!
        self.smooth_control_inputs = (
            self.alpha_smooth * u_new +
            (1 - self.alpha_smooth) * self.smooth_control_inputs
        )

        # ========================================
        # 7. Apply safety clipping (AFTER smoothing, as in training)
        # ========================================
        u_safe = check_control_inputs(self.smooth_control_inputs)

        # ========================================
        # 7b. Rate limiting (prevent sudden jumps)
        # ========================================
        if self.max_rate_limit > 0:
            delta_u = u_safe - self.prev_published_u
            delta_u_clipped = np.clip(delta_u, -self.max_rate_limit, self.max_rate_limit)
            u_safe = self.prev_published_u + delta_u_clipped
            self.prev_published_u = u_safe.copy()

        # ========================================
        # 8. Publish control
        # ========================================
        self.publish_control_inputs(u_safe)

        # ========================================
        # 9. Save to CSV
        # ========================================
        self.save_to_csv()

        # ========================================
        # 10. Lap detection (only after hover time)
        # ========================================
        if self.num_laps > 0 and elapsed > self.hover_time:
            # TIME-BASED lap detection (more reliable than angle-based for small radii)
            # Calculate expected time for one lap
            lap_duration = self.period / self.vel_coeff

            # Calculate how many laps should have been completed based on time
            time_since_start = elapsed - self.hover_time
            laps_from_time = int(time_since_start / lap_duration)

            if laps_from_time > self.laps_completed:
                self.laps_completed = laps_from_time
                self.get_logger().info(
                    f'===== LAP {self.laps_completed}/{self.num_laps} COMPLETED at t={elapsed:.2f}s ===== '
                    f'(time-based: {time_since_start:.2f}s / {lap_duration:.2f}s per lap)'
                )

                # Check if we've completed all requested laps
                if self.laps_completed >= self.num_laps:
                    self.get_logger().info(
                        f'All {self.num_laps} lap(s) completed! Stopping controller...'
                    )
                    self.finished = True
                    return

        # ========================================
        # 11. Update state for next iteration
        # ========================================
        self.pos_prev = self.pos_curr.copy()

        # CRITICAL: Store SCALED actions to match training environment!
        # Training env stores: u_prev = action * control_scale (not normalized!)
        self.u_prev = u_new.copy()  # Store SCALED action (not normalized)
        
        # ========================================
        # 12. Logging (every 1 second)
        # ========================================
        if int(elapsed * self.control_rate) % int(self.control_rate) == 0:
            error = self.pos_curr - target_curr
            error_norm = np.linalg.norm(error)
            self.get_logger().info(
                f't={elapsed:.2f}s | '
                f'pos=[{self.pos_curr[0]:.3f}, {self.pos_curr[1]:.3f}]m | '
                f'target=[{target_curr[0]:.3f}, {target_curr[1]:.3f}]m | '
                f'error={error_norm:.4f}m | '
                f'u=[{u_safe[0]:.1f}, {u_safe[1]:.1f}, '
                f'{u_safe[2]:.1f}, {u_safe[3]:.1f}, '
                f'{u_safe[4]:.1f}, {u_safe[5]:.1f}]° (deg)'
            )

    def publish_control_inputs(self, control_inputs):
        """
        Publish control inputs to /all_motors_control
        """
        if self.debug:
            self.get_logger().info(f"Publishing control inputs: {control_inputs}")

        control_message = AllMotorsControl()
        control_message.motors_control = tuple(control_inputs.tolist())
        self.controls_publisher.publish(control_message)

    def initialize_csv(self):
        """
        Initialize the CSV file with headers matching the ROB format
        """
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        with open(self.results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Header: ID, positions (3 RBs), orientations (3 RBs), motor angles, motor currents
            writer.writerow([
                'ID',
                'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3',
                'qx1', 'qy1', 'qz1', 'w1', 'qx2', 'qy2', 'qz2', 'w2', 'qx3', 'qy3', 'qz3', 'w3',
                'phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6',
                'current1', 'current2', 'current3', 'current4', 'current5', 'current6'
            ])

    def save_to_csv(self):
        """
        Save control data to CSV file in ROB format
        Format: ID, x1,y1,z1, x2,y2,z2, x3,y3,z3, qx1,qy1,qz1,w1, qx2,qy2,qz2,w2, qx3,qy3,qz3,w3, phi1-6, current1-6
        """
        # Prepare row data
        row = [self.csv_row_id]

        # Add positions for all 3 rigid bodies (x, y, z for each)
        if self.rigid_bodies_data is not None:
            for i in range(3):
                if i < len(self.rigid_bodies_data['positions']):
                    pos = self.rigid_bodies_data['positions'][i]
                    row.extend([pos.x, pos.y, pos.z])
                else:
                    row.extend([0.0, 0.0, 0.0])
        else:
            # No mocap data yet, fill with zeros
            row.extend([0.0] * 9)

        # Add orientations (quaternions) for all 3 rigid bodies (qx, qy, qz, w for each)
        if self.rigid_bodies_data is not None:
            for i in range(3):
                if i < len(self.rigid_bodies_data['orientations']):
                    quat = self.rigid_bodies_data['orientations'][i]
                    row.extend([quat.x, quat.y, quat.z, quat.w])
                else:
                    row.extend([0.0, 0.0, 0.0, 1.0])
        else:
            # No mocap data yet, fill with identity quaternions
            row.extend([0.0, 0.0, 0.0, 1.0] * 3)

        # Add motor positions (phi1-phi6) in degrees
        row.extend(self.motor_positions.tolist())

        # Add motor currents (current1-current6)
        row.extend(self.motor_currents.tolist())

        # Write to CSV
        with open(self.results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # Increment row counter
        self.csv_row_id += 1


def main(args=None):
    """
    Run the ROS2 RL node with multi-threaded executor
    """
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ROS2 Node for Closed-Loop RL Control')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained RL policy checkpoint (.zip file)'
    )

    # Parse known args (ROS2 args handled by rclpy)
    parsed_args, unknown = parser.parse_known_args()

    # Check checkpoint exists
    from pathlib import Path
    checkpoint_path = Path(parsed_args.checkpoint)
    if not checkpoint_path.exists():
        print(f'ERROR: Checkpoint not found: {checkpoint_path}')
        return 1

    # Initialize ROS2
    rclpy.init(args=args)

    try:
        rl_node = RLNode(str(checkpoint_path))

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(rl_node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            rl_node.get_logger().info('Keyboard interrupt, shutting down.')
        finally:
            rl_node.destroy_node()
            rclpy.shutdown()

    except Exception as e:
        print(f'ERROR: {e}')
        return 1

    return 0


if __name__ == '__main__':
    exit(main())