import numpy as np
import pandas as pd

class ReferenceTrajectoryGenerator:
    def __init__(self, traj_config, dt):
        """
        Initialize the reference trajectory generator.
        """
        self.traj_type = traj_config["type"]
        self.traj_speed = traj_config["speed"]
        self.center = np.array(traj_config.get("center", [0.0, 0.0]))
        self.traj_params = traj_config.get("parameters", {})
        self.z_level = self.traj_params.get("z_level", 0.0)
        self.include_velocity = traj_config.get("include_velocity", True)
        self.rest_pos = traj_config["rest_pos"]
        self.trajectory = None  # Will hold the pre-sampled trajectory if requested.
        self.dt = dt
        self.times = None
        self.csv_path = traj_config["csv_path"] if "csv_path" in traj_config else None
        self.csv_pt_ct = 0  # counter for csv points
        
        if self.csv_path is None:
                raise ValueError("CSV path must be provided in mpc_initializer_node for 'controlled_csv' trajectory type.")
        else:
            self.data = np.loadtxt(self.csv_path, delimiter=',', skiprows=1)


    def _init_flower_resampling(self):
        """Precompute equidistant star points & velocities for one full cycle."""
        # parameters
        amplitude = self.traj_params.get("amplitude", 1.0)
        inner_ratio = self.traj_params.get("inner_ratio", 0.5)
        m = 6  # number of prongs
        N = self.traj_params.get("flower_samples", 500)  # resolution

        # helper functions
        def radius(theta):
            R = amplitude / (1 + inner_ratio)
            return R * (1 + inner_ratio * np.cos(m * theta))

        def xy(theta):
            r = radius(theta)
            return np.array([self.center[0] + r * np.cos(theta),
                             self.center[1] + r * np.sin(theta)])

        # 1. sample theta finely and build arc‑length
        thetas = np.linspace(0, 2 * np.pi, N * 10)
        pts = np.stack([xy(t) for t in thetas], axis=0)
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate(([0], np.cumsum(ds)))
        L = s[-1]

        # 2. choose N equidistant arc‑length targets
        s_targ = np.linspace(0, L, N)
        thetas_e = np.interp(s_targ, s, thetas)

        # 3. final equidistant points
        pts_e = np.stack([xy(t) for t in thetas_e], axis=0)  # shape (N,2)

        # 4. approximate velocities by forward difference
        period = 2 * np.pi / self.traj_speed
        dt = period / N
        vels = (np.roll(pts_e, -1, axis=0) - pts_e) / dt  # (N,2)

        # stash on the object
        self._flower_pts = pts_e
        self._flower_vels = vels
        self._flower_N = N
        self._flower_period = period
        self._flower_init = True

    def compute_point(self, t):
        """
        Compute the reference point at time t (position, and velocity if enabled).

        Args:
            t (float): The time at which to compute the trajectory.

        Returns:
            np.ndarray: If include_velocity is False: shape (3,) [x, y, z].
                        If include_velocity is True: shape (6,) [x, y, z, vx, vy, vz].
        """
        

        # print(f"traj_type: {self.traj_type}")
        if self.traj_type == "circle":
            radius = self.traj_params.get("radius", 1.0)
            theta = self.traj_speed * t
            x = self.center[0] + radius * np.cos(theta)
            y = self.center[1] + radius * np.sin(theta)
            pos = np.array([x, y, self.z_level])
            pos += self.rest_pos
            if self.include_velocity:
                # Derivatives: dx/dt = -radius * traj_speed * sin(theta), dy/dt = radius * traj_speed * cos(theta)
                vx = -radius * self.traj_speed * np.sin(theta)
                vy = radius * self.traj_speed * np.cos(theta)
                vz = 0.0
                vel = np.array([vx, vy, vz])
                return np.concatenate([pos, vel])
            else:
                return pos
        
        elif self.traj_type == "spiral":
            """
            Spiral trajectory that starts at the origin and moves outward in radius
            while ascending to z_level over a fixed duration.

            Uses the SAME conventions as other trajectories:
              - radius comes from traj_params["radius"]
              - z comes from self.z_level
              - angular speed is self.traj_speed
              - position ordering: [x, z, y]
              - velocity ordering: [vx, vz, vy]

            Additional parameter:
              - duration (float): total spiral time [s]
            """
            radius = self.traj_params.get("radius", 1.0)
            duration = float(self.traj_params.get("duration", 10.0))
            duration = max(duration, 1e-9)

            # normalized time in [0, 1]
            if t <= 0.0:
                tau = 0.0
            elif t >= duration:
                tau = 1.0
            else:
                tau = t / duration

            # linearly increase radius and height
            r = radius * tau
            z = self.z_level * tau

            # angular position (loops set implicitly by traj_speed)
            theta = self.traj_speed * min(t, duration)

            x = self.center[0] + r * np.cos(theta)
            y = self.center[1] + r * np.sin(theta)

            pos = np.array([x, z, y]) + self.rest_pos

            if self.include_velocity:
                if t >= duration:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                else:
                    r_dot = radius / duration
                    z_dot = self.z_level / duration
                    theta_dot = self.traj_speed

                    vx = r_dot * np.cos(theta) - r * np.sin(theta) * theta_dot
                    vy = r_dot * np.sin(theta) + r * np.cos(theta) * theta_dot
                    vz = z_dot

                vel = np.array([vx, vz, vy])
                return np.concatenate([pos, vel])
            else:
                return pos


        elif self.traj_type == "controlled_csv":
            # CSV has columns: ID,x1,y1,z1,x2,y2,z2,x3,y3,z3,qx1,qy1,qz1,w1,qx2,qy2,qz2,w2,qx3,qy3,qz3,w3,phi1,phi2,phi3,phi4,phi5,phi6,current1,current2,current3,current4,current5,current6
            # we want to just track the tip position (x3,y3,z3)

            #TODO: should we subtract the rest position here? No, roshan's model takes raw positions.
            pos = np.array(self.data[self.csv_pt_ct, 7:10])  # x3, y3, z3

            # rest_positions = np.array([0.10368,-0.14215,0.11343,0.10405,-0.27971,0.11191,0.10730,-0.40798,0.12463]) #updated 10/28/25
            # pos = pos - rest_positions[6:9]  # center around rest position
            # print(f"pos shape: {pos.shape}")
            # print(f'pos: {pos}')
            # print(f'csv_pt_ct: {self.csv_pt_ct}')

            # csv is always sampled at 100 Hz (0.01 s), so we need to align that with the dt of the mpc
            # i.e. if dt = 0.02, then we need to skip every other point
            skip_rate = int(self.dt / 0.01)
            self.csv_pt_ct += (skip_rate)  # already incremented

            # TODO: need to make sure the dt is right
            # print(pos)
            return pos

        elif self.traj_type == "circle_with_ramp":
            radius = self.traj_params.get("radius", 1.0)
            ramp_duration = 4.0
            v_tangent = radius / ramp_duration
            theta = self.traj_speed * max(t - ramp_duration, 0.0)  # offset time for circle start
            
            if t < ramp_duration:
                # Linear ramp phase before circle
                frac = t / ramp_duration
                
                # Ramp x from center to center + radius
                x = self.center[0] + radius * frac
                
                # Ramp z from initial z to target z_level
                # Assuming you want to ramp from some initial z value to self.z_level
                # You may need to define self.z_initial or adjust this logic
                z = self.z_level * frac  # Ramps from 0 to z_level
                
                # Ramp y - stays at center during ramp (but ramping from 0 velocity)
                y = self.center[1]
                
                # Output in x, z, y order
                pos = np.array([x, z, y])
                pos += self.rest_pos
                # print(self.z_level)
                
                if self.include_velocity:
                    vx = v_tangent  # x velocity during ramp
                    vz = self.z_level / ramp_duration  # z velocity during ramp
                    vy = 0.0  # y stays constant during ramp
                    # Velocity also in x, z, y order
                    vel = np.array([vx, vz, vy])
                    return np.concatenate([pos, vel])
                else:
                    return pos
            else:
                # Circle phase
                x = self.center[0] + radius * np.cos(theta)
                y = self.center[1] + radius * np.sin(theta)
                z = self.z_level
                
                # Output in x, z, y order
                pos = np.array([x, z, y])
                pos += self.rest_pos
                
                if self.include_velocity:
                    vx = -radius * self.traj_speed * np.sin(theta)
                    vy = radius * self.traj_speed * np.cos(theta)
                    vz = 0.0  # Fixed: was showing random variables
                    # Velocity also in x, z, y order
                    vel = np.array([vx, vz, vy])
                    return np.concatenate([pos, vel])
                else:
                    return pos


        elif self.traj_type == "eight":
            amplitude = self.traj_params.get("amplitude", 1.0)
            theta = self.traj_speed * t
            x = self.center[0] + amplitude * np.sin(theta)
            y = self.center[1] + amplitude * np.sin(2 * theta)
            pos = np.array([x, self.z_level, y])
            pos += self.rest_pos
            if self.include_velocity:
                # Derivatives: dx/dt = amplitude * traj_speed * cos(theta),
                #              dy/dt = 2 * amplitude * traj_speed * cos(2*theta)
                vx = amplitude * self.traj_speed * np.cos(theta)
                vy = 2 * amplitude * self.traj_speed * np.cos(2 * theta)
                vz = 0.0
                vel = np.array([vx, vy, vz])
                return np.concatenate([pos, vel])
            else:
                return pos

        elif self.traj_type == "pacman":
            # Pacman trajectory: a circular arc with a missing wedge ("mouth")
            # replaced by two line segments: one from the arc endpoint to the center,
            # and one from the center to the other endpoint.
            radius = self.traj_params.get("radius", 1.0)
            mouth_angle = self.traj_params.get("mouth_angle", np.pi / 4)  # default mouth angle
            # Define arc boundaries:
            start_angle = mouth_angle / 2  # arc starts at this angle
            end_angle = 2 * np.pi - mouth_angle / 2  # arc ends at this angle
            arc_span = end_angle - start_angle  # equals (2π - mouth_angle)
            # Compute segment lengths:
            L_arc = radius * arc_span  # length of the circular arc
            L_line1 = radius  # from endpoint at end_angle to center
            L_line2 = radius  # from center to endpoint at start_angle
            L_total = L_arc + L_line1 + L_line2  # total length of the trajectory
            # Assume a constant linear speed along the entire trajectory.
            # For the arc, linear speed v = radius * traj_speed.
            v = self.traj_speed * radius
            # Total distance traveled along the path (with periodicity)
            s = (v * t) % L_total
            if s < L_arc:
                # On the arc segment.
                # The relation: s = radius * (angle - start_angle)
                angle = start_angle + s / radius
                x = self.center[0] + radius * np.cos(angle)
                y = self.center[1] + radius * np.sin(angle)
                pos = np.array([x, y, self.z_level])
                pos += self.rest_pos
                if self.include_velocity:
                    # Angular rate is constant: d(angle)/dt = self.traj_speed.
                    angle_dot = self.traj_speed
                    vx = -radius * np.sin(angle) * angle_dot
                    vy = radius * np.cos(angle) * angle_dot
                    vel = np.array([vx, vy, 0.0])
                    return np.concatenate([pos, vel])
                else:
                    return pos
            elif s < L_arc + L_line1:
                # On the first line segment: from the arc endpoint at end_angle to the center.
                s_line = s - L_arc
                u = s_line / L_line1  # normalized parameter [0, 1]
                P1 = np.array([self.center[0] + radius * np.cos(end_angle),
                               self.center[1] + radius * np.sin(end_angle),
                               self.z_level])
                C = np.array([self.center[0], self.center[1], self.z_level])
                pos = (1 - u) * P1 + u * C
                pos += self.rest_pos
                if self.include_velocity:
                    T_line = L_line1 / v  # time to traverse this line segment
                    vel = (C - P1) / T_line
                    return np.concatenate([pos, vel])
                else:
                    return pos
            else:
                # On the second line segment: from the center to the arc endpoint at start_angle.
                s_line = s - (L_arc + L_line1)
                u = s_line / L_line2  # normalized parameter [0, 1]
                P2 = np.array([self.center[0] + radius * np.cos(start_angle),
                               self.center[1] + radius * np.sin(start_angle),
                               self.z_level])
                C = np.array([self.center[0], self.center[1], self.z_level])
                pos = (1 - u) * C + u * P2
                pos += self.rest_pos
                if self.include_velocity:
                    T_line = L_line2 / v  # time to traverse this line segment
                    vel = (P2 - C) / T_line
                    return np.concatenate([pos, vel])
                else:
                    return pos

        elif self.traj_type == "pacman_with_ramp":
            radius = self.traj_params.get("radius", 1.0)
            mouth_angle = self.traj_params.get("mouth_angle", np.pi / 4)
            ramp_duration = 4.0
            v_ramp = radius / ramp_duration
            
            # Define angles and lengths like in pacman
            start_angle = mouth_angle / 2
            end_angle = 2 * np.pi - mouth_angle / 2
            arc_span = end_angle - start_angle
            L_arc = radius * arc_span
            L_line1 = radius
            L_line2 = radius
            L_total = L_arc + L_line1 + L_line2
            v = self.traj_speed * radius
            
            if t < ramp_duration:
                # Ramp phase: move toward starting point of arc
                frac = t / ramp_duration
                
                # Target point at end of ramp (start of arc)
                x_target = self.center[0] + radius * np.cos(start_angle)
                y_target = self.center[1] + radius * np.sin(start_angle)
                
                # Ramp x from center to target x position
                x = self.center[0] + (x_target - self.center[0]) * frac
                
                # Ramp y from center to target y position  
                y = self.center[1] + (y_target - self.center[1]) * frac
                
                # Ramp z from 0 to z_level
                z = self.z_level * frac
                
                # Output in x, z, y order
                pos = np.array([x, z, y])
                pos += self.rest_pos
                
                if self.include_velocity:
                    vx = (x_target - self.center[0]) / ramp_duration
                    vz = self.z_level / ramp_duration
                    vy = (y_target - self.center[1]) / ramp_duration
                    # Velocity also in x, z, y order
                    vel = np.array([vx, vz, vy])
                    return np.concatenate([pos, vel])
                else:
                    return pos
            else:
                # Shift time and compute regular pacman trajectory
                t_shifted = t - ramp_duration
                s = (v * t_shifted) % L_total
                
                if s < L_arc:
                    angle = start_angle + s / radius
                    x = self.center[0] + radius * np.cos(angle)
                    y = self.center[1] + radius * np.sin(angle)
                    z = self.z_level
                    
                    # Output in x, z, y order
                    pos = np.array([x, z, y])
                    pos += self.rest_pos
                    
                    if self.include_velocity:
                        angle_dot = self.traj_speed
                        vx = -radius * np.sin(angle) * angle_dot
                        vy = radius * np.cos(angle) * angle_dot
                        vz = 0.0
                        # Velocity in x, z, y order
                        vel = np.array([vx, vz, vy])
                        return np.concatenate([pos, vel])
                    else:
                        return pos
                        
                elif s < L_arc + L_line1:
                    s_line = s - L_arc
                    u = s_line / L_line1
                    
                    P1_x = self.center[0] + radius * np.cos(end_angle)
                    P1_y = self.center[1] + radius * np.sin(end_angle)
                    P1_z = self.z_level
                    
                    C_x = self.center[0]
                    C_y = self.center[1]
                    C_z = self.z_level
                    
                    # Interpolate each coordinate
                    x = (1 - u) * P1_x + u * C_x
                    y = (1 - u) * P1_y + u * C_y
                    z = (1 - u) * P1_z + u * C_z
                    
                    # Output in x, z, y order
                    pos = np.array([x, z, y])
                    pos += self.rest_pos
                    
                    if self.include_velocity:
                        T_line = L_line1 / v
                        vx = (C_x - P1_x) / T_line
                        vy = (C_y - P1_y) / T_line
                        vz = (C_z - P1_z) / T_line
                        # Velocity in x, z, y order
                        vel = np.array([vx, vz, vy])
                        return np.concatenate([pos, vel])
                    else:
                        return pos
                else:
                    s_line = s - (L_arc + L_line1)
                    u = s_line / L_line2
                    
                    P2_x = self.center[0] + radius * np.cos(start_angle)
                    P2_y = self.center[1] + radius * np.sin(start_angle)
                    P2_z = self.z_level
                    
                    C_x = self.center[0]
                    C_y = self.center[1]
                    C_z = self.z_level
                    
                    # Interpolate each coordinate
                    x = (1 - u) * C_x + u * P2_x
                    y = (1 - u) * C_y + u * P2_y
                    z = (1 - u) * C_z + u * P2_z
                    
                    # Output in x, z, y order
                    pos = np.array([x, z, y])
                    pos += self.rest_pos
                    
                    if self.include_velocity:
                        T_line = L_line2 / v
                        vx = (P2_x - C_x) / T_line
                        vy = (P2_y - C_y) / T_line
                        vz = (P2_z - C_z) / T_line
                        # Velocity in x, z, y order
                        vel = np.array([vx, vz, vy])
                        return np.concatenate([pos, vel])
                    else:
                        return pos

        elif self.traj_type == "pacman_3d":
            radius = self.traj_params.get("radius", 1.0)
            mouth_angle = self.traj_params.get("mouth_angle", np.pi / 4)
            
            # Define angles for the pacman arc
            start_angle = mouth_angle / 2
            end_angle = 2 * np.pi - mouth_angle / 2
            arc_span = end_angle - start_angle
            
            # Calculate path lengths
            # The two mouth lines are vertical ramps
            L_line1 = self.z_level  # First mouth line: vertical UP from origin
            L_arc = radius * arc_span  # Arc portion at constant z_level
            L_line2 = self.z_level  # Second mouth line: vertical DOWN to origin
            L_total = L_line1 + L_arc + L_line2
            
            v = self.traj_speed * radius  # Speed along the path
            s = (v * t) % L_total  # Current position along path
            
            # Calculate the starting and ending points of the arc at z_level
            # These are where the mouth lines connect to the arc
            x_start = self.center[0] + radius * np.cos(start_angle)
            y_start = self.center[1] + radius * np.sin(start_angle)
            x_end = self.center[0] + radius * np.cos(end_angle)
            y_end = self.center[1] + radius * np.sin(end_angle)
            
            if s < L_line1:
                # First mouth line: vertical ramp UP from origin to start of arc
                u = s / L_line1  # Progress along first line [0, 1]
                
                # Interpolate from origin to start point of arc
                x = self.center[0] + u * (x_start - self.center[0])
                y = self.center[1] + u * (y_start - self.center[1])
                z = u * self.z_level
                
                # Output in x, z, y order
                pos = np.array([x, z, y])
                pos += self.rest_pos
                
                if self.include_velocity:
                    T_line1 = L_line1 / v
                    vx = (x_start - self.center[0]) / T_line1
                    vz = self.z_level / T_line1
                    vy = (y_start - self.center[1]) / T_line1
                    # Velocity in x, z, y order
                    vel = np.array([vx, vz, vy])
                    return np.concatenate([pos, vel])
                else:
                    return pos
                    
            elif s < L_line1 + L_arc:
                # Arc portion: pacman shape at constant z_level
                s_arc = s - L_line1
                angle = start_angle + s_arc / radius
                
                x = self.center[0] + radius * np.cos(angle)
                y = self.center[1] + radius * np.sin(angle)
                z = self.z_level  # Constant z during arc
                
                # Output in x, z, y order
                pos = np.array([x, z, y])
                pos += self.rest_pos
                
                if self.include_velocity:
                    angle_dot = self.traj_speed
                    vx = -radius * np.sin(angle) * angle_dot
                    vz = 0.0  # No vertical motion during arc
                    vy = radius * np.cos(angle) * angle_dot
                    # Velocity in x, z, y order
                    vel = np.array([vx, vz, vy])
                    return np.concatenate([pos, vel])
                else:
                    return pos
                    
            else:
                # Second mouth line: vertical ramp DOWN from end of arc back to origin
                s_line2 = s - (L_line1 + L_arc)
                u = s_line2 / L_line2  # Progress along second line [0, 1]
                
                # Interpolate from end point of arc back to origin
                x = x_end + u * (self.center[0] - x_end)
                y = y_end + u * (self.center[1] - y_end)
                z = (1 - u) * self.z_level  # Ramp down from z_level to 0
                
                # Output in x, z, y order
                pos = np.array([x, z, y])
                pos += self.rest_pos
                
                if self.include_velocity:
                    T_line2 = L_line2 / v
                    vx = (self.center[0] - x_end) / T_line2
                    vz = -self.z_level / T_line2  # Negative (going down)
                    vy = (self.center[1] - y_end) / T_line2
                    # Velocity in x, z, y order
                    vel = np.array([vx, vz, vy])
                    return np.concatenate([pos, vel])
                else:
                    return pos



        elif self.traj_type == "flower":
            # lazy‐init the equidistant star
            if not getattr(self, "_flower_init", False):
                self._init_flower_resampling()

            # wrap t into [0, period)
            tp = (t % self._flower_period)
            # find index in [0..N)
            idx = int((tp / self._flower_period) * self._flower_N) % self._flower_N

            # build position
            xy = self._flower_pts[idx]
            pos = np.array([xy[0], self.z_level, xy[1]])
            pos += self.rest_pos

            if self.include_velocity:
                vx, vy = self._flower_vels[idx]
                vel = np.array([vx, vy, 0.0])
                return np.concatenate([pos, vel])
            else:
                return pos

        else:
            raise ValueError(f"Unknown trajectory type: {self.traj_type}")

    def sample_trajectory(self, total_duration):
        """
        Sample the trajectory over a given duration and store it internally.

        Args:
            total_duration (float): Total time duration over which to sample.

        Returns:
            np.ndarray: Array of shape (num_samples, d) where d is 3 or 6.
        """
        self.times = np.arange(0, total_duration + self.dt, self.dt)
        traj = [self.compute_point(t) for t in self.times]
        self.trajectory = np.array(traj)
        return self.trajectory

    def eval(self):
        """
        Get a future segment of the trajectory starting at start_time.

        Args:

        Returns:
            np.ndarray: Array of shape (num_steps, d), where d is 3 if velocities are not included,
                        or 6 if they are.
        """
        return self.trajectory
