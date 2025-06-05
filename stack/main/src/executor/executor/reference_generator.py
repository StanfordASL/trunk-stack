import numpy as np


class ReferenceTrajectoryGenerator:
    def __init__(self, traj_config, dt):
        """
        Initialize the reference trajectory generator.
        """
        self.traj_type = traj_config["type"]
        self.traj_speed = traj_config["speed"]
        self.center = np.array(traj_config.get("center", [0.0, 0.0]))
        self.traj_params = traj_config.get("parameters", {})
        self.z_level = traj_config.get("z_level", 0.0)
        self.include_velocity = traj_config.get("include_velocity", True)
        self.trajectory = None  # Will hold the pre-sampled trajectory if requested.
        self.dt = dt

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
        if self.traj_type == "circle":
            radius = self.traj_params.get("radius", 1.0)
            theta = self.traj_speed * t
            x = self.center[0] + radius * np.cos(theta)
            y = self.center[1] + radius * np.sin(theta)
            pos = np.array([x, y, self.z_level])
            if self.include_velocity:
                # Derivatives: dx/dt = -radius * traj_speed * sin(theta), dy/dt = radius * traj_speed * cos(theta)
                vx = -radius * self.traj_speed * np.sin(theta)
                vy = radius * self.traj_speed * np.cos(theta)
                vz = 0.0
                vel = np.array([vx, vy, vz])
                return np.concatenate([pos, vel])
            else:
                return pos

        elif self.traj_type == "eight":
            amplitude = self.traj_params.get("amplitude", 1.0)
            theta = self.traj_speed * t
            x = self.center[0] + amplitude * np.sin(theta)
            y = self.center[1] + amplitude * np.sin(2 * theta)
            pos = np.array([x, y, self.z_level])
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
                if self.include_velocity:
                    T_line = L_line2 / v  # time to traverse this line segment
                    vel = (P2 - C) / T_line
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
            pos = np.array([xy[0], xy[1], self.z_level])

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
        times = np.arange(0, total_duration + self.dt, self.dt)
        traj = [self.compute_point(t) for t in times]
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
