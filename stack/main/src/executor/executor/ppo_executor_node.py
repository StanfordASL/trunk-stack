#!/usr/bin/env python3
import os
import re
import glob
import csv
import time
from typing import Optional, Tuple, List

import numpy as np

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore

from interfaces.msg import AllMotorsControl, TrunkRigidBodies, AllMotorsStatus
from stable_baselines3 import PPO

# Use the exact same ReferenceTrajectoryGenerator implementation as Koopman/MPC
from .reference_generator import ReferenceTrajectoryGenerator


def _find_latest_checkpoint(models_dir: str) -> str:
    patt = os.path.join(models_dir, "ppo_after_update_*.zip")
    paths = sorted(glob.glob(patt))
    if not paths:
        raise FileNotFoundError(f"No PPO checkpoints found under: {patt}")

    def extract_update(p: str) -> int:
        m = re.search(r"ppo_after_update_(\d+)\.zip$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths_by_update = sorted(paths, key=lambda p: extract_update(p))
    if extract_update(paths_by_update[-1]) >= 0:
        return paths_by_update[-1]

    return max(paths, key=lambda p: os.path.getmtime(p))


def _checkpoint_for_update(models_dir: str, update_num: int) -> str:
    path = os.path.join(models_dir, f"ppo_after_update_{int(update_num)}.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Requested checkpoint does not exist: {path}")
    return path


class PPOExecutorNode(Node):
    """
    PPO closed-loop executor with training-matching RL logic and training-matching timing.

    Key guarantees:
      - 100 Hz loop enforced with perf_counter (like training step pacing)
      - No per-step disk I/O (buffer rows, write at end) to avoid slowing loop
      - Action mapping + obs normalization matches training exactly
      - ReferenceTrajectoryGenerator used unchanged
      - Stops when trajectory ends (k == T)
    """

    def __init__(self):
        super().__init__("ppo_executor_node")

        self.declare_parameters(namespace="", parameters=[
            ("debug", False),
            ("config_name", "ppo_config"),
            ("results_name", "ppo_test_experiment"),
            ("checkpoint_update_num", -1),  # overrides YAML if != -1
            ("deterministic", True),
            ("start_requires_motor_status", True),
            ("flush_every_n", 0),  # 0 => only write at end; else chunk-write every N steps
        ])

        self.debug = bool(self.get_parameter("debug").value)
        self.config_name = str(self.get_parameter("config_name").value)
        self.results_name = str(self.get_parameter("results_name").value)
        self.ckpt_update_override = int(self.get_parameter("checkpoint_update_num").value)
        self.det_override = bool(self.get_parameter("deterministic").value)
        self.start_requires_motor_status = bool(self.get_parameter("start_requires_motor_status").value)
        self.flush_every_n = int(self.get_parameter("flush_every_n").value)

        self.data_dir = os.getenv("TRUNK_DATA", "/home/trunk/Documents/trunk-stack/stack/main/data")
        self.config_path = os.path.join(self.data_dir, "trajectories/closed_loop/ppo", f"{self.config_name}.yaml")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config YAML not found: {self.config_path}")

        import yaml
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        if "trajectory" not in cfg or "ppo" not in cfg:
            raise ValueError(f"Config must contain 'trajectory' and 'ppo' sections: {self.config_path}")

        self.traj_cfg = cfg["trajectory"]
        self.ppo_cfg = cfg["ppo"]

        # Timing (match training)
        self.control_hz = float(self.ppo_cfg.get("control_hz", 100.0))
        self.dt = float(self.ppo_cfg.get("dt", 0.01))
        if abs(self.dt - 1.0 / self.control_hz) > 5e-4:
            self.get_logger().warn(
                f"dt ({self.dt}) not close to 1/control_hz ({1.0/self.control_hz}). "
                "Proceeding, but for 100Hz you want dt=0.01."
            )

        # Safety (match training)
        self.u_max = np.array(self.ppo_cfg.get("motor_limits_deg", [50, 100, 70, 100, 70, 50]), dtype=np.float32)
        self.u_min = -self.u_max
        self.du_max_deg_per_sec = float(self.ppo_cfg.get("du_max_deg_per_sec", 100.0))
        self.du_max_step = float(self.du_max_deg_per_sec * self.dt)
        self.du_max = np.ones(6, dtype=np.float32) * self.du_max_step

        # Deterministic inference (default True for deployment)
        self.deterministic = bool(self.ppo_cfg.get("deterministic", True))
        self.deterministic = bool(self.det_override)

        # ROS I/O (match koopman/training style)
        qos = QoSProfile(depth=10)

        self.mocap_subscription = self.create_subscription(
            TrunkRigidBodies, "/trunk_rigid_bodies", self.mocap_callback, qos
        )
        self.motor_status_subscription = self.create_subscription(
            AllMotorsStatus, "/all_motors_status", self.motor_status_callback, qos
        )
        self.controls_publisher = self.create_publisher(
            AllMotorsControl, "/all_motors_control", qos
        )

        self.latest_tip_position: Optional[np.ndarray] = None
        self.latest_motor_positions: Optional[np.ndarray] = None
        self._tip_index: Optional[int] = None
        self._got_motor_status = False

        # Reference trajectory (unchanged implementation)
        self.ref_gen = ReferenceTrajectoryGenerator(self.traj_cfg, self.dt)
        duration = float(self.traj_cfg["duration"])
        self.ref_gen.sample_trajectory(duration)
        self.ref_traj = np.array(self.ref_gen.trajectory, dtype=np.float32)
        self.ref_times = np.array(self.ref_gen.times, dtype=np.float32)
        if self.ref_traj.ndim != 2 or self.ref_traj.shape[1] < 3:
            raise ValueError(f"Reference trajectory shape unexpected: {self.ref_traj.shape}")
        self.T = self.ref_traj.shape[0]

        # Load PPO model
        models_dir = os.path.join(self.data_dir, "ppo", "models")
        yaml_update_num = int(self.ppo_cfg.get("checkpoint_update_num", -1))
        use_update_num = self.ckpt_update_override if self.ckpt_update_override != -1 else yaml_update_num
        if use_update_num == -1:
            ckpt_path = _find_latest_checkpoint(models_dir)
        else:
            ckpt_path = _checkpoint_for_update(models_dir, use_update_num)

        self.get_logger().info(f"Loading PPO checkpoint: {ckpt_path}")
        self.model = PPO.load(ckpt_path, device="cpu")

        self.get_logger().info(
            f"PPO Executor ready. Traj length={self.T} steps, duration≈{self.ref_times[-1]:.2f}s, "
            f"target rate={self.control_hz:.1f} Hz, dt={self.dt:.4f}s"
        )

        # Results path
        self.results_file = os.path.join(self.data_dir, f"trajectories/closed_loop/ppo/{self.results_name}.csv")
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        # Runtime state (match training logic)
        self.u_cmd = np.zeros(6, dtype=np.float32)
        self.du_prev = np.zeros(6, dtype=np.float32)
        self.p_prev: Optional[np.ndarray] = None

        self.ise = 0.0
        self.k = 0
        self.started = False
        self.finished = False

        # Buffer rows to avoid slowing loop
        self._rows: List[List[object]] = []
        self._write_header()

        # Main loop runs in main() via self.run()

    # ---------- ROS callbacks ----------
    def mocap_callback(self, msg: TrunkRigidBodies):
        if self._tip_index is None:
            names = list(msg.rigid_body_names)
            tip_name = "3"
            if tip_name in names:
                self._tip_index = names.index(tip_name)
            else:
                candidates = [i for i, n in enumerate(names) if str(n).strip().endswith(tip_name)]
                self._tip_index = candidates[0] if candidates else max(0, len(names) - 1)
            self.get_logger().info(f"Tip rigid body index set to {self._tip_index} (name='{names[self._tip_index]}').")

        pos = msg.positions[self._tip_index]
        self.latest_tip_position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

    def motor_status_callback(self, msg: AllMotorsStatus):
        self.latest_motor_positions = np.array(msg.positions, dtype=np.float32)
        if not self._got_motor_status:
            self._got_motor_status = True
            self.get_logger().info("Received first /all_motors_status message.")

    # ---------- Publishing (same style as training node) ----------
    def publish_motor_command(self, u_cmd: np.ndarray):
        m = AllMotorsControl()
        m.motors_control = [float(x) for x in u_cmd.tolist()]
        self.controls_publisher.publish(m)

    # ---------- Training-matching obs normalization ----------
    def _normalize_obs(self, p, v, p_star, u_cmd, du_prev) -> np.ndarray:
        p_scale = 0.5
        v_scale = 0.5
        p_n = p / p_scale
        v_n = v / v_scale
        p_star_n = p_star / p_scale
        u_n = u_cmd / self.u_max
        du_n = du_prev / (self.du_max_step + 1e-8)
        return np.concatenate([p_n, v_n, p_star_n, u_n, du_n]).astype(np.float32)

    # ---------- Training-matching action mapping + safety ----------
    def _apply_action_mapping(self, a: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        du_pos = np.minimum(self.du_max, self.u_max - u)
        du_neg = np.maximum(-self.du_max, self.u_min - u)

        du = np.zeros(6, dtype=np.float32)
        pos_mask = a >= 0
        du[pos_mask] = a[pos_mask] * du_pos[pos_mask]
        du[~pos_mask] = a[~pos_mask] * np.abs(du_neg[~pos_mask])

        u_next = np.clip(u + du, self.u_min, self.u_max).astype(np.float32)
        return u_next, du

    # ---------- CSV ----------
    def _write_header(self):
        with open(self.results_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "topt", "uopt", "y",          # koopman-compatible
                "p_star", "v", "du", "err2", "ise", "reward", "obs", "action"
            ])

    def _flush_rows(self):
        if not self._rows:
            return
        with open(self.results_file, "a", newline="") as f:
            w = csv.writer(f)
            w.writerows(self._rows)
        self._rows = []

    # ---------- Main run loop (training-style pacing) ----------
    def run(self):
        self.get_logger().info("Waiting for mocap (and motor status if required) before starting PPO loop...")
        t0 = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            have_mocap = self.latest_tip_position is not None
            have_motor = self.latest_motor_positions is not None
            if have_mocap and (have_motor or not self.start_requires_motor_status):
                break
            if time.time() - t0 > 5.0:
                self.get_logger().warn(f"Still waiting... mocap={have_mocap}, motor_status={have_motor}")
                t0 = time.time()

        if not rclpy.ok():
            return

        self.started = True
        self.p_prev = self.latest_tip_position.copy()
        self.get_logger().info("PPO loop STARTED.")

        # Training did not require motor status to set u_cmd, but for deployment this avoids mismatch:
        # if you want strict "command from zero", comment this out.
        if self.latest_motor_positions is not None:
            self.u_cmd = np.clip(self.latest_motor_positions.astype(np.float32), self.u_min, self.u_max)
            if self.debug:
                self.get_logger().info(f"Initialized internal u_cmd from motor status: {np.round(self.u_cmd, 2).tolist()}")

        next_t = time.perf_counter()
        last_log = time.perf_counter()

        while rclpy.ok() and (self.k < self.T):
            step_start = time.perf_counter()

            # Keep subscriptions fresh
            rclpy.spin_once(self, timeout_sec=0.0)

            # Read state
            p = self.latest_tip_position.copy()
            p_star = self.ref_traj[self.k, :3].copy()
            v = (p - self.p_prev) / float(self.dt) if self.p_prev is not None else np.zeros(3, dtype=np.float32)
            self.p_prev = p.copy()

            obs = self._normalize_obs(p, v, p_star, self.u_cmd, self.du_prev)

            # Policy
            action, _ = self.model.predict(obs, deterministic=self.deterministic)

            # Map + safety
            u_next, du = self._apply_action_mapping(action, self.u_cmd)

            # Publish
            self.publish_motor_command(u_next)

            # Metrics
            err = p - p_star
            err2 = float(np.dot(err, err))
            reward = -err2
            self.ise += err2 * float(self.dt)

            # Buffer CSV row (koopman-compatible first cols)
            t = float(self.ref_times[self.k]) if self.k < len(self.ref_times) else float(self.k * self.dt)
            row = [
                [t],
                [float(x) for x in u_next.tolist()],
                [float(x) for x in p.tolist()],
                [float(x) for x in p_star.tolist()],
                [float(x) for x in v.tolist()],
                [float(x) for x in du.tolist()],
                float(err2),
                float(self.ise),
                float(reward),
                [float(x) for x in obs.tolist()],
                [float(x) for x in np.asarray(action, dtype=np.float32).reshape(-1).tolist()],
            ]
            self._rows.append(row)

            # Optional chunk flush
            if self.flush_every_n > 0 and (len(self._rows) >= self.flush_every_n):
                self._flush_rows()

            # Advance
            self.u_cmd = u_next
            self.du_prev = du
            self.k += 1

            # 1 Hz log showing effective rate
            now = time.perf_counter()
            if now - last_log >= 1.0:
                eff_hz = 1.0 / max(1e-6, (now - step_start))
                self.get_logger().info(
                    f"k={self.k}/{self.T} ISE={self.ise:.4f} last_step_time={(now-step_start)*1000:.2f}ms"
                )
                last_log = now

            # Enforce dt pacing (training-style)
            next_t += self.dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # If we’re behind, resync to avoid drift explosion
                next_t = time.perf_counter()

        # Done
        self.get_logger().info("Trajectory complete. Commanding zero and saving results.")
        self.publish_motor_command(np.zeros(6, dtype=np.float32))
        self._flush_rows()
        self.get_logger().info(f"Saved: {self.results_file}")


def main(args=None):
    rclpy.init(args=args)
    node = PPOExecutorNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        try:
            node.publish_motor_command(np.zeros(6, dtype=np.float32))
            time.sleep(0.1)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
