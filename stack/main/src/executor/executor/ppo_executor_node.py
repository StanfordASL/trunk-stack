#!/usr/bin/env python3
import os
import re
import glob
import csv
import time
from typing import Optional, Tuple

import numpy as np

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.callback_groups import ReentrantCallbackGroup  # type: ignore
from rclpy.executors import MultiThreadedExecutor  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore

from interfaces.msg import AllMotorsControl, TrunkRigidBodies, AllMotorsStatus
from stable_baselines3 import PPO

# IMPORTANT: must match your package layout. This is the same idea as Koopman nodes.
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
    PPO closed-loop executor.

    Publishing is made to match your training node + Koopman executor:
      - QoSProfile(depth=3)
      - AllMotorsControl().motors_control = tuple(...)
      - explicit print() each time we publish
    """

    def __init__(self):
        super().__init__("ppo_executor_node")

        self.declare_parameters(namespace="", parameters=[
            ("debug", False),
            ("config_name", "ppo_config"),
            ("results_name", "ppo_test_experiment"),
            ("checkpoint_update_num", -1),          # override YAML; -1 => use YAML; YAML -1 => latest
            ("deterministic", True),                # override YAML if desired
            ("require_motor_status_to_start", True) # set False if you want to start with only mocap
        ])

        self.debug = bool(self.get_parameter("debug").value)
        self.config_name = str(self.get_parameter("config_name").value)
        self.results_name = str(self.get_parameter("results_name").value)
        self.ckpt_update_override = int(self.get_parameter("checkpoint_update_num").value)
        self.det_override = self.get_parameter("deterministic").value
        self.require_motor_status_to_start = bool(self.get_parameter("require_motor_status_to_start").value)

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

        # Safety (match training)
        self.u_max = np.array(self.ppo_cfg.get("motor_limits_deg", [50, 100, 70, 100, 70, 50]), dtype=np.float32)
        self.u_min = -self.u_max
        self.du_max_deg_per_sec = float(self.ppo_cfg.get("du_max_deg_per_sec", 100.0))
        self.du_max_step = float(self.du_max_deg_per_sec * self.dt)
        self.du_max = np.ones(6, dtype=np.float32) * self.du_max_step

        # Deterministic inference
        self.deterministic = bool(self.ppo_cfg.get("deterministic", True))
        # allow param override
        if self.det_override is not None:
            self.deterministic = bool(self.det_override)

        # Callback group + QoS (match Koopman style)
        self.callback_group = ReentrantCallbackGroup()
        qos_sub = QoSProfile(depth=3)
        qos_pub = QoSProfile(depth=3)

        # Subscribers
        self._tip_index: Optional[int] = None
        self.latest_tip_position: Optional[np.ndarray] = None
        self.latest_motor_positions: Optional[np.ndarray] = None
        self._got_motor_status = False

        self.mocap_subscription = self.create_subscription(
            TrunkRigidBodies,
            "/trunk_rigid_bodies",
            self.mocap_callback,
            qos_sub,
            callback_group=self.callback_group
        )

        self.motor_status_subscription = self.create_subscription(
            AllMotorsStatus,
            "/all_motors_status",
            self.motor_status_callback,
            qos_sub,
            callback_group=self.callback_group
        )

        # Publisher (MATCH Koopman)
        self.controls_publisher = self.create_publisher(
            AllMotorsControl,
            "/all_motors_control",
            qos_pub
        )

        # Reference trajectory (unchanged)
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
            f"rate={self.control_hz:.1f} Hz, dt={self.dt:.4f}s"
        )

        # Logging file (koopman-compatible first cols)
        self.results_file = os.path.join(self.data_dir, f"trajectories/closed_loop/ppo/{self.results_name}.csv")
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        self._init_csv()

        # Runtime state
        self.u_cmd = np.zeros(6, dtype=np.float32)
        self.du_prev = np.zeros(6, dtype=np.float32)
        self.p_prev: Optional[np.ndarray] = None
        self.ise = 0.0
        self.k = 0
        self.started = False
        self.finished = False
        self.tick = 0

        # Call once immediately (like Koopman calls mpc_callback once)
        self.control_loop()

        # Timer (like Koopman)
        self.exec_timer = self.create_timer(
            self.dt,
            self.control_loop,
            callback_group=self.callback_group
        )

    # ---------------- ROS callbacks ----------------
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

        i = self._tip_index
        pos = msg.positions[i]
        self.latest_tip_position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

    def motor_status_callback(self, msg: AllMotorsStatus):
        self.latest_motor_positions = np.array(msg.positions, dtype=np.float32)
        if not self._got_motor_status:
            self._got_motor_status = True
            self.get_logger().info("Received first /all_motors_status message.")

    # ---------------- Publishing (MATCH Koopman) ----------------
    def publish_control_inputs(self, u_cmd: np.ndarray):
        # mimic koopman: explicit print
        print(f"Publishing control inputs: {u_cmd}")

        control_message = AllMotorsControl()
        # MATCH koopman exactly: tuple(...)
        control_message.motors_control = tuple([float(x) for x in u_cmd.tolist()])
        self.controls_publisher.publish(control_message)

        if self.debug:
            self.get_logger().info("Published new motor control setting: " + str(u_cmd))

    # ---------------- CSV ----------------
    def _init_csv(self):
        with open(self.results_file, mode="w", newline="") as file:
            w = csv.writer(file)
            w.writerow([
                "topt", "uopt", "y",          # koopman-compatible
                "p_star", "v", "du", "err2", "ise", "reward", "obs"
            ])

    def _append_csv(self, t: float, u: np.ndarray, y: np.ndarray,
                    p_star: np.ndarray, v: np.ndarray, du: np.ndarray,
                    err2: float, ise: float, reward: float, obs: np.ndarray):
        with open(self.results_file, mode="a", newline="") as file:
            w = csv.writer(file)
            w.writerow([
                [float(t)],
                [float(x) for x in u.tolist()],
                [float(x) for x in y.tolist()],
                [float(x) for x in p_star.tolist()],
                [float(x) for x in v.tolist()],
                [float(x) for x in du.tolist()],
                float(err2),
                float(ise),
                float(reward),
                [float(x) for x in obs.tolist()],
            ])

    # ---------------- PPO obs + action mapping (match training) ----------------
    def _normalize_obs(self, p, v, p_star, u_cmd, du_prev) -> np.ndarray:
        p_scale = 0.5
        v_scale = 0.5
        p_n = p / p_scale
        v_n = v / v_scale
        p_star_n = p_star / p_scale
        u_n = u_cmd / self.u_max
        du_n = du_prev / (self.du_max_step + 1e-8)
        return np.concatenate([p_n, v_n, p_star_n, u_n, du_n]).astype(np.float32)

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

    # ---------------- Main loop ----------------
    def control_loop(self):
        if self.finished:
            return

        self.tick += 1

        # Throttle: print once per second to prove timer is running
        if self.tick % int(max(1, round(self.control_hz))) == 0:
            self.get_logger().info(
                f"tick={self.tick} started={self.started} have_mocap={self.latest_tip_position is not None} "
                f"have_motor_status={self.latest_motor_positions is not None} k={self.k}/{self.T}"
            )

        # Wait for required signals
        if self.latest_tip_position is None:
            return
        if self.require_motor_status_to_start and (self.latest_motor_positions is None):
            return

        if not self.started:
            self.started = True
            self.p_prev = self.latest_tip_position.copy()
            self.get_logger().info("PPO control loop STARTED.")
            # immediate publish a tiny nonzero "ping" to confirm topic wiring (safe, tiny)
            # Comment this out if you absolutely don't want it.
            ping = np.zeros(6, dtype=np.float32)
            ping[0] = 0.01
            self.publish_control_inputs(ping)
            self.publish_control_inputs(np.zeros(6, dtype=np.float32))

        # Done?
        if self.k >= self.T:
            self.get_logger().info("Trajectory complete. Commanding zero and finishing.")
            self.publish_control_inputs(np.zeros(6, dtype=np.float32))
            self.finished = True
            return

        # Build obs
        p = self.latest_tip_position.copy()
        p_star = self.ref_traj[self.k, :3].copy()
        v = (p - self.p_prev) / float(self.dt) if self.p_prev is not None else np.zeros(3, dtype=np.float32)
        self.p_prev = p.copy()

        obs = self._normalize_obs(p, v, p_star, self.u_cmd, self.du_prev)

        # Policy
        action, _ = self.model.predict(obs, deterministic=self.deterministic)

        # Command mapping + safety
        u_next, du = self._apply_action_mapping(action, self.u_cmd)

        # Publish (THIS is what you should see on /all_motors_control)
        self.publish_control_inputs(u_next)

        # Metrics + logging
        err = p - p_star
        err2 = float(np.dot(err, err))
        reward = -err2
        self.ise += err2 * float(self.dt)

        t = float(self.ref_times[self.k]) if self.k < len(self.ref_times) else float(self.k * self.dt)
        self._append_csv(t, u_next, p, p_star, v, du, err2, self.ise, reward, obs)

        # Update state
        self.u_cmd = u_next
        self.du_prev = du
        self.k += 1


def main(args=None):
    rclpy.init(args=args)
    node = PPOExecutorNode()

    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        try:
            node.publish_control_inputs(np.zeros(6, dtype=np.float32))
            time.sleep(0.1)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
