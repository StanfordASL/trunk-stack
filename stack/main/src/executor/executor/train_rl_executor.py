#!/usr/bin/env python3
import os
import time
import glob
import csv
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from interfaces.msg import AllMotorsControl, TrunkRigidBodies, AllMotorsStatus

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


# -----------------------------
# Config
# -----------------------------

@dataclass
class PPOConfig:
    # Timing
    control_hz: float = 100.0
    dt: float = 0.01

    # Init: start the policy command at the target command (+ tiny noise)
    init_u_noise_std_deg: float = 0.25     # small, e.g. 0.1–0.5 deg


    # Episode / rollout
    episode_seconds: float = 10.0          # 10s
    ramp_seconds: float = 2.0              # 2s ramp in/out (only used when not in sanity mode)

    # PPO
    n_steps: int = 8000                    # only change vs SB3 defaults
    total_rollouts: int = 180                # 180 = 4 hours of training
    seed: int = 0

    # Safety (rate limit)
    du_max_deg_per_sec: float = 100.0       # 50 deg/s

    # Motor limits (ordering you provided)
    # Order: BASE TIP MID TIP MID BASE
    # limits: base +/-50, mid +/-70, tip +/-100
    motor_limits_deg: Tuple[float, float, float, float, float, float] = (50.0, 100.0, 70.0, 100.0, 70.0, 50.0)

    # Reward
    lambda_action: float = 0.0             # tracking-only for now

    # Data locations
    data_root: str = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    ppo_dirname: str = "ppo"
    trajectories_subdir: str = "trajectories/dynamic"

    # Dataset file numbers
    dataset_nums: Tuple[int, ...] = (430, 431, 432, 433) # for debug small dataset (320, 321)

    # Mocap rigid body name for tip
    tip_body_name: str = "3"

    # ROS topics (update if your stack uses different names)
    mocap_topic: str = "/trunk_rigid_bodies"
    motor_status_topic: str = "/all_motors_status"
    motor_control_topic: str = "/all_motors_control"


# -----------------------------
# Utility: dataset loading
# -----------------------------

def _find_csvs(cfg: PPOConfig) -> Dict[str, List[str]]:
    base = os.path.join(cfg.data_root, cfg.trajectories_subdir)
    control_files = []
    obs_files = []
    for n in cfg.dataset_nums:
        patt_control = os.path.join(base, f"control_inputs_controlled_{n}*")
        patt_obs = os.path.join(base, f"observations_controlled_{n}*")
        control_files += sorted(glob.glob(patt_control))
        obs_files += sorted(glob.glob(patt_obs))

    control_files = [p for p in control_files if p.lower().endswith(".csv")]
    obs_files = [p for p in obs_files if p.lower().endswith(".csv")]
    if len(control_files) == 0 or len(obs_files) == 0:
        raise FileNotFoundError(
            f"Could not find dataset CSVs under {base}. "
            f"Expected control_inputs_controlled_{{430..433}}.csv and observations_controlled_{{430..433}}.csv"
        )
    return {"control": control_files, "obs": obs_files}


def _read_csv_header(path: str) -> List[str]:
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header


def _load_control_csv(path: str) -> np.ndarray:
    header = _read_csv_header(path)
    h = [s.strip().lower() for s in header]

    def idx_of(prefix: str, i: int) -> Optional[int]:
        key = f"{prefix}{i}"
        try:
            return h.index(key)
        except ValueError:
            return None

    cols = []
    for prefix in ["u", "phi", "motor", "pos"]:
        candidate = []
        for i in range(1, 7):
            j = idx_of(prefix, i)
            if j is None:
                candidate = []
                break
            candidate.append(j)
        if len(candidate) == 6:
            cols = candidate
            break

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if len(cols) == 6:
        return data[:, cols].astype(np.float32)

    # Fallback: assume col0 is ID and next 6 are motor commands
    if data.shape[1] < 7:
        raise ValueError(f"Control CSV {path} has too few columns ({data.shape[1]}). Can't infer 6 motor commands.")
    return data[:, 1:7].astype(np.float32)


def _load_obs_csv_for_tip(path: str) -> np.ndarray:
    header = _read_csv_header(path)
    h = [s.strip() for s in header]

    def find_col(name: str) -> int:
        if name in h:
            return h.index(name)
        lower = [s.lower() for s in h]
        if name.lower() in lower:
            return lower.index(name.lower())
        raise ValueError(f"Could not find column '{name}' in {path}. Header starts with: {h[:20]}")

    ix = find_col("x3")
    iy = find_col("y3")
    iz = find_col("z3")

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data[:, [ix, iy, iz]].astype(np.float32)


def _sample_clip(control_u: np.ndarray, target_p: np.ndarray, clip_len: int) -> Tuple[np.ndarray, np.ndarray, int]:
    T = min(control_u.shape[0], target_p.shape[0])
    if T < clip_len + 1:
        raise ValueError(f"Trajectory too short: T={T} < clip_len={clip_len} in dataset.")
    start = np.random.randint(0, T - clip_len)
    u_clip = control_u[start:start + clip_len].copy()
    p_clip = target_p[start:start + clip_len].copy()
    return u_clip, p_clip, start


# -----------------------------
# Stats helpers
# -----------------------------

def _traj_stats(u: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """
    u: [T,6] motor positions (deg)
    Returns dict with per-motor:
      max_abs_u, mean_abs_u, max_abs_speed, mean_abs_speed
    speed computed as finite difference du/dt (deg/s), length T-1 (we pad with 0 for convenience).
    """
    u = np.asarray(u, dtype=np.float32)
    max_abs_u = np.max(np.abs(u), axis=0)
    mean_abs_u = np.mean(np.abs(u), axis=0)

    if u.shape[0] >= 2:
        du = np.diff(u, axis=0) / float(dt)
        max_abs_speed = np.max(np.abs(du), axis=0)
        mean_abs_speed = np.mean(np.abs(du), axis=0)
    else:
        max_abs_speed = np.zeros(6, dtype=np.float32)
        mean_abs_speed = np.zeros(6, dtype=np.float32)

    return {
        "max_abs_u": max_abs_u,
        "mean_abs_u": mean_abs_u,
        "max_abs_speed": max_abs_speed,
        "mean_abs_speed": mean_abs_speed,
    }


def _fmt_vec(v: np.ndarray, fmt: str = "{:.2f}") -> str:
    return "[" + ", ".join(fmt.format(float(x)) for x in v.tolist()) + "]"


# -----------------------------
# Gym environment wrapping ROS
# -----------------------------

class TrunkTrajectoryTrackingEnv(gym.Env):
    """
    Real-robot environment:
      obs = [p(3), v(3), p*(3), u(6), du_prev(6)]  => 21D
      action a in [-1,1]^6 => mapped to delta motor command per-step (deg), state-dependent bound,
                             then integrated and clipped to absolute motor limits.
      reward = -||p - p*||^2  (tracking only)
    """

    metadata = {}

    def __init__(self, node: Node, cfg: PPOConfig,
                 dataset_pairs: List[Tuple[np.ndarray, np.ndarray]],
                 sanity_check: bool,
                 sanity_noise_std: float):
        super().__init__()
        self.node = node
        self.cfg = cfg
        self.dataset_pairs = dataset_pairs
        self.sanity_check = bool(sanity_check)
        self.sanity_noise_std = float(sanity_noise_std)

        self.dt = cfg.dt
        self.control_hz = cfg.control_hz
        self.steps_per_episode = int(round(cfg.episode_seconds * cfg.control_hz))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        self.u_max = np.array(cfg.motor_limits_deg, dtype=np.float32)
        self.u_min = -self.u_max

        self.du_max_step = float(cfg.du_max_deg_per_sec * cfg.dt)
        self.du_max = np.ones(6, dtype=np.float32) * self.du_max_step

        # runtime state
        self._episode_step = 0
        self._p_prev = None
        self._du_prev = np.zeros(6, dtype=np.float32)
        self._u_cmd = np.zeros(6, dtype=np.float32)

        # targets
        self._p_star = None            # [steps,3]
        self._u_star = None            # [steps,6] target control clip for stats
        self._u_start = None           # [6,]
        self._clip_id = None           # (dataset_idx, start_row)

        # ISE / return
        self._ise = 0.0
        self._episode_return = 0.0

        # expose last transition for callback logging
        self.last_transition: Optional[Dict[str, Any]] = None

        # for sanity stats: store executed u trajectory for this episode
        self._u_rl_hist: List[np.ndarray] = []

    def _normalize_obs(self, p, v, p_star, u_cmd, du_prev) -> np.ndarray:
        # Choose conservative scales (won't be perfect, but far better than raw mixed units)
        # If your trunk tip typically stays within ~0.5 m workspace, this is reasonable.
        p_scale = 0.5
        v_scale = 0.5  # m/s scale guess; adjust if needed

        p_n = p / p_scale
        v_n = v / v_scale
        p_star_n = p_star / p_scale

        # Motor terms: normalize by per-motor limits and per-step delta limit
        u_n = u_cmd / self.u_max
        du_n = du_prev / (self.du_max_step + 1e-8)

        return np.concatenate([p_n, v_n, p_star_n, u_n, du_n]).astype(np.float32)


    def _get_tip_position_real(self) -> np.ndarray:
        p = self.node.latest_tip_position
        if p is None:
            raise RuntimeError("No mocap tip position received yet.")
        return p.astype(np.float32)

    def _spin_until_ready(self, timeout_s: float = 1.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self.node, timeout_sec=0.0)
            if (self.node.latest_tip_position is not None) and (self.node.latest_motor_positions is not None):
                return
            time.sleep(0.001)

    def _publish_motor_command(self, u_cmd: np.ndarray) -> None:
        # In sanity check mode, we NEVER publish nonzero commands.
        if self.sanity_check:
            self.node.publish_motor_command(np.zeros(6, dtype=np.float32))
        else:
            self.node.publish_motor_command(u_cmd)

    def _ramp_command(self, u_from: np.ndarray, u_to: np.ndarray, seconds: float) -> None:
        # Ramp is only used in non-sanity mode. In sanity mode, we hold zeros.
        if self.sanity_check:
            self.node.publish_motor_command(np.zeros(6, dtype=np.float32))
            return

        steps = int(round(seconds * self.control_hz))
        if steps <= 0:
            self._publish_motor_command(u_to)
            return
        for k in range(steps):
            alpha = (k + 1) / steps
            u = (1.0 - alpha) * u_from + alpha * u_to
            self._publish_motor_command(u)
            rclpy.spin_once(self.node, timeout_sec=0.0)
            time.sleep(self.dt)

    def _sample_new_episode_target(self) -> None:
        idx = np.random.randint(0, len(self.dataset_pairs))
        control_u, target_p = self.dataset_pairs[idx]
        u_clip, p_clip, start = _sample_clip(control_u, target_p, self.steps_per_episode)
        self._p_star = p_clip
        self._u_star = u_clip
        self._u_start = u_clip[0].copy()
        self._clip_id = (idx, start)

    def _get_tip_position(self) -> np.ndarray:
        if self.sanity_check:
            # synthetic feedback: target + Gaussian noise
            eps = np.random.randn(3).astype(np.float32) * self.sanity_noise_std
            return (self._p_star[self._episode_step] + eps).astype(np.float32)
        return self._get_tip_position_real()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # choose target clip
        self._sample_new_episode_target()

        # init internal controller state
        u0 = np.zeros(6, dtype=np.float32)
        self._u_cmd = u0.copy()
        self._du_prev = np.zeros(6, dtype=np.float32)
        self._u_rl_hist = []

        # Choose u_init (dataset start + tiny noise)
        u_init = self._u_start.copy()
        std = float(getattr(self.cfg, "init_u_noise_std_deg", 0.25))
        if std > 0.0:
            u_init = u_init + np.random.normal(0.0, std, size=u_init.shape).astype(np.float32)
        u_init = np.clip(u_init, self.u_min, self.u_max).astype(np.float32)

        if self.sanity_check:
            self._u_cmd = u_init
            self._publish_motor_command(u0)
        else:
            # Single ramp: 0 -> u_init
            self._ramp_command(u_from=u0, u_to=u_init, seconds=self.cfg.ramp_seconds)
            self._u_cmd = u_init



        # init kinematics
        if not self.sanity_check:
            self._spin_until_ready(timeout_s=2.0)

        self._episode_step = 0
        self._ise = 0.0
        self._episode_return = 0.0
        self.last_transition = None

        p = self._get_tip_position()
        self._p_prev = p.copy()

        p_star = self._p_star[self._episode_step]
        v = np.zeros(3, dtype=np.float32)
        obs = self._normalize_obs(p, v, p_star, self._u_cmd, self._du_prev)
        info = {"clip_id": self._clip_id}
        return obs, info

    def step(self, action: np.ndarray):
        step_start = time.time()

        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        # map action -> state-dependent delta bounds
        u = self._u_cmd
        du_pos = np.minimum(self.du_max, self.u_max - u)
        du_neg = np.maximum(-self.du_max, self.u_min - u)

        du = np.zeros(6, dtype=np.float32)
        pos_mask = a >= 0
        du[pos_mask] = a[pos_mask] * du_pos[pos_mask]
        du[~pos_mask] = a[~pos_mask] * np.abs(du_neg[~pos_mask])

        u_next = np.clip(u + du, self.u_min, self.u_max)

        # publish (or hold zeros if sanity)
        self._publish_motor_command(u_next)

        # internal controller state always advances, even in sanity mode (for stats)
        self._u_cmd = u_next
        self._du_prev = du
        self._u_rl_hist.append(u_next.copy())

        if not self.sanity_check:
            rclpy.spin_once(self.node, timeout_sec=0.0)

        # observation
        p = self._get_tip_position()
        v = (p - self._p_prev) / self.dt if self._p_prev is not None else np.zeros(3, dtype=np.float32)
        self._p_prev = p.copy()

        p_star = self._p_star[self._episode_step]

        err = p - p_star
        err2 = float(np.dot(err, err))

        reward = -err2 - float(self.cfg.lambda_action) * float(np.dot(du, du))
        self._episode_return += reward
        self._ise += err2 * self.dt

        t = self._episode_step * self.dt
        self._episode_step += 1

        terminated = False
        truncated = self._episode_step >= self.steps_per_episode

        obs = self._normalize_obs(p, v, p_star, self._u_cmd, self._du_prev)
        info = {
            "ise": self._ise,
            "episode_return": self._episode_return,
            "clip_id": self._clip_id,
            # expose motor trajectories for sanity stats
            "u_star": self._u_star,
            "u_rl": np.stack(self._u_rl_hist, axis=0) if len(self._u_rl_hist) > 0 else np.zeros((0, 6), dtype=np.float32),
        }

        self.last_transition = {
            "t": t,
            "p": p.copy(),
            "v": v.copy(),
            "p_star": p_star.copy(),
            "u": self._u_cmd.copy(),
            "du": du.copy(),
            "reward": float(reward),
            "err2": float(err2),
            "clip_id": self._clip_id,
            "step_idx": int(self._episode_step - 1),
        }

        # keep 100 Hz pacing during collection (even in sanity mode, to mimic timing)
        elapsed = time.time() - step_start
        sleep_t = self.dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

        # ramp down at episode end (only non-sanity)
        if (truncated or terminated) and (not self.sanity_check):
            self._ramp_command(u_from=self._u_cmd, u_to=np.zeros(6, dtype=np.float32), seconds=self.cfg.ramp_seconds)
            self._u_cmd = np.zeros(6, dtype=np.float32)
            self._publish_motor_command(self._u_cmd)

        # In sanity mode, always hold zeros
        if truncated or terminated:
            self._publish_motor_command(np.zeros(6, dtype=np.float32))

        return obs, reward, terminated, truncated, info


# -----------------------------
# SB3 Callback: per-episode logging + CSV dumps + sanity stats
# -----------------------------

class EpisodeAndRolloutLoggerCallback(BaseCallback):
    def __init__(self, node: Node, cfg: PPOConfig,
                 rollouts_dir: str, metrics_path: str,
                 sanity_check: bool,
                 verbose: int = 0):
        super().__init__(verbose=verbose)
        self.node = node
        self.cfg = cfg
        self.rollouts_dir = rollouts_dir
        self.metrics_path = metrics_path
        self.sanity_check = bool(sanity_check)

        self._episode_buf: List[Dict[str, Any]] = []
        self._episode_in_rollout = 0
        self._last_rollout_idx_seen = 1

    def _get_env(self) -> TrunkTrajectoryTrackingEnv:
        # training_env.envs[0] is typically a Monitor wrapper
        e = self.training_env.envs[0]
        return self._unwrap_env(e)


    def _infer_rollout_idx(self) -> int:
        nt = max(int(self.num_timesteps), 1)
        return int((nt - 1) // int(self.cfg.n_steps) + 1)

    def _write_episode_csv(self, rollout_idx: int, episode_in_rollout: int, rows: List[Dict[str, Any]]) -> str:
        path = os.path.join(self.rollouts_dir, f"rollout_{rollout_idx:04d}_episode_{episode_in_rollout:02d}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t",
                "px", "py", "pz",
                "vx", "vy", "vz",
                "pstarx", "pstary", "pstarz",
                "u0","u1","u2","u3","u4","u5",
                "du0","du1","du2","du3","du4","du5",
                "reward",
                "err2",
                "dataset_idx",
                "clip_start_row",
                "step_idx",
            ])
            for r in rows:
                (ds_idx, start_row) = r["clip_id"]
                w.writerow([
                    r["t"],
                    r["p"][0], r["p"][1], r["p"][2],
                    r["v"][0], r["v"][1], r["v"][2],
                    r["p_star"][0], r["p_star"][1], r["p_star"][2],
                    *r["u"].tolist(),
                    *r["du"].tolist(),
                    r["reward"],
                    r["err2"],
                    ds_idx,
                    start_row,
                    r["step_idx"],
                ])
        return path

    def _append_metrics_row(self, rollout_idx: int, update_idx: int,
                            episode_in_rollout: int,
                            episode_ise: float, episode_return: float,
                            clip_id: Tuple[int, int],
                            update_wall_time_s: float = np.nan):
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                time.time(),
                rollout_idx,
                update_idx,
                episode_in_rollout,
                episode_ise,
                episode_return,
                update_wall_time_s,
                clip_id[0],
                clip_id[1],
            ])

    def _print_sanity_stats(self, info0: Dict[str, Any]) -> None:
        u_star = info0.get("u_star", None)
        u_rl = info0.get("u_rl", None)
        if u_star is None or u_rl is None:
            return

        u_star = np.asarray(u_star, dtype=np.float32)
        u_rl = np.asarray(u_rl, dtype=np.float32)

        s_star = _traj_stats(u_star, self.cfg.dt)
        s_rl = _traj_stats(u_rl, self.cfg.dt) if u_rl.shape[0] > 0 else _traj_stats(np.zeros((1, 6), dtype=np.float32), self.cfg.dt)

        u_max = np.array(self.cfg.motor_limits_deg, dtype=np.float32)
        du_limit = float(self.cfg.du_max_deg_per_sec)

        # violations
        star_u_viol = s_star["max_abs_u"] > (u_max + 1e-2)
        rl_u_viol = s_rl["max_abs_u"] > (u_max + 1e-2)
        star_du_viol = s_star["max_abs_speed"] > (du_limit + 1e-2)
        rl_du_viol = s_rl["max_abs_speed"] > (du_limit + 1e-2)

        self.node.get_logger().info(
            "SANITY STATS (target vs policy) — per motor order [BASE, TIP, MID, TIP, MID, BASE]"
        )
        self.node.get_logger().info(
            f"  max|u| deg   target={_fmt_vec(s_star['max_abs_u'])}  policy={_fmt_vec(s_rl['max_abs_u'])}"
        )
        self.node.get_logger().info(
            f"  mean|u| deg  target={_fmt_vec(s_star['mean_abs_u'])}  policy={_fmt_vec(s_rl['mean_abs_u'])}"
        )
        self.node.get_logger().info(
            f"  max|du/dt| deg/s target={_fmt_vec(s_star['max_abs_speed'])}  policy={_fmt_vec(s_rl['max_abs_speed'])}"
        )
        self.node.get_logger().info(
            f"  mean|du/dt| deg/s target={_fmt_vec(s_star['mean_abs_speed'])}  policy={_fmt_vec(s_rl['mean_abs_speed'])}"
        )

        if np.any(star_u_viol) or np.any(rl_u_viol) or np.any(star_du_viol) or np.any(rl_du_viol):
            self.node.get_logger().warn(
                "SANITY VIOLATION: One or more motors exceeded limits. "
                f"u_limit={_fmt_vec(u_max, '{:.0f}')}, du_limit={du_limit:.1f} deg/s"
            )
            if np.any(star_u_viol):
                self.node.get_logger().warn(f"  target max|u| exceeds limit motors: {np.where(star_u_viol)[0].tolist()}")
            if np.any(rl_u_viol):
                self.node.get_logger().warn(f"  policy max|u| exceeds limit motors: {np.where(rl_u_viol)[0].tolist()}")
            if np.any(star_du_viol):
                self.node.get_logger().warn(f"  target max|du/dt| exceeds limit motors: {np.where(star_du_viol)[0].tolist()}")
            if np.any(rl_du_viol):
                self.node.get_logger().warn(f"  policy max|du/dt| exceeds limit motors: {np.where(rl_du_viol)[0].tolist()}")

    def _unwrap_env(self, e):
        """
        SB3 often wraps envs in Monitor/TimeLimit/etc.
        This unwraps .env chains until the base env.
        """
        while hasattr(e, "env"):
            e = e.env
        return e


    def _on_step(self) -> bool:
        env = self._get_env()

        rollout_idx = self._infer_rollout_idx()
        if rollout_idx != self._last_rollout_idx_seen:
            self._episode_in_rollout = 0
            self._last_rollout_idx_seen = rollout_idx

        # --- NEW: buffer the env's rich transition dict (has clip_id, p, v, u, etc.)
        lt = getattr(env, "last_transition", None)

        if isinstance(lt, dict) and len(lt) > 0:
            # copy arrays safely
            row = {
                "t": float(lt["t"]),
                "p": np.asarray(lt["p"], dtype=np.float32).copy(),
                "v": np.asarray(lt["v"], dtype=np.float32).copy(),
                "p_star": np.asarray(lt["p_star"], dtype=np.float32).copy(),
                "u": np.asarray(lt["u"], dtype=np.float32).copy(),
                "du": np.asarray(lt["du"], dtype=np.float32).copy(),
                "reward": float(lt["reward"]),
                "err2": float(lt["err2"]),
                "clip_id": tuple(lt["clip_id"]) if not isinstance(lt["clip_id"], tuple) else lt["clip_id"],
                "step_idx": int(lt["step_idx"]),
            }
            self._episode_buf.append(row)
        else:
            # Fallback: don't crash; just skip this step if env didn't populate last_transition yet
            pass

        dones = self.locals.get("dones", None)
        if dones is None:
            return True
        done0 = bool(dones[0])

        if done0:
            self._episode_in_rollout += 1

            infos = self.locals.get("infos", [{}])
            info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
            episode_ise = float(info0.get("ise", np.nan))
            episode_return = float(info0.get("episode_return", np.nan))

            clip_id = info0.get("clip_id", (-1, -1))
            if not isinstance(clip_id, tuple):
                clip_id = tuple(clip_id)

            # Write episode CSV using the buffered env transitions (which include clip_id)
            csv_path = self._write_episode_csv(rollout_idx, self._episode_in_rollout, self._episode_buf)

            self._append_metrics_row(
                rollout_idx=rollout_idx,
                update_idx=rollout_idx,
                episode_in_rollout=self._episode_in_rollout,
                episode_ise=episode_ise,
                episode_return=episode_return,
                clip_id=clip_id,
                update_wall_time_s=np.nan,
            )

            self.node.get_logger().info(
                f"[Rollout {rollout_idx}] Episode {self._episode_in_rollout}: "
                f"ISE={episode_ise:.6f}, Return={episode_return:.6f}, clip={clip_id}, saved={csv_path}"
            )

            if self.sanity_check:
                self._print_sanity_stats(info0)

            self._episode_buf = []

        return True



# -----------------------------
# ROS2 node that owns everything
# -----------------------------

class TrainRLExecutor(Node):
    def __init__(self, cfg: PPOConfig):
        super().__init__("train_rl_executor")

        self.cfg = cfg
        np.random.seed(cfg.seed)

        # ROS params
        self.declare_parameter("sanity_check", True)
        self.declare_parameter("sanity_noise_std", 0.002)  # meters; adjust if your mocap scale differs
        self.sanity_check = bool(self.get_parameter("sanity_check").value)
        self.sanity_noise_std = float(self.get_parameter("sanity_noise_std").value)

        qos = QoSProfile(depth=10)

        # Subscribers (still OK in sanity mode; we just don't require data)
        self.mocap_sub = self.create_subscription(TrunkRigidBodies, cfg.mocap_topic, self.mocap_callback, qos)
        self.motor_status_sub = self.create_subscription(AllMotorsStatus, cfg.motor_status_topic, self.motor_status_callback, qos)

        # Publisher
        self.motor_control_pub = self.create_publisher(AllMotorsControl, cfg.motor_control_topic, qos)

        # Latest sensor state
        self.latest_tip_position: Optional[np.ndarray] = None
        self.latest_motor_positions: Optional[np.ndarray] = None
        self._tip_index: Optional[int] = None

        # Output directories
        self.ppo_root = os.path.join(cfg.data_root, cfg.ppo_dirname)
        self.rollouts_dir = os.path.join(self.ppo_root, "rollouts")
        self.models_dir = os.path.join(self.ppo_root, "models")
        self.metrics_dir = os.path.join(self.ppo_root, "metrics")
        os.makedirs(self.rollouts_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.metrics_dir, "metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp_unix",
                    "rollout_idx",
                    "update_idx",
                    "episode_idx_in_rollout",
                    "episode_ise",
                    "episode_return",
                    "update_wall_time_s",
                    "clip_dataset_idx",
                    "clip_start_row",
                ])

        # Load dataset targets/controls
        files = _find_csvs(cfg)
        control_files = sorted(files["control"])
        obs_files = sorted(files["obs"])
        n_pairs = min(len(control_files), len(obs_files))

        self.dataset_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_pairs):
            u = _load_control_csv(control_files[i])
            pstar = _load_obs_csv_for_tip(obs_files[i])
            self.dataset_pairs.append((u, pstar))

        self.get_logger().info(f"Loaded {len(self.dataset_pairs)} dataset file-pairs for target sampling.")
        self.get_logger().info(f"Saving PPO outputs under: {self.ppo_root}")
        self.get_logger().info(f"SANITY CHECK mode: {'ON' if self.sanity_check else 'OFF'}")
        if self.sanity_check:
            self.get_logger().warn(
                "SANITY CHECK is ON: this node will HOLD ZERO motor commands (no robot motion). "
                "It will still run PPO rollouts/updates using synthetic feedback (target+noise) and print safety stats."
            )

        # Env + model
        self.env = TrunkTrajectoryTrackingEnv(
            node=self,
            cfg=cfg,
            dataset_pairs=self.dataset_pairs,
            sanity_check=self.sanity_check,
            sanity_noise_std=self.sanity_noise_std
        )

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            n_steps=cfg.n_steps,
            seed=cfg.seed,
            verbose=1,
            device="cpu",
            policy_kwargs=dict(log_std_init=0.0) # was -2.0 in run 1
        )

        self.logger_cb = EpisodeAndRolloutLoggerCallback(
            node=self,
            cfg=cfg,
            rollouts_dir=self.rollouts_dir,
            metrics_path=self.metrics_path,
            sanity_check=self.sanity_check,
            verbose=0
        )

    def ramp_to_zero(self, seconds: float):
        """Smoothly ramp from the env's current commanded u to zeros, then hold zeros."""
        if self.sanity_check:
            self.publish_motor_command(np.zeros(6, dtype=np.float32))
            return

        env = self.env  # this is your TrunkTrajectoryTrackingEnv
        u_from = np.array(getattr(env, "_u_cmd", np.zeros(6, dtype=np.float32)), dtype=np.float32)
        u_to = np.zeros(6, dtype=np.float32)

        # Use the env's ramp (100 Hz timed, publishes along the way)
        env._ramp_command(u_from=u_from, u_to=u_to, seconds=seconds)

        # Ensure internal and commanded state agree
        env._u_cmd = u_to.copy()
        env._du_prev = np.zeros(6, dtype=np.float32)

        # Final "hold" publish
        self.publish_motor_command(u_to)


    # ROS callbacks
    def mocap_callback(self, msg: TrunkRigidBodies):
        if self._tip_index is None:
            names = list(msg.rigid_body_names)
            if self.cfg.tip_body_name in names:
                self._tip_index = names.index(self.cfg.tip_body_name)
            else:
                candidates = [i for i, n in enumerate(names) if str(n).strip().endswith(self.cfg.tip_body_name)]
                if len(candidates) > 0:
                    self._tip_index = candidates[0]
                else:
                    self._tip_index = min(2, len(names) - 1)
            self.get_logger().info(f"Tip rigid body index set to {self._tip_index} (name='{names[self._tip_index]}').")

        i = self._tip_index
        pos = msg.positions[i]
        self.latest_tip_position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

    def motor_status_callback(self, msg: AllMotorsStatus):
        self.latest_motor_positions = np.array(msg.positions, dtype=np.float32)

    def publish_motor_command(self, u_cmd: np.ndarray):
        m = AllMotorsControl()
        m.motors_control = [float(x) for x in u_cmd.tolist()]
        self.motor_control_pub.publish(m)

    def run_initial_training(self):
        total_timesteps = int(self.cfg.n_steps * self.cfg.total_rollouts)

        self.get_logger().info("=== Starting initial PPO training run ===")
        self.get_logger().info(f"Episode length: {self.cfg.episode_seconds}s ({int(self.cfg.episode_seconds*self.cfg.control_hz)} steps)")
        self.get_logger().info(f"Rollout length (n_steps): {self.cfg.n_steps} steps (~{self.cfg.n_steps*self.cfg.dt:.1f}s)")
        self.get_logger().info(f"Total timesteps: {total_timesteps} => {self.cfg.total_rollouts} updates")
        self.get_logger().info(f"Delta limit: {self.cfg.du_max_deg_per_sec} deg/s => {self.cfg.du_max_deg_per_sec*self.cfg.dt:.3f} deg/step")
        self.get_logger().info("Motor order: BASE TIP MID TIP MID BASE")
        self.get_logger().info(f"Motor limits: {self.cfg.motor_limits_deg}")

        # Always hold zeros at start
        self.publish_motor_command(np.zeros(6, dtype=np.float32))
        time.sleep(0.5)

        for rollout_idx in range(self.cfg.total_rollouts):
            self.get_logger().info(f"\n=== Rollout {rollout_idx+1}/{self.cfg.total_rollouts}: collecting {self.cfg.n_steps} steps ===")

            t0 = time.time()
            self.model.learn(
                total_timesteps=self.cfg.n_steps,
                reset_num_timesteps=False,
                callback=self.logger_cb,
                progress_bar=False
            )
            wall = time.time() - t0

            ckpt_path = os.path.join(self.models_dir, f"ppo_after_update_{rollout_idx+1}.zip")
            self.model.save(ckpt_path)

            self.get_logger().info(f"Update {rollout_idx+1} complete. learn() wall time: {wall:.2f} s")
            self.get_logger().info(f"Saved model: {ckpt_path}")

            # Log update row (episode rows already logged)
            with open(self.metrics_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([time.time(), rollout_idx+1, rollout_idx+1, -1, np.nan, np.nan, wall, -1, -1])

            # Ramp to zero after rollouts
            self.ramp_to_zero(self.cfg.ramp_seconds)
            time.sleep(0.2)

        self.get_logger().info("=== Initial PPO training run finished ===")
        self.get_logger().info(f"All outputs saved under: {self.ppo_root}")


def main():
    cfg = PPOConfig()

    rclpy.init()
    node = TrainRLExecutor(cfg)

    # If sanity mode is OFF, wait for initial messages (real robot loop)
    if not node.sanity_check:
        node.get_logger().info("Waiting for initial mocap + motor status messages...")
        t0 = time.time()
        while rclpy.ok() and (node.latest_tip_position is None or node.latest_motor_positions is None):
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t0 > 5.0:
                node.get_logger().warn("Still waiting for mocap/motor status... check topics if this persists.")
                t0 = time.time()
    else:
        # sanity mode: still spin a bit so tip index can be learned if mocap is running
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        node.run_initial_training()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        # safety: always hold zeros
        node.publish_motor_command(np.zeros(6, dtype=np.float32))
        time.sleep(0.2)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
