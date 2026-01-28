#!/usr/bin/env python3
import os
import yaml

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore


class PPOInitializerNode(Node):
    """
    Writes a PPO test config YAML under:
      <TRUNK_DATA>/trajectories/closed_loop/ppo/<config_name>.yaml

    The 'trajectory' section is intentionally structured to match Koopman/MPC configs
    so ReferenceTrajectoryGenerator can be used unchanged across methods.
    """
    def __init__(self):
        super().__init__("ppo_initializer_node")

        self.declare_parameters(namespace="", parameters=[
            ("debug", False),
            ("config_name", "ppo_config"),
            # Optional: override model selection from params instead of YAML
            ("checkpoint_update_num", -1),   # -1 => use latest
        ])

        self.debug = bool(self.get_parameter("debug").value)
        config_name = str(self.get_parameter("config_name").value)
        ckpt_update_num = int(self.get_parameter("checkpoint_update_num").value)

        self.data_dir = os.getenv("TRUNK_DATA", "/home/trunk/Documents/trunk-stack/stack/main/data")

        # NOTE: trajectory block shape mirrors koopman initializer.
        # You can edit these defaults in one place and reuse for all controllers.
        config = {
            "trajectory": {
                "type": "controlled_csv",  # must match ReferenceTrajectoryGenerator supported types
                "duration": 30.0,
                "speed": 0.25,
                "include_velocity": False,
                "csv_path": os.path.join(self.data_dir, "trajectories/dynamic/observations_controlled_310.csv"),
                "rest_pos": [0.1008, -0.39455, 0.117025],
                "parameters": {
                    "center": [0.0, 0.0],
                    "radius": 0.2,
                    "amplitude": 0.2,
                    "z_level": 0.09,
                    "mouth_angle": 0.7854
                }
            },
            "ppo": {
                # Execution timing (match training)
                "control_hz": 100.0,
                "dt": 0.01,

                # Safety (match training)
                "du_max_deg_per_sec": 100.0,
                "motor_limits_deg": [50.0, 100.0, 70.0, 100.0, 70.0, 50.0],  # BASE TIP MID TIP MID BASE

                # Model selection
                # -1 means "latest checkpoint in data/ppo/models"
                "checkpoint_update_num": ckpt_update_num,

                # Inference mode
                "deterministic": True,
            }
        }

        config_dir = os.path.join(self.data_dir, "trajectories/closed_loop/ppo")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"{config_name}.yaml")

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.get_logger().info(f"PPO config saved to: {config_path}")
        self.get_logger().info("Edit this YAML to define your test trajectory + PPO checkpoint selection.")


def main(args=None):
    rclpy.init(args=args)
    node = PPOInitializerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
