import json
import os
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore

import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", False)

from controller.mpc.gusto import GuSTOConfig                # type: ignore
from controller.mpc_solver_node import run_mpc_solver_node  # type: ignore
from .utils.models import SSMR
from .utils.misc import HyperRectangle
from .reference_generator import ReferenceTrajectoryGenerator


class MPCInitializerNode(Node):
    """
    This node initializes all that is needed for MPC.
    """
    def __init__(self):
        super().__init__('mpc_initializer_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', True)                               # False or True (print debug messages)
        ])

        config = {
            "trajectory": {
                "type": "circle_with_ramp",
                "duration": 30.0,  # Duration of the simulation in seconds
                "speed": 0.62831853071,  # Angular speed (rad/s)
                "include_velocity": False,
                "parameters": {
                    "center": [0.0, 0.0],  # Center of the (x,y) trajectory
                    "radius": 0.05,  # [m]  For "circle" and "pacman"
                    "amplitude": 0.05,  # [m]  For "eight"
                    "z_level": 0.0,  # [m]  Constant z-coordinate
                    "mouth_angle": 0.7854  # [rad] Defines the size of the pacman mouth
                },
                "dt": 0.01
            },
            "model_type": "ssm",
            "model": "ssmr_orth_baseline"
        }

        self.debug = self.get_parameter('debug').value
        self.model_name = config["model"]  # self.get_parameter('model_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack-ssmr/stack/main/data')

        # Load the model
        self._load_model(config["model_type"])

        # Generate reference trajectory
        dt = config["trajectory"]["dt"]

        traj_config = config["trajectory"]
        self.model_name = config["model"]
        
        # Works for ssm
        # MPC constraints
        U = HyperRectangle([-45, -70, -25, -70, -25, -45], [45, 70, 25, 70, 25, 45])  # HyperRectangle([-0.05]*2, [0.05]*2)
        # dU = HyperRectangle([0.05]*2, [-0.05]*2)
        U = None
        dU = None

        # Load MPC config from JSON
        mpc_config_path = '/home/trunk/Documents/trunk-stack-ssmr/stack/main/src/executor/executor/mpc_config.json'
        with open(mpc_config_path, 'r') as f:
            mpc_config = json.load(f)
        
        # MPC cost:
        Qz = jnp.diag(jnp.array(mpc_config["Qz"]))
        Qzf = jnp.diag(jnp.array(mpc_config["Qzf"]))
        R = jnp.diag(jnp.array(mpc_config["R"])) * mpc_config["R_scale"]
        R_du = jnp.diag(jnp.array(mpc_config["R_du"])) * mpc_config["R_du_scale"]
        N = mpc_config["N"]
        
        gusto_config = GuSTOConfig(
            Qz=Qz,
            Qzf=Qzf,
            R=R,
            R_du=R_du,
            x_char=jnp.ones(self.model.n_x),
            f_char=jnp.ones(self.model.n_x),
            N=N,
            dt=dt,
            verbose=0
        )
        self.ref_traj = ReferenceTrajectoryGenerator(traj_config, traj_config["dt"])
        self.ref_traj.sample_trajectory(traj_config["duration"])
        times = self.ref_traj.times

        x0 = jnp.zeros(self.model.n_x)
        self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0, t=times, dt=dt, ref_traj=self.ref_traj, U=U,
                                                   dU=dU, solver="CLARABEL")  # According to Paul, use CLARABEL for SSM models, and OSQP for Koopman

    def _load_model(self, model_type):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """
        # Load the model
        if model_type == "ssm":
            model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}.npz')
            self.model = SSMR(model_path=model_path)
        else:
            KeyError(f"The requested model type {model_type} was not recognized.")
        
        print(f'---- Model loaded: {self.model_name}')
        print('Dimensions:')
        print('     n_x:', self.model.n_x)
        print('     n_u:', self.model.n_u)
        print('     n_z:', self.model.n_z)
        print('     n_y:', self.model.n_y)


def main(args=None):
    rclpy.init(args=args)
    mpc_initializer_node = MPCInitializerNode()
    rclpy.spin(mpc_initializer_node)
    mpc_initializer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
