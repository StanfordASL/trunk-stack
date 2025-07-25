import os
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore

import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

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
                "type": "eight",
                "duration": 20.0,  # Duration of the simulation in seconds
                "speed": 1.5,  # Angular speed (rad/s)
                "include_velocity": False,
                "parameters": {
                    "center": [0.0, 0.0],  # Center of the (x,y) trajectory
                    "radius": 0.08,  # [m]  For "circle" and "pacman"
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
        dt = 0.01

        traj_config = config["trajectory"]
        self.model_name = config["model"]
        
        # Works for ssm
        # MPC constraints
        U = HyperRectangle([0.9]*2, [-0.9]*2)
        dU = HyperRectangle([0.05]*2, [-0.05]*2)
        
        # MPC cost:
        Qz = 700.0 * jnp.eye(3)  # jnp.eye(self.model.n_z)
        Qz = Qz.at[2, 2].set(0)
        Qzf = 2000.0 * jnp.eye(3)  # hardcode for the moment jnp.eye(self.model.n_z)
        Qzf = Qzf.at[2, 2].set(0)
        R = 0.0 * jnp.eye(self.model.n_u)
        R_du = 16.0 * jnp.eye(self.model.n_u)
        N = 7
        
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
