import os
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore

import jax                          # type: ignore
import jax.numpy as jnp             # type: ignore
import logging

from controller.mpc.gusto import GuSTOConfig                # type: ignore
from controller.mpc_solver_node import run_mpc_solver_node  # type: ignore
from .utils.models import SlowAdiabaticSSM
from .utils.misc import HyperRectangle
from .utils.delay_embedded_state import DelayEmbeddedState
from .utils.reference_generator import ReferenceTrajectoryGenerator

logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


class MPCInitializerNode(Node):
    """
    This node initializes all that is needed for MPC.
    """
    def __init__(self):
        super().__init__('mpc_initializer_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False)                               # False or True (print debug messages)
        ])

        # TODO: eventually move this to a config file (e.g., YAML or JSON)
        # so that results have a corresponding config file and can be reproduced
        config = {
            "mpc": {
                "Q_rows": [0, 1, 2],
                "Qz": 10.0,
                "Qzf": 10.0,
                "R": 0.0,
                "Rdu": 1.0,
                "U_constraint": "none",
                "dU_constraint": "none",
                "N": 10,
                "dt": 0.01
            },
            "trajectory": {
                "type": "eight",  # Options: "circle", "circle_with_ramp", "eight", "pacman", "pacman_with_ramp", "flower"
                "duration": 5,  # Duration of the simulation in seconds
                "speed": 0.5,  # Angular speed (rad/s)
                "include_velocity": False,
                "parameters": {
                    "center": [0.0, 0.0],  # Center of the (x,y) trajectory
                    "radius": 0.07,  # [m]  For "circle" and "pacman"
                    "amplitude": 0.05,  # [m]  For "eight"
                    "z_level": 0.0,  # [m]  Constant z-coordinate
                    "mouth_angle": 0.7854  # [rad] Defines the size of the pacman mouth (default π/4)
                }
            },
            "delay_embedding": {
                "num_delay": 3,
                "also_embedd_u": False
            },
            "model": "adiabatic/first_slow_aSSM.npz",
            "critical_manifold": "adiabatic/first_crit_mani_rbf.npz"
        }

        mpc_config, traj_config, self.delay_config = config["mpc"], config["trajectory"], config["delay_embedding"]

        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        self.debug = self.get_parameter('debug').value
        self.model_name = config["model"]
        self.manifold_name = config["critical_manifold"]

        # Load the model
        self._load_model()

        # Number of delay embeddings
        num_delay = self.delay_config["num_delay"]
        if num_delay is None:
            num_delay = self.model.n_y // 3 - 1  # NOTE: this assumes observations are just 3D (x, y, z), no velocities

        # Not used downstream, only to now initialize
        delay_emb_state = DelayEmbeddedState(
            n_y=self.model.n_y,
            n_u=self.model.n_u,
            num_delay=num_delay,
            also_embedd_u=self.delay_config["also_embedd_u"]
        )

        self.ref_traj = ReferenceTrajectoryGenerator(traj_config, mpc_config["dt"])
        self.ref_traj.sample_trajectory(traj_config["duration"])
        self.times = self.ref_traj.times

        # Build the cost matrices for the MPC controller
        qz = jnp.zeros((self.model.n_z, self.model.n_z))
        qzf = jnp.zeros((self.model.n_z, self.model.n_z))
        for row in mpc_config["Q_rows"]:
            qz = qz.at[row, row].set(mpc_config["Qz"])
            qzf = qzf.at[row, row].set(mpc_config["Qzf"])

        # Build the gusto configuration object
        gusto_config = GuSTOConfig(
            Qz=qz,
            Qzf=qzf,
            R=mpc_config["R"] * jnp.eye(self.model.n_u),
            R_du=mpc_config["Rdu"] * jnp.eye(self.model.n_u),
            x_char=jnp.ones(self.model.n_x),
            f_char=jnp.ones(self.model.n_x),
            N=mpc_config["N"],
            dt=mpc_config["dt"],
            U_constraint=mpc_config["U_constraint"],
            dU_constraint=mpc_config["dU_constraint"]
        )

        # Input constraints
        uc = mpc_config["U_constraint"]
        if uc is None or str(uc).lower() == 'none':
            u = None
        else:
            u = HyperRectangle([float(uc)] * self.model.n_u, [-float(uc)] * self.model.n_u)

        duc = mpc_config["dU_constraint"]
        if duc is None or str(duc).lower() == 'none':
            du = None
        else:
            du = HyperRectangle([float(duc)] * self.model.n_u, [-float(duc)] * self.model.n_u)

        x0 = 0.1 * jnp.ones(self.model.n_x)  # initial state (assumed zero for now)
        self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0, t=self.times, dt=mpc_config["dt"],
                                                   ref_traj=self.ref_traj, U=u, dU=du, solver="CLARABEL")  # Was GUROBI

    def _load_model(self):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """
        model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}')
        manifold_path = os.path.join(self.data_dir, f'models/ssm/{self.manifold_name}')

        # Load the model
        self.model = SlowAdiabaticSSM(model_path, manifold_path)

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
