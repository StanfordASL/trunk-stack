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


class MPCInitializerNode(Node):
    """
    This node initializes all that is needed for MPC.
    """
    def __init__(self):
        super().__init__('mpc_initializer_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('model_name', 'ssm_origin_300g_4D_slow'),      # 'ssmr_200g' (what model to use)
        ])
        self.debug = self.get_parameter('debug').value
        self.model_name = self.get_parameter('model_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Load the model
        self._load_model()

        # Generate reference trajectory
        z_ref, t = self._generate_ref_trajectory(10, 0.01, 'circle', 0.05)

        # MPC configuration
        U = HyperRectangle([0.4]*6, [-0.4]*6)
        dU = HyperRectangle([0.1]*6, [-0.1]*6)
        # U = None
        dU = None

        Qz = 5 * jnp.eye(self.model.n_z)
        Qz = Qz.at[1, 1].set(0)
        Qzf = 10 * jnp.eye(self.model.n_z)
        Qzf = Qzf.at[1, 1].set(0)
        R_tip, R_mid, R_top = 0.001, 0.005, 0.01
        R = 0.1*jnp.diag(jnp.array([R_tip, R_mid, R_top, R_mid, R_top, R_tip]))

        gusto_config = GuSTOConfig(
            Qz=Qz,
            Qzf=Qzf,
            R=R,
            x_char=jnp.ones(self.model.n_x),
            f_char=jnp.ones(self.model.n_x),
            N=6
        )

        x0 = jnp.zeros(self.model.n_x)
        self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0, t=t, z=z_ref, U=U, dU=dU, solver="GUROBI")


    def _load_model(self):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """
        model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}.npz')

        # Load the model
        self.model = SSMR(model_path=model_path)
        print(f'---- Model loaded: {self.model_name}')
        print('Dimensions:')
        print('     n_x:', self.model.n_x)
        print('     n_u:', self.model.n_u)
        print('     n_z:', self.model.n_z)
        print('     n_y:', self.model.n_y)

    def _generate_ref_trajectory(self, T, dt, traj_type, size):
        """
        Generate a reference trajectory of dimension n_z for the system to track.
        """
        t = jnp.linspace(0, T, int(T/dt))
        z_ref = jnp.zeros((len(t), self.model.n_z))

        # NOTE: y is vertically up here

        if self.model.n_z == 2:
            if traj_type == 'circle':
                z_ref = z_ref.at[:, 0].set(size * (jnp.cos(2 * jnp.pi / T * t) - 1))
                z_ref = z_ref.at[:, 1].set(size * jnp.sin(2 * jnp.pi / T * t))
            elif traj_type == 'figure_eight':
                z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 1].set(size * jnp.sin(4 * jnp.pi / T * t))
            elif traj_type == 'periodic_line':
                m = -1
                z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 1].set(m * size * jnp.sin(2 * jnp.pi / T * t))
            elif traj_type == 'arc':
                m = -1
                l_trunk = 0.35
                R = l_trunk / 2
                z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 1].set(m * size * jnp.sin(2 * jnp.pi / T * t))
            else:
                raise ValueError('Invalid trajectory type: ' + traj_type + '. Valid options are: "circle" or "figure_eight".')
        elif self.model.n_z == 3:
            if traj_type == 'circle':
                z_ref = z_ref.at[:, 0].set(size * (jnp.cos(2 * jnp.pi / T * t) - 1))
                z_ref = z_ref.at[:, 1].set(size / 2 * jnp.ones_like(t))
                z_ref = z_ref.at[:, 2].set(size * jnp.sin(2 * jnp.pi / T * t))
            elif traj_type == 'figure_eight':
                z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 1].set(size / 2 * jnp.ones_like(t))
                z_ref = z_ref.at[:, 2].set(size * jnp.sin(4 * jnp.pi / T * t))
            elif traj_type == 'periodic_line':
                m = -1
                z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 1].set(jnp.zeros_like(t))
                z_ref = z_ref.at[:, 2].set(m * size * jnp.sin(2 * jnp.pi / T * t))
            elif traj_type == 'arc':
                m = -1
                l_trunk = 0.35
                R = l_trunk / 2
                z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 2].set(m * size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 1].set(R - jnp.sqrt(R**2 - z_ref[:, 0]**2 - z_ref[:, 0]**2))
            else:
                raise ValueError('Invalid trajectory type: ' + traj_type + '. Valid options are: "circle" or "figure_eight".')
        return z_ref, t


def main(args=None):
    rclpy.init(args=args)
    mpc_initializer_node = MPCInitializerNode()
    rclpy.spin(mpc_initializer_node)
    mpc_initializer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
