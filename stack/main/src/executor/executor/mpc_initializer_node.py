import os
import jax
import jax.numpy as jnp
import logging
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from controller.mpc.gusto import GuSTOConfig  # type: ignore
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
            ('model_name', 'origin_ssmr_200g'),                    # 'ssmr_200g' (what model to use)
        ])
        self.debug = self.get_parameter('debug').value
        self.model_name = self.get_parameter('model_name').value
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Generate reference trajectory
        z_ref, t = self._generate_ref_trajectory(10, 0.01, 'circle', 0.1)
        # t = jnp.arange(0, 5, 0.01)
        # z_ref = jnp.zeros((len(t), 3))

        # Load the model
        self._load_model()

        # MPC configuration
        Qz = jnp.eye(self.model.n_z)
        # Qz = Qz.at[1, 1].set(0)
        Qzf = jnp.eye(self.model.n_z)
        # Qzf = Qzf.at[1, 1].set(0)
        gusto_config = GuSTOConfig(
            Qz=Qz,
            Qzf=10*Qzf,
            R=0.04*jnp.eye(self.model.n_u),
            x_char=0.05*jnp.ones(self.model.n_x),
            f_char=0.5*jnp.ones(self.model.n_x),
            N=9
        )
        U = HyperRectangle([0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                           [-0.4, -0.4, -0.4, -0.4, -0.4, -0.4])
        # dU = HyperRectangle([0.1]*6, [-0.1]*6)
        dU = None
        x0 = jnp.zeros(self.model.n_x)
        self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0, t=t, z=z_ref, U=U, dU=dU)

    def _load_model(self):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """
        model_path = os.path.join(self.data_dir, f'models/ssmr/{self.model_name}.npz')

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
        Generate a 3D reference trajectory for the system to track.
        """
        t = jnp.linspace(0, T, int(T/dt))
        z_ref = jnp.zeros((len(t), 3))

        # Note that y is up
        if traj_type == 'circle':
            z_ref = z_ref.at[:, 0].set(size * jnp.cos(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(size / 2 * jnp.ones_like(t))
            z_ref = z_ref.at[:, 2].set(size * jnp.sin(2 * jnp.pi / T * t))
        elif traj_type == 'figure_eight':
            z_ref = z_ref.at[:, 0].set(size * jnp.sin(jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(size / 2 * jnp.ones_like(t))
            z_ref = z_ref.at[:, 2].set(size * jnp.sin(2 * jnp.pi / T * t))
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
