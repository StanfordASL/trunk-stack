import os
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
import yaml

import jax
import jax.numpy as jnp
import logging

from controller.mpc.gusto import GuSTOConfig                # type: ignore
from controller.mpc_solver_node import run_mpc_solver_node  # type: ignore
from .utils.models import control_SSMR
from .utils.misc import HyperRectangle

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
            ('debug', False),                               # False or True (print debug messages)
        ])

        config_path = os.path.join(os.path.dirname(__file__), "mpc_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        mpc_config, delay_config = config["mpc"], config["delay_embedding"]

        self.debug = self.get_parameter('debug').value
        self.model_name = config["model"]
        self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Load the model
        self._load_model()

        # Generate reference trajectory
        dt = 0.02
        z_ref, t = self._generate_ref_trajectory(10, dt, 'figure_eight', 0.03)

        # 6) build warm-start arrays
        x0_red = self.model.encode(jnp.array(self.delay_emb_state.get_current_state()))
        shift = self.model.ssm.specified_params["shift_steps"]  # Is 0 if there is no subsampling
        num_delay = self.embedding_up_to
        pad_length = self.model.n_u * ((1 + shift) * num_delay - shift)
        u_ref_init = jnp.zeros((pad_length,))
        x0_red_u_init = jnp.concatenate([x0_red, u_ref_init], axis=0)

        # 7) Build the cost matrices for the MPC controller
        qz = jnp.zeros((self.model.n_z, self.model.n_z))
        qzf = jnp.zeros((self.model.n_z, self.model.n_z))

        for row in mpc_config["Q_rows"]:
            qz = qz.at[row, row].set(mpc_config["Qz"])
            qzf = qzf.at[row, row].set(mpc_config["Qzf"])

        # 8) build the gusto configuration object
        gusto_config = GuSTOConfig(
            Qz=qz,
            Qzf=qzf,
            R=mpc_config["R"] * jnp.eye(self.model.n_u),
            R_du=mpc_config["Rdu"] * jnp.eye(self.model.n_u),
            x_char=jnp.ones(x0_red_u_init.shape[0]),
            f_char=jnp.ones(x0_red_u_init.shape[0]),
            N=mpc_config["N"],
            dt=mpc_config["dt"],
            U_constraint=mpc_config["U_constraint"],
            dU_constraint=mpc_config["dU_constraint"]
        )

        # 9) input constraints
        uc = self.config["U_constraint"]
        if uc is None or str(uc).lower() == 'none':
            u = None
        else:
            u = HyperRectangle([float(uc)] * self.model.n_u, [-float(uc)] * self.model.n_u)

        duc = self.config["dU_constraint"]
        if duc is None or str(duc).lower() == 'none':
            du = None
        else:
            du = HyperRectangle([float(duc)] * self.model.n_u, [-float(duc)] * self.model.n_u)

        x0 = jnp.zeros(self.model.n_x)
        self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0, t=t, dt=dt, z=z_ref, U=u, dU=du,
                                                   solver="GUROBI")

    def _load_model(self):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """

        model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}.pkl')

        delay_config = {
            "perf_var_dim": 3,
            "also_embedd_u": True
        }

        # Load the model
        self.model = control_SSMR(delay_config, model_path)
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
            elif traj_type == 'point':
                z_ref = z_ref.at[:, 0].set(jnp.zeros_like(t))
                z_ref = z_ref.at[:, 2].set(-size * jnp.ones_like(t))
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
                z_ref = z_ref.at[:, 0].set(size * jnp.sin(2 * jnp.pi / T * t))
                z_ref = z_ref.at[:, 1].set(size / 2 * jnp.ones_like(t))
                z_ref = z_ref.at[:, 2].set(size * (jnp.cos(2 * jnp.pi / T * t) - 1))
            elif traj_type == 'point':
                z_ref = z_ref.at[:, 0].set(jnp.zeros_like(t))
                z_ref = z_ref.at[:, 1].set(jnp.zeros_like(t))
                z_ref = z_ref.at[:, 2].set(-size * jnp.ones_like(t))
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
