import os
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore

import jax
import jax.numpy as jnp
import logging

from controller.mpc.gusto import GuSTOConfig                # type: ignore
from controller.mpc_solver_node import run_mpc_solver_node  # type: ignore
from .utils.models import control_SSMR
from .utils.misc import HyperRectangle
from .delay_embedded_state import DelayEmbeddedState
from .reference_generator import ReferenceTrajectoryGenerator

logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

run_on_pauls_computer = False


class MPCInitializerNode(Node):
    """
    This node initializes all that is needed for MPC.
    """
    def __init__(self):
        super().__init__('mpc_initializer_node')
        self.declare_parameters(namespace='', parameters=[
            ('debug', False)                               # False or True (print debug messages)
        ])

        config = {
            "mpc": {
                "Q_rows": [0, 1],
                "Qz": 200.0,
                "Qzf": 2000.0,  # was 640
                "R": 0.0,
                "Rdu": 8.0,
                "U_constraint": 0.25,
                "dU_constraint": 0.02,
                "N": 10,
                "dt": 0.04
            },
            "trajectory": {
                "type": "circle_with_ramp",  # Options: "circle", "circle_with_ramp", "eight", "pacman", "flower"
                "duration": 20.0,  # Duration of the simulation in seconds
                "speed": 0.5,  # Angular speed (rad/s)
                "include_velocity": False,
                "parameters": {
                    "center": [0.0, 0.0],  # Center of the (x,y) trajectory
                    "radius": 0.03,  # [m]  For "circle" and "pacman"
                    "amplitude": 0.03,  # [m]  For "eight"
                    "z_level": 0.0,  # [m]  Constant z-coordinate
                    "mouth_angle": 0.7854  # [rad] Defines the size of the pacman mouth (default π/4)
                }
            },
            "delay_embedding": {
                "perf_var_dim": 3,
                "also_embedd_u": True
            },
            # "model": "sim_origin_best.pkl"
            "model": "first_mpc_model_real_trunk.pkl"
        }

        mpc_config, traj_config, self.delay_config = config["mpc"], config["trajectory"], config["delay_embedding"]

        self.debug = self.get_parameter('debug').value
        self.model_name = config["model"]

        if run_on_pauls_computer:
            self.data_dir = os.getenv('TRUNK_DATA', '/Users/paulleonardwolff/Desktop/trunk-stack/stack/main/data')
        else:
            self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Load the model
        self._load_model()

        # 3) figure out measurement dims for delay embedding
        meas_var_dim = len([x - 1 for x in self.model.ssm.specified_params["measured_rows"] if x <= 18])
        num_u = self.model.ssm.specified_params["num_u"]
        embedding_up_to = self.model.ssm.specified_params["embedding_up_to"]
        include_velocity = bool(self.model.ssm.specified_params["include_velocity"])
        shift = self.model.ssm.specified_params["shift_steps"]  # Is 0 if there is no subsampling
        pad_length = self.model.n_u * ((1 + shift) * embedding_up_to - shift)

        # 4) initial DelayEmbeddedState -> not used downstream...only to now initialize
        delay_emb_state = DelayEmbeddedState(
            meas_var_dim * (1 + int(include_velocity)),
            num_u,
            embedding_up_to,
            self.delay_config["also_embedd_u"],
            initial_state=None
        )

        # Generate reference trajectory
        # z_ref, t = self._generate_ref_trajectory(10, dt, 'figure_eight', 0.03)

        self.ref_traj = ReferenceTrajectoryGenerator(traj_config, mpc_config["dt"])
        self.ref_traj.sample_trajectory(traj_config["duration"])
        self.times = self.ref_traj.times

        # 6) build warm-start arrays
        u_ref_init = jnp.zeros((pad_length,))
        x0_red = self.model.encode(jnp.array(delay_emb_state.get_current_state()))
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

        self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0_red_u_init, t=self.times, dt=mpc_config["dt"],
                                                   ref_traj=self.ref_traj, U=u, dU=du, solver="OSQP")  # Was GUROBI

    def _load_model(self):
        """
        Load the learned (non-autonomous) dynamics model of the system.
        """

        model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}')

        # Load the model
        self.model = control_SSMR(self.delay_config, model_path)
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
