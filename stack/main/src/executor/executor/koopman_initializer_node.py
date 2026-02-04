open('/tmp/LOADED.txt', 'w').write('FILE LOADED\n')
import os
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore

import yaml
import jax
import jax.numpy as jnp
import logging

from controller.mpc.gusto_upgrade_trunk import GuSTOConfig                # type: ignore
from controller.koopman_solver_node import run_mpc_solver_node  # type: ignore
from .utils.models import control_SSMR, control_SSMR_simplified_ref_vec
from .utils.misc import HyperRectangle
from .delay_embedded_state import DelayEmbeddedState
from .reference_generator import ReferenceTrajectoryGenerator

from .koopman_model import setup_koopman
from control import dlqr
logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", False)

run_on_pauls_computer = False


class MPCInitializerNode(Node):
    """
    This node initializes all that is needed for MPC.
    """
    def __init__(self):
        super().__init__('koopman_initializer_node')

        with open('/home/trunk/Desktop/test.txt', 'w') as f:
                f.write('INIT STARTED\n')

        
        # Declare parameters including the config filename
        self.declare_parameters(namespace='', parameters=[
            ('debug', False),                               # False or True (print debug messages)
            ('config_name', 'mpc_config')                   # Name for the config file (without .yaml extension)
        ])
        
        self.data_dir = '/home/trunk/Documents/trunk-stack/stack/main/data'
        


        config = {
            "mpc": {
                "Qz_diag": [1000, 100, 1000]*3,
                "R": 10.0,
                "U_constraint": 0.9,
                "dU_constraint": 0.4,
                "dt": 0.02,
                "N": 10
            },
            "trajectory": {
                "type": "spiral",  # Options: "circle", "circle_with_ramp", "eight", "pacman_3d", "pacman", "pacman_with_ramp", "flower", "controlled_csv"
                "duration": 30.0,  # Duration of the simulation in seconds
                "speed": 0.5,  # Angular speed (rad/s)
                "include_velocity": False,
                "csv_path": os.path.join(self.data_dir, 'trajectories/dynamic/observations_controlled_311.csv'),
                "rest_pos": [0.10643, -0.38418, 0.10180], # in x, y, z mocap order for tip
                "parameters": {
                    "center": [0.0, 0.0],  # Center of the (x,y) trajectory
                    "radius": 0.2,  # [m]  For "circle" and "pacman"
                    "amplitude": 0.2,  # [m]  For "eight"
                    "z_level": 0.09,  # [m]  Constant z-coordinate
                    "mouth_angle": 0.7854  # [rad] Defines the size of the pacman mouth (default π/4)
                }
            },
            "delay_embedding": {
                "perf_var_dim": 3,
                "also_embedd_u": False
            },
            # "model": "first_mpc_model_real_trunk.pkl"
            # "model": "real_origin_faster_v2.pkl"
            # "model": "real_model_60.pkl"
            "model": "best_51_v3.pkl"
        }
        # TODO add failsafe if there is not a ref traj csv file
         # Get the config filename from ROS parameter
        config_name = self.get_parameter('config_name').value
        
        # Save config as YAML in the specified path
        config_dir = os.path.join(self.data_dir, 'trajectories/closed_loop/koopman')
        os.makedirs(config_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        config_file = os.path.join(config_dir, f"{config_name}.yaml")
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        self.get_logger().info(f"Config saved to: {config_file}")
        
    
    # TODO add failsafe if there is not a ref traj csv file
        mpc_config, traj_config, self.delay_config = config["mpc"], config["trajectory"], config["delay_embedding"]

        # set koopman parameters
        Q_full = jnp.diag(jnp.array(mpc_config["Qz_diag"]))
        R = mpc_config["R"]*jnp.eye(6)

        self.debug = self.get_parameter('debug').value
        self.model_name = config["model"]

        if run_on_pauls_computer:
            self.data_dir = os.getenv('TRUNK_DATA', '/Users/paulleonardwolff/Desktop/trunk-stack/stack/main/data')
        else:
            self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')


        print("before", flush=True)
        self.get_logger().info("STEP 1: Starting initialization")
        self.ref_traj = ReferenceTrajectoryGenerator(traj_config, mpc_config["dt"])
        self.get_logger().info("STEP 2: Sample trahectory")
        self.ref_traj.sample_trajectory(traj_config["duration"])
        self.get_logger().info("STEP 3: Trajectory sampled")
        self.get_logger().info(f"STEP 4: Shape is {self.ref_traj.trajectory.shape}")

        print(self.ref_traj.trajectory.shape, flush=True)
        print(self.ref_traj.trajectory, flush=True)
        self.get_logger().info("before")
        self.get_logger().info(f"Trajectory shape: {self.ref_traj.trajectory.shape}")
        

        self.times = self.ref_traj.times
        self.get_logger().info(f"times: {self.times}")
        
        self.ref_traj = self.ref_traj.trajectory

        # 9) input constraints
        uc = mpc_config["U_constraint"]
        if uc is None or str(uc).lower() == 'none':
            u = None
        else:
            #u = HyperRectangle([float(uc)] * self.model.n_u, [-float(uc)] * self.model.n_u)
            u = None

        duc = mpc_config["dU_constraint"]
        if duc is None or str(duc).lower() == 'none':
            du = None
        else:
            #du = HyperRectangle([float(duc)] * self.model.n_u, [-float(duc)] * self.model.n_u)
            du = None
     
        A_koop, B_koop, exps_I, I = setup_koopman(dt=mpc_config["dt"])
        K, S, E = dlqr(A_koop, B_koop, Q_full, R)
        print("LQR gain K:", K)
        print("LQR gain shape:", K.shape)
        K_red = K  # Only use position feedback
        x0 = jnp.zeros((5,))

        self.mpc_solver_node = run_mpc_solver_node(K_red, I, exps_I, x0, t=self.times, dt=mpc_config["dt"], ref_traj=self.ref_traj,
                       U=u, dU=du, init_guess_type='shift',init_node=False)


def main(args=None):
    with open('/home/trunk/Desktop/test_debug/main_started.txt', 'w') as f:
        f.write('MAIN CALLED\n')
    rclpy.init(args=args)
    mpc_initializer_node = MPCInitializerNode()
    rclpy.spin(mpc_initializer_node)
    mpc_initializer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
