open('/tmp/LOADED.txt', 'w').write('FILE LOADED\n')
import os
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore

import yaml
import jax
import jax.numpy as jnp
import logging

from controller.mpc.gusto_upgrade_trunk import GuSTOConfig                # type: ignore
from controller.mpc_solver_node import run_mpc_solver_node  # type: ignore
from .utils.models import control_SSMR, control_SSMR_simplified_ref_vec
from .utils.misc import HyperRectangle
from .delay_embedded_state import DelayEmbeddedState
from .reference_generator import ReferenceTrajectoryGenerator

from .adiabatic_ssm_trunk import setup_aSSM

logging.getLogger('jax').setLevel(logging.ERROR)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", False)

run_on_pauls_computer = False


class MPCInitializerNode(Node):
    """
    This node initializes all that is needed for MPC.
    """
    def __init__(self):
        super().__init__('mpc_initializer_node')

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
                "H_shape": [3,24],
                "Qz_diag": [1000, 100, 1000],
                "Qzf": 0.0,
                "R": 0.0,
                "R_du_diag": [0.001, 0.0005, 0.001, 0.0005, 0.001, 0.001],
                "x_char": [0.1, 0.1, 0.1, 0.1, 0.1],
                "f_char": [0.1, 0.1, 0.1, 0.1, 0.1],
                "U_constraint": 0.9,
                "dU_constraint": 0.4,
                "dt": 0.02,
                "N": 10
            },
            "trajectory": {
                "type": "controlled_csv",  # Options: "circle", "circle_with_ramp", "eight", "pacman_3d", "pacman", "pacman_with_ramp", "flower", "controlled_csv"
                "duration": 30.0,  # Duration of the simulation in seconds
                "speed": 0.25,  # Angular speed (rad/s)
                "include_velocity": False,
                "csv_path": os.path.join(self.data_dir, 'trajectories/dynamic/observations_controlled_310.csv'),
                "rest_pos": [0.1008, -0.39455, 0.117025], # in x, y, z mocap order for tip
                "parameters": {
                    "center": [0.0, 0.0],  # Center of the (x,y) trajectory
                    "radius": 0.1,  # [m]  For "circle" and "pacman"
                    "amplitude": 0.1,  # [m]  For "eight"
                    "z_level": 0.05,  # [m]  Constant z-coordinate
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
        config_dir = os.path.join(self.data_dir, 'trajectories/closed_loop/adiabatic')
        os.makedirs(config_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        config_file = os.path.join(config_dir, f"{config_name}.yaml")
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        self.get_logger().info(f"Config saved to: {config_file}")
        
        # # If you still need results_file:
        # self.results_file = os.path.join(self.data_dir, f"trajectories/closed_loop/adiabatic/{self.results_name}.csv")
        # self.initialize_csv()
    
    # TODO add failsafe if there is not a ref traj csv file
        mpc_config, traj_config, self.delay_config = config["mpc"], config["trajectory"], config["delay_embedding"]

        self.debug = self.get_parameter('debug').value
        self.model_name = config["model"]

        if run_on_pauls_computer:
            self.data_dir = os.getenv('TRUNK_DATA', '/Users/paulleonardwolff/Desktop/trunk-stack/stack/main/data')
        else:
            self.data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

        # Load MPC parameters
        H = jnp.eye(mpc_config["H_shape"][0], mpc_config["H_shape"][1])
        Qz = jnp.diag(jnp.array(mpc_config["Qz_diag"]))
        R_du = jnp.diag(jnp.array(mpc_config["R_du_diag"]))
        Qzf = mpc_config["Qzf"]*jnp.eye(3)
        R = mpc_config["R"]*jnp.eye(6)
        x_char = jnp.array(mpc_config["x_char"])*10
        f_char = jnp.array(mpc_config["f_char"])*10


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

        x0_red = jnp.zeros((5,))


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
        
        # SET STATE CONSTRAINTS (X)
        X = None
        

        gusto_config = GuSTOConfig(
            Qz=Qz,
            Qzf=Qzf,
            R=R,
            R_du = R_du,
            x_char=x_char,
            f_char=f_char,
            N=mpc_config['N'],        # MPC horizon
            H=H
        )
        # CHANGED
        # self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0_red_u_init, t=self.times, dt=mpc_config["dt"],
        #                                           ref_traj=self.ref_traj, U=u, dU=du, solver="CLARABEL")  # Was GUROBI
        self.model, self.exps_M, self.M, self.U_select, self.W_sm, self.epsilon_sm, self.case_rbf, self.V = setup_aSSM(dt=mpc_config["dt"], rk4=False)

        self.mpc_solver_node = run_mpc_solver_node(self.model, gusto_config, x0_red, t=self.times, dt=mpc_config["dt"],
                                                   ref_traj=self.ref_traj, U=u, dU=du, X=X, exps_M = self.exps_M , M = self.M, U_select = self.U_select, W_sm = self.W_sm, epsilon_sm = self.epsilon_sm, case_rbf = self.case_rbf, V = self.V , init_guess_type='shift', solver="OSQP")  # Was GUROBI
        
    

    # def _load_model(self):
    #     """
    #     Load the learned (non-autonomous) dynamics model of the system.
    #     """

    #     model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}')

    #     # Load the model
    #     # CHANGED
    #     # self.model = control_SSMR(self.delay_config, model_path)
    #     self.model = control_SSMR_simplified_ref_vec(self.delay_config, model_path)
    #     print(f'---- Model loaded: {self.model_name}')
    #     print('Dimensions:')
    #     print('     n_x:', self.model.n_x)
    #     print('     n_u:', self.model.n_u)
    #     print('     n_z:', self.model.n_z)
    #     print('     n_y:', self.model.n_y)


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
