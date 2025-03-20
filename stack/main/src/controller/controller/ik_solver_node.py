import os
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from interfaces.srv import ControlSolver
import pandas as pd
from sklearn.neighbors import NearestNeighbors #type: ignore


class MLP(nn.Module):
    def __init__(self,
                 num_inputs: int = 9,
                 num_outputs: int = 6,
                 num_neurons: list = [32, 32],
                 act: nn.Module = nn.Tanh(),  # should be relu for spectral norm to make sense
                 spectral_normalize: bool = False):
        super(MLP, self).__init__()

        layers = []

        # Input layer
        if spectral_normalize:
            input_layer = spectral_norm(nn.Linear(num_inputs, num_neurons[0]))
        else:
            input_layer = nn.Linear(num_inputs, num_neurons[0])
        layers.append(input_layer)
        layers.append(act)
        
        # Hidden layers
        for i in range(len(num_neurons) - 1):
            if spectral_normalize:
                hidden_layer = spectral_norm(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            else:
                hidden_layer = nn.Linear(num_neurons[i], num_neurons[i + 1])
            layers.append(hidden_layer)
            layers.append(act)
        
        # Output layer
        if spectral_normalize:
            output_layer = spectral_norm(nn.Linear(num_neurons[-1], num_outputs))
        else:
            output_layer = nn.Linear(num_neurons[-1], num_outputs)
        
        layers.append(output_layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, y):
        x = y
        for layer in self.layers:
            x = layer(x)
        u = x
        return u


class IKSolverNode(Node):
    def __init__(self):
        super().__init__('ik_solver_node')
        self.declare_parameters(namespace='', parameters=[
            ('ik_type', 'slow_manifold'),  # 'nn' or 'lq' or 'interp' or 'slow_manifold' or 'ffpid_sm'
            ('u2y_file', 'u2y.npy'),  # for least=squares (lq)
            ('y2u_file', 'y2u_8seeds.npy'),  # for least=squares (lq)
            ('u_min', -0.25),
            ('u_max', 0.25),
            ('du_max', 0.15),        # 0.2 is too reactive, 0.1 is too slow
            ('limit_delta', False),  # False or True, if limit_delta, constrains the difference in u between timesteps
            ('tip_only', False),     # False or True
            ('knn_k', 3),            # k, number of nearest neighbors in KNN
            ('alpha', 0.25)          # smoothing coefficient, closer to 0 is more smooth
        ])
        
        self.ik_type = self.get_parameter('ik_type').value
        self.u2y_file = self.get_parameter('u2y_file').value
        self.y2u_file = self.get_parameter('y2u_file').value
        self.u_min = self.get_parameter('u_min').value
        self.u_max = self.get_parameter('u_max').value
        self.limit_delta = self.get_parameter('limit_delta').value
        self.du_max = self.get_parameter('du_max').value
        self.tip_only = self.get_parameter('tip_only').value
        self.alpha = self.get_parameter('alpha').value

        # Define rest positions
        self.rest_positions = np.array([0.10056, -0.10541, 0.10350,
                                        0.09808, -0.20127, 0.10645,
                                        0.09242, -0.31915, 0.09713]) #from 2/2/25

        # Initializations
        self.u_opt_previous = np.array([0., 0., 0., 0., 0., 0.])  # initially no control input
        self.smooth_stat = self.u_opt_previous  # expontential smoothing

        # Get inverse kinematics mappings

        data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        self.data_dir = data_dir
        if self.ik_type == 'lq':
            self.u2y = np.load(os.path.join(data_dir, f'models/ik/{self.u2y_file}'))
            self.y2u = np.load(os.path.join(data_dir, f'models/ik/{self.y2u_file}'))
        elif self.ik_type == 'slow_manifold':
            self.model_name = 'slow_manifold'
            
            self._load_slow_model() # Load the model

            # # Print sizes of matrices
            # self.get_logger().info(f'decoder_exp shape = {self.decoder_exp.shape}')
            # self.get_logger().info(f'const_coeff shape = {self.const_coeff.shape}')
            # self.get_logger().info(f'decoder_coeff shape = {self.decoder_coeff.shape}')
        elif self.ik_type == 'ffpid_sm':
            # Gain matrix K
            c = 0  # tunable parameter
            theta = 15 * np.pi / 180
            self.K = -c * np.array([
                                    [np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]
                    ])
            
        elif self.ik_type == 'nn':
            self.neural_ik_model = MLP()
            self.neural_ik_model.load_state_dict(torch.load(os.path.join(data_dir, 'models/ik/neural_ik_model_state.pth'), weights_only=False))
            self.neural_ik_model.eval()
        elif self.ik_type == 'interp': 
            # load position the data
            self.ys_ik = pd.read_csv(data_dir + '/trajectories/steady_state/observations_circle_seed0.csv')
            max_seed = 10
            for seed in range(1, max_seed+1):
                self.ys_ik = pd.concat([self.ys_ik, pd.read_csv(data_dir +f'/trajectories/steady_state/observations_circle_seed{seed}.csv')])
            self.ys_ik = self.ys_ik.drop(columns='ID')
            self.ys_ik = self.ys_ik - self.rest_positions # center about zero
            self.ys_ik = self.ys_ik.values  # Convert to numpy array

             # Load control inputs data
            self.us_ik = pd.read_csv(data_dir + '/trajectories/steady_state/control_inputs_circle_seed0.csv')
            for seed in range(1, max_seed + 1):
                self.us_ik = pd.concat([self.us_ik, pd.read_csv(data_dir +f'/trajectories/steady_state/control_inputs_circle_seed{seed}.csv')])
            self.us_ik = self.us_ik.drop(columns='ID')
            self.us_ik = self.us_ik.values  # Convert to numpy array

            # Initialize NearestNeighbors
            self.n_neighbors = self.get_parameter('knn_k').value
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
            # Fit the k-nearest neighbors model
            self.knn.fit(self.ys_ik[:, -3:])  
        else:
            raise ValueError(f"{self.ik_type} is not a valid option, choose from 'lq' or 'nn'")

        # Define service, which uses the ik callback function
        if self.ik_type == 'lq':
            self.srv = self.create_service(ControlSolver, 'ik_solver', self.lq_ik_callback)
            self.get_logger().info('Control solver (lq) service has been created.')
        elif self.ik_type == 'nn':
            self.srv = self.create_service(ControlSolver, 'ik_solver', self.nn_ik_callback)
            self.get_logger().info('Control solver (nn) service has been created.')
        elif self.ik_type == 'interp':
            self.srv = self.create_service(ControlSolver, 'ik_solver', self.interp_ik_callback)
            self.get_logger().info('Control solver (interp) service has been created.')
        elif self.ik_type == 'slow_manifold':
            self.srv = self.create_service(ControlSolver, 'ik_solver', self.slow_manifold_callback)
            self.get_logger().info('Control solver (slow manifold) service has been created')
    

    def phi(self, xi, exps):
        """ Returns monomials for slow manifold.
        Args: 
            xi: Input array of shape (n_dimensions,) of positions
            exps: Exponent matrix of shape (n_monomials, n_dimensions). 
        Returns:
            monomials: Monomial matrix of shape (n_monomials). """ 
        #self.get_logger().info(f"exps: {exps.shape}")
        x = np.reshape(xi, (1,-1)) # reshaped to dim (n_dimensions,1)
        #self.get_logger().info(f"x: {x.shape}")
        x_expanded = np.tile(x, (exps.shape[0], 1)) #repeat for each monomial (n_monomials, n_dimensions)
        #self.get_logger().info(f"x_expanded: {x_expanded.shape}")
        monomials = np.prod(x_expanded**exps, axis=1) 
        return monomials

    def _load_slow_model(self):
        "Loads model for slow manifold predictions"
        model_path = os.path.join(self.data_dir, f'models/ssm/{self.model_name}.npz')

        # Load the model
        self.model = np.load(model_path)

        self.decoder_exp = np.array(self.model['decoder_exp'])
        self.const_coeff = np.array(self.model['Const_coeff'])
        self.decoder_coeff = np.array(self.model['decoder_coeff'])
        self.get_logger().info('Loaded slow manifold model')


    def slow_manifold_callback(self, request, response):
        """
        Callback function that runs when the service is queried.
        Request contains: z (desired performance variable trajectory)
        Response contains: uopt (the found control inputs)
        """
        # decoder_exp shape = (83, 6)
        # const_coeff shape = (6, 1)
        # decoder_coeff shape = (6, 83)
        zf_des = np.array(request.zf) # desired positions
        # self.get_logger().info(f"zf_des (zero centered from AVP) {zf_des}")
        zf_des += self.rest_positions # It is already zero centered straight from the AVP... so we need to offset away from zero into the frame we learned the model in
        # self.get_logger().info(f"offset zf_des: {zf_des}")

        zf_des_spec = zf_des[3:] # only command desired positions for mid and tip
        monomials = self.phi(zf_des_spec, self.decoder_exp)
        u = (self.decoder_coeff @ monomials).reshape(self.const_coeff.shape[0], 1)

        self.const_coeff = self.const_coeff.reshape(self.const_coeff.shape[0],1)
        u_opt = u + self.const_coeff # add constant coefficients
        #self.get_logger().info(f"u_opt {u_opt}")

        # check control inputs are within the workspace
        u_opt = self.check_control_inputs(u_opt.flatten())
        response.uopt = u_opt.tolist()

        return response
    
    # todo: debug
    def ffpid_callback(self, request, response):
        zf_des = np.array(request.zf) # desired positions
        zf_des += self.rest_positions # It is already zero centered straight from the AVP... so we need to offset away from zero into the frame we learned the model in

        zf_des_spec = zf_des[3:] # only command desired positions for mid and tip
        monomials = self.phi(zf_des_spec, self.decoder_exp)
        u = (self.decoder_coeff @ monomials).reshape(self.const_coeff.shape[0], 1)

        self.const_coeff = self.const_coeff.reshape(self.const_coeff.shape[0],1)
        u_sm = u + self.const_coeff # add constant coefficients

        # We only actually use u1 and u6 for now
        u_ik = u_ik[[1, -1]]

        # Calculate the error
        y = np.array(request.y0)
        e = zf_des.flatten() - y

        # Calculate the P(ID) control inputs
        u_pid = self.K @ e
        # u_pid = 1 / np.linalg.norm(y) * self.K @ e

        print('IK: ', u_ik)
        print('e: ', e)
        print('PID: ', u_pid)

        # Combine feed-forward and PID control inputs
        u_opt = u_ik + u_pid

        # Do exponential smoothing
        self.smooth_stat = self.alpha * u_opt + (1 - self.alpha) * self.smooth_stat
        u_opt = self.smooth_stat

        # check control inputs !!!

        # Convert back to format for all control inputs
        u_ffpid = np.array([u_opt[0], 0, 0, 0, 0, u_opt[1]])

        response.uopt = u_ffpid.tolist()
        return response

    def interp_ik_callback(self, request, response):
        """
        Callback function that runs when the service is queried.
        Request contains: z (desired performance variable trajectory)
        Response contains: uopt (the found control inputs)
        """
        zf_des = np.array(request.zf)

        # Extract the relevant indices (x and z position of tip)
        zf_des_relevant = zf_des[-3:].reshape(1, -1)
        distances, indices = self.knn.kneighbors(zf_des_relevant)

        # Get the corresponding u values from us_ik
        u_neighbors = self.us_ik[indices.flatten()]

        # Inverse distance weighting
        weights = 1 / distances.flatten()
        weights /= weights.sum()  # Normalize weights

        # Calculate the optimal control inputs u_opt
        u_opt = np.dot(weights, u_neighbors)

        # check control inputs are within the workspace
        u_opt = self.check_control_inputs(u_opt)

        # Do exponential smoothing
        self.smooth_stat = self.alpha * u_opt + (1 - self.alpha) * self.smooth_stat
        u_opt = self.smooth_stat

        # Update previous u_opt
        self.u_opt_previous = u_opt

        response.uopt = u_opt.tolist()
        return response

    def lq_ik_callback(self, request, response):
        """
        Callback function that runs when the service is queried.
        Request contains: z (desired performance variable trajectory)
        Response contains: uopt (the found control inputs)
        """
        zf_des = np.array(request.zf)
        if self.tip_only:
            zf_des = np.array([0,0,0,0,0,0,zf_des[6], zf_des[7], zf_des[8]]) #trying only commanding tip position
        u_opt = self.check_control_inputs(self.y2u @ zf_des)

        if self.limit_delta:
            du = u_opt - self.u_opt_previous  # delta u between timesteps
            du_clipped = np.clip(du, -self.du_max, self.du_max)  # clip delta u
            u_opt = self.u_opt_previous + du_clipped  # update u with clipped du
            u_opt = self.check_control_inputs(u_opt) 
        
        self.u_opt_previous = u_opt # update previous u
        response.uopt = u_opt.tolist()
        return response
    
    def nn_ik_callback(self, request, response):
        """
        Callback function that runs when the service is queried.
        Request contains: z (desired performance variable trajectory)
        Response contains: uopt (the found control inputs)
        """
        zf_des = np.array(request.zf)
        if self.tip_only:
            zf_des = np.array([0,0,0,0,0,0,zf_des[6], zf_des[7], zf_des[8]]) #trying only commanding tip position
        with torch.no_grad():
            # Forward pass
            nn_output = self.neural_ik_model(torch.tensor(zf_des, dtype=torch.float))
        u_opt = nn_output.numpy()

        if self.limit_delta: # smoothing on,reject large changes in u
            du = u_opt - self.u_opt_previous # delta u between timesteps
            du_clipped = np.clip(du, -self.du_max, self.du_max) # clip delta u
            u_opt = self.u_opt_previous + du_clipped # update u with clipped du
            u_opt = self.check_control_inputs(u_opt) 
        else:  # smoothing off
            u_opt = self.check_control_inputs(u_opt)

        self.u_opt_previous = u_opt # update previous u
        response.uopt = u_opt.astype(np.float64).tolist()
        return response
    
    def check_control_inputs(self, u_opt):
        # reject vector norms of u that are too large
        tip_range, mid_range, base_range = 0.45, 0.35, 0.3 

        u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

        # First we clip to max and min values
        u1 = np.clip(u1, -tip_range, tip_range)
        u6 = np.clip(u6, -tip_range, tip_range)
        u2 = np.clip(u2, -mid_range, mid_range)
        u5 = np.clip(u5, -mid_range, mid_range)
        u3 = np.clip(u3, -base_range, base_range)
        u4 = np.clip(u4, -base_range, base_range)

        # Compute control input vectors
        u1_vec = u1 * np.array([-np.cos(15 * np.pi/180), np.sin(15 * np.pi/180)])
        u2_vec = u2 * np.array([np.cos(45 * np.pi/180), np.sin(45 * np.pi/180)])
        u3_vec = u3 * np.array([-np.cos(15 * np.pi/180), -np.sin(15 * np.pi/180)])
        u4_vec = u4 * np.array([-np.cos(75 * np.pi/180), np.sin(75 * np.pi/180)])
        u5_vec = u5 * np.array([np.cos(45 * np.pi/180), -np.sin(45 * np.pi/180)])
        u6_vec = u6 * np.array([-np.cos(75 * np.pi/180), -np.sin(75 * np.pi/180)])

        # Calculate the norm based on the constraint
        vector_sum = (
            0.75 * (u3_vec + u4_vec) +
            1.0 * (u2_vec + u5_vec) +
            1.4 * (u1_vec + u6_vec)
        )
        norm_value = np.linalg.norm(vector_sum)

        # Check the constraint: if the constraint is met, then keep previous control command
        if norm_value > 0.9:
            u_opt = self.u_opt_previous
        else:
            # Else the clipped command is published
            u_opt = np.array([u1, u2, u3, u4, u5, u6])

        return u_opt
    

def main(args=None):
    rclpy.init(args=args)
    ik_solver_node = IKSolverNode()
    rclpy.spin(ik_solver_node)
    ik_solver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
