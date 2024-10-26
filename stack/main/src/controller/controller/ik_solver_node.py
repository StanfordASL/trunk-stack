import os
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from interfaces.srv import ControlSolver


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
            ('ik_type', 'nn'),  # 'nn' or 'lq'
            ('u2y_file', 'u2y.npy'),  # for least=squares (lq)
            ('y2u_file', 'y2u_8seeds.npy'),  # for least=squares (lq)
            ('u_min', -0.25),
            ('u_max', 0.25),
            ('du_max', 0.02),
            ('limit_delta', False), # False or True, if limit_delta, constrains the difference in u between timesteps
            ('tip_only', False)     # False or True
        ])
        
        self.ik_type = self.get_parameter('ik_type').value
        self.u2y_file = self.get_parameter('u2y_file').value
        self.y2u_file = self.get_parameter('y2u_file').value
        self.u_min = self.get_parameter('u_min').value
        self.u_max = self.get_parameter('u_max').value
        self.limit_delta = self.get_parameter('limit_delta').value
        self.du_max = self.get_parameter('du_max').value
        self.tip_only = self.get_parameter('tip_only').value
        self.u_opt_previous = np.array([0., 0., 0., 0., 0., 0.]) # initially no control input

        # Get inverse kinematics mappings
        data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        if self.ik_type == 'lq':
            self.u2y = np.load(os.path.join(data_dir, f'models/ik/{self.u2y_file}'))
            self.y2u = np.load(os.path.join(data_dir, f'models/ik/{self.y2u_file}'))
        elif self.ik_type == 'nn':
            self.neural_ik_model = MLP()
            self.neural_ik_model.load_state_dict(torch.load(os.path.join(data_dir, 'models/ik/neural_ik_model_state.pth'), weights_only=False))
            self.neural_ik_model.eval()
        else:
            raise ValueError(f"{self.ik_type} is not a valid option, choose from 'lq' or 'nn'")

        # Define service, which uses the ik callback function
        if self.ik_type == 'lq':
            self.srv = self.create_service(ControlSolver, 'ik_solver', self.lq_ik_callback)
            self.get_logger().info('Control solver (lq) service has been created.')
        elif self.ik_type == 'nn':
            self.srv = self.create_service(ControlSolver, 'ik_solver', self.nn_ik_callback)
            self.get_logger().info('Control solver (nn) service has been created.')


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
        tip_range, mid_range, base_range = 0.4, 0.35, 0.3

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
        if norm_value > 0.7:
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
