import os
import torch
import numpy as np
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from interfaces.srv import ControlSolver


class IKSolverNode(Node):
    def __init__(self):
        super().__init__('ik_solver_node')
        self.declare_parameters(namespace='', parameters=[
            ('ik_type', 'nn')  # 'nn' or 'lq'
            ('u2y_file', 'u2y.npy'),  # for least=squares (lq)
            ('y2u_file', 'y2u.npy'),  # for least=squares (lq)
            ('u_min', -0.25),
            ('u_max', 0.25),
            ('du_max', 0.04),
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
        self.u_opt_previous = np.array([0, 0, 0, 0, 0, 0]) # initially no control input

        # Get inverse kinematics mappings
        data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        if self.ik_type == 'lq':
            self.u2y = np.load(os.path.join(data_dir, f'models/ik/{self.u2y_file}'))
            self.y2u = np.load(os.path.join(data_dir, f'models/ik/{self.y2u_file}'))
        elif self.ik_type == 'nn':
            neural_ik_model = torch.load(os.path.join(data_dir, 'models/ik/neural_ik_model.pth'))
            neural_ik_model.eval()
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
        u_opt = np.clip(self.y2u @ zf_des, self.u_min, self.u_max) # not limiting du (default)

        if self.limit_delta:
            du = u_opt - self.u_opt_previous # delta u between timesteps
            du_clipped = np.clip(du, -self.du_max, self.du_max) # clip delta u
            u_opt = self.u_opt_previous + du_clipped # update u with clipped du
            u_opt = np.clip(u_opt, self.u_min, self.u_max) # clip u_opt again for safety
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
            nn_output = self.neural_ik_model(torch.tensor(zf_des))
        u_opt = nn_output.numpy()

        if self.limit_delta:
            du = u_opt - self.u_opt_previous # delta u between timesteps
            du_clipped = np.clip(du, -self.du_max, self.du_max) # clip delta u
            u_opt = self.u_opt_previous + du_clipped # update u with clipped du
            u_opt = np.clip(u_opt, self.u_min, self.u_max) # clip u_opt again for safety
            self.u_opt_previous = u_opt # update previous u

        response.uopt = u_opt.tolist()
        return response


def main(args=None):
    rclpy.init(args=args)
    ik_solver_node = IKSolverNode()
    rclpy.spin(ik_solver_node)
    ik_solver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
