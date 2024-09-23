import os
import numpy as np
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from interfaces.srv import ControlSolver

#TODO: delta u max 
class IKSolverNode(Node):
    def __init__(self):
        super().__init__('ik_solver_node')
        self.declare_parameters(namespace='', parameters=[
            ('u2y_file', 'u2y.npy'),
            ('y2u_file', 'y2u.npy'),
            ('u_min', -0.25),
            ('u_max', 0.25),
            ('du_max', 0.04),
            ('limit_delta', False), # False or True -- if limit_delta, constrains the difference in u between timesteps'
            ('tip_only', False)     # False or True -- 
        ])

        self.u2y_file = self.get_parameter('u2y_file').value
        self.y2u_file = self.get_parameter('y2u_file').value
        self.u_min = self.get_parameter('u_min').value
        self.u_max = self.get_parameter('u_max').value
        self.limit_delta = self.get_parameter('limit_delta').value
        self.du_max = self.get_parameter('du_max').value
        self.tip_only = self.get_parameter('tip_only').value
        self.u_opt_previous = np.array([0, 0, 0, 0, 0, 0]) # initially no control input

        # Get mappings
        self.data_dir = os.getenv('TRUNK_DATA', '/home/asl/Documents/asl_trunk_ws/data')
        self.u2y = np.load(os.path.join(self.data_dir, f'models/ik/{self.u2y_file}'))
        self.y2u = np.load(os.path.join(self.data_dir, f'models/ik/{self.y2u_file}'))

        # Define service, which uses the ik callback function
        self.srv = self.create_service(ControlSolver, 'ik_solver', self.ik_callback)
        self.get_logger().info('Control solver service has been created.')


    def ik_callback(self, request, response):
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


def main(args=None):
    rclpy.init(args=args)
    ik_solver_node = IKSolverNode()
    rclpy.spin(ik_solver_node)
    ik_solver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
