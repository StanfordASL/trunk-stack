import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from interfaces.srv import ControlSolver


class FFPIDSolverNode(Node):
    """
    Feed-forward PID controller ROS node.
    Currently, only interpolation-based feed-forward term is implemented as then desired observables are easily altered.
    """
    def __init__(self):
        super().__init__('ffpid_solver_node')
        self.declare_parameters(namespace='', parameters=[
            ('alpha', 1.0)          # smoothing coefficient, closer to 0 is more smoothing
        ])

        self.alpha = self.get_parameter('alpha').value

        # Define rest positions
        self.rest_position = np.array([0.10056, -0.10541, 0.10350,
                                       0.09808, -0.20127, 0.10645,
                                       0.09242, -0.31915, 0.09713])

        # Initialization
        self.smooth_stat = np.array([0., 0., 0., 0., 0., 0.])

        data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
        self.data_dir = data_dir

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
        
        # Fit the k-nearest neighbors model to only x, z position of the tip
        self.knn.fit(self.ys_ik[:, [-3, -1]])

        # Gain matrix K
        c = 0.1  # tunable parameter
        theta = 15 * np.pi / 180
        self.K = -c * np.array([
                                [np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]
                ])

        # Define service, which uses the ik callback function
        self.srv = self.create_service(ControlSolver, 'ffpid_solver', self.ffpid_callback)
        self.get_logger().info('Control solver (feed-forward PID) service has been created.')

    def ffpid_callback(self, request, response):
        """
        Callback function that runs when the service is queried.
        """
        z_des = np.array(request.z).reshape(1, -1)
        distances, indices = self.knn.kneighbors(z_des)

        # Get the corresponding u values from us_ik
        u_neighbors = self.us_ik[indices.flatten()]

        # Inverse distance weighting
        weights = 1 / distances.flatten()
        weights /= weights.sum()

        # Calculate the feed-forward control inputs
        u_ik = np.dot(weights, u_neighbors)

        # Calculate the error
        y = np.array(request.y0)
        e = z_des - y

        # Calculate the P(ID) control inputs
        u_pid = self.K @ e

        # Combine feed-forward and PID control inputs
        u_opt = u_ik + u_pid

        # Do exponential smoothing
        self.smooth_stat = self.alpha * u_opt + (1 - self.alpha) * self.smooth_stat
        u_opt = self.smooth_stat

        response.uopt = u_opt.tolist()
        return response
    

def main(args=None):
    rclpy.init(args=args)
    ffpid_solver_node = FFPIDSolverNode
    rclpy.spin(ffpid_solver_node)
    ffpid_solver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
