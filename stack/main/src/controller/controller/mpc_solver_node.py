import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from scipy.interpolate import interp1d
from interfaces.srv import ControlSolver
from .mpc.gusto import GuSTO
from .mpc.locp import LOCP
import numpy as np


def run_mpc_solver_node(model, config, x0, t=None, dt=None, ref_traj=None, u=None, zf=None,
                       U=None, X=None, Xf=None, dU=None, init_node=False, koopman=False, **kwargs):
    """
    Function that builds a ROS node to run MPC and runs it continuously. This node
    provides a service that at each query will run MPC once.

    :model: the model
    :config: GuSTOConfig object with parameters for GuSTO
    :x0: initial condition (n_x,)
    :t: (optional) desired trajectory time vector (M,), required if z or u variables are
                   2D arrays, used for interpolation of z and u
    :z: (optional) desired tracking trajectory for objective function. Can either be array
                   of size (M, n_z) to correspond to t, or can be a constant 1D array (n_z,)
    :u: (optional) desired control for objective function. Can either be array of size (M, n_u)
                   to correspond to t, or it can be a constant 1D array (n_u,)
    :zf: (optional) terminal target state (n_z,), defaults to 0 if Qzf provided
    :U: (optional) control constraint (Polyhedron object)
    :X: (optional) state constraint (Polyhedron object)
    :Xf: (optional) terminalstate constraint (Polyhedron object)
    :dU: (optional) u_k - u_{k-1} constraint Polyhedron object
    :init_node: (optional) whether to initialize, False if run from a different ROS node
    :kwargs: (optional): Keyword args for GuSTO (see gusto.py GuSTO __init__.py and and optionally for the solver
    (https://osqp.org/docs/interfaces/solver_settings.html)
    """
    assert t is not None or dt is not None, "Either t array or dt must be provided."
    if init_node:
        rclpy.init()

    if not koopman:
        node = MPCSolverNode(model, config, x0, t=t, dt=dt, ref_traj=ref_traj, u=u, zf=zf,
                               U=U, X=X, Xf=Xf, dU=dU, **kwargs)
    else:
        node = Koopman_MPCSolverNode(model, config, x0, t=t, dt=dt, ref_traj=ref_traj, u=u, zf=zf,
                               U=U, X=X, Xf=Xf, dU=dU, **kwargs)
    rclpy.spin(node)
    rclpy.shutdown()


def arr2jnp(x, dim, squeeze=False):
    """
    Converts python list to (-1, dim) shape jax numpy array
    """
    if squeeze:
        return jnp.asarray(x, dtype='float64').reshape(-1, dim).squeeze()
    else:
        return jnp.asarray(x, dtype='float64').reshape(-1, dim)


def jnp2arr(x):
    """ 
    Converts from jax numpy array to python list.
    """
    return x.flatten().tolist()


class MPCSolverNode(Node):
    """
    Defines a service provider node that will run the GuSTO MPC implementation.
    """

    def __init__(self, model, config, x0, t=None, zf=None, dt=None, ref_traj=None, u=None,
                 U=None, dU=None, **kwargs):
        self.model = model
        if dt is not None:
            self.dt = dt
        elif dt is None and t is not None:
            self.dt = t[1] - t[0]
        self.N = config.N
        self.t = t

        # Define target values
        self.ref_traj = ref_traj
        self.u = u
        """
        if z is not None and z.ndim == 2:
            self.z_interp = interp1d(t, z, axis=0,
                                     bounds_error=False, fill_value=(z[0, :], z[-1, :]))

        if u is not None and u.ndim == 2:
            self.u_interp = interp1d(t, u, axis=0,
                                     bounds_error=False, fill_value=(u[0, :], u[-1, :]))
        """

        # Set up GuSTO and run first solve with a simple initial guess
        self.u_init = jnp.zeros((config.N, self.model.n_u))
        self.x_init = self.model.rollout(x0, self.u_init, self.dt)

        print(f"[INIT] ref_traj.eval().shape = {ref_traj.eval().shape}")
        print(f"[INIT] z (first call to GuSTO) shape = {ref_traj.eval()[:self.N+1].shape}")
        print(f"[INIT] zf shape = {ref_traj.eval()[self.N+1].shape}")
        print(f"[INIT] Qz.shape = {config.Qz.shape}")
        print(f"[INIT] R.shape = {config.R.shape}")
        print(f"[INIT] x_init.shape = {self.x_init.shape}")

        print(f"x0 values:\n{x0}")
        print(f"x_init[0] values:\n{self.x_init[0]}")

        z = jnp.array(self.ref_traj.eval())[:self.N+1]
        zf = self.ref_traj.eval()[-1]
        print(f"z[0] (first reference point):\n{z[0]}")
        print(f"zf (final reference point):\n{zf}")

        self.gusto = GuSTO(self.model, config, x0, self.u_init, self.x_init, z=jnp.array(self.ref_traj.eval())[:self.N+1], # u=u,
                           zf=jnp.array(self.ref_traj.eval())[self.N+1], U=U, dU=dU, **kwargs)  # X=X, Xf=Xf,

        self.xopt, self.uopt, _, _ = self.gusto.get_solution()
        self.topt = self.dt * jnp.arange(self.N + 1)

        # Also force JIT-compilation of encoder mapping and conversions
        model.encode(jnp.zeros(self.model.n_y))

        # Initialize the ROS node
        super().__init__('mpc_solver_node')

        # Define the service, which uses the gusto callback function
        self.srv = self.create_service(ControlSolver, 'mpc_solver', self.gusto_callback)
        self.get_logger().info('MPC solver service has been created.')

    def gusto_callback(self, request, response):
        """
        Callback function that runs when the service is queried, request message contains:
        t0, y0, u0

        and the response message will contain:

        t, xopt, uopt, zopt
        """
        t0 = request.t0

        full_ref = np.array(self.ref_traj.eval())  # shape = (M, n_z)
        M = full_ref.shape[0]
        T_final = (M - 1) * self.dt        # 2) If t0 is beyond the last valid reference time, return done=True immediately
        if t0 > T_final:
            response.done = True
            return response
        else:
            response.done = False
        
        y0 = arr2jnp(request.y0, self.model.n_y, squeeze=True)

        # TODO: include interp for adiabatic here 

        print("DEBUG: Shape of incoming y0 is ", y0.shape)
        x0 = self.model.encode(y0)

        start_idx = int(t0 / self.dt)
        end_idx = start_idx + (self.N + 1)        
        if end_idx <= M:
            # We still have at least (N+1) points remaining
            slice_np = full_ref[start_idx:end_idx, :]  # shape = (N+1, n_z)
        else:
            # We are near the end; slice what remains, then pad
            available = full_ref[start_idx:M, :]  # shape = (M - start_idx, n_z)
            n_missing = (self.N + 1) - (M - start_idx)  # how many rows we’re short
            last_row = full_ref[M - 1, :].reshape(1, -1)  # shape = (1, n_z)
            pad_rows = np.repeat(last_row, n_missing, axis=0)  # shape = (n_missing, n_z)
            slice_np = np.vstack([available, pad_rows])  # shape = (N+1, n_z)        # Convert to JAX arrays for solver
        ref_window = jnp.array(slice_np)  # (N+1, n_z)
        ref_final = jnp.array(slice_np[-1, :])  # (n_z,)

        # Get initial guess
        idx0 = jnp.searchsorted(self.topt, t0, side='right')
        n_remaining_u = self.N - idx0
        n_remaining_x = self.N + 1 - idx0

        u_init_temp = self.u_init.copy()  # Create a copy to modify
        x_init_temp = self.x_init.copy()

        for i in range(n_remaining_u):
            u_init_temp = u_init_temp.at[i].set(self.uopt[idx0 + i, :])
        for i in range(n_remaining_u, self.N):
            u_init_temp = u_init_temp.at[i].set(self.uopt[-1, :])

        for i in range(n_remaining_x):
            x_init_temp = x_init_temp.at[i].set(self.xopt[idx0 + i, :])
        for i in range(n_remaining_x, self.N + 1):
            x_init_temp = x_init_temp.at[i].set(self.xopt[-1, :])

        self.u_init = u_init_temp  # Assign the modified copy back
        self.x_init = x_init_temp

        # Update LOCP parameter with the previously applied control
        self.gusto.locp.u0_prev.value = np.asarray(request.u0)

        # Solve GuSTO and get solution
        self.gusto.solve(x0, self.u_init, self.x_init, z=ref_window, zf=ref_final)
        self.xopt, self.uopt, zopt, t_solve = self.gusto.get_solution()

        print("Shape of self.uopt: ", self.uopt.shape)
        self.topt = t0 + self.dt * jnp.arange(self.N + 1)
        response.t = jnp2arr(self.topt)
        response.xopt = jnp2arr(self.xopt)
        response.uopt = jnp2arr(self.uopt)
        response.zopt = jnp2arr(zopt)
        response.solve_time = t_solve

        return response


class Koopman_MPCSolverNode(Node):
    """
    Defines a service provider node that will run MPC using LOCP
    """

    def __init__(self, model, config, x0, t=None, zf=None, dt=None, ref_traj=None, u=None,
                 U=None, X=None, Xf=None, dU=None, verbose=0, warm_start=True, **kwargs):
        self.model = model
        self.planning_horizon = config.N  # Assuming config contains the planning horizon
        self.dt = dt

        # Extract necessary matrices from the Koopman model
        self.A_d = self.model.A_d
        self.B_d = self.model.B_d
        self.C = self.model.C
        self.H = self.model.H

        # Print out the model matrices for verification
        print(f"[INIT] A_d.shape = {self.A_d.shape}")
        print(f"[INIT] B_d.shape = {self.B_d.shape}")
        if self.C is not None:
            print(f"[INIT] C.shape = {self.C.shape}")
        print(f"[INIT] H.shape = {self.H.shape}")

        # Define target values and setup interpolation if needed
        self.cost_params = config  # Assuming config contains the cost parameters

        self.ref_traj = ref_traj

        self.verbose = verbose

        # LOCP problem setup
        if self.verbose == 2:
            locp_verbose = True
        else:
            locp_verbose = False

        # Initialize the LOCP object with Koopman model matrices
        self.locp = LOCP(self.planning_horizon, self.H, self.cost_params.Q, self.cost_params.R,
                         Qzf=self.cost_params.Qf, U=U, X=X, Xf=Xf, dU=dU, verbose=locp_verbose, warm_start=warm_start,
                         is_tr_active=False, **kwargs)

        # Get the linear model matrices for use in the optimization
        self.d_d = [self.model.d_d for i in range(self.planning_horizon)] if hasattr(self.model, 'd_d') else [np.zeros(self.A_d.shape[0]) for i in range(self.planning_horizon)]

        self.X = X
        self.xopt = None
        self.uopt = None
        self.topt = None

        # Initialize the ROS node
        super().__init__('koopman_mpc_solver_node')

        # Define the service, which uses the mpc_callback function
        self.srv = self.create_service(ControlSolver, 'mpc_solver', self.mpc_callback)

    def mpc_callback(self, request, response):
        """
        Callback function that runs when the service is queried, request message contains:
        t0, x0

        and the response message will contain:

        t, xopt, uopt, zopt
        """
        t0 = request.t0
        x0 = arr2jnp(request.x0, self.model.N, squeeze=True)

        z, zf, u = self.get_target(t0)
        self.locp.update(self.A_d, self.B_d, self.d_d, x0, None, 0, 0, z=z, zf=zf, u=u)
        Jstar, success, solver_stats = self.locp.solve()

        if success:
            if self.verbose:
                try:
                    print('{:.3f} s from LOCP solve'.format(solver_stats.solve_time))
                except:
                    print('Solver failed.')
            self.xopt, self.uopt, _ = self.locp.get_solution()

        else:
            print('No solution found, extending previous solution')
            self.xopt = np.concatenate((self.xopt[1:, :], np.expand_dims(self.xopt[-1, :], axis=0)), axis=0)
            self.uopt = np.concatenate((self.uopt[1:, :], np.expand_dims(self.uopt[-1, :], axis=0)), axis=0)

        self.topt = t0 + self.dt * np.arange(self.planning_horizon + 1)
        response.t = jnp2arr(self.topt)
        response.xopt = jnp2arr(self.xopt)
        response.uopt = jnp2arr(self.uopt)
        try:
            response.solve_time = solver_stats.solve_time
        except:
            response.solve_time = 0.0
        return response

    def get_target(self, t0):
        """
        Returns z, zf, u arrays for GuSTO solve
        """

        # Get target z terms directly from ref_traj
        start_idx = int(t0 / self.dt)
        end_idx = start_idx + (self.planning_horizon + 1)
        if end_idx <= len(self.ref_traj.eval()):
            z = self.ref_traj.eval()[start_idx:end_idx, :]
        else:
            available = self.ref_traj.eval()[start_idx:, :]
            n_missing = (self.planning_horizon + 1) - len(available)
            last_row = self.ref_traj.eval()[-1, :].reshape(1, -1)
            pad_rows = np.repeat(last_row, n_missing, axis=0)
            z = np.vstack([available, pad_rows])

        # Get target zf term for cost function (last value)
        zf = z[-1, :]

        # Get target u terms directly from ref_traj
        u = self.ref_traj.u[start_idx:end_idx] if self.ref_traj.u is not None else None

        return z, zf, u
