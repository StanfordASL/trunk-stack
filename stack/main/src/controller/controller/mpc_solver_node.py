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
import time


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

    def __init__(self, model, config, x0, t=None, zf=None, dt=None, ref_traj=None, u=None, U=None, X=None, Xf=None, dU=None, verbose=2, warm_start=True, **kwargs):
        self.model = model
        self.N = config.N
        self.Qz = config.Qz
        self.Qzf = config.Qzf

        self.R = config.R
        self.R_du = config.R_du
        self.x_char = config.x_char 

        self.dt = dt

        # Extract necessary matrices from the Koopman model
        self.A_d = [self.model.A_d for i in range(self.N)]
        self.B_d = [self.model.B_d for i in range(self.N)]
        self.C = self.model.C
        self.H = self.model.H

        self.ref_traj = ref_traj
        self.verbose = verbose

        # LOCP problem setup
        if self.verbose == 2:
            locp_verbose = True
        else:
            locp_verbose = False

        # Initialize LOCP
        self.locp = LOCP(self.N, self.H, self.Qz, self.R, Qzf=self.Qzf,
                         U=U, X=X, Xf=Xf, dU=dU,
                         verbose=locp_verbose, warm_start=warm_start, 
                         x_char=self.x_char, R_du=self.R_du, **kwargs)

        # Get the linear model matrices for use in the optimization
        # Defaults to zero for sure
        self.d_d = [self.model.d_d for i in range(self.N)] if hasattr(self.model, 'd_d') else [np.zeros(self.A_d[0].shape[0]) for i in range(self.N)]

        self.X = X
        self.xopt = None
        self.uopt = None
        self.topt = None

        # Initialize the ROS node
        super().__init__('koopman_mpc_solver_node')

        # --- Warm start: initial solve for JIT compilation ---
        try:
            print("[WARM START] Starting initial solve to trigger JIT compilation...")
            dummy_y = jnp.zeros(self.model.n_y)
            dummy_x = self.model.encode(dummy_y)
            xk = jnp.tile(dummy_x.reshape(1, -1), (self.N + 1, 1))
            z_dummy = jnp.tile(jnp.zeros(self.H.shape[0]), (self.N + 1, 1))
            zf_dummy = z_dummy[-1]

            # Dummy solve to trigger JIT compile
            # self.locp.update(self.A_d, self.B_d, self.d_d, dummy_x, xk, 0.0, 0.0, z=z_dummy, zf=zf_dummy)
            self.locp.update(self.A_d, self.B_d, self.d_d, dummy_x, None, 0.0, 0.0, z=z_dummy, zf=zf_dummy)
            J_init, success, stats = self.locp.solve()

            if success:
                self.xopt, self.uopt, _ = self.locp.get_solution()
                self.get_logger().info(f'[WARM START] Initial solve completed successfully in {stats.solve_time:.4f} s.')
            else:
                self.xopt = xk
                self.uopt = jnp.zeros((self.N, self.model.n_u))
                self.get_logger().warn('[WARM START] Initial dummy solve failed. Defaulting to zero input rollout.')
        except Exception as e:
            self.get_logger().warn(f'[WARM START] Exception during warm start solve: {e}')
            self.xopt = jnp.tile(jnp.zeros(self.model.n_x), (self.N + 1, 1))
            self.uopt = jnp.zeros((self.N, self.model.n_u))

        # Define the service, which uses the mpc_callback function
        self.srv = self.create_service(ControlSolver, 'mpc_solver', self.mpc_callback)
        self.get_logger().info('MPC solver service has been created.')

    def mpc_callback(self, request, response):
        t0 = request.t0
        self.get_logger().info(f"Received t0 = {t0}")

        y0 = arr2jnp(request.y0, self.model.n_y, squeeze=True)
        self.get_logger().info(f"y0.shape = {y0.shape}, y0 = {y0}")

        x0 = self.model.encode(y0)
        self.get_logger().info(f"Encoded x0.shape = {x0.shape}")

        # xk = np.tile(x0.reshape(1, -1), (self.locp.N + 1, 1))
        # self.get_logger().info(f"Encoded xk.shape = {xk.shape}")

        full_ref = np.array(self.ref_traj.eval())
        M = full_ref.shape[0]
        T_final = (M - 1) * self.dt
        self.get_logger().info(f"full_ref.shape = {full_ref.shape}, T_final = {T_final}")

        if t0 > T_final:
            self.get_logger().info("t0 exceeds reference time horizon, setting response.done = True")
            response.done = True
            return response
        else:
            response.done = False

        start_idx = int(t0 / self.dt)
        end_idx = start_idx + (self.N + 1)
        self.get_logger().info(f"Slicing ref from start_idx = {start_idx} to end_idx = {end_idx}")

        if end_idx <= M:
            slice_np = full_ref[start_idx:end_idx, :]
        else:
            available = full_ref[start_idx:M, :]
            n_missing = (self.N + 1) - (M - start_idx)
            last_row = full_ref[M - 1, :].reshape(1, -1)
            pad_rows = np.repeat(last_row, n_missing, axis=0)
            slice_np = np.vstack([available, pad_rows])
            self.get_logger().info(f"Near end of ref: padded {n_missing} rows with last ref point")

        z = slice_np
        zf = slice_np[-1, :]
        self.get_logger().info(f"z.shape = {z.shape}, zf.shape = {zf.shape}")

        # THIS IS NOT PASSED IN JOHNS NODE - But it doesnt have Rdu
        self.locp.u0_prev.value = np.asarray(request.u0)

        # CHECK WHAT IS PASSED FOR D_D AND XK
        try:
            # self.locp.update(self.A_d, self.B_d, self.d_d, x0, xk, 0, 0, z=z, zf=zf, u=u)
            self.locp.update(self.A_d, self.B_d, self.d_d, x0, None, 0, 0, z=z, zf=zf, u=None)
            self.get_logger().info("locp.update() successful")
        except Exception as e:
            # Log the full traceback for better debugging
            self.get_logger().error(f"Exception during locp.update.")

        self.get_logger().info("locp.update() successful")

        try:
            start_time = time.time()
            Jstar, success, solver_stats = self.locp.solve()
            elapsed_time = time.time() - start_time
            self.get_logger().info(f"[SOLVE] LOCP solve success={success}, J*={Jstar:.4f}, elapsed_time={elapsed_time:.4f} s")
        except Exception as e:
            self.get_logger().error(f"Exception during solve: {e}")
            success = False

        if success:
            try:
                self.get_logger().debug('{:.3f} s from LOCP solve'.format(solver_stats.solve_time))
            except Exception:
                self.get_logger().debug('Solver succeeded but no solve_time available.')
            self.xopt, self.uopt, _ = self.locp.get_solution()
            self.get_logger().debug(f"xopt.shape = {self.xopt.shape}, uopt.shape = {self.uopt.shape}")
        else:
            self.get_logger().debug('No solution found, extending previous solution')
            self.xopt = np.concatenate((self.xopt[1:, :], np.expand_dims(self.xopt[-1, :], axis=0)), axis=0)
            self.uopt = np.concatenate((self.uopt[1:, :], np.expand_dims(self.uopt[-1, :], axis=0)), axis=0)

        self.topt = t0 + self.dt * np.arange(self.N + 1)
        response.t = jnp2arr(self.topt)
        response.xopt = jnp2arr(self.xopt)
        response.uopt = jnp2arr(self.uopt)

        try:
            response.solve_time = solver_stats.solve_time
        except Exception:
            response.solve_time = 0.0

        try:
            response.zopt = jnp2arr(z)
            self.get_logger().debug(f"response.zopt.shape = {np.array(z).shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to assign response.zopt: {e}")
            response.zopt = []

        self.get_logger().debug("Response packaged and sent back.")
        return response
