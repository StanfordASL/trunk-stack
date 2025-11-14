import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", False)
import rclpy                        # type: ignore
from rclpy.node import Node         # type: ignore
from interfaces.srv import ControlSolver
from .mpc.gusto_upgrade_trunk import GuSTO
import numpy as np


def run_mpc_solver_node(K_r=None , I=None, exps_I = None, x0=None, t=None, dt=None, ref_traj=None, u=None, zf=None,
                       U=None, X=None, Xf=None, dU=None, init_guess_type='shift', init_node=False, **kwargs):
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
    node = MPCSolverNode( K_r = K_r , I=I, exps_I = exps_I, x0=None, t=t, dt=dt, ref_traj=ref_traj, u=u, zf=zf, U=U, X=X, Xf=Xf, dU=dU, init_guess_type=init_guess_type,
                         **kwargs)
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

    def __init__(self, K_r=None, I=None, exps_I=None, x0=None, t=None, zf=None, dt=None, ref_traj=None, u=None,
                 U=None, dU=None, init_guess_type='shift',**kwargs):
        
       
        self.dt = dt
        self.U = U
        self.dU = dU

        self.K_r = K_r
        self.I = I 
        self.exps_I = exps_I

        
        # Extract dimensions
        self.n_x = 9     # state dimension
        self.n_u = 6     # control dimension
        self.n_z = 3 # performance dimension
        
        # shift = self.model.ssm.specified_params["shift_steps"]  # Is 0 if there is no subsampling
        # num_delay = self.model.ssm.specified_params["embedding_up_to"]
        # pad_length = self.model.ssm.specified_params["num_u"] * ((1 + shift) * num_delay - shift)
        # self.u_ref_init = jnp.zeros((pad_length,))
        print("dt is: ", dt)

        if dt is not None:
            self.dt = dt
        elif dt is None and t is not None:
            self.dt = t[1] - t[0]
        self.N = 10
        self.t = t

        # Define target values
        self.ref_traj = ref_traj
        self.z_ref = self.ref_traj
        print(self.ref_traj)
        self.u = u
        
        # New code

        #  # Initialize GuSTO with zeros as initial guess
        # self.u_init = jnp.zeros((self.N, self.n_u))
        # self.x_init = self.model.multistep_dynamics(x0, self.u_init)
        # self.x_init = self.x_init[:self.N]
        # # can be such that x0 is full state...
        # z_ref_win = self.ref_traj[0:self.N]
        # print(z_ref_win.shape)
        # #print(f"x_init shape: {x_init.shape}, z_ref_win shape: {z_ref_win.shape}")
        # self.gusto = GuSTO(
        #     self.model, 
        #     self.config,
        #     x0,
        #     self.u_init,
        #     self.x_init,
        #     z=z_ref_win,
        #     zf=z_ref_win[-1],
        #     U=U,
        #     dU=dU,
        #     start_with_solve=True,
        #     exps_M=self.exps_M,
        #     M=self.M,
        #     U_select = self.U_select,
        #     W_sm = self.W_sm,
        #     epsilon_sm = self.epsilon_sm,
        #     case_rbf = self.case_rbf,
        #     solver='CLARABEL',
        # )
        # """
        # if z is not None and z.ndim == 2:
        #     self.z_interp = interp1d(t, z, axis=0, bounds_error=False, fill_value=(z[0, :], z[-1, :]))

        # if u is not None and u.ndim == 2:
        #     self.u_interp = interp1d(t, u, axis=0, bounds_error=False, fill_value=(u[0, :], u[-1, :]))
        # """
        # # Set up GuSTO and run first solve with a simple initial guess
        # # self.u_init = jnp.zeros((config.N, self.model.n_u))
        
        # # self.x_init = self.model.rollout(x0, self.u_init, self.dt)

        # # self.gusto = GuSTO(self.model, config, x0, self.u_init, self.x_init, z=jnp.array(self.ref_traj.eval())[:self.N+1],
        # #                    zf=jnp.array(self.ref_traj.eval())[self.N+1], U=U, dU=dU, **kwargs)

        # self.xopt, self.uopt, _, _ = self.gusto.get_solution()

        # self.x_prev = self.xopt
        # self.u_prev = self.uopt
        # self.t_idx = 0

        # self.topt = self.dt * jnp.arange(self.N) # additional time-keeper not involved compute_control

        
       
         # For the slew rate cost
        self.last_applied_u = None

        # What type of initial guess to use (shift, dyn_feasible, zeros)
        self.init_guess_type = init_guess_type
        # Also force JIT-compilation of encoder mapping and conversions
        #self.model.encode(jnp.zeros(self.model.n_y))

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
        self.t0 = request.t0

        # 1) Compute the reference’s final time based on its length and dt
        full_ref = np.array(self.ref_traj)  # shape = (M, n_z)
        M_end = full_ref.shape[0]
        T_final = (M_end - 1) * self.dt
        # print(M_end)
        # print(T_final)
        # print(self.t)
        
        # 2) If t0 is beyond the last valid reference time, return done=True immediately
        if self.t0 > T_final:
            response.done = True
            return response
        else:
            response.done = False

        # 3) Set initial condition for solve
        current_state = np.array(request.y0[:3])
        self.t_idx = int(self.t0 / self.dt)
        us_xs_0_ubar = self.I @ self.eval_monomials_single_sm(full_ref[self.t_idx, :], self.exps_I) 
        u = -self.K_r @ np.tile((current_state - full_ref[self.t_idx, :]).reshape(-1, 1), (3, 1)) + us_xs_0_ubar.reshape(-1, 1)# LQR control law around current position
     
        #print(current_state)

        #print(reduced_x)
        #reduced_x = 0*reduced_x
        # # 3) Reconstruct y0 from the delayed embedding
        # y0_np = np.array(request.y0)  # purely for debugging or sanity checks
        # print("Received request.y0 of shape:", y0_np.shape)
        # y0 = arr2jnp(request.y0, self.model.n_y, squeeze=True)
        # num_blocks = self.model.ssm.specified_params["embedding_up_to"] + 1
        # block_size = self.model.n_y // num_blocks
        # y0_blocks = y0.reshape((num_blocks, block_size))

        # state_part = y0_blocks[:, : (block_size - self.model.n_u)]
        # u_part = y0_blocks[:, (block_size - self.model.n_u):]

        # y0_scaled = jnp.concatenate([state_part, u_part], axis=1)
        # y0 = y0_scaled.reshape((self.model.n_y,))

        # x0 = self.model.encode(y0)

        # # 4) Recover previous control
        # if self.u_prev0 is None:
        #     self.u_prev0 = np.zeros((self.model.n_u,))
        # else:
        #     self.u_prev0 = np.array(request.u0)

        # self.u_prev = jnp.array(request.u0) # in case of delays we have this input - maybe bring this back for delays?
        # print(self.u_prev.shape)
        # print(t_solve)
        # CHANGED
        # 5) Update u_ref_init by shifting in the previous input
        # if self.u_ref_init.shape[0] >= self.model.n_u:
        #     self.u_ref_init = jnp.concatenate(
        #        [self.u_prev0, self.u_ref_init[:-self.model.n_u]],
        #        axis=0
        #    )
        # x0_aug = jnp.concatenate([x0, self.u_ref_init], axis=0)

        # 6) Build ref_window of length (N+1) rows, padding with the last row if we run out

        # start_idx = int(t0 / self.dt)
        # end_idx = start_idx + (self.N + 1)

        # if end_idx <= M:
        #     # We still have at least (N+1) points remaining
        #     slice_np = full_ref[start_idx:end_idx, :]  # shape = (N+1, n_z)
        # else:
        #     # We are near the end; slice what remains, then pad
        #     available = full_ref[start_idx:M, :]  # shape = (M - start_idx, n_z)
        #     n_missing = (self.N + 1) - (M - start_idx)  # how many rows we’re short
        #     last_row = full_ref[M - 1, :].reshape(1, -1)  # shape = (1, n_z)
        #     pad_rows = np.repeat(last_row, n_missing, axis=0)  # shape = (n_missing, n_z)
        #     slice_np = np.vstack([available, pad_rows])  # shape = (N+1, n_z)

        # # Convert to JAX arrays for solver
        # ref_window = jnp.array(slice_np)  # (N+1, n_z)
        # ref_final = jnp.array(slice_np[-1, :])  # (n_z,)

        # # 7) Build the MPC initial guesses (rollout‐based warm start)
        # idx0 = int(jnp.searchsorted(self.topt, t0, side='right'))
        # n_remaining_u = self.N - idx0
        # n_remaining_x = self.N + 1 - idx0

        # u_init_temp = self.u_init.copy()
        # x_init_temp = self.x_init.copy()

        # for i in range(n_remaining_u):
        #     u_init_temp = u_init_temp.at[i].set(self.uopt[idx0 + i, :])
        # for i in range(n_remaining_u, self.N):
        #     u_init_temp = u_init_temp.at[i].set(self.uopt[-1, :])

        # for i in range(n_remaining_x):
        #     x_init_temp = x_init_temp.at[i].set(self.xopt[idx0 + i, :])
        # for i in range(n_remaining_x, self.N + 1):
        #     x_init_temp = x_init_temp.at[i].set(self.xopt[-1, :])

        # self.u_init = u_init_temp
        # self.x_init = x_init_temp

        # # 8) Update the LOCP parameter for previous input
        # # print("Shape of self.u_prev0:", self.u_prev0.shape)
        # self.gusto.locp.u0_prev.value = self.u_prev0

        # # 9) Solve GuSTO with the (possibly padded) reference

        # self.gusto.solve(
        #     # CHANGED
        #     # x0_aug,
        #     x0,
        #     self.u_init,
        #     self.x_init,
        #     z=ref_window,
        #     zf=ref_final
        # )

        # self.xopt, self.uopt, zopt, t_solve = self.gusto.get_solution()
        # xopt_extracted = self.xopt[:, : self.model.n_x]

        # 10) Package the response
        self.topt = jnp.array([self.t0, self.t0])
        self.uopt = u 
        print(self.topt)
        response.t = jnp2arr(self.topt)
        response.uopt = jnp2arr(self.uopt)
        
        return response
    
    def compute_control(self, state: jnp.ndarray):
        """
        Compute the control action for the current observation.
        
        Args:
            obs: Current observation (could be full or partial state)
            
        Returns:
            u: Optimal control action
            info: Dictionary containing additional information
        """
        #t_start = time.time()
        #import time
        #t_start = time.time()
        
        self.t_idx = int(self.t0 / self.dt)
        #t_after_tidx = time.time()
        
        # Get reference trajectory for current MPC window
        max_ind = min(self.t_idx + self.N, len(self.z_ref))
        z_ref_win = self.z_ref[self.t_idx:max_ind]
        # Pad if not length N+1
        if len(z_ref_win) < self.N:
            k = self.N + 1 - len(z_ref_win)
            print(f"Warning: z_ref_win is shorter than N+1, padding with last reference {k} times")
            last_z = jnp.tile(self.z_ref[-1], (k, 1))
            z_ref_win = jnp.concatenate([z_ref_win, last_z])

        #t_after_ref = time.time()

        # Initialize next MPC problem
        if self.init_guess_type == 'shift':
            # We shift x_prev by one step and then re-insert 'state' as x_init[0].
            # This helps the solver solve from the correct initial state.
            x_init = jnp.concatenate([self.x_prev[1:], 
                                    self.model.discrete_dynamics(self.x_prev[-1], self.u_prev[-1])[None, :]], axis=0)
            
            #print(f"x_init shape: {x_init.shape}, z_ref_win shape: {z_ref_win.shape}")
            #print(f"state shape: {state.shape}, x_init[0] shape: {x_init[0].shape}")
            x_init = x_init.at[0].set(state)  # Force the first predicted state to match the real current state 
            # here we have a problem if x_init is only a subset.
            u_init = jnp.concatenate([self.u_prev[1:], self.u_prev[-1:]], axis=0)

        elif self.init_guess_type == 'dyn_feasible':
            u_init = jnp.concatenate([self.u_prev[1:], self.u_prev[-1:]], axis=0)
            x_init = self.model.multistep_dynamics(state, u_init)
            #print(x_init.shape)
            x_init = x_init[:self.N]
            
        else:
            u_init = jnp.zeros((self.N, self.n_u))
            x_init = self.model.multistep_dynamics(state, u_init)
            x_init = x_init[:self.N]
        
        #t_after_init = time.time()

        # Update LOCP parameter with the previously applied control
        if self.last_applied_u is not None:
            self.gusto.locp.u0_prev.value = np.asarray(self.last_applied_u)
        
        # Solve MPC problem
        self.gusto.solve(state, u_init, x_init, z=z_ref_win, zf=z_ref_win[-1])
        #t_after_solve = time.time()

        x_opt, u_opt, z_opt, solve_time = self.gusto.get_solution()
        #t_after_get = time.time()
        
         # Log timing breakdown
        # if hasattr(self, 'solve_count'):
        #     self.solve_count += 1
        # else:
        #     self.solve_count = 1
        
    #     if self.solve_count % 10 == 0:  # Log every 10 solves
    #         self.get_logger().info(f"""
    # Timing breakdown (ms):
    # t_idx computation: {(t_after_tidx - t_start)*1000:.2f}
    # Reference prep: {(t_after_ref - t_after_tidx)*1000:.2f}
    # Init guess prep: {(t_after_init - t_after_ref)*1000:.2f}
    # GuSTO solve: {(t_after_solve - t_after_init)*1000:.2f}
    # Get solution: {(t_after_get - t_after_solve)*1000:.2f}
    # Total: {(t_after_get - t_start)*1000:.2f}
    # Reported solve_time: {solve_time*1000:.2f}
    #         """)

        # Store solution for warm start
        self.x_prev = x_opt
        self.u_prev = u_opt

        # Increment time index
        #self.t_idx += 1
        
        # Prepare info dictionary
        # info = {
        #     'solve_time': solve_time,
        #     'total_time': time.time() - t_start,
        #     'predicted_trajectory': z_opt,
        #     'control_trajectory': u_opt
        # }
        
        self.last_applied_u = u_opt[0]

        return x_opt, u_opt, z_opt, solve_time
    

    # def get_target(self, t0):
    #     """
    #     Returns z, zf, u arrays for GuSTO solve.
    #     """
    #     t = t0 + self.dt * jnp.arange(self.N + 1)

    #     # Get target z terms for cost function
    #     if self.z is not None:
    #         if self.z.ndim == 2:
    #             z = self.z_interp(t)
    #         else:
    #             z = self.z.reshape(1, -1).repeat(self.N + 1)
    #     else:
    #         z = None

    #     # Get target zf term for cost function 
    #     if z is not None:
    #         zf = z[-1, :]
    #     else:
    #         zf = None

    #     # Get target u terms for cost function
    #     if self.u is not None:
    #         if self.u.ndim == 2:
    #             u = self.u_interp(t)
    #         else:
    #             u = self.u.reshape(1, -1).repeat(self.N)
    #     else:
    #         u = None

    #     return z, zf, u

    def eval_monomials_single_sm(self, xi, exps):
        """Evaluate monomials at a single point with constant term added"""
        xi = jnp.asarray(xi).flatten()
        x = xi.reshape(1, -1, 1)  # Shape: (1, 9, 1)
        
        # Add dimension to exps for broadcasting
        exps_expanded = exps[:, :, None]  # Shape: (219, 9, 1)
        
        powered = x ** exps_expanded  # Now broadcasting works: (1, 9, 1) ** (219, 9, 1)
        u = jnp.prod(powered, axis=1, keepdims=True)  # Shape: (219, 1)
        # Reshape u to 2D before concatenating
        u = u.squeeze(axis=-1)  # Shape: (219, 1)
        
        # Add constant term (1)
        u = jnp.vstack([u, jnp.ones((1, 1))])
        return u