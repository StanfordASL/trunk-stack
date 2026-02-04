"""
GuSTO (Guaranteed Sequential Trajectory Optimization) implementation, adopted from original code.
"""

import time
import jax
import jax.numpy as jnp
from functools import partial
from .locp_upgrade import LOCP
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.distance import cdist

@dataclass
class GuSTOConfig:
    """
    GuSTOConfig class for storing GuSTO parameters.
    """
    Qz: jnp.ndarray                     # positive semi-definite performance variable weighting matrix
    Qzf: jnp.ndarray                    # positive semi-definite terminal performance variable weighting matrix
    R: jnp.ndarray                      # positive definite control weighting matrix
    x_char: jnp.ndarray                 # characteristic quantities for x, for scaling
    f_char: jnp.ndarray                 # characteristic quantities for f, for scaling
    N: int = 8                          # integer optimization horizon
    epsilon: float = 0.01               # constraint violation threshold
    max_gusto_iters: int = 1           # maximum number of GuSTO iterations
    delta0: float = 1e4                 # trust region
    omega0: float = 1                   # slack variable weighting
    rho: float = 0.1                    # model compute_accuracy
    beta_fail: float = 0.5              # trust region update TODO: specify
    beta_succ: float = 2                # trust region update TODO: specify
    gamma_fail: float = 5               # cost function penalty term TODO: specify
    omega_max: float = 1e10             # cost function penalty term TODO: specify
    convg_thresh: float = 0.01          # convergence threshold
    verbose: int = 0                    # verbosity level (0, 1, 2)
    warm_start: bool = True             # warm start the solver
    H: jnp.ndarray = None               # performance mapping matrix
    R_du: jnp.ndarray = None  # control rate weighting matrix

@jax.tree_util.register_static
class GuSTO:
    """
    GuSTO class for solving trajectory optimization problems via SQP.

    :model: TemplateModel object describing dynamics
    :config: GuSTOConfig object with parameters for GuSTO (see above)
    :x0: initial condition (n_x,)
    :u_init: control initial guess (N, n_u)
    :x_init: state initial guess (N+1, n_x)
    :z: (optional) desired tracking trajectory for objective function (N+1, n_z)
    :u: (optional) desired control for objective function (N, n_u)
    :zf: (optional) terminal target state (n_z,), defaults to 0 if Qzf provided
    :U: (optional) control constraint (Polyhedron object)
    :X: (optional) state constraint (Polyhedron object)
    :Xf: (optional) terminalstate constraint (Polyhedron object)
    :dU: (optional) u_k - u_{k-1} slew rate constraint (Polyhedron object)
    :kwargs: Keyword arguments for the solver and what solver to use, e.g.
    (https://osqp.org/docs/interfaces/solver_settings.html)
    """
    def __init__(self, model, config, x0, u_init, x_init,
                z=None, u=None, zf=None, U=None, X=None, Xf=None, dU=None,
                start_with_solve=True, exps_M = None, M = None, U_select = None, W_sm = None, epsilon_sm = None, case_rbf = None, **kwargs):
        self.model = model
        self.dt = self.model.dt

        # Extract configuration parameters
        self._extract_config(config)

        # Problem dimensions
        self.n_x = x0.shape[0]
        # here the problem should change..., it should be size of x_init 
        self.n_u = self.R.shape[0]
        self.n_z = self.Qz.shape[0]

        # Constraints - State, control, final state
        self.U = U
        self.X = X
        self.Xf = Xf
        self.dU = dU
        self.exps_M = exps_M
        self.M = M

        self.U_select = U_select
        self.W_sm = W_sm
        self.epsilon_sm = epsilon_sm
        self.case_rbf = case_rbf

        # Characteristic quantities
        self.x_scale = 1. / jnp.abs(self.x_char)
        self.f_scale = 1. / jnp.abs(self.f_char)

        # Problem parameters
        self.x_k = None  # previous state
        self.u_k = None  # previous input
        self.locp_solve_time = None  # time spent in LOCP solve

        # LOCP problem
        if self.verbose == 2:
            locp_verbose = True
        else:
            locp_verbose = False
        
        # Assert that there is a performance mapping matrix
        assert self.H is not None, 'Performance mapping matrix H must be provided'
        self.nonlinear_perf_mapping = True

        # Initialize LOCP
        self.locp = LOCP(self.N, self.H, self.Qz, self.R, Qzf=self.Qzf,
                         U=self.U, X=self.X, Xf=self.Xf, dU=self.dU,
                         verbose=locp_verbose, warm_start=self.warm_start, x_char=self.x_char,
                         nonlinear_perf_mapping=self.nonlinear_perf_mapping, R_du=self.R_du, **kwargs)

        # First SCP solve
        if start_with_solve:
            print('First solve may take a while due to factorization and caching.')
            self.solve(x0, u_init, x_init, z, zf, u)
    
    def rbf_eval_batch(u, U_select, W, epsilon, case_rbf):
        """
        RBF evaluation function for multiple points (batch)
        
        Inputs:
            u: (N, d) array, where each row is a d-dimensional evaluation point
            U_select: (M, d) array, where each row is a d-dimensional center
            W: (M, d_prime) array, where each row is the weight vector for a center
            epsilon: shape parameter for the Gaussian RBF
            case_rbf: boolean, if True use Gaussian RBF, else use distance
        
        Output:
            final_value: (N, d_prime) array, where each row is the RBF interpolation
                        evaluated at the corresponding row of u
        """
        u = np.asarray(u)
        
        # Ensure u is 2D
        if u.ndim == 1:
            u = u.reshape(1, -1)
        
        N, d = u.shape           # N: number of evaluation points, d: dimension
        M, d_check = U_select.shape  # M: number of centers
        M_check, d_prime = W.shape   # d_prime: dimension of output values
        
        # Validate dimensions
        if d_check != d:
            raise ValueError(f'Dimension of U_select ({d_check}) must match dimension of u ({d})')
        if M_check != M:
            raise ValueError(f'Number of rows in W ({M_check}) must match number of centers in U_select ({M})')
        
        # Compute pairwise Euclidean distances between u and U_select
        # dist: (N, M) matrix, where dist[i,j] is the distance between u[i] and U_select[j]
        dist = cdist(u, U_select, metric='euclidean')  # Shape: (N, M)
        
        # Compute Gaussian RBF: phi(r) = exp(-(epsilon * r)^2)
        # A: (N, M) matrix, where A[i,j] = phi(||u[i] - U_select[j]||)
        if case_rbf:
            A = np.exp(-(epsilon * dist)**2)
        else:
            A = dist
        
        # Compute RBF interpolation: final_value = A @ W
        # A: (N, M), W: (M, d_prime) -> final_value: (N, d_prime)
        final_value = A @ W  # Shape: (N, d_prime)
        
        return final_value

    @partial(jax.jit, static_argnums=(0,))
    def rbf_eval_single_jax(self, u):
        """
        JAX-compatible version of rbf_eval_single
        Expects MATLAB-style format: U_select is (d, M), W is (M, d_prime)
        
        Args:
            u: (d,) or (1, d) - evaluation point
            U_select: (d, M) - each column is a center
            W: (M, d_prime) - weights
            epsilon: scalar - shape parameter
            case_rbf: bool - if True use Gaussian RBF, else use distance
        
        Returns:
            final_value: (d_prime, 1) - RBF evaluation result
        """
        u = jnp.asarray(u).reshape(-1)  # Flatten to (d,) - handles both (d,) and (1,d)
        
        # U_select: (d, M) - each column is a center
        # u: (d,) - evaluation point
        # Compute distances from u to each center
        u_expanded = u[:, None]  # Shape: (d, 1)
        diff = self.U_select - u_expanded  # Broadcasting: (d, M) - (d, 1) = (d, M)

        # Define scaling factors as JAX array
        scaling = jnp.array([50.0, 100.0, 70.0, 100.0, 70.0, 50.0])[:, None]  # Shape: (d, 1)
    
        # Apply scaling (element-wise division)
        diff = diff / scaling  # Broadcasting: (d, M) / (d, 1) = (d, M)
    
        dist = jnp.sqrt(jnp.sum(diff**2, axis=0, keepdims=True)+1e-12)  # Shape: (1, M)
        
        # Compute RBF
        # if case_rbf:
        #     A = jnp.exp(-(epsilon * dist)**2)  # Shape: (1, M)
        # else:
        #     A = dist  # Shape: (1, M)
        A = dist
        
        # Compute interpolation
        # A: (1, M), W: (M, d_prime) -> result: (1, d_prime)
        result = A @ self.W_sm  # Shape: (1, d_prime)
        final_value = result.T  # Shape: (d_prime, 1)
        
        return final_value
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_aSSM_exps(self, xi):
        """
        Evaluate monomial terms using JAX arrays.
        """
        xi = jnp.asarray(xi).flatten()
        x = xi.reshape(1, -1, 1)
        exps_expanded = self.exps_M[:, :, None]  # Add the missing dimension
        powered = x ** exps_expanded      # Now broadcasting works
        u = jnp.prod(powered, axis=1, keepdims=True)
        # Reshape u to 2D before concatenating
        u = u.squeeze(axis=-1)  # Shape: (*, 1)
        return u
    
    def eval_monomials_batch_sm(self, xi, exps):
        """Evaluate monomials at multiple points with constant term added"""
        xi = jnp.asarray(xi)  # Shape: (N, 9) where N is number of points
        
        # Ensure xi is 2D
        if xi.ndim == 1:
            xi = xi.reshape(1, -1)
        
        N, n_dims = xi.shape
        x = xi.reshape(N, n_dims, 1)  # Shape: (N, 9, 1)
        
        # Add dimension to exps for broadcasting
        exps_expanded = exps[:, :, None]  # Shape: (219, 9, 1)
        
        # Broadcasting: (N, 9, 1) with (219, 9, 1) -> (N, 219, 9, 1)
        # We need to reshape for proper broadcasting
        x_expanded = x[:, None, :, :]  # Shape: (N, 1, 9, 1)
        exps_broadcast = exps_expanded[None, :, :, :]  # Shape: (1, 219, 9, 1)
        
        powered = x_expanded ** exps_broadcast  # Shape: (N, 219, 9, 1)
        u = jnp.prod(powered, axis=2)  # Product over dimension axis, Shape: (N, 219, 1)
        u = u.squeeze(axis=-1)  # Shape: (N, 219)
        
        # Add constant term (1) for each point
        ones_term = jnp.ones((N, 1))  # Shape: (N, 1)
        u = jnp.concatenate([u, ones_term], axis=1)  # Shape: (N, 220)
        
        return u

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

    
    """
    def performance_mapping(self, x):
        
        Default performance mapping is linear.
        Here need to change the function definition as input should be y and xi. 
       
        if self.nonlinear_perf_mapping:
            perf_var = jnp.matmul(self.H, self.M @ self.eval_aSSM_exps(jnp.concatenate([x,self.xs[:2]]), self.exps_M)).squeeze() + self.xs
            #print(f"perf_var shape: {perf_var.shape}, x shape: {x.shape}, xs shape: {self.xs.shape}")
        else:
            perf_var = jnp.matmul(self.H, x.T).T

        return perf_var
    """
    @partial(jax.jit, static_argnums=(0,))
    def single_perf_mapping(self, xi):
        return jnp.matmul(self.H, self.M @ self.eval_aSSM_exps(xi)).squeeze()

    @partial(jax.jit, static_argnums=(0,))
    def performance_mapping(self, x, u):
        # Ensure both x and u are batched (N, 6) and (N, ?)
        
        x = jnp.reshape(x,(5,-1)).T
        u = jnp.atleast_2d(u)
        
        # Evaluate nonlinear feature for each u
        # xs_batch_full = jax.vmap(
        #     lambda ui: self.rbf_eval_single_jax(ui, self.U_select, self.W_sm, self.epsilon_sm, self.case_rbf)
        # )(u)

        xs_batch_full = jax.vmap(
        lambda ui: jnp.ravel(
            self.rbf_eval_single_jax(ui)
        ))(u)
        #print(xs_batch_full.shape)
        xs_batch = xs_batch_full[:, :3]

        # Concatenate to form (N, 8)
        x_inputs = jnp.concatenate([x, xs_batch], axis=1)

        # Vectorized performance map core
        perf_vars = jax.vmap(self.single_perf_mapping)(x_inputs) + xs_batch_full[:, :3]

        # Return squeezed if single
        return jnp.squeeze(perf_vars)

    """"
    @partial(jax.jit, static_argnums=(0,))
    def performance_mapping(self, x, u):
        
        Performance mapping that handles both single samples and batches.
    
        Supports both:
        - Single sample: x.shape = (6,)
        - Batched: x.shape = (N, 6) or x.shape = (6, N) 
        
        # if not self.nonlinear_perf_mapping:
        #     # Linear case - simple matrix multiplication
        #     return jnp.matmul(self.H, x.T).T + self.xs
    
        # Nonlinear case - handle both single samples and batches
       
        # Handle single sample case
        if x.ndim == 1:
            # x shape: (6,)
            #print(u.shape)
            xs_partial_full = self.rbf_eval_single_jax(u, self.U_select, self.W_sm, self.epsilon_sm, self.case_rbf)
            xs_partial = xs_partial_full[:3].squeeze()  # shape (2,)
            x_input = jnp.concatenate([x, xs_partial])  # shape (8,)
            perf_var = jnp.matmul(self.H, self.M @ self.eval_aSSM_exps(x_input, self.exps_M)).squeeze() + xs_partial_full[:3].squeeze()
            return perf_var
    
        # Handle batch case - need to determine if it's (N, 6) or (6, N)
        if x.shape[0] == 2:
            # Assume it's (6, N) format - transpose to (N, 6)
            print(x.shape)
            x_batch = x.T  # shape (N, 6)
        else:
            # Already in (N, 6) format
            x_batch = x

        
        xs_batch_full = jax.vmap(lambda ui: self.rbf_eval_single_jax(ui, self.U_select, self.W_sm, self.epsilon_sm, self.case_rbf).squeeze())(u) 
        
        xs_batch = xs_batch_full[:, :3].squeeze()  # shape (batch_size, 2)
        
        # Broadcast xs_partial to match batch size
        #xs_batch = jnp.tile(xs_partial, (batch_size, 1))  # shape (batch_size, 2)
        #xs_batch_0 = jnp.tile(xs_partial_0, (batch_size, 1))  # shape (batch_size, 2)
        # Prepare batched inputs
        x_inputs = jnp.concatenate([x_batch, xs_batch], axis=1)  # shape (batch_size, 8)
    
        # Apply to all batch elements using vmap
        def single_perf_mapping(xi):
            return jnp.matmul(self.H, self.M @ self.eval_aSSM_exps(xi, self.exps_M)).squeeze() 
    
        # Vectorize over batch
        perf_vars = jax.vmap(single_perf_mapping)(x_inputs) + xs_batch_full[:, :3].squeeze()  # shape (batch_size, output_dim)
        #print(perf_vars.shape)
        # Return in same format as input
        if x.shape[0] == 2:
            return perf_vars.T  # Return as (output_dim, N) if input was (6, N)
        else:
            return perf_vars    # Return as (N, output_dim) if input was (N, 6)
    """

    def solve(self, x0, u_init, x_init, z=None, zf=None, u=None):
        """
        Solve with minimal overhead.
        """
        import time
        
        t_total_start = time.time()
        
        # Initialization
        itr = 0
        self.x_k = x_init
        self.u_k = u_init
        new_solution = True
        Jstar_prev = jnp.inf
        delta_prev = jnp.inf
        omega_prev = jnp.inf
        converged = False
        delta = self.delta0
        omega = self.omega0
        
        #x0_np = np.asarray(x0, dtype=np.float64)

        # First Jacobians
        t0 = time.time()
        A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k, self.u_k)
        t_first_dyn = time.time() - t0
        
        t0 = time.time()
        if self.nonlinear_perf_mapping:
            H_d, G_d, c_d = self._get_perf_mapping_linearizations(self.x_k, self.u_k)
        else:
            H_d, G_d, c_d = None, None, None
        t_first_perf = time.time() - t0
        
        # Timing accumulators
        t_locp_update = 0.0
        t_locp_solve = 0.0
        t_iter_jac = 0.0
        
        # Main optimization loop
        while self._is_valid_iteration(itr) and not converged and omega <= self.omega_max:
            rho_k = -1
            max_violation = -1
            dsol = -1
            delta_cur = delta
            omega_cur = omega
            
            # LOCP Update
            #x_k_np = np.asarray(self.x_k, dtype=np.float64)
            t0 = time.time()
            if new_solution:
                self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u, Hd=H_d, Gd=G_d, cd=c_d)
                new_solution = False
            else:
                self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u, Hd=H_d, Gd=G_d, cd=c_d, full=False)
            t_locp_update += time.time() - t0
            
            # LOCP Solve
            t0 = time.time()
            Jstar, success, stats = self.locp.solve()
            t_locp_solve += time.time() - t0
            
            if not success:
                self.xopt = jnp.copy(self.x_k)
                self.uopt = jnp.copy(self.u_k)
                if self.nonlinear_perf_mapping:
                    self.zopt = self.performance_mapping(self.xopt.T, self.uopt.T).T
        
                else:
                    self.zopt = jnp.transpose(self.H @ self.xopt.T)
                return
            
            x_next, u_next, _ = self.locp.get_solution()
            e_tr, tr_satisfied = self._is_in_trust_region(self.x_k, x_next, delta)
            
            if tr_satisfied:
                rho_k = self._compute_accuracy(self.x_k, self.u_k, x_next, u_next, Jstar)
                
                if rho_k > self.rho and itr != 1:
                    delta = self.beta_fail * delta
                else:
                    if delta_prev == delta and omega_prev == omega and Jstar_prev <= Jstar:
                        delta = self.beta_fail * delta
                    delta_prev = delta
                    Jstar_prev = Jstar
                    omega_prev = omega
                    
                    max_violation, X_satisfied = self._state_constraints_violated(x_next)
                    
                    if not X_satisfied:
                        omega = self.gamma_fail * omega
                    
                    dsol, converged = self._is_converged(self.x_k, x_next, u_next)
                    
                    if not X_satisfied:
                        converged = False
                    
                    new_solution = True
            else:
                omega = self.gamma_fail * omega
            
            itr += 1
            
            # Update for next iteration
            if new_solution:
                self.x_k = x_next.copy()
                self.u_k = u_next.copy()
                
                if self.max_gusto_iters >= 1:
                    t0 = time.time()
                    A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k, self.u_k)

                    if self.nonlinear_perf_mapping:
                        H_d, G_d, c_d = self._get_perf_mapping_linearizations(self.x_k, self.u_k)
                    else:
                        H_d, G_d, c_d = None, None, None
                    t_iter_jac += time.time() - t0
        
        t_total = time.time() - t_total_start
        
        # Simple summary (minimal printing)
        print(f'Solved in {itr} iterations/{t_total*1000:.2f}ms')
        print(f'  Jacobians: {(t_first_dyn + t_first_perf + t_iter_jac)*1000:.2f}ms')
        print(f'  LOCP update: {t_locp_update*1000:.2f}ms, solve: {t_locp_solve*1000:.2f}ms')
        
        # Save optimal solution
        self.xopt = jnp.copy(self.x_k)
        self.uopt = jnp.copy(self.u_k)
        if self.nonlinear_perf_mapping:
            self.zopt = self.performance_mapping(self.xopt.T, self.uopt).T
        else:
            self.zopt = jnp.transpose(self.H @ self.xopt.T)
        self.locp_solve_time = t_locp_solve

    def get_solution(self):
        return self.xopt, self.uopt, self.zopt, self.locp_solve_time

    def _extract_config(self, config):
        """
        Extract configuration parameters from GuSTOConfig object.
        """
        for key, value in asdict(config).items():
            setattr(self, key, value)

    @partial(jax.jit, static_argnums=(0,))
    def _is_converged(self, x_k, x, u):
        """
        Sequential problem has converged when current and previous state input pairs are close.
        """
        dx = (1. / self.n_x) * jnp.sum(jnp.linalg.norm(jnp.multiply(self.x_scale, x - x_k), axis=1))
        dsol = (1. / self.N) * dx
        converged = jnp.where(dsol <= self.convg_thresh, True, False)
        return dsol, converged

    def _is_valid_iteration(self, itr):
        """
        Is the current iteration within the limits.
        """
        return jnp.less_equal(itr, self.max_gusto_iters)

    @partial(jax.jit, static_argnums=(0,))
    def _is_in_trust_region(self, x_k, x, delta):
        """
        Check if the new state is within the trust region of the previous state.
        """
        max_diff = jnp.max(jnp.linalg.norm(jnp.multiply(self.x_scale, x - x_k), ord=jnp.inf, axis=1))
        def outside_region(_):
            return max_diff, False
        def inside_region(_):
            return 0.0, True
        return jax.lax.cond(max_diff - delta > self.epsilon, outside_region, inside_region, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _state_constraints_violated(self, x):
        """
        For GuSTO, state constraints get enforced as penalties, not as strict constraints. Computes whether the state
        constraints are within a user-chosen tolerance epsilon.
        """
        def compute_violation(x_row):
            return self.X.get_constraint_violation(x_row)

        if self.X is not None:
            # Vectorize the constraint violation computation
            violations = jax.vmap(compute_violation)(x)
            max_violation = jnp.max(violations)
        else:
            max_violation = 0.0

        def outside_threshold(_):
            return max_violation, False
        def inside_threshold(_):
            return max_violation, True

        return jax.lax.cond(max_violation > self.epsilon, outside_threshold, inside_threshold, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_accuracy(self, x_k, u_k, x, u, J):
        Ak_list, Bk_list, _ = self.model.dynamics_jac(x_k, u_k, continuous=True)
        # Compute the accuracy of the model
        fk_list = self.model.continuous_dynamics(x_k, u_k)
        #print(fk_list.shape, Ak_list.shape, Bk_list.shape, x.shape, u.shape)
        f_list = self.model.continuous_dynamics(x, u)
        #print(f_list.shape)
        f_approx = fk_list + jnp.einsum("nij,nj->ni", Ak_list, (x - x_k)) + jnp.einsum("nij,nj->ni", Bk_list, (u - u_k))
        error = self.dt * jnp.linalg.norm(self.f_scale * (f_list - f_approx), ord=2, axis=1).sum()
        approx = self.dt * jnp.linalg.norm(self.f_scale * f_approx, ord=2, axis=1).sum()
        
        rho_k = error / (J + approx)
        return rho_k
    
    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def _perform_dynamics_linearization(self, x, u):
        """
        Obtain the affine dynamics of each point along trajectory in a list.
        """
        f = partial(self.model.discrete_dynamics)
        A, B = jax.jacfwd(f, argnums=(0, 1))(x, u)
        d = f(x, u) - A @ x - B @ u
        return A, B, d

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def _perform_perf_mapping_linearization(self, x, u):
        """
        Obtain the affine performance mappings at each point along trajectory in a list.
        """
        g = self.performance_mapping
        H, G = jax.jacfwd(g, argnums=(0, 1))(x, u)
        c = g(x, u) - H @ x - G @ u
        return H, G, c

    def _get_dynamics_linearizations(self, x, u):
        """
        Wrapper method that calls self.model.get_dynamics_linearizations if it exists,
        otherwise it calls the local method _perform_dynamics_linearization.
        """
        if hasattr(self.model, 'dynamics_jac') and callable(getattr(self.model, 'dynamics_jac')):
            return self.model.dynamics_jac(x, u)
        else:
            return self._perform_dynamics_linearization(x, u)
       
    def _get_perf_mapping_linearizations(self, x, u):
        """
        Wrapper method that calls self.model.get_perf_mapping_linearizations if it exists,
        otherwise it calls the local method _perform_perf_mapping_linearization.
        """
        return self._perform_perf_mapping_linearization(x, u)