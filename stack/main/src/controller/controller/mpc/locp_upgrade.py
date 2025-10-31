"""
LOCP (Linear Optimal Control Problem) implementation, adopted from original GuSTO code.
"""

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from functools import partial
from cvxpy.atoms.affine.reshape import reshape as cvxpy_reshape
reshape = partial(cvxpy_reshape, order= 'F')
import time
import scipy.sparse as sp
import jax
import jax.numpy as jnp


class LOCP:
    """
    Linear Optimal Control Problem (LOCP) class for MPC.

    :N: number of steps in OCP horizon
    :H: performance variable matrix (n_z, n_x)
    :R: control cost matrix np.array (n_u, n_u)
    :Qz: performance cost matrix (n_z, n_z)
    :Qzf: (optional) terminal performance cost matrix (n_z, n_z)
    :U: (optional) control constraints, Polyhedron object
    :X: (optional) state constraints, Polyhedron object
    :Xf: (optional) terminal set, Polyhedron object
    :dU: (optional) u_k - u_{k-1} / slew rate constraint, Polyhedron object
    :verbose: (optional) boolean
    :warm_start: (optional) boolean
    :nonlinear_perf_mapping: (optional) boolean
    :x_char: (optional) characteristic quantities for state (for scaling)
    :kwargs: (optional) additional arguments for the solver
    """
    def __init__(self, N, H, Qz, R, Qzf=None, U=None, X=None, Xf=None, dU=None, verbose=False, warm_start=True,
                 nonlinear_perf_mapping=False, x_char=None, R_du=None, **kwargs):
        self.N = N
        self.H = H
        self.Qz = Qz
        self.R = R
        self.R_du = R_du
        self.Qzf = Qzf
        self.U = U
        self.X = X
        self.Xf = Xf
        self.dU = dU
        self.verbose = verbose
        self.warm_start = warm_start
        self.nonlinear_perf_mapping = nonlinear_perf_mapping

        # Ensure we have a self.H in SSM class such that 2nd dim is dim of RO state
        self.n_x = 2
        self.n_z = Qz.shape[0]
        self.n_u = R.shape[0]

        # Characteristic values for scaling
        if x_char is None:
            self.x_scale = np.ones(self.n_x)  # default to no scaling
        else:
            self.x_scale = 1. / np.abs(x_char)

        # Build CVX problem
        self.x = cp.Variable((self.N) * self.n_x)
        self.u = cp.Variable(self.N * self.n_u)
        
        # Trust region slack variable
        self.tr_active = kwargs.pop('is_tr_active', True) 
        if self.tr_active:
            self.st = cp.Variable(self.N)
        else:
            self.st = None
        
        # Solver arguments
        self.solver_args = kwargs
        if not 'solver' in self.solver_args:
            self.solver_args['solver'] = 'OSQP'
        else:
            self.solver_args = {'solver': self.solver_args['solver']}

        # Input nullspace
        self.input_nullspace = kwargs.pop('input_nullspace', None)

        # Parameters
        if self.warm_start:
            self.delta = cp.Parameter(nonneg=True)
            self.omega = cp.Parameter(nonneg=True)
            self.z = cp.Parameter((self.N) * self.n_z)
            self.u_des = cp.Parameter(self.N * self.n_u)
            self.Ad = [cp.Parameter((self.n_x, self.n_x)) for i in range(self.N)]
            self.Bd = [cp.Parameter((self.n_x, self.n_u)) for i in range(self.N)]
            self.dd = cp.Parameter(self.N * self.n_x)

            # Adding observer linearization parameters here. Expect parameters to be None
            # If dynamics class has nonlinear_perf_mapping = False. In this case this class should
            # use self.H as shown above. This seems to be different from when not warm_starting
            if self.nonlinear_perf_mapping:
                self.Hd = [cp.Parameter((self.n_z, self.n_x)) for i in range(self.N)]
                self.Gd = [cp.Parameter((self.n_z, self.n_u)) for i in range(self.N)]
                self.cd = cp.Parameter((self.N) * self.n_z)

            self.x0 = cp.Parameter(self.n_x)
            self.xk = cp.Parameter((self.N, self.n_x)) # Linearization points for trust region
            if self.Qzf is not None:
                self.zf = cp.Parameter(self.n_z)

            # Extra parameter for the previously applied input, will be updated each time
            self.u0_prev = cp.Parameter(self.n_u, value=np.zeros(self.n_u))  

            self._problem_setup()

    def _convert_to_numpy(self, arr):
        """
        Convert JAX array to NumPy array properly.
        CVXPY requires actual NumPy arrays, not JAX DeviceArrays.
        """
        import jax.numpy as jnp
        
        # If it's a JAX array, convert using __array__() interface
        if isinstance(arr, jnp.ndarray):
            return np.asarray(arr.__array__(), dtype=np.float64)
        # If it's already numpy, ensure float64
        elif isinstance(arr, np.ndarray):
            return arr.astype(np.float64, copy=False)
        # Otherwise try standard conversion
        else:
            return np.asarray(arr, dtype=np.float64)


    def update(self, Ad, Bd, dd, x0, xk, delta, omega, z=None, zf=None, u=None, full=True, **kwargs):
        """
        Update the potentially changing LOCP data. xk is updated solution trajectory.
        """
        import jax.numpy as jnp
        
        # Debug: Check if we're receiving JAX arrays
        if self.verbose >= 2:
            if isinstance(Ad[0], jnp.ndarray):
                print("WARNING: Ad contains JAX arrays - converting to NumPy")
            if isinstance(xk, jnp.ndarray):
                print("WARNING: xk is JAX array - converting to NumPy")

        # If using warm start, set the parameters to their current values
        if self.warm_start:
            # Set parameters
            if full:
                if z is not None:
                    z_flat = np.ravel(z[:self.N])
                    if z_flat.shape[0] != self.N * self.n_z:
                        raise ValueError(f"z shape mismatch: expected {self.N * self.n_z}, got {z_flat.shape[0]}")
                    self.z.value = self._convert_to_numpy(z_flat)
                else:
                    self.z.value = np.zeros((self.N) * self.n_z, dtype=np.float64)

                if u is not None:
                    u_flat = np.ravel(u)
                    if u_flat.shape[0] != self.N * self.n_u:
                        raise ValueError(f"u_des shape mismatch: expected {self.N * self.n_u}, got {u_flat.shape[0]}")
                    self.u_des.value = self._convert_to_numpy(u_flat)
                else:
                    self.u_des.value = np.zeros(self.N * self.n_u, dtype=np.float64)

                if self.Qzf is not None and zf is not None:
                    self.zf.value = self._convert_to_numpy(zf)
                elif self.Qzf is not None and zf is None:
                    self.zf.value = np.zeros(self.n_z, dtype=np.float64)

                # Update linearization matrices - Convert JAX to NumPy properly
                for j in range(self.N):
                    Ad_j = self._convert_to_numpy(Ad[j])
                    Bd_j = self._convert_to_numpy(Bd[j])
                    
                    if Ad_j.shape != (self.n_x, self.n_x):
                        raise ValueError(f"Ad[{j}] shape mismatch: expected {(self.n_x, self.n_x)}, got {Ad_j.shape}")
                    if Bd_j.shape != (self.n_x, self.n_u):
                        raise ValueError(f"Bd[{j}] shape mismatch: expected {(self.n_x, self.n_u)}, got {Bd_j.shape}")
                    
                    # Ensure contiguous C-order arrays
                    self.Ad[j].value = np.ascontiguousarray(Ad_j)
                    self.Bd[j].value = np.ascontiguousarray(Bd_j)

                if self.nonlinear_perf_mapping:
                    Hd = kwargs.get('Hd')
                    Gd = kwargs.get('Gd')
                    cd = kwargs.get('cd')
                    
                    if Hd is None or Gd is None or cd is None:
                        raise ValueError("nonlinear_perf_mapping=True but Hd, Gd, or cd not provided")
                    
                    for j in range(self.N):
                        Hd_j = self._convert_to_numpy(Hd[j])
                        Gd_j = self._convert_to_numpy(Gd[j])
                        
                        if Hd_j.shape != (self.n_z, self.n_x):
                            raise ValueError(f"Hd[{j}] shape mismatch: expected {(self.n_z, self.n_x)}, got {Hd_j.shape}")
                        if Gd_j.shape != (self.n_z, self.n_u):
                            raise ValueError(f"Gd[{j}] shape mismatch: expected {(self.n_z, self.n_u)}, got {Gd_j.shape}")
                        
                        self.Hd[j].value = np.ascontiguousarray(Hd_j)
                        self.Gd[j].value = np.ascontiguousarray(Gd_j)

                    cd_flat = np.ravel(self._convert_to_numpy(cd))
                    if cd_flat.shape[0] != self.N * self.n_z:
                        raise ValueError(f"cd shape mismatch: expected {self.N * self.n_z}, got {cd_flat.shape[0]}")
                    self.cd.value = np.ascontiguousarray(cd_flat)

                dd_flat = np.ravel(self._convert_to_numpy(dd))
                if dd_flat.shape[0] != self.N * self.n_x:
                    raise ValueError(f"dd shape mismatch: expected {self.N * self.n_x}, got {dd_flat.shape[0]}")
                self.dd.value = np.ascontiguousarray(dd_flat)
                
                xk_array = self._convert_to_numpy(xk)
                if xk_array.shape != (self.N, self.n_x):
                    raise ValueError(f"xk shape mismatch: expected {(self.N, self.n_x)}, got {xk_array.shape}")
                self.xk.value = np.ascontiguousarray(xk_array)
                
                x0_array = self._convert_to_numpy(x0)
                if x0_array.shape[0] != self.n_x:
                    raise ValueError(f"x0 shape mismatch: expected {self.n_x}, got {x0_array.shape[0]}")
                self.x0.value = np.ascontiguousarray(x0_array)

            # Always update delta and omega
            self.omega.value = float(omega)
            self.delta.value = float(delta)

        else:
            # Non-warm-start path
            self.delta = delta
            self.omega = omega
            if z is not None:
                self.z = self._convert_to_numpy(np.ravel(z[:self.N]))
            else:
                self.z = np.zeros((self.N) * self.n_z, dtype=np.float64)

            if u is not None:
                self.u_des = self._convert_to_numpy(np.ravel(u))
            else:
                self.u_des = np.zeros(self.N * self.n_u, dtype=np.float64)

            if self.Qzf is not None and zf is not None:
                self.zf = self._convert_to_numpy(zf)
            elif self.Qzf is not None and zf is None:
                self.zf = np.zeros(self.n_z, dtype=np.float64)

            self.Ad = [self._convert_to_numpy(Ad[j]) for j in range(self.N)]
            self.Bd = [self._convert_to_numpy(Bd[j]) for j in range(self.N)]
            self.dd = self._convert_to_numpy(np.ravel(dd))
            self.x0 = self._convert_to_numpy(x0)
            self.xk = self._convert_to_numpy(xk)

            if self.nonlinear_perf_mapping:
                self.Hd = [self._convert_to_numpy(kwargs.get('Hd')[j]) for j in range(self.N)]
                self.Gd = [self._convert_to_numpy(kwargs.get('Gd')[j]) for j in range(self.N)]
                self.cd = self._convert_to_numpy(kwargs.get('cd'))

            self._problem_setup()

    # def update(self, Ad, Bd, dd, x0, xk, delta, omega, z=None, zf=None, u=None, full=True, **kwargs):
    #     """
    #     Update the potentially changing LOCP data. xk is updated solution trajectory.
    #     """

    #     # If using warm start, set the parameters to their current values
    #     if self.warm_start:
    #         # Set parameters
    #         if full:
    #             if z is not None:
    #                 self.z.value = np.ravel(z[:self.N])
    #             else:
    #                 self.z.value = np.zeros((self.N) * self.n_z)  # default set to 0

    #             if u is not None:
    #                 self.u_des.value = np.ravel(u)
    #             else:
    #                 self.u_des.value = np.zeros(self.N * self.n_u)  # default set to 0

    #             if self.Qzf is not None and zf is not None:
    #                 self.zf.value = np.asarray(zf)
    #             elif self.Qzf is not None and zf is None:
    #                 self.zf.value = np.zeros(self.n_z)  # default set to 0

                

    #             # Added observer linearizations here. Make sure to propogate Hd, Gd and cd as parameters in kwargs
    #             for j in range(self.N):
    #                 self.Ad[j].value = np.asarray(Ad[j])
    #                 self.Bd[j].value = np.asarray(Bd[j])

    #             if self.nonlinear_perf_mapping:
    #                 for j in range(self.N):
    #                     self.Hd[j].value = np.asarray(kwargs.get('Hd')[j])
    #                     self.Gd[j].value = np.asarray(kwargs.get('Gd')[j])

    #             self.dd.value = np.ravel(np.asarray(dd))
    #             if self.nonlinear_perf_mapping:
    #                 cd = kwargs.get('cd')
    #                 self.cd.value = np.ravel(np.asarray(cd))

    #             self.xk.value = np.asarray(xk)
    #             self.x0.value = np.asarray(x0)

    #         # Always update delta and omega
    #         self.omega.value = omega
    #         self.delta.value = delta

    #     # Otherwise just build a new problem from scratch each time
    #     else:
    #         self.delta = delta
    #         self.omega = omega
    #         if z is not None:
    #             self.z = np.ravel(z[:self.N])
    #         else:
    #             self.z = np.zeros((self.N) * self.n_z)

    #         if u is not None:
    #             self.u_des = np.ravel(u)
    #         else:
    #             self.u_des = np.zeros(self.N * self.n_u)

    #         if self.Qzf is not None and zf is not None:
    #             self.zf = zf
    #         elif self.Qzf is not None and zf is None:
    #             self.zf = np.zeros(self.n_z)

    #         self.Ad = np.asarray(Ad)
    #         self.Bd = np.asarray(Bd)
    #         self.dd = np.ravel(np.asarray(dd))
    #         self.x0 = np.asarray(x0)
    #         self.xk = np.asarray(xk)

    #         # Observer params here
    #         if self.nonlinear_perf_mapping:
    #             self.Hd = np.asarray(kwargs.get('Hd'))
    #             self.Gd = np.asarray(kwargs.get('Gd'))
    #             self.cd = np.asarray(kwargs.get('cd'))

    #         self._problem_setup()

    def solve(self):
        """
        Solve the LOCP quadratic program.
        """
        t0 = time.time()
        try:
            # A,P,q,l,u = self.prob.get_problem_data('OSQP')
            # print(type(P),P.dtype,P.format)
            Jstar = self.prob.solve(warm_start=self.warm_start, verbose=self.verbose,ignore_dpp=True, **self.solver_args)
        except Exception as e:
            print('Solving with warm-start failed, so turning off')
            print(f'Solving with warm-start failed due to: {e}')
            try:
                Jstar = self.prob.solve(warm_start=False, verbose=self.verbose,ignore_dpp=True, **self.solver_args)
            except cp.SolverError:
                print('Solver still failed, returning inf')
                return np.inf, False, None
        t1 = time.time()
        if self.verbose >= 2:
            print('DEBUG: Solve routing in LOCP computed in {:.3f} seconds'.format(t1 - t0))
        if self.prob.status == 'optimal':
            return Jstar, True, self.prob.solver_stats
        else:
            return np.inf, False, None

    # def get_solution(self):
    #     """
    #     Extract the most recent solution from calling solve().
    #     """
    #     # NOTE: Dan mentioned that this reshape is inefficient, keep in numpy or something (can ask Dan)
    #     x = jnp.reshape(self.x.value, (self.N, self.n_x))
    #     #x = x[:self.N]  # Get first N entries, excluding the last one
    #     u = jnp.reshape(self.u.value, (self.N, self.n_u))
    #     if self.tr_active:
    #         s = jnp.asarray(self.st.value)
    #     else:
    #         s = None
    #     return x, u, s
    
    def get_solution(self):
        """
        Extract the most recent solution from calling solve().
        Returns arrays in shape (N, n_x) and (N, n_u) for compatibility.
        """
        # Get raw solution values
        x_val = self.x.value
        u_val = self.u.value
        
        # Convert to JAX arrays and reshape to 2D
        x = jnp.asarray(x_val).reshape(self.N, self.n_x)
        u = jnp.asarray(u_val).reshape(self.N, self.n_u)
        
        # Get slack variables if active
        if self.tr_active:
            s = jnp.asarray(self.st.value)
        else:
            s = None
        
        # Validation
        if x.shape != (self.N, self.n_x):
            raise ValueError(f"x solution has wrong shape: {x.shape}, expected {(self.N, self.n_x)}")
        if u.shape != (self.N, self.n_u):
            raise ValueError(f"u solution has wrong shape: {u.shape}, expected {(self.N, self.n_u)}")
        
        return x, u, s


    def _problem_setup(self):
        """
        Define the CVX problem.
        """
        J = self._set_objective()
        constraints = self._set_constraints()
        self.prob = cp.Problem(cp.Minimize(J), constraints)

    def _set_objective(self):
        """
        Compute the quadratic part of the objective.
        """
        J = 0

        # Control cost
        Rfull = sp.csc_matrix(block_diag(*[self.R for j in range(self.N)]))
        J += cp.quad_form(self.u - self.u_des, Rfull)

        # Performance cost (we expect all trajectories to be non-shifted i.e., about origin)
        # Assuming a map from reduced-ordered state to performance variable (which we linearize)
        #if self.Qzf is None:
        #    Qzfull = sp.csc_matrix(block_diag(*[self.Qz for _ in range(self.N + 1)]))
        #else:
        #    Qz_list = [self.Qz for _ in range(self.N)] + [self.Qzf]
        #    Qzfull = sp.csc_matrix(block_diag(*Qz_list))

        Qz_list = [self.Qz for _ in range(self.N)]
        Qzfull = sp.csc_matrix(block_diag(*Qz_list))

        if self.nonlinear_perf_mapping:
            #cdfull = np.reshape(self.cd, ((self.N)*self.n_z,)) if isinstance(self.cd, list) else \
            #    reshape(self.cd, ((self.N)*self.n_z,))
            #print(self.cd.shape)

            cdfull = np.reshape(self.cd, ((self.N)*self.n_z,)) if isinstance(self.cd, list) else \
                reshape(self.cd, ((self.N)*self.n_z,))
            
            
            if self.warm_start:
                Hfull = []
                for j in range(self.N):
                    cur = [np.zeros((self.n_z, self.n_x))] * (self.N)
                    cur[j] = self.Hd[j]
                    Hfull.append(cur)
                Hfull = cp.bmat(Hfull)
                Gfull = []
                for j in range(self.N):
                    cur = [np.zeros((self.n_z, self.n_u))] * (self.N)
                    cur[j] = self.Gd[j]
                    Gfull.append(cur)
                #Gfull.append([np.zeros((self.n_z, self.n_u))] * (self.N))    
                Gfull = cp.bmat(Gfull)
            else:
                Hfull = block_diag(*[self.Hd[j] for j in range(self.N)])
                Gfull = block_diag(*[self.Gd[j] for j in range(self.N)])
                #Gd_list = [self.Gd[j] for j in range(self.N)] + [np.zeros((self.n_z, self.n_u))]
                #Gfull = block_diag(*Gd_list)
                #Gfull = block_diag(*[self.Gd[j] for j in range(self.N)])
            #print( Hfull.shape, Gfull.shape, self.x.shape, self.u.shape, cdfull.shape, self.z.shape)
            #u_padded = cp.hstack([self.u, np.zeros(self.n_u)])
            J += cp.quad_form(Hfull @ self.x + Gfull @ self.u + cdfull - self.z, Qzfull)
        else:
            Hfull = block_diag(*[self.H for j in range(self.N + 1)])
            J += cp.quad_form(Hfull @ self.x - self.z, Qzfull)

        # Slack variables
        if self.tr_active:
            J += self.omega * cp.sum(self.st)

        # Nullspace contribution
        if self.input_nullspace is not None:
            nullSpace = np.tile(self.input_nullspace, self.N)
            J += cp.norm2(nullSpace @ self.u)

        # Control rate cost
        if self.R_du is not None:
            # First difference, u[0] - u0_prev
            J += cp.quad_form(self.u[:self.n_u] - self.u0_prev, self.R_du)
            if self.N > 1:
                # Differences within the horizon
                R_du_full = sp.block_diag([self.R_du]*(self.N-1), format='csc')
                u_diff = self.u[self.n_u:] - self.u[:-self.n_u]
                J += cp.quad_form(u_diff, R_du_full)

        return J

    def _set_constraints(self):
        constr = []

        # Dynamics constraints
        if self.warm_start:
            Adfull = []
            for j in range(self.N - 1):
                cur = [np.zeros((self.n_x, self.n_x))] * (self.N - 1)
                cur[j] = self.Ad[j]
                Adfull.append(cur)
            Adfull = cp.bmat(Adfull)

            Bdfull = []
            for j in range(self.N - 1):
                cur = [np.zeros((self.n_x, self.n_u))] * (self.N - 1)
                cur[j] = self.Bd[j]
                Bdfull.append(cur)
            Bdfull = cp.bmat(Bdfull)
        else:
            Adfull = block_diag(*self.Ad)
            Bdfull = block_diag(*self.Bd)
        #print(Adfull.shape, Bdfull.shape, self.x.shape, self.u.shape, self.dd.shape)
        #print(self.x[self.n_x:].shape,self.x[:-self.n_x].shape,self.u[:-self.n_x].shape)
        constr += [self.x[self.n_x:] == Adfull @ self.x[:-self.n_x] + Bdfull @ self.u[:-self.n_u] + self.dd[:-self.n_x]]
        

        # Trust region constraints
        if self.tr_active:
            X_scale = self.x_scale.reshape(-1, 1).repeat(self.N, axis=1)
            dx = reshape(self.x, (self.n_x, self.N)) - self.xk.T
            dx_scaled = cp.multiply(X_scale, dx)
            constr += [cp.norm(dx_scaled, 'inf', axis=0) <= self.delta + self.st]

            # Slack variable positivity
            constr += [self.st >= 0]

        # Control constraints
        if self.U is not None:
            UAfull = block_diag(*[self.U.A for j in range(self.N)])
            Ubfull = np.tile(self.U.b, self.N)
            constr += [UAfull @ self.u <= Ubfull]

        if self.dU is not None:
            dUAfull = block_diag(*[self.dU.A for j in range(self.N - 1)])
            dUbfull = np.tile(self.dU.b, self.N - 1)
            constr += [dUAfull @ (self.u[self.n_u:] - self.u[:-self.n_u]) <= dUbfull]

        # State constraints
        if self.X is not None:
            if self.nonlinear_perf_mapping:
                cdfull = np.reshape(self.cd, ((self.N + 1) * self.n_z,)) if isinstance(self.cd, list) else \
                    reshape(self.cd, ((self.N + 1) * self.n_z,))
                if self.warm_start:
                    Hfull = []
                    for j in range(self.N):
                        cur = [np.zeros((self.n_z, self.n_x))] * self.N
                        cur[j] = self.Hd[j + 1]
                        Hfull.append(cur)
                    Hfull = cp.bmat(Hfull)
                    Gfull = []
                    for j in range(self.N):
                        cur = [np.zeros((self.n_z, self.n_u))] * self.N
                        cur[j] = self.Gd[j + 1]
                        Gfull.append(cur)
                    Gfull = cp.bmat(Gfull)
                else:
                    Hfull = block_diag(*[self.Hd[j + 1] for j in range(self.N)])
                    Gfull = block_diag(*[self.Gd[j + 1] for j in range(self.N)])

                # Take only last N of cdfull
                cdfull = cdfull[self.n_z:]
                XAfull = block_diag(*[self.X.A for j in range(self.N)]) @ Hfull
                Xbfull = np.tile(self.X.b, self.N) - block_diag(*[self.X.A for j in range(self.N)]) @ cdfull
                constr += [XAfull @ self.x[self.n_z:] <= Xbfull]
            else:
                XAfull = block_diag(*[self.X.A for j in range(self.N)])
                Xbfull = np.tile(self.X.b, self.N)
                constr += [XAfull @ self.x[self.n_x:] <= Xbfull]

        # Terminal constraints
        if self.Xf is not None:
            constr += [self.Xf.A @ self.x[-self.n_x:] <= self.Xf.b]

        # Initial condition
        constr += [self.x[:self.n_x] == self.x0]

        return constr
    
    def validate_problem_data(self):
        """
        Validate all parameter shapes and values before solving.
        Call this method before solve() to diagnose issues.
        """
        print("\n=== LOCP Problem Validation ===")
        print(f"Horizon N: {self.N}")
        print(f"State dim n_x: {self.n_x}")
        print(f"Control dim n_u: {self.n_u}")
        print(f"Perf var dim n_z: {self.n_z}")
        
        if self.warm_start:
            print("\n--- Parameter Values ---")
            print(f"x0: {self.x0.value.shape if self.x0.value is not None else 'None'}")
            print(f"xk: {self.xk.value.shape if self.xk.value is not None else 'None'}")
            print(f"delta: {self.delta.value if self.delta.value is not None else 'None'}")
            print(f"omega: {self.omega.value if self.omega.value is not None else 'None'}")
            print(f"z: {self.z.value.shape if self.z.value is not None else 'None'}")
            print(f"u_des: {self.u_des.value.shape if self.u_des.value is not None else 'None'}")
            print(f"dd: {self.dd.value.shape if self.dd.value is not None else 'None'}")
            
            print("\n--- Dynamics Linearization ---")
            for j in range(min(3, self.N)):  # Show first 3
                Ad_val = self.Ad[j].value
                Bd_val = self.Bd[j].value
                print(f"Ad[{j}]: {Ad_val.shape if Ad_val is not None else 'None'}")
                print(f"Bd[{j}]: {Bd_val.shape if Bd_val is not None else 'None'}")
                
                if Ad_val is not None:
                    if np.any(np.isnan(Ad_val)) or np.any(np.isinf(Ad_val)):
                        print(f"  WARNING: Ad[{j}] contains NaN or Inf!")
                if Bd_val is not None:
                    if np.any(np.isnan(Bd_val)) or np.any(np.isinf(Bd_val)):
                        print(f"  WARNING: Bd[{j}] contains NaN or Inf!")
            
            if self.nonlinear_perf_mapping:
                print("\n--- Performance Mapping Linearization ---")
                print(f"cd: {self.cd.value.shape if self.cd.value is not None else 'None'}")
                for j in range(min(3, self.N)):
                    Hd_val = self.Hd[j].value
                    Gd_val = self.Gd[j].value
                    print(f"Hd[{j}]: {Hd_val.shape if Hd_val is not None else 'None'}")
                    print(f"Gd[{j}]: {Gd_val.shape if Gd_val is not None else 'None'}")
                    
                    if Hd_val is not None:
                        if np.any(np.isnan(Hd_val)) or np.any(np.isinf(Hd_val)):
                            print(f"  WARNING: Hd[{j}] contains NaN or Inf!")
                    if Gd_val is not None:
                        if np.any(np.isnan(Gd_val)) or np.any(np.isinf(Gd_val)):
                            print(f"  WARNING: Gd[{j}] contains NaN or Inf!")
        
        print("\n--- Variable Shapes ---")
        print(f"x: {self.x.shape}")
        print(f"u: {self.u.shape}")
        if self.tr_active:
            print(f"st (slack): {self.st.shape}")
        
        print("\n--- Expected Shapes ---")
        print(f"x should be: ({self.N * self.n_x},)")
        print(f"u should be: ({self.N * self.n_u},)")
        print(f"z should be: ({self.N * self.n_z},)")
        print(f"dd should be: ({self.N * self.n_x},)")
        if self.nonlinear_perf_mapping:
            print(f"cd should be: ({self.N * self.n_z},)")
        
        print("=" * 40 + "\n")

