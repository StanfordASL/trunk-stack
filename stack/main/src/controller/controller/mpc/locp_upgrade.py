"""
LOCP (Linear Optimal Control Problem) implementation, adopted from original GuSTO code.
"""

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from cvxpy.atoms.affine.reshape import reshape
import time
import scipy.sparse as sp
from scipy.sparse import csc_matrix
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

    def update(self, Ad, Bd, dd, x0, xk, delta, omega, z=None, zf=None, u=None, full=True, **kwargs):
        """
        Update the potentially changing LOCP data. xk is updated solution trajectory.
        """

        # If using warm start, set the parameters to their current values
        if self.warm_start:
            # Set parameters
            if full:
                if z is not None:
                    self.z.value = np.ravel(z[:self.N], order = 'F')
                else:
                    self.z.value = np.zeros((self.N) * self.n_z)  # default set to 0

                if u is not None:
                    self.u_des.value = np.ravel(u, order = 'F')
                else:
                    self.u_des.value = np.zeros(self.N * self.n_u)  # default set to 0

                if self.Qzf is not None and zf is not None:
                    self.zf.value = np.asarray(zf)
                elif self.Qzf is not None and zf is None:
                    self.zf.value = np.zeros(self.n_z)  # default set to 0

                

                # Added observer linearizations here. Make sure to propogate Hd, Gd and cd as parameters in kwargs
                for j in range(self.N):
                    self.Ad[j].value = np.asarray(Ad[j])
                    self.Bd[j].value = np.asarray(Bd[j])

                if self.nonlinear_perf_mapping:
                    for j in range(self.N):
                        self.Hd[j].value = np.asarray(kwargs.get('Hd')[j])
                        self.Gd[j].value = np.asarray(kwargs.get('Gd')[j])

                self.dd.value = np.ravel(np.asarray(dd), order='F')
                if self.nonlinear_perf_mapping:
                    cd = kwargs.get('cd')
                    self.cd.value = np.ravel(np.asarray(cd), order='F')

                self.xk.value = np.asarray(xk)
                self.x0.value = np.asarray(x0)

            # Always update delta and omega
            self.omega.value = omega
            self.delta.value = delta

        # Otherwise just build a new problem from scratch each time
        else:
            self.delta = delta
            self.omega = omega
            if z is not None:
                self.z = np.ravel(z[:self.N], order='F')
            else:
                self.z = np.zeros((self.N) * self.n_z)

            if u is not None:
                self.u_des = np.ravel(u, order='F')
            else:
                self.u_des = np.zeros(self.N * self.n_u)

            if self.Qzf is not None and zf is not None:
                self.zf = zf
            elif self.Qzf is not None and zf is None:
                self.zf = np.zeros(self.n_z)

            self.Ad = np.asarray(Ad)
            self.Bd = np.asarray(Bd)
            self.dd = np.ravel(np.asarray(dd), order='F')
            self.x0 = np.asarray(x0)
            self.xk = np.asarray(xk)

            # Observer params here
            if self.nonlinear_perf_mapping:
                self.Hd = np.asarray(kwargs.get('Hd'))
                self.Gd = np.asarray(kwargs.get('Gd'))
                self.cd = np.asarray(kwargs.get('cd'))

            self._problem_setup()

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

    def get_solution(self):
        """
        Extract the most recent solution from calling solve().
        """
        # NOTE: Dan mentioned that this reshape is inefficient, keep in numpy or something (can ask Dan)
        x = jnp.reshape(self.x.value, (self.N, self.n_x))
        #x = x[:self.N]  # Get first N entries, excluding the last one
        u = jnp.reshape(self.u.value, (self.N, self.n_u))
        if self.tr_active:
            s = jnp.asarray(self.st.value)
        else:
            s = None
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
        Rfull = csc_matrix(block_diag(*[self.R for j in range(self.N)]), dtype =np.float64)
        J += cp.quad_form(self.u - self.u_des, Rfull)

        # Performance cost (we expect all trajectories to be non-shifted i.e., about origin)
        # Assuming a map from reduced-ordered state to performance variable (which we linearize)
        #if self.Qzf is None:
        #    Qzfull = sp.csc_matrix(block_diag(*[self.Qz for _ in range(self.N + 1)]))
        #else:
        #    Qz_list = [self.Qz for _ in range(self.N)] + [self.Qzf]
        #    Qzfull = sp.csc_matrix(block_diag(*Qz_list))

        Qz_list = [self.Qz for _ in range(self.N)]
        Qzfull = csc_matrix(block_diag(*Qz_list), dtype=np.float64)

        if self.nonlinear_perf_mapping:
            #cdfull = np.reshape(self.cd, ((self.N)*self.n_z,)) if isinstance(self.cd, list) else \
            #    reshape(self.cd, ((self.N)*self.n_z,))
            #print(self.cd.shape)

            cdfull = np.reshape(self.cd, ((self.N)*self.n_z,), order = 'F') if isinstance(self.cd, list) else \
                reshape(self.cd, ((self.N)*self.n_z,), order = 'F')
            
            
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
            dx = cp.reshape(self.x, (self.n_x, self.N)) - self.xk.T
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
                cdfull = np.reshape(self.cd, ((self.N + 1) * self.n_z,), order = 'F') if isinstance(self.cd, list) else \
                    reshape(self.cd, ((self.N + 1) * self.n_z,), order = 'F')
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
