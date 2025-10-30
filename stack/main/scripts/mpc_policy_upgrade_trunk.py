import jax.numpy as jnp
from typing import Optional, Tuple, Any
import time
from gusto_upgrade_trunk import GuSTO, GuSTOConfig
import numpy as np


class MPCPolicy():
    """
    Model Predictive Control policy implementation using GuSTO optimizer.
    
    This policy uses a GuSTO-based MPC controller to compute optimal control actions
    based on the current state observation and a reference trajectory.
    """
    
    def __init__(self,
                 model,
                 config: GuSTOConfig,
                 U: Optional[Any] = None,
                 dU: Optional[Any] = None,
                 exps_M: Optional[Any] = None,
                 M: Optional[Any] = None,
                 U_select: Optional[Any] = None, 
                 W_sm: Optional[Any] = None, 
                 epsilon_sm: Optional[Any] = None, 
                 case_rbf: Optional[Any] = None,
                 init_guess_type='shift'):
        """
        Initialize the MPC policy.
        
        Args:
            model: Model object representing the dynamical system
            config: GuSTO configuration parameters
            z_ref: Reference trajectory for the MPC controller
            U: Control constraints Polyhedron object
            dU: Control rate constraints Polyhedron object
            smoothing_func: Optional function to smooth observations
        """
        super().__init__()
        
        self.model = model
        self.dt = model.dt
        self.config = config
        self.U = U
        self.dU = dU

        self.exps_M = exps_M
        self.M = M
        self.U_select = U_select
        self.W_sm = W_sm
        self.epsilon_sm = epsilon_sm
        self.case_rbf = case_rbf

        

        # Extract dimensions
        self.n_x = model.n_x     # state dimension
        self.n_u = model.n_u     # control dimension
        self.n_z = self.config.H.shape[0] # performance dimension
        
        # MPC parameters
        self.N = config.N

        # Initialize warm start variables
        self.x_prev = jnp.zeros(self.n_x)
        self.u_prev = jnp.zeros(self.n_u)

        # For the slew rate cost
        self.last_applied_u = None

        # What type of initial guess to use (shift, dyn_feasible, zeros)
        self.init_guess_type = init_guess_type
        
    def reset(self, x0: jnp.ndarray, obs, z_ref: jnp.ndarray, start_with_solve=True):
        """
        Reset the policy with a new goal state.
        
        Args:
            x0: Initial state of the system
            obs: Initial observation
            z_ref: Reference trajectory for the MPC controller
        """
        self.z_ref = z_ref

        # Initialize GuSTO with zeros as initial guess
        u_init = jnp.zeros((self.N, self.n_u))
        x_init = self.model.multistep_dynamics(x0, u_init)
        x_init = x_init[:self.N]
        # can be such that x0 is full state...
        z_ref_win = self.z_ref[0:self.N]
        #print(f"x_init shape: {x_init.shape}, z_ref_win shape: {z_ref_win.shape}")
        self.gusto = GuSTO(
            self.model, 
            self.config,
            x0,
            u_init,
            x_init,
            z=z_ref_win,
            zf=z_ref_win[-1],
            U=self.U,
            dU=self.dU,
            start_with_solve=start_with_solve,
            exps_M=self.exps_M,
            M=self.M,
            U_select = self.U_select,
            W_sm = self.W_sm,
            epsilon_sm = self.epsilon_sm,
            case_rbf = self.case_rbf,
            solver='CLARABEL'
        )

        # Get initial solution x_opt (N+1 x n), u_opt (N x m)
        if start_with_solve:
            x_opt, u_opt, _, _ = self.gusto.get_solution()
            self.x_prev = x_opt
            self.u_prev = u_opt
        else:
            self.x_prev = jnp.zeros(self.n_x)
            self.u_prev = jnp.zeros(self.n_u)
    
        # Reset where index into reference
        self.t_idx = 0


    def update_model_param(self, param, param2):
        """
        Update the model's parameter and reinitialize GuSTO if necessary.
        """
        if hasattr(self.model, 'update_param'):
            self.model.update_param(param)
            self.xs = param2
            # Reinitialize GuSTO to ensure dynamics and Jacobians reflect new param
            u_init = self.u_prev if self.u_prev is not None else jnp.zeros((self.N, self.n_u))
            x_init = self.model.multistep_dynamics(self.x_prev[0], u_init)
            #print(x_init.shape)
            x_init = x_init[:self.N]
            z_ref_win = self.z_ref[self.t_idx:min(self.t_idx + self.N, len(self.z_ref))]
            if len(z_ref_win) < self.N:
                k = self.N  - len(z_ref_win)
                z_ref_win = jnp.concatenate([z_ref_win, jnp.tile(self.z_ref[-1], (k, 1))])
            self.gusto = GuSTO(
                self.model,
                self.config,
                self.x_prev[0],
                u_init,
                x_init,
                z=z_ref_win,
                zf=z_ref_win[-1],
                U=self.U,
                dU=self.dU,
                start_with_solve=False,
                exps_M=self.exps_M,
                M=self.M,
                exps_S = self.exps_S,
                Su = self.Su,
                xs=self.xs,
                solver='CLARABEL'
            )

    def compute_control(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the control action for the current observation.
        
        Args:
            obs: Current observation (could be full or partial state)
            
        Returns:
            u: Optimal control action
            info: Dictionary containing additional information
        """
        t_start = time.time()
        
        # Get reference trajectory for current MPC window
        max_ind = min(self.t_idx + self.N, len(self.z_ref))
        z_ref_win = self.z_ref[self.t_idx:max_ind]
        # Pad if not length N+1
        if len(z_ref_win) < self.N:
            k = self.N + 1 - len(z_ref_win)
            print(f"Warning: z_ref_win is shorter than N+1, padding with last reference {k} times")
            last_z = jnp.tile(self.z_ref[-1], (k, 1))
            z_ref_win = jnp.concatenate([z_ref_win, last_z])

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
        
        # Update LOCP parameter with the previously applied control
        if self.last_applied_u is not None:
            self.gusto.locp.u0_prev.value = np.asarray(self.last_applied_u)
        
        # Solve MPC problem
        self.gusto.solve(state, u_init, x_init, z=z_ref_win, zf=z_ref_win[-1])
        x_opt, u_opt, z_opt, solve_time = self.gusto.get_solution()

        # Store solution for warm start
        self.x_prev = x_opt
        self.u_prev = u_opt
        
        # Increment time index
        self.t_idx += 1
        
        # Prepare info dictionary
        info = {
            'solve_time': solve_time,
            'total_time': time.time() - t_start,
            'predicted_trajectory': z_opt,
            'control_trajectory': u_opt
        }
        
        self.last_applied_u = u_opt[0]

        return u_opt[0], info