import numpy as np
import torch
from abc import ABC, abstractmethod 
import jax
import jax.numpy as jnp
from typing import Union, Tuple
from functools import partial

class System(ABC):
    def __init__(self, dt, n_x, n_u, rk4=False):
        self.dt = dt
        self.n_x = n_x
        self.n_u = n_u
        self.rk4 = rk4

    @abstractmethod
    def continuous_dynamics(self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        """
        Compute the continuous time dynamics dx/dt = f(x,u) at several state, action pairs
        
        Args:
            x: State vector of shape (N, n_x)
            u: Control input vector of shape (N, n_u)
            
        Returns:
            dx/dt: State derivative vector of shape (N, n_x)
        """
        pass

    def discrete_dynamics(self, x: Union[np.ndarray,jnp.ndarray], u: Union[np.ndarray,jnp.ndarray]) -> Union[np.ndarray,jnp.ndarray]:
        """
        Compute the discrete time dynamics x_{k+1} = f_d(x_k, u_k) using RK4 integration.
        
        Args:
            x: Current state vector of shape (N, n_x)
            u: Current control input vector of shape (N, n_u)
            dt: Time step
            
        Returns:
            x_next: Next state vector of shape (N, n_x)
        """
        # RK4 integration
        if self.rk4:
            k1 = self.continuous_dynamics(x, u)
            k2 = self.continuous_dynamics(x + 0.5*self.dt*k1, u)
            k3 = self.continuous_dynamics(x + 0.5*self.dt*k2, u)
            k4 = self.continuous_dynamics(x + self.dt*k3, u)

            return x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        # Euler
        else:
            return x + self.continuous_dynamics(x, u) * self.dt
        
    def multistep_dynamics(self, x: Union[np.ndarray, jnp.ndarray], controls: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray,jnp.ndarray]:
        """Repeatedly apply the discrete dynamics to get an (N+1 x n_x) array of states."""
        N, n_u = controls.shape
        states = [x]
        for i in range(N):
            curr_state = states[-1][None,:]
            curr_control = controls[i][None,:]
            next_state = self.discrete_dynamics(curr_state, curr_control).squeeze()

            # Convert back to numpy array from tensor if the starting state indicates that user expects numpy output
            # Note: user should ensure x is tensor if multistep_dynamics is to be used in backpropagation/loss
            if isinstance(x, np.ndarray) and isinstance(next_state, torch.Tensor):
                next_state = next_state.detach().cpu().numpy()

            states.append(next_state)

        if isinstance(x, np.ndarray):
            states = np.array(states)
        elif isinstance(x, torch.Tensor):
            states = torch.stack(states).to(x.device, dtype=torch.float32)
        else:
            # jnp for gusto
            states = jnp.array(states)
            
        return states
    
    def dynamics_jac(self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray], continuous=False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get linearized discrete dynamics matrices A_list, B_list, d_list around several state, action pairs.
        Can be overridden for analytical derivatives.
        """
        if continuous:
            f = partial(self.continuous_dynamics)
        else:
            f = partial(self.discrete_dynamics)
        
        if len(x.shape) == 2:
            # Compute Jacobians using JAX automatic differentiation
            A, B = jax.vmap(jax.jacfwd(f, argnums=(0, 1)))(x, u)
            
            # Compute affine terms
            d = jax.vmap(f)(x, u) - jnp.einsum('ijk,ik->ij', A, x) - jnp.einsum('ijk,ik->ij', B, u)
        else:
            A, B = jax.jacfwd(f, argnums=(0, 1))(x, u)
            d = f(x, u) - A @ x - B @ u
        
        return A, B, d
    
    def update_dynamics_using_obs(self, curr_state, obs):
        """Potentially update the dynamics using current state and latest observation."""
        pass

class LTISys(System):
    def __init__(self, A, B, C, *sup_args):
        self.A = A
        self.B = B
        self.C = C
        super(LTISys, self).__init__(*sup_args)
    
    def continuous_dynamics(self, x, u):
        # A is n_x x n_x, x.T is n_x x N -> n_x x N
        return (self.A @ x.T + self.B @ u.T + self.C).T

class SimpleTerrainBicycle(System):
    """
    A single-patch “terrain” bicycle model that can be auto‐differentiated by JAX.
    Controls: u = (vc, delta)
    State: x = (x, y, psi, vxb, vyb, omega)
    Tire forces come from:
       Fx = Cm * vc - Co
       Fyr = Cy * alpha_r
       Fyf = Cy * alpha_f
    with alpha_r, alpha_f computed from slip geometry.
    """

    def __init__(self, m, Iz, a, b, Cy, Cm, Co, *sup_args):
        super().__init__(*sup_args)
        self.m = m
        self.Iz = Iz
        self.a = a
        self.b = b
        self.Cy = Cy
        self.Cm = Cm
        self.Co = Co

    def continuous_dynamics(self, x, u):
        """
        x = [px, py, psi, vxb, vyb, omega]
        u = [vc, delta]
        Returns dx/dt as a jnp.array([...]).
        """
        px, py, psi, vxb, vyb, omega = x.T
        vc, delta = u.T

        # Use JAX-friendly clip:
        vxb_safe = jnp.maximum(vxb, 1e-2)  # Prevent division by zero

        # Slip angles
        alpha_r = jnp.arctan2(vyb - self.b * omega, vxb_safe)
        alpha_f = jnp.arctan2(vyb + self.a * omega, vxb_safe) - delta

        # Tire forces
        Fx  = self.Cm * vc - self.Co
        Fyr = self.Cy * alpha_r
        Fyf = self.Cy * alpha_f

        # Kinematic relationships
        dx   = vxb * jnp.cos(psi) - vyb * jnp.sin(psi)
        dy   = vxb * jnp.sin(psi) + vyb * jnp.cos(psi)
        dpsi = omega

        # EOM: mass and moment of inertia
        dvxb  = (1.0 / self.m) * (Fx - Fyf * jnp.sin(delta) + self.m * vyb * omega)
        dvyb  = (1.0 / self.m) * (Fyr + Fyf * jnp.cos(delta) - self.m * vxb * omega)
        domega = (1.0 / self.Iz) * (Fyf * self.a * jnp.cos(delta) - Fyr * self.b)

        return jnp.array([dx, dy, dpsi, dvxb, dvyb, domega])
    



def test_linearization(system, x0, u0, step=1):
    # Taylor centering point is x0, u0
    # Amount to move from centering point is step
    num_states = len(x0)
    num_actions = len(u0)
    
    # Move slightly away from Taylor center point
    x = x0 + step * np.random.rand(num_states)
    u = u0 + step * np.random.rand(num_actions)
    
    # Compute nonlinear step
    next_state = system.discrete_dynamics(x[None,:], u[None,:]).squeeze()

    # Compare to linearized prediction about Taylor center
    A, B, C = system.dynamics_jac(x0[None,:], u0[None,:])
    A, B, C = A.squeeze(), B.squeeze(), C.squeeze()
    if not isinstance(A, np.ndarray):
        A, B, C = A.cpu().detach().numpy(), B.cpu().detach().numpy(), C.cpu().detach().numpy()

    lin_next_state = A @ x + B @ u + C

    return next_state, lin_next_state

if __name__ == '__main__':
    #### Initialize system ####
    dt = 0.05 # s
    n_x = 6
    n_u = 2
    rk4 = False

    # Fill in these values with something reasonable for testing
    m = 1.2 # kg
    l = 0.176 # m
    w = 0.183 # m
    a = l/2 # m
    b = l/2 # m
    Iz = m/12 * (l**2 + w**2) # kg * m^2
    Cy = -10 # N/rad
    Cm = 1 # kg/s
    Co = 0.1 # N

    system = SimpleTerrainBicycle(m, Iz, a, b, Cy, Cm, Co, dt, n_x, n_u, rk4)

    #### Test system ####
    x0 = np.random.rand(n_x)
    u0 = np.random.rand(n_u)
    next_state, lin_next_state = test_linearization(system, x0, u0, step=0.1)
    print('next_state', next_state, 'lin_next_state', lin_next_state)