"""
Quick diagnostic to identify bottleneck in 0.095s solve time

Add this temporarily to your GuSTO solve() method to profile
"""

import time

# Add these timing prints in gusto_upgrade_trunk.py solve() method:

# Around line 469 (after initial Jacobian computation):
"""
print(f'Initial setup time: {(time.time() - t0)*1000:.2f} ms')
"""

# Around line 490 (inside the while loop, after LOCP solve):
"""
print(f'  - LOCP solve: {stats.solve_time*1000:.2f} ms')
print(f'  - Trust region check: {(time.time() - t_locp_end)*1000:.2f} ms')  # Add t_locp_end = time.time() after solve
"""

# Around line 544 (after recomputing Jacobians):
"""
print(f'  - Jacobian recomputation: {(time.time() - t_jac_start)*1000:.2f} ms')
"""

# BETTER: Add comprehensive timing to the solve() method
# Replace the solve() method with this instrumented version:

def solve_instrumented(self, x0, u_init, x_init, z=None, zf=None, u=None):
    """Instrumented version of solve() with detailed timing."""

    timing = {
        'total': 0,
        'initial_jac': 0,
        'locp_solve': 0,
        'trust_region': 0,
        'jacobian_recomp': 0,
        'other': 0
    }

    t0 = time.time()
    t_locp = 0.0

    itr = 0
    self.x_k = x_init
    self.u_k = u_init

    # Initial Jacobian computation
    t_jac_0 = time.time()
    A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k, self.u_k)
    if self.nonlinear_perf_mapping:
        H_d, G_d, c_d = self._get_perf_mapping_linearizations(self.x_k, self.u_k)
    else:
        H_d, G_d, c_d = None, None, None
    timing['initial_jac'] = time.time() - t_jac_0

    new_solution = True
    Jstar_prev = jnp.inf
    delta_prev = jnp.inf
    omega_prev = jnp.inf
    converged = False
    delta = self.delta0
    omega = self.omega0

    print(f"\n{'='*60}")
    print(f"GuSTO SOLVE TIMING BREAKDOWN")
    print(f"{'='*60}")
    print(f"Initial Jacobian: {timing['initial_jac']*1000:.2f} ms")

    iteration_times = []

    while self._is_valid_iteration(itr) and not converged and omega <= self.omega_max:
        t_iter = time.time()

        # LOCP update and solve
        if new_solution:
            self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u, Hd=H_d, Gd=G_d, cd=c_d)
            new_solution = False
        else:
            self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u, Hd=H_d, Gd=G_d, cd=c_d, full=False)

        t_solve = time.time()
        Jstar, success, stats = self.locp.solve()
        t_locp += stats.solve_time
        locp_time = time.time() - t_solve

        if not success:
            print(f"Iteration {itr} failed")
            break

        x_next, u_next, _ = self.locp.get_solution()

        # Trust region check
        t_tr = time.time()
        e_tr, tr_satisfied = self._is_in_trust_region(self.x_k, x_next, delta)
        tr_time = time.time() - t_tr

        # Convergence and constraint checking
        t_conv = time.time()
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

        conv_time = time.time() - t_conv

        # Jacobian recomputation
        jac_recomp_time = 0
        if new_solution:
            self.x_k = x_next.copy()
            self.u_k = u_next.copy()
            if self.max_gusto_iters >= 1:
                t_jac = time.time()
                A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k, self.u_k)
                if self.nonlinear_perf_mapping:
                    H_d, G_d, c_d = self._get_perf_mapping_linearizations(self.x_k, self.u_k)
                else:
                    H_d, G_d, c_d = None, None, None
                jac_recomp_time = time.time() - t_jac

        iter_total = time.time() - t_iter
        iteration_times.append({
            'locp': locp_time * 1000,
            'trust_region': tr_time * 1000,
            'convergence': conv_time * 1000,
            'jacobian': jac_recomp_time * 1000,
            'total': iter_total * 1000
        })

        print(f"\nIteration {itr}:")
        print(f"  LOCP solve:    {locp_time*1000:6.2f} ms")
        print(f"  Trust region:  {tr_time*1000:6.2f} ms")
        print(f"  Convergence:   {conv_time*1000:6.2f} ms")
        print(f"  Jacobian:      {jac_recomp_time*1000:6.2f} ms")
        print(f"  Iteration tot: {iter_total*1000:6.2f} ms")

        itr += 1

    t_gusto = time.time() - t0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Iterations: {itr}")
    print(f"Total GuSTO time: {t_gusto*1000:.2f} ms")
    print(f"Total LOCP time:  {t_locp*1000:.2f} ms")
    print(f"Initial Jacobian: {timing['initial_jac']*1000:.2f} ms")

    if iteration_times:
        avg_iter = sum(t['total'] for t in iteration_times) / len(iteration_times)
        print(f"Avg iteration:    {avg_iter:.2f} ms")

    print(f"{'='*60}\n")

    # Save optimal solution (same as before)
    self.xopt = jnp.copy(self.x_k)
    self.uopt = jnp.copy(self.u_k)
    if self.nonlinear_perf_mapping:
        self.zopt = self.performance_mapping(self.xopt.T, self.uopt).T
    else:
        self.zopt = jnp.transpose(self.H @ self.xopt.T)
    self.locp_solve_time = t_locp


# USAGE:
# Temporarily replace solve() with solve_instrumented() to get detailed breakdown
# OR just add selective print statements at key points
