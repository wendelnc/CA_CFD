# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def time_step(q_sys,t):

    # 1) Convert q_sys = [rho, rho*u, E]^T to Primitive Variables w_sys = [rho, u, p]^T
    all_rho = q_sys[0, :]
    all_vex = q_sys[1, :] / q_sys[0, :]
    all_pre = (q_sys[2, :] - 0.5 * q_sys[0, :] * (q_sys[1, :] / q_sys[0, :])**2) * (cfg.gamma - 1)

    # 2) Compute Sound Speed
    a = np.sqrt(cfg.gamma * all_pre / all_rho)

    # 3) Find Maximum Wave Speed
    max_eigen_1 = np.max(np.abs(all_vex-a))
    max_eigen_2 = np.max(np.abs(all_vex))
    max_eigen_3 = np.max(np.abs(all_vex+a))
    
    alpha = np.array([max_eigen_1, max_eigen_2 , max_eigen_3])

    max_speed = np.max(alpha)

    # 4) Compute Time Step
    dt = cfg.CFL * cfg.dx / max_speed

    # Stops our td from going past t1
    if t + dt > cfg.tf:
        dt = cfg.tf - t

    return dt, alpha