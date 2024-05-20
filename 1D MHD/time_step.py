# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg       # Input Parameters
import eigenvalues as ev          # Compute Eigenvalues

# @njit
def time_step(q_sys,t):

    em = 1.0e-15

    # 1) Compute Eigenvalues
    alpha = ev.eigenvalues(q_sys)
    
    # 2) Find Maximum Wave Speed
    max_speed   = max(em, max(alpha[0],alpha[5]))

    # 3) Compute Time Step
    dt = cfg.CFL * cfg.dx / max_speed

    # 4) Stops our dt from going past tf
    if t + dt > cfg.tf:
        dt = cfg.tf - t

    return dt