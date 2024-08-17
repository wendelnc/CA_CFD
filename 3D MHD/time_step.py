# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg       # Input Parameters
import eigenvalues as ev          # Compute Eigenvalues

# @njit
def time_step(q_sys,t):

    em = 1.0e-15

    alphax, alphay, alphaz = ev.eigenvalues(q_sys)

    max_speed_x   = max(em, max(alphax[4],alphax[5]))
    max_speed_y   = max(em, max(alphay[4],alphay[5]))
    max_speed_z   = max(em, max(alphaz[4],alphaz[5]))

    max_speed = max(max_speed_x,max_speed_y,max_speed_z)

    # 4) Compute Time Step
    dt = cfg.CFL * cfg.dx / max_speed
    # dt = CFL * min(dx / max_speed_x, dy / max_speed_y, dz / max_speed_z)

    # Stops our dt from going past tf
    if t + dt > cfg.tf:
        dt = cfg.tf - t

    return dt