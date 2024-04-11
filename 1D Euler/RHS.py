# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import lf_flux as lf             # Compute Lax-Friedrichs Flux Vector Splitting

@njit
def RHS(q_sys, q_sys_new, alpha, dt):

    for i in range(cfg.nghost,cfg.nx1+cfg.nghost):

            # Compute Lax-Friedrichs Flux Vector Splitting
            f_l = lf.lf_flux(np.array([q_sys[:,i-3],q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2]]),alpha)
            f_r = lf.lf_flux(np.array([q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2],q_sys[:,i+3]]),alpha)
                
            # Update Solution
            q_sys_new[:,i] = q_sys[:,i] - ((dt)/(cfg.dx))*(f_r - f_l)  
    
    return q_sys_new
