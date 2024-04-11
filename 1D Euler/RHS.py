# Standard Python Libraries
import numpy as np
from numba import njit, jit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import init as ic                # Initialize Test Problem
import plotting as eplt          # Plotting Solution
import ghost as ght              # Add Ghost Cells
import boundary_conditions as bc # Update Boundary Conditions
import time_step as ts           # Compute Time Step
import cons2prim as c2p          # Convert Conserved to Primitive Variables
import get_flux as gf            # Compute Flux
import w_half as wh              # Compute w_{i+1/2} (or w_{i-1/2})
import eigenvectors as ev        # Compute Right and Left Eigenvectors
import weno as wn                # Compute WENO Reconstruction
import lf_flux as lf             # Compute Lax-Friedrichs Flux Vector Splitting

@njit
def RHS(q_sys, q_sys_new, alpha, dt):

    for i in range(cfg.nghost,cfg.nx1+cfg.nghost):

            # Compute Lax-Friedrichs Flux Vector Splitting
            f_l = lf.lf_flux(np.array([q_sys[:,i-3],q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2]]),alpha)
            f_r = lf.lf_flux(np.array([q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2],q_sys[:,i+3]]),alpha)
        
            # f_l = lf.lf_flux(q_sys[:,i-3],q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2],alpha)
            # f_r = lf.lf_flux(q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2],q_sys[:,i+3],alpha)
        
            # Update Solution
            q_sys_new[:,i] = q_sys[:,i] - ((dt)/(cfg.dx))*(f_r - f_l)  
    
    return q_sys_new
