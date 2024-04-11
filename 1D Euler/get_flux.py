# Standard Python Libraries
import numpy as np
from numba import njit

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
def get_flux(q_sys):
    '''
    Function Name:      get_flux
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         get_flux computes the flux of conserved variables

    Inputs:             q_sys = [rho, rho*u, E]^T

    Outputs:            f1,f2,f3: fluxes 

    Dependencies:       cons2prim
    '''
    
    # den, vex, pre = c2p.cons2prim(q_sys)
    den = q_sys[0]
    vex   = q_sys[1] / q_sys[0]
    pre   = (cfg.gamma - 1.) * (q_sys[2] - (0.5 * q_sys[0] * (vex)**2.))

    E = q_sys[2] 

    f1 = den * vex
    f2 = den * vex * vex + pre 
    f3 = vex * (E + pre)

    flux = np.array([f1,f2,f3])
    
    return flux