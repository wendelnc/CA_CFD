# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

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

    Dependencies:       none
    '''
    
    den = q_sys[0]
    vex   = q_sys[1] / q_sys[0]
    pre   = (cfg.gamma - 1.) * (q_sys[2] - (0.5 * q_sys[0] * (vex)**2.))

    E = q_sys[2] 

    f1 = den * vex
    f2 = den * vex * vex + pre 
    f3 = vex * (E + pre)

    flux = np.array([f1,f2,f3])
    
    return flux