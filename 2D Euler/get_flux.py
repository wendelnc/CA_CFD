# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def get_flux(q_sys,nx,ny):
    '''
    Function Name:      get_flux_half
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         get_flux computes the flux of conserved variables

    Inputs:             q_sys = [rho, rho*u, rho*v, E]^T

    Outputs:            f1,f2,f3,f4: fluxes 

    Dependencies:       none
    '''
    
    den = q_sys[0]
    vex   = q_sys[1] / q_sys[0]
    vey   = q_sys[2] / q_sys[0]
    E = q_sys[3] 
    pre   = (cfg.gamma - 1.) * (E - (0.5 * den * ((vex)**2. + (vey)**2.)))
    
    vn = vex * nx + vey * ny

    f1 = den * vn
    f2 = den * vex * vn + pre * nx
    f3 = den * vey * vn + pre * ny
    f4 = vn * (E + pre)

    flux = np.array([f1,f2,f3,f4])
    
    return flux