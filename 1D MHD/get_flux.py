# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables

@njit
def get_flux(q_sys):
    '''
    Function Name:      get_flux
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 04-15-2024

    Definition:         get_flux computes the flux of conserved variables

    Inputs:             q_sys = [ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z]^T

    Outputs:            fluxes 

    Dependencies:       cons2prim
    '''

    rho, vex, vey, vez, pre, Bx, By, Bz = c2p.cons2prim(q_sys)
    Eng = q_sys[4]

    f1 = rho * vex
    f2 = (rho * vex * vex) + pre + (0.5 * (Bx**2 + By**2 + Bz**2)) - (Bx * Bx)
    f3 = (rho * vex * vey) - (Bx * By)
    f4 = (rho * vex * vez) - (Bx * Bz)
    f5 = (vex * (Eng + pre + (0.5 * (Bx**2 + By**2 + Bz**2)))) - (Bx * (Bx * vex + By * vey + Bz * vez))
    f6 = 0.0
    f7 = (vex * By) - (vey * Bx)
    f8 = (vex * Bz) - (vez * Bx)
    
    flux = np.array([f1,f2,f3,f4,f5,f6,f7,f8])
    
    return flux