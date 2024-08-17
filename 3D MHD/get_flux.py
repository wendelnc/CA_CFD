# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables

@njit
def get_flux(q_sys, nx, ny, nz):
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

    vn = vex*nx + vey*ny + vez*nz
    Bn = Bx*nx  + By*ny  + Bz*nz

    Bnm = Bx**2 + By**2 + Bz**2

    f1 = (rho * vn)
    f2 = (rho * vex * vn) + nx*(pre + 0.5*Bnm) - (Bx * Bn)
    f3 = (rho * vey * vn) + ny*(pre + 0.5*Bnm) - (By * Bn)
    f4 = (rho * vez * vn) + nz*(pre + 0.5*Bnm) - (Bz * Bn)
    f5 = (vn * (Eng + pre + 0.5*Bnm)) - (Bn * (Bx * vex + By * vey + Bz * vez))
    f6 = (vn * Bx) - (vex * Bn)
    f7 = (vn * By) - (vey * Bn)
    f8 = (vn * Bz) - (vez * Bn)
    
    flux = np.array([f1,f2,f3,f4,f5,f6,f7,f8])
    
    return flux