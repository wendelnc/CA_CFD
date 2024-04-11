# Standard Python Libraries
import numpy as np

# User Defined Libraries
import cons2prim as c2p          # Convert Conserved to Primitive Variables

def get_flux(q_sys,nx,ny):
    '''
    Function Name:      get_flux_half
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         get_flux computes the flux of conserved variables

    Inputs:             q_sys = [rho, rho*u, rho*v, E]^T

    Outputs:            f1,f2,f3,f4: fluxes 

    Dependencies:       cons2prim
    '''
    
    den, vex, vey, pre = c2p.cons2prim(q_sys)
    
    E = q_sys[3] 

    vn = vex * nx + vey * ny

    f1 = den * vn
    f2 = den * vex * vn + pre * nx
    f3 = den * vey * vn + pre * ny
    f4 = vn * (E + pre)

    flux = np.array([f1,f2,f3,f4])
    
    return flux