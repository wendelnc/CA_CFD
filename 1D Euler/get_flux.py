# Standard Python Libraries
import numpy as np

# User Defined Libraries
import cons2prim as c2p          # Convert Conserved to Primitive Variables

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
    
    den, vex, pre = c2p.cons2prim(q_sys)
    
    E = q_sys[2] 

    f1 = den * vex
    f2 = den * vex * vex + pre 
    f3 = vex * (E + pre)

    flux = np.array([f1,f2,f3])
    
    return flux