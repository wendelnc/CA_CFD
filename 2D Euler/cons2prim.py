# Standard Python Libraries
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def cons2prim(q_sys):
    '''
    Function Name:      cons2prim
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         cons2prim converts the conserved variables to primative variables

    Inputs:             q_sys = [rho, rho*u, rho*v, E]^T

    Outputs:            rho: density
                        u: x velocity
                        v: y velocity
                        p: pressure

    Dependencies:       none
    '''

    rho = q_sys[0]
    u   = q_sys[1] / q_sys[0]
    v   = q_sys[2] / q_sys[0]
    p   = (cfg.gamma - 1.) * (q_sys[3] - (0.5 * q_sys[0] * ((u)**2. + (v)**2.)))

    return rho, u, v, p