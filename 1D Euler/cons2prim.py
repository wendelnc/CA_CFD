# Standard Python Libraries

# User Defined Libraries
import configuration as cfg      # Input Parameters

def cons2prim(q_sys):
    '''
    Function Name:      cons2prim
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         cons2prim converts the conserved variables to primative variables

    Inputs:             q_sys = [rho, rho*u, E]^T

    Outputs:            rho: density
                        u: x velocity
                        p: pressure

    Dependencies:       none
    '''

    rho = q_sys[0]
    u   = q_sys[1] / q_sys[0]
    p   = (cfg.gamma - 1.) * (q_sys[2] - (0.5 * q_sys[0] * (u)**2.))

    return rho, u, p