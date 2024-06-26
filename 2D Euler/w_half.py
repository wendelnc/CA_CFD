# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def w_half(q1,q2):
    '''
    Function Name:      w_half
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         w_half computes w_{1/2} given two conserved variables arrays

    Inputs:             q1: conserved variables at cell q_sys_{i,j}
                        q2: conserved variables at cell q_sys_{i+1,j} (or q_sys_{i,j+1})

    Outputs:            qavg: average of the conserved variables 

    Dependencies:       none
    '''

    rho1 = q1[0]
    u1   = q1[1] / q1[0]
    v1   = q1[2] / q1[0]
    p1   = (cfg.gamma - 1.) * (q1[3] - (0.5 * rho1 * ((u1)**2. + (v1)**2.)))
    
    rho2 = q2[0]
    u2   = q2[1] / q2[0]
    v2   = q2[2] / q2[0]
    p2   = (cfg.gamma - 1.) * (q2[3] - (0.5 * rho2 * ((u2)**2. + (v2)**2.)))

    den = 0.5 * ( rho1 + rho2 )
    vex = 0.5 * ( u1 + u2 )
    vey = 0.5 * ( v1 + v2 )
    pre = 0.5 * ( p1 + p2 )
    E = pre / (cfg.gamma - 1) + (0.5 * den * (vex**2 + vey**2))
    H = (E + pre) / den
    a = np.sqrt( (cfg.gamma*pre) / den )

    return den, vex, vey, pre, E, H, a