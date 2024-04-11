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

    Definition:         w_half computes w_{i+{1/2}} given two conserved variables arrays

    Inputs:             q1: conserved variables at cell q_sys_{j}
                        q2: conserved variables at cell q_sys_{i+1} 

    Outputs:            qavg: average of the conserved variables 

    Dependencies:       none
    '''

    rho1 = q1[0]
    rho2 = q2[0]
    u1   = q1[1] / q1[0]
    u2   = q2[1] / q2[0]
    p1   = (cfg.gamma - 1.) * (q1[2] - (0.5 * q1[0] * (u1)**2.))
    p2   = (cfg.gamma - 1.) * (q2[2] - (0.5 * q2[0] * (u2)**2.))

    den = 0.5 * ( rho1 + rho2 )
    vex = 0.5 * ( u1 + u2 )
    pre = 0.5 * ( p1 + p2 )
    E = pre / (cfg.gamma - 1) + (0.5 * den * vex**2 )
    H = (E + pre) / den
    a = np.sqrt( (cfg.gamma*pre) / den )

    return den, vex, pre, E, H, a