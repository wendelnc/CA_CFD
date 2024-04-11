# Standard Python Libraries
import numpy as np

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables

def w_half(q1,q2):
    '''
    Function Name:      w_half
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         w_half computes w_{1/2} given two conserved variables arrays

    Inputs:             q1: conserved variables at cell q_sys_{i,j}
                        q2: conserved variables at cell q_sys_{i+1,j} or (q_sys_{i,j+1})

    Outputs:            qavg: average of the conserved variables 

    Dependencies:       cons2prim
    '''

    rho1, u1, v1, p1 = c2p.cons2prim(q1)
    rho2, u2, v2, p2 = c2p.cons2prim(q2)

    den = 0.5 * ( rho1 + rho2 )
    vex = 0.5 * ( u1 + u2 )
    vey = 0.5 * ( v1 + v2 )
    pre = 0.5 * ( p1 + p2 )
    E = pre / (cfg.gamma - 1) + (0.5 * den * (vex**2 + vey**2))
    H = (E + pre) / den
    a = np.sqrt( (cfg.gamma*pre) / den )

    return den, vex, vey, pre, E, H, a