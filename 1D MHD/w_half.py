# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables

@njit
def w_half(q1,q2):
    '''
    Function Name:      w_half
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 04-15-2024

    Definition:         w_half computes w_{i+{1/2}} given two conserved variables arrays

    Inputs:             q1: conserved variables at cell q_sys_{i}
                        q2: conserved variables at cell q_sys_{i+1} 

    Outputs:            averaged values of the conserved variables 

    Dependencies:       cons2prim
    '''

    den1, vex1, vey1, vez1, pre1, Bx1, By1, Bz1 = c2p.cons2prim(q1)
    den2, vex2, vey2, vez2, pre2, Bx2, By2, Bz2 = c2p.cons2prim(q2)

    den = 0.5 * ( den1 + den2 )
    vex = 0.5 * ( vex1 + vex2 )
    vey = 0.5 * ( vey1 + vey2 )
    vez = 0.5 * ( vez1 + vez2 )
    pre = 0.5 * ( pre1 + pre2 )
    Bx  = 0.5 * (  Bx1 + Bx2 )
    By  = 0.5 * (  By1 + By2 )
    Bz  = 0.5 * (  Bz1 + Bz2 )

    return den, vex, vey, vez, pre, Bx, By, Bz