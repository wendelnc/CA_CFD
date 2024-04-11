# Standard Python Libraries
import numpy as np
from numba import njit, jit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import init as ic                # Initialize Test Problem
import plotting as eplt          # Plotting Solution
import ghost as ght              # Add Ghost Cells
import boundary_conditions as bc # Update Boundary Conditions
import time_step as ts           # Compute Time Step
import cons2prim as c2p          # Convert Conserved to Primitive Variables
import get_flux as gf            # Compute Flux
import w_half as wh              # Compute w_{i+1/2} (or w_{i-1/2})
import eigenvectors as ev        # Compute Right and Left Eigenvectors
import weno as wn                # Compute WENO Reconstruction
import lf_flux as lf             # Compute Lax-Friedrichs Flux Vector Splitting

@njit
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

    # rho1, u1, p1 = c2p.cons2prim(q1)
    # rho2, u2, p2 = c2p.cons2prim(q2)

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