# Standard Python Libraries
import numpy as np
from numba import njit

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
def eigenvectors(u,H,a):
    '''
    Function Name:      eigenvectors
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-28-2024

    Definition:         eigenvectors computes the left and right eigenvectors for 1D Euler Equations

    Inputs:             u: velocity
                        H: total enthalpy
                        a: speed of sound

    Outputs:            r: right eigenvectors
                        l: left eigenvectors

    Dependencies:       none
    '''

    R = np.zeros((3,3),dtype=np.float64)

    R[0,0] = 1.0
    R[1,0] = u - a
    R[2,0] = H - u*a

    R[0,1] = 1.0
    R[1,1] = u
    R[2,1] = 0.5 * u**2.

    R[0,2] = 1.0
    R[1,2] = u + a
    R[2,2] = H + u*a

    L = np.linalg.inv(R)

    return R, L