# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def eigenvectors(u,H,a):
    '''
    Function Name:      eigenvectors
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 04-10-2024

    Definition:         eigenvectors computes the left and right eigenvectors for 1D Euler Equations

    Inputs:             u: velocity
                        H: total enthalpy
                        a: speed of sound

    Outputs:            R: right eigenvectors
                        L: left eigenvectors

    Dependencies:       none
   
    Here we are following this paper:

    "EIGENVALUES AND EIGENVECTORS OF THE EULER EQUATIONS IN GENERAL GEOMETRIES" by AXEL ROHDE
    http://microcfd.com/download/pdf/AIAA-2001-2609.pdf (See Equations 11 and 16)
    For R we remove the 4th and 5th columns and 2nd and 3rd rows
    For L we remove the 4th and 5th rows and 3rd and 4th columns
    
     '''

    R = np.zeros((3,3),dtype=np.float64)
    L = np.zeros((3,3),dtype=np.float64)

    ek = 0.5 * u**2.

    R[0,0] = 1.0
    R[1,0] = u - a
    R[2,0] = H - u*a

    R[0,1] = 1.0
    R[1,1] = u
    R[2,1] = ek

    R[0,2] = 1.0
    R[1,2] = u + a
    R[2,2] = H + u*a

    L[0,0] = ((cfg.gamma-1.0)*ek + a*u) / (2.0 * a**2)
    L[0,1] = ((1.0-cfg.gamma)*u - a) / (2.0 * a**2)
    L[0,2] = (cfg.gamma-1.0) / (2.0 * a**2)

    L[1,0] = (a**2 - (cfg.gamma-1.0)*ek) / a**2
    L[1,1] = ((cfg.gamma-1.0)*u) / a**2
    L[1,2] = (1.0-cfg.gamma) / a**2

    L[2,0] = ((cfg.gamma-1.0)*ek - a*u) / (2.0 * a**2)
    L[2,1] = ((1.0-cfg.gamma)*u + a) / (2.0 * a**2)
    L[2,2] = (cfg.gamma-1.0) / (2.0 * a**2)
    
    return R, L