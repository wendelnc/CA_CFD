# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries

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