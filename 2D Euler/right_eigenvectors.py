# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries

@njit
def right_eigenvectors(u,v,H,a,nx,ny):
    '''
    Compute the right eigenvectors of the Euler equations
    Following this work 
    "EIGENVALUES AND EIGENVECTORS OF THE EULER EQUATIONS IN GENERAL GEOMETRIES" by AXEL ROHDE
    http://microcfd.com/download/pdf/AIAA-2001-2609.pdf (See Equations 11 and 16)
    Except it is R1, R4, R2, R3 instead of R1, R2, R3, R4
    to keep the eigenvalues as vn-a, vn, vn, vn+a 
    As seen in this work 
    "ATHENA: A NEW CODE FOR ASTROPHYSICAL MHD" by Stone et al.
    https://iopscience.iop.org/article/10.1086/588755/pdf (See Appendix B1)
    '''
    ek = 0.5 * (u**2 + v**2)
    vn = u*nx + v*ny 

    R = np.zeros((4,4),dtype=np.float64)

    R[0,0] = 1.0
    R[1,0] = u - a*nx
    R[2,0] = v - a*ny
    R[3,0] = H - a*vn

    R[0,1] = 0.0
    R[1,1] = ny
    R[2,1] = -nx
    R[3,1] = u*ny - v*nx

    R[0,2] = 1.0
    R[1,2] = u
    R[2,2] = v
    R[3,2] = ek

    R[0,3] = 1.0
    R[1,3] = u + a*nx
    R[2,3] = v + a*ny
    R[3,3] = H + a*vn

    return R
