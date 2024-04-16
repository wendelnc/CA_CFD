# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def left_eigenvectors(u,v,H,a,nx,ny):
    '''
    Compute the left eigenvectors of the Euler equations
    Following this work
    "EIGENVALUES AND EIGENVECTORS OF THE EULER EQUATIONS IN GENERAL GEOMETRIES" by AXEL ROHDE
    http://microcfd.com/download/pdf/AIAA-2001-2609.pdf (See Equations 11 and 16)
    Except it is L1, L4, L2, L3 instead of L1, L2, L3, L4
    to keep the eigenvalues as vn-a, vn, vn, vn, vn+a 
    As seen in this work 
    "ATHENA: A NEW CODE FOR ASTROPHYSICAL MHD" by Stone et al.
    https://iopscience.iop.org/article/10.1086/588755/pdf (See Appendix B1)
    '''
    ek = 0.5 * (u**2 + v**2)
    vn = u*nx + v*ny

    L = np.zeros((4,4),dtype=np.float64)

    L[0,0] = ((cfg.gamma-1.0)*ek + a*vn) / (2.0 * a**2)
    L[0,1] = ((1.0-cfg.gamma)*u - a*nx) / (2.0 * a**2)
    L[0,2] = ((1.0-cfg.gamma)*v - a*ny) / (2.0 * a**2)
    L[0,3] = (cfg.gamma-1.0) / (2.0 * a**2)
    
    if nx == 1.0: 
        L[1,0] = (v - vn*ny) / nx 
        L[1,1] = ny
        L[1,2] = (ny**2 - 1.0) / nx
        L[1,3] = 0.0
    else:
        L[1,0] = (vn*nx - u) / ny
        L[1,1] = (1 - nx**2) / ny
        L[1,2] = -nx
        L[1,3] = 0.0

    L[2,0] = (a**2 - (cfg.gamma-1.0)*ek) / a**2
    L[2,1] = ((cfg.gamma-1.0)*u) / a**2
    L[2,2] = ((cfg.gamma-1.0)*v) / a**2
    L[2,3] = (1.0-cfg.gamma) / a**2

    L[3,0] = ((cfg.gamma-1.0)*ek - a*vn) / (2.0 * a**2)
    L[3,1] = ((1.0-cfg.gamma)*u + a*nx) / (2.0 * a**2)
    L[3,2] = ((1.0-cfg.gamma)*v + a*ny) / (2.0 * a**2)
    L[3,3] = (cfg.gamma-1.0) / (2.0 * a**2)

    return L