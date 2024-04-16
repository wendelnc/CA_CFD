# Standard Python Libraries
import numpy as np
# from numba import njit

# User Defined Libraries

# @njit
def right_eigenvectors(vex, vey, vez, a, Ca, Cf, Cs):
    '''
    Compute the right eigenvectors of the MHD equations
    Following this work 
    "ATHENA: A NEW CODE FOR ASTROPHYSICAL MHD" by Stone et al.
    https://iopscience.iop.org/article/10.1086/588755/pdf (See Appendix B3)
    '''

    R = np.zeros((7,7),dtype=np.float64)

    alpha_f = np.sqrt((a**2 - Cs**2) / (Cf**2 - Cs**2))
    alpha_s = np.sqrt((Cf**2 - a**2) / (Cf**2 - Cs**2))

    Cff = Cf * alpha_f
    Css = Cs * alpha_s

    Vxf = vex * alpha_f
    Vxs = vex * alpha_s

    Vyf = vey * alpha_f
    Vys = vey * alpha_s

    Vzf = vez * alpha_f
    Vzs = vez * alpha_s

    R[0,0] = alpha_f
    R[1,0] = 2.0 
    R[2,0] = 3.0
    R[3,0] = 4.0
    R[4,0] = 5.0
    R[5,0] = 6.0
    R[6,0] = 7.0

    R[0,1] = 0.0
    R[1,1] = 2.1
    R[2,1] = 3.1
    R[3,1] = 4.1
    R[4,1] = 5.1
    R[5,1] = 6.1
    R[6,1] = 7.1

    R[0,2] = alpha_s
    R[1,2] = 2.2
    R[2,2] = 3.2
    R[3,2] = 4.2
    R[4,2] = 5.2
    R[5,2] = 6.2
    R[6,2] = 7.2

    R[0,3] = 1.0
    R[1,3] = 2.3
    R[2,3] = 3.3
    R[3,3] = 4.3
    R[4,3] = 5.3
    R[5,3] = 6.3
    R[6,3] = 7.3

    R[0,4] = alpha_s
    R[1,4] = 2.4
    R[2,4] = 3.4
    R[3,4] = 4.4
    R[4,4] = 5.4
    R[5,4] = 6.4
    R[6,4] = 7.4

    R[0,5] = 0.0
    R[1,5] = 2.5
    R[2,5] = 3.5
    R[3,5] = 4.5
    R[4,5] = 5.5
    R[5,5] = 6.5
    R[6,5] = 7.5

    R[0,6] = alpha_f
    R[1,6] = 2.6
    R[2,6] = 3.6
    R[3,6] = 4.6
    R[4,6] = 5.6
    R[5,6] = 6.6
    R[6,6] = 7.6

    print(R)

    return R
