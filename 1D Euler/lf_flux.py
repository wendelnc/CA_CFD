# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import get_flux as gf            # Compute Flux
import w_half as wh              # Compute w_{i+1/2} (or w_{i-1/2})
import eigenvectors as ev        # Compute Right and Left Eigenvectors
import weno as wn                # Compute WENO Reconstruction

@njit
def lf_flux(q_arr,alpha):
# def lf_flux(q0,q1,q2,q3,q4,q5,alpha):
    '''
    Function Name:      lf_flux
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 03-26-2024

    Definition:         lf_flux computes the Lax-Friedrichs flux vector splitting

    Inputs:             q_arr: array of conserved variables
                        alpha: max wave speeds over all grid points

    Outputs:            f_hafl: result of the Lax-Friedrichs flux vector splitting

    Dependencies:       cons2prim, get_flux, w_half, eigenvectors
    '''

                  # x_{i+1/2}   # x_{i-1/2}
                  #----------   #----------
    q0 = q_arr[0] # cell i-2    # cell i-3
    q1 = q_arr[1] # cell i-1    # cell i-2
    q2 = q_arr[2] # cell i      # cell i-1
    q3 = q_arr[3] # cell i+1    # cell i
    q4 = q_arr[4] # cell i+2    # cell i+1
    q5 = q_arr[5] # cell i+3    # cell i+2

    # 1. Compute the physical flux at each grid point:
    f0 = gf.get_flux(q0)
    f1 = gf.get_flux(q1)
    f2 = gf.get_flux(q2)
    f3 = gf.get_flux(q3)
    f4 = gf.get_flux(q4)
    f5 = gf.get_flux(q5)

    # 2. At each x_{i+1/2}:
    # (a) Compute the average state w_{i+1/2} in the primitive variables:
    den, vex, pre, E, H, a = wh.w_half(q2,q3)
        
    #(b) Compute the right and left eigenvectors of the flux Jacobian matrix, ∂f/∂x, at x = x_{i+1/2}:
    r, l = ev.eigenvectors(vex,H,a)

    # (c) Project the solution and physical flux into the right eigenvector space:
    
    vj0 = np.dot(l, q0)
    vj1 = np.dot(l, q1)
    vj2 = np.dot(l, q2)
    vj3 = np.dot(l, q3)
    vj4 = np.dot(l, q4)
    vj5 = np.dot(l, q5)

    gj0 = np.dot(l, f0)
    gj1 = np.dot(l, f1)
    gj2 = np.dot(l, f2)
    gj3 = np.dot(l, f3)
    gj4 = np.dot(l, f4)
    gj5 = np.dot(l, f5)

    # (d) Perform a Lax-Friedrichs flux vector splitting for each component of the characteristic variables. 
    # Specifically, assume that the mth components of Vj and Gj are vj and gj, respectively, then compute
    # g^{±}_{j}= 0.5* (g_j ± α^{m} v_j) where α(m) = max_k | λ^{m} q_k | 
    # is the maximal wave speed of the m^{th} component of characteristic variables over all grid points
    
    gp0 = 0.5 * (gj0 + 1.1 * alpha * vj0)
    gp1 = 0.5 * (gj1 + 1.1 * alpha * vj1)
    gp2 = 0.5 * (gj2 + 1.1 * alpha * vj2)
    gp3 = 0.5 * (gj3 + 1.1 * alpha * vj3)
    gp4 = 0.5 * (gj4 + 1.1 * alpha * vj4)
    # gp5 = 0.5 * (gj5 + 1.1 * alpha * vj5) # Not needed for WENO

    # gm0 = 0.5 * (gj0 - 1.1 * alpha * vj0) # Not needed for WENO
    gm1 = 0.5 * (gj1 - 1.1 * alpha * vj1)
    gm2 = 0.5 * (gj2 - 1.1 * alpha * vj2)
    gm3 = 0.5 * (gj3 - 1.1 * alpha * vj3)
    gm4 = 0.5 * (gj4 - 1.1 * alpha * vj4)
    gm5 = 0.5 * (gj5 - 1.1 * alpha * vj5)


    # (e) Perform a WENO reconstruction on each of the computed flux components gj± to obtain 
    # the corresponding component of the numerical flux
    
    g_half = wn.weno(gp0,gp1,gp2,gp3,gp4,gm1,gm2,gm3,gm4,gm5)

    # (f) Project the numerical flux back to the conserved variables

    f_half = np.dot(r, g_half)

    return f_half