# Standard Python Libraries
import numpy as np
# from numba import njit

# User Defined Libraries
import configuration as cfg       # Input Parameters
import cons2prim as c2p           # Convert Conserved to Primitive Variables
import get_flux as gf             # Compute Flux
import w_half as wh               # Compute w_{i+1/2} (or w_{i-1/2}
import set_rght_eigEntropy as rev # Compute Right Eigenvectors
import set_left_eigEntropy as lev # Compute Left Eigenvectors
import weno as wn                 # Compute WENO Reconstruction

# @njit
def lf_flux(q_arr,alpha,nx,ny,nz):

                  # x_{i+1/2}   # x_{i-1/2}
                  #----------   #----------
    q0 = q_arr[0] # cell i-2    # cell i-3
    q1 = q_arr[1] # cell i-1    # cell i-2
    q2 = q_arr[2] # cell i      # cell i-1
    q3 = q_arr[3] # cell i+1    # cell i
    q4 = q_arr[4] # cell i+2    # cell i+1
    q5 = q_arr[5] # cell i+3    # cell i+2

    # 1. Compute the physical flux at each grid point:
    flux0 = gf.get_flux(q0,nx,ny,nz) 
    flux1 = gf.get_flux(q1,nx,ny,nz) 
    flux2 = gf.get_flux(q2,nx,ny,nz) 
    flux3 = gf.get_flux(q3,nx,ny,nz) 
    flux4 = gf.get_flux(q4,nx,ny,nz) 
    flux5 = gf.get_flux(q5,nx,ny,nz) 

    # 2. At each x_{i+1,j}:
    # (a) Compute the average state w_{i+1/2} in the primitive variables:
    den, vex, vey, vez, pre, Bx, By, Bz = wh.w_half(q2,q3) 

    if nx == 1:
        t1 = 0.0

        Bnm = np.sqrt(Bz**2 + By**2)
        if Bnm > 1e-7:
            t2 = By / Bnm
            t3 = Bz / Bnm
        else:
            t2 = np.sin(np.pi / 4.0)
            t3 = np.cos(np.pi / 4.0)

    if ny == 1:
        t2 = 0.0

        Bnm = np.sqrt(Bx**2 + Bz**2)
        if Bnm > 1e-7:
            t1 = Bx / Bnm
            t3 = Bz / Bnm
        else:
            t1 = np.sin(np.pi / 4.0)
            t3 = np.cos(np.pi / 4.0)

    # (b) Compute the right and left eigenvectors of the flux Jacobian matrix, ∂f/∂x, at x = x_{i+1/2,j}:
    r = rev.set_rght_eigEntropy(den, vex, vey, vez, pre, Bx, By, Bz, nx, ny, nz, t1, t2, t3)
    l = lev.set_left_eigEntropy(den, vex, vey, vez, pre, Bx, By, Bz, nx, ny, nz, t1, t2, t3)

    # (c) Project the solution and physical flux into the right eigenvector space:

    vj0 = np.dot(l, q0)
    vj1 = np.dot(l, q1)
    vj2 = np.dot(l, q2)
    vj3 = np.dot(l, q3)
    vj4 = np.dot(l, q4)
    vj5 = np.dot(l, q5)

    gj0 = np.dot(l, flux0)
    gj1 = np.dot(l, flux1)
    gj2 = np.dot(l, flux2)
    gj3 = np.dot(l, flux3)
    gj4 = np.dot(l, flux4)
    gj5 = np.dot(l, flux5)

    # (d) Perform a Lax-Friedrichs flux vector splitting for each component of the characteristic variables. 
    # Specifically, assume that the mth components of Vj and Gj are vj and gj, respectively, then compute
    # g^{±}_{j}= 0.5* (g_j ± α^{m} v_j) where α(m) = max_k | λ^{m} q_k | 
    # is the maximal wave speed of the m^{th} component of characteristic variables over all grid points

    gp0 = 0.5 * (gj0 + 1.1 * alpha * vj0)
    gp1 = 0.5 * (gj1 + 1.1 * alpha * vj1)
    gp2 = 0.5 * (gj2 + 1.1 * alpha * vj2)
    gp3 = 0.5 * (gj3 + 1.1 * alpha * vj3)
    gp4 = 0.5 * (gj4 + 1.1 * alpha * vj4)
    # gp5 = 0.5 * (gj5 + 1.1 * alpha * vj5)     # not needed for WENO

    # gm0 = 0.5 * (gj0 - 1.1 * alpha * vj0)     # not needed for WENO
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