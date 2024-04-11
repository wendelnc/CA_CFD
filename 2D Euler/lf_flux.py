# Standard Python Libraries
import numpy as np

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables
import get_flux as gf            # Compute Flux
import w_half as wh              # Compute w_{i+1/2} (or w_{i-1/2}
import right_eigenvectors as rev # Compute Right Eigenvectors
import left_eigenvectors as lev  # Compute Left Eigenvectors
import weno as wn                # Compute WENO Reconstruction

def lf_flux(q_arr,nx,ny):

                  # x_{i+1/2}   # x_{i-1/2}
                  #----------   #----------
    q0 = q_arr[0] # cell i-2    # cell i-3
    q1 = q_arr[1] # cell i-1    # cell i-2
    q2 = q_arr[2] # cell i      # cell i-1
    q3 = q_arr[3] # cell i+1    # cell i
    q4 = q_arr[4] # cell i+2    # cell i+1
    q5 = q_arr[5] # cell i+3    # cell i+2

    # 1. Compute the physical flux at each grid point:
    flux0 = gf.get_flux(q0,nx,ny) 
    flux1 = gf.get_flux(q1,nx,ny) 
    flux2 = gf.get_flux(q2,nx,ny) 
    flux3 = gf.get_flux(q3,nx,ny) 
    flux4 = gf.get_flux(q4,nx,ny) 
    flux5 = gf.get_flux(q5,nx,ny) 

    # 2. At each x_{i+1,j}:
    # (a) Compute the average state w_{i+1/2} in the primitive variables:
    den, vex, vey, pre, E, H, a = wh.w_half(q2,q3) 
  
    # (b) Compute the right and left eigenvectors of the flux Jacobian matrix, ∂f/∂x, at x = x_{i+1/2,j}:
    # if nx == 1.0 and ny == 0.0:
    #     r = f_Right(vex,vey,a,H)
    #     l = f_Left(vex,vey,a,H)
    # else:
    #     r = g_Right(vex,vey,a,H)
    #     l = g_Left(vex,vey,a,H)

    r = rev.right_eigenvectors(vex,vey,H,a,nx,ny)
    l = lev.left_eigenvectors(vex,vey,H,a,nx,ny)

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

    # First we need α(m)
    # Compute the primitive variables at each grid point: 
    rho0, u0, v0, p0 = c2p.cons2prim(q0)
    rho1, u1, v1, p1 = c2p.cons2prim(q1)
    rho2, u2, v2, p2 = c2p.cons2prim(q2)
    rho3, u3, v3, p3 = c2p.cons2prim(q3)
    rho4, u4, v4, p4 = c2p.cons2prim(q4)
    rho5, u5, v5, p5 = c2p.cons2prim(q5)

    # Sound speed at each grid point:
    a0 = np.sqrt(cfg.gamma * p0 / rho0)
    a1 = np.sqrt(cfg.gamma * p1 / rho1)
    a2 = np.sqrt(cfg.gamma * p2 / rho2)
    a3 = np.sqrt(cfg.gamma * p3 / rho3)
    a4 = np.sqrt(cfg.gamma * p4 / rho4)
    a5 = np.sqrt(cfg.gamma * p5 / rho5)

    # Normalized velocity at each grid point:
    vn0 = u0 * nx + v0 * ny
    vn1 = u1 * nx + v1 * ny
    vn2 = u2 * nx + v2 * ny
    vn3 = u3 * nx + v3 * ny
    vn4 = u4 * nx + v4 * ny
    vn5 = u5 * nx + v5 * ny

    # α^{m} at each grid point:
    alpha0 = np.array([np.abs(vn0 - a0), np.abs(vn0), np.abs(vn0), np.abs(vn0 + a0)])
    alpha1 = np.array([np.abs(vn1 - a1), np.abs(vn1), np.abs(vn1), np.abs(vn1 + a1)])
    alpha2 = np.array([np.abs(vn2 - a2), np.abs(vn2), np.abs(vn2), np.abs(vn2 + a2)])
    alpha3 = np.array([np.abs(vn3 - a3), np.abs(vn3), np.abs(vn3), np.abs(vn3 + a3)])
    alpha4 = np.array([np.abs(vn4 - a4), np.abs(vn4), np.abs(vn4), np.abs(vn4 + a4)])
    alpha5 = np.array([np.abs(vn5 - a5), np.abs(vn5), np.abs(vn5), np.abs(vn5 + a5)])

    # g^{+}_{j}= 0.5* (g_j + α^{m} v_j)
    gp0 = 0.5 * (gj0 + 1.1 * alpha0 * vj0)
    gp1 = 0.5 * (gj1 + 1.1 * alpha1 * vj1)
    gp2 = 0.5 * (gj2 + 1.1 * alpha2 * vj2)
    gp3 = 0.5 * (gj3 + 1.1 * alpha3 * vj3)
    gp4 = 0.5 * (gj4 + 1.1 * alpha4 * vj4)
    gp5 = 0.5 * (gj5 + 1.1 * alpha5 * vj5)

    gp = np.array([gp0,gp1,gp2,gp3,gp4,gp5])

    # g^{-}_{j}= 0.5* (g_j - α^{m} v_j)
    gm0 = 0.5 * (gj0 - 1.1 * alpha0 * vj0)
    gm1 = 0.5 * (gj1 - 1.1 * alpha1 * vj1)
    gm2 = 0.5 * (gj2 - 1.1 * alpha2 * vj2)
    gm3 = 0.5 * (gj3 - 1.1 * alpha3 * vj3)
    gm4 = 0.5 * (gj4 - 1.1 * alpha4 * vj4)
    gm5 = 0.5 * (gj5 - 1.1 * alpha5 * vj5)

    gm = np.array([gm0,gm1,gm2,gm3,gm4,gm5])

    # (e) Perform a WENO reconstruction on each of the computed flux components gj± to obtain 
    # the corresponding component of the numerical flux

    g_half = wn.weno(gp[0:5],gm[1:6])

    # (f) Project the numerical flux back to the conserved variables

    f_half = np.dot(r, g_half)

    return f_half