'''
####################################################################################################
1D Magnetohydrodynamic Equations
Carolyn Wendeln
04/13/2024
####################################################################################################

The 1D Magnetohydrodynamic Equations are given by

∂(ρ)/∂t     + ∇·(ρu)                                = 0
∂(ρu)/∂t    + ∇·(ρu⊗u + (p + 0.5*∥B∥^2 )I - B⊗B )    = 0
∂(B)/∂t     + ∇·( u⊗B - B⊗u )                       = 0
∂(E)/∂t     + ∇·( u(E + p + 0.5*∥B∥^2 ) - B(u·B))    = 0
 
with the additional requirement that ∇·B = 0

ρ is the density
u is the velocity vector (u_x,u_y,u_z)
B is the magnetic field vector (B_x,B_y,B_z)
E is the total energy
p is the  pressure
I is the identity matrix
E is the total energy = (p / (γ-1)) + 0.5ρ*∥u∥^2 + 0.5*∥B∥^2
γ is the adiabatic index = 5/3 
∥·∥ is the Euclidean vector norm

Written slightly different we have

∂q/∂t + ∇·f = 0

where the conserved variables q are

    | ρ    |
    | ρu_x |
    | ρu_y |
q = | ρu_z |
    | E    |
    | B_x  |
    | B_y  |
    | B_z  |

    |              ρu_x                |
    | ρu_xu_x + p + 0.5*∥B∥^2 - B_xB_x  |
    |        ρu_xu_y - B_xB_y          |
f = |        ρu_xu_z - B_xB_z          |
    | u_x(E + p + 0.5*∥B∥^2) - B_x(u·B) |
    |               0                  |
    |        u_xB_y - u_yB_x           |
    |        u_xB_z - u_zB_x           |

For convenience, we also introduce the primative variables w

    | ρ   |
    | u_x |
    | u_y |
w = | u_z |
    | p   |
    | B_x |
    | B_y |
    | B_z |

I am solving this system of equations using a Finite Difference Lax-Friedrichs Flux Vector Splitting method
Following Procedure 2.10 from the following paper:
"Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws"
By: Chi-Wang Shu
https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf

'''

# Standard Python Libraries
import time
import numpy as np

# User Defined Libraries (not all needed, but all defined here)
import configuration as cfg       # Input Parameters
import init as ic                 # Initialize Test Problem
import plotting as eplt           # Plotting Solution
import ghost as ght               # Add Ghost Cells
import boundary_conditions as bc  # Update Boundary Conditions
import eigenvalues as ev          # Compute Eigenvalues
import time_step as ts            # Compute Time Step
import cons2prim as c2p           # Convert Conserved to Primitive Variables
import get_flux as gf             # Compute Flux
import w_half as wh               # Compute w_{i+1/2} (or w_{i-1/2})
import set_rght_eigEntropy as rev # Compute Right Eigenvectors
import set_rght_eigEntropy as lev # Compute Left Eigenvectors
import lf_flux as lf              # Compute Lax-Friedrichs Flux Vector Splitting
import ssp_rk as rk               # Strong Stability Preserving Runge-Kutta (SSP-RK3 and SSP-RK4)

def main():

    start = time.time()

    # Initialize the Test Problem
    q_sys = ic.initial_condition()

    # Add Ghost Cells
    q_sys = ght.add_ghost_cells(q_sys)
    q_new = q_sys.copy()

    # Plot Initial Condition
    # eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],cfg.ti)

    t = cfg.ti

    # All Solutions for Movie Making
    all_solns = []
    all_t = []
    all_solns.append(q_sys[:,cfg.nghost:-cfg.nghost])
    all_t.append(t)

    while t < (cfg.tf):
        
        # Update Boundary Conditions 
        q_sys = bc.boundary_conditions(q_sys)
        
        # Compute Time Step ∆t from CFL Condition
        dt = ts.time_step(q_sys,t)

        # SSP-RK4 Time Integration Scheme
        q_sys = rk.rk4(q_sys, dt)
         
        # Update Time Step
        t += dt

        # All Solutions for Movie Making
        all_solns.append(q_sys[:,cfg.nghost:-cfg.nghost])
        all_t.append(t)


    print(f"Finished after {time.time() - start:.5f} seconds")

    eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],t)

    # rho, vex, vey, vez, pre, Bx, By, Bz = c2p.cons2prim(q_sys[:, cfg.nghost:-cfg.nghost])
    # print("den = np.array({})".format(rho.tolist()))
    # print("vex = np.array({})".format(vex.tolist()))
    # print("vey = np.array({})".format(vey.tolist()))
    # print("vez = np.array({})".format(vez.tolist()))
    # print("pre = np.array({})".format(pre.tolist()))
    # print("Bx = np.array({})".format(Bx.tolist()))
    # print("By = np.array({})".format(By.tolist()))
    # print("Bz = np.array({})".format(Bz.tolist()))

    print("let's make a movie!")
    eplt.movie_maker(all_solns,all_t)

if __name__ == "__main__":
    main()
