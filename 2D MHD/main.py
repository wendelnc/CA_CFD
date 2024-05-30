'''
####################################################################################################
2D Magnetohydrodynamic Equations
Carolyn Wendeln
04/18/2024
####################################################################################################

The 2D Magnetohydrodynamic Equations are given by

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

∂q/∂t + ∂F/∂x + ∂G/∂y + = 0

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
F = |        ρu_xu_z - B_xB_z          |
    | u_x(E + p + 0.5*∥B∥^2) - B_x(u·B) |
    |               0                  |
    |        u_xB_y - u_yB_x           |
    |        u_xB_z - u_zB_x           |

    |              ρu_y                |
    |        ρu_yu_x - B_yB_x          |
    | ρu_yu_y + p + 0.5*∥B∥^2 - B_yB_y  |
G = |        ρu_yu_z - B_yB_z          |
    | u_y(E + p + 0.5*∥B∥^2) - B_y(u·B) |
    |        u_yB_x - u_xB_y           |
    |               0                  |
    |        u_yB_z - u_zB_y           |

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
import time_step as ts            # Compute Time Step
import cons2prim as c2p           # Convert Conserved to Primitive Variables
import get_flux as gf             # Compute Flux
import w_half as wh               # Compute w_{i+1/2} (or w_{i-1/2})
import set_rght_eigEntropy as rev # Compute Right Eigenvectors
import set_rght_eigEntropy as lev # Compute Left Eigenvectors
import lf_flux as lf              # Compute Lax-Friedrichs Flux Vector Splitting

def main():

    start = time.time()

    # Initialize the Test Problem
    q_sys = ic.initial_condition()
 
    # Add Ghost Cells
    q_sys = ght.add_ghost_cells(q_sys)

    q_sys_new = q_sys.copy()

    # Plot Initial Condition
    # eplt.plot_all_prim(q_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost],cfg.ti)
    eplt.plot_1D(q_sys[:, cfg.nx2//2, cfg.nghost:-cfg.nghost],cfg.ti)
    # eplt.plot_all_cons(q_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost],cfg.ti)
    
    t = cfg.ti

    # All Solutions for Movie Making
    # all_solns = []
    # all_t = []
    # all_solns.append(q_sys[:,cfg.nghost:-cfg.nghost])
    # all_t.append(t)

    while t < (cfg.tf):

        # Update Boundary Conditions 
        q_sys = bc.boundary_conditions(q_sys)

        # Compute Time Step ∆t from CFL Condition
        dt, alphax, alphay = ts.time_step(q_sys,t)

        for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
            for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
                f_l = lf.lf_flux(np.array([q_sys[:,j,i-3],q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2]]),alphax,1,0,0)
                f_r = lf.lf_flux(np.array([q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2],q_sys[:,j,i+3]]),alphax,1,0,0)
                g_l = lf.lf_flux(np.array([q_sys[:,j-3,i],q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i]]),alphay,0,1,0)
                g_r = lf.lf_flux(np.array([q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i],q_sys[:,j+3,i]]),alphay,0,1,0)

                q_sys_new[:,j,i] = q_sys[:,j,i] - ((dt)/(cfg.dx))*(f_r - f_l)  -  ((dt)/(cfg.dy))*(g_r - g_l) 

        q_sys = np.copy(q_sys_new)

        # Update Time Step
        t += dt

        # All Solutions for Movie Making
        # all_solns.append(q_sys[:,cfg.nghost:-cfg.nghost])
        # all_t.append(t)


    print(f"Finished after {time.time() - start:.5f} seconds")
    
    # eplt.plot_soln(q_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost],t)
    eplt.plot_1D(q_sys[:, cfg.nx2//2, cfg.nghost:-cfg.nghost],cfg.ti)

    # print("let's make some movies!")
    # eplt.movie_maker_all_prim(all_solns,all_t)
    # eplt.movie_maker_all_cons(all_solns,all_t)

if __name__ == "__main__":
    main()
