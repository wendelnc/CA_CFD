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

∂q/∂t + ∂F(q)/∂x + ∂G(q)/∂y = 0

where the conserved variables q are

       | ρ    |
       | ρu_x |
       | ρu_y |
q =    | ρu_z |
       | E    |
       | B_x  |
       | B_y  |
       | B_z  |

       |              ρu_x                |
       | ρu_xu_x + p + 0.5*∥B∥^2 - B_xB_x  |
       |        ρu_xu_y - B_xB_y          |
F(q) = |        ρu_xu_z - B_xB_z          |
       | u_x(E + p + 0.5*∥B∥^2) - B_x(u·B) |
       |               0                  |
       |        u_xB_y - u_yB_x           |
       |        u_xB_z - u_zB_x           |

       |              ρu_y                |
       |        ρu_yu_x - B_yB_x          |
       | ρu_yu_y + p + 0.5*∥B∥^2 - B_yB_y  |
G(q) = |        ρu_yu_z - B_yB_z          |
       | u_y(E + p + 0.5*∥B∥^2) - B_y(u·B) |
       |        u_yB_x - u_xB_y           |
       |               0                  |
       |        u_yB_z - u_zB_y           |

For convenience, we also introduce the primative variables w

       | ρ   |
       | u_x |
       | u_y |
w =    | u_z |
       | p   |
       | B_x |
       | B_y |
       | B_z |

I am solving this system of equations using a Finite Difference Lax-Friedrichs Flux Vector Splitting method
Following Procedure 2.10 from the following paper:
"Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws"
By: Chi-Wang Shu
https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf

To satisfy the ∇·B = 0 constraint, I am using an unstaggered constraint transport method from the following paper:
"An Unstaggered Constrained Transport Method for the 3D Ideal Magnetohydrodynamic Equations"
By: Christiane Helzel, James A. Rossmanith, and Bertram Taetz
https://arxiv.org/abs/1007.2606

To solve the magnetic potential, I am using a Hamilton-Jacobi Lax-Friedrichs Flux Vector Splitting method
seen in the following paper:
"Weighted ENO Schemes for Hamilton-Jacobi Equations"
By: Guang-Shan Jiang and Danping Peng
https://epubs.siam.org/doi/10.1137/S106482759732455X

A single time step of the CT method consists of the following steps:

    0) Start with Q_{MHD}^{n} and Q_{A}^{n}

    1) Build the right hand sides of both semi-discrete systems

        Q_{MHD}^{*} = Q_{MHD}^{n} + ∆t \mathcal{L}(Q_{MHD}^{n})
        Q_{A}^{n+1} = Q_{A}^{n}   + ∆t \mathcal{H}(Q_{A}^{n}, \textbf{u}^{n})

        where Q_{MHD}^{*} = (ρ^{n+1}, ρ\textbf{u}^{n+1}, E^{*}, B^{*})

        B^{*} means the predicted magnetic field without satisfying the divergence free constraint
        E^{*} will be updated based on the option in Step 3
        
    2) Correct B^{*} by the magnetic potential Q_{A}^{n+1} at new stage by a discrete curl operator

        B^{n+1} = \nabla \times Q_{A}^{n+1}

    3) Set the total energy density E^{n+1} based on the following options

        Option 1: Keep the total energy conserved
        E^{n+1} = E^{*}

        Option 2: Keep the pressure the same after the correction of magnetic field
        E^{n+1} = E^{*} + 0.5(∥B^{n+1}∥^2 - ∥B^{n}∥^2)

        Option 1 gives us energy conservation
        Option 2 helps preserve the positivity of the pressure for low pressure problems

        In this code we use Option 1
'''

# Standard Python Libraries
import time
import numpy as np
import matplotlib.pyplot as plt

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
import hj_flux as hj              # Compute Hamilton-Jacobi Lax-Friedrichs Flux Vector Splitting
import ssp_rk as rk               # Strong Stability Preserving Runge-Kutta (SSP-RK3 and SSP-RK4)
import check_divergence as cd     # Check Divergence
import save as sv                 # Save Data

def main():

    start = time.time()
    print(f"Started at {time.strftime('%H:%M')}")

    # Initialize the Test Problem
    q_sys, a_sys = ic.initial_condition()

    t = cfg.ti

    nt = 0  # Initialize the time step counter

    ################################################################################################
    # Saving Results
    ################################################################################################
   
    # All Solutions for Movie Making
    all_q_sys = []
    all_a_sys = []
    all_t = []
    all_div = []

    all_q_sys.append(q_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost])
    all_a_sys.append(a_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost])
    all_t.append(t)

    q_sys = bc.periodic_bc(q_sys)
    a_sys = bc.ct_periodic_bc(a_sys)
    
    div = cd.check_divergence(q_sys)
    all_div.append(div[cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost])

    sv.save_q_sys(q_sys,t)
    sv.save_a_sys(a_sys,t)
    sv.save_div(div,t)

    ################################################################################################
    # Time Integration
    ################################################################################################

    while t < (cfg.tf):

        # eplt.plot_all_prim(q_sys,t)

        # Update Boundary Conditions 
        q_sys = bc.periodic_bc(q_sys)
        a_sys = bc.ct_periodic_bc(a_sys)

        # Compute Time Step ∆t from CFL Condition
        dt = ts.time_step(q_sys,t)

        # Check Divergence ∇·B = 0
        div = cd.check_divergence(q_sys)
        
        q_sys, a_sys = rk.ct_rk3(q_sys, a_sys, dt)

        # Update Time Step
        t += dt
        
        # All Solutions for Movie Making
        all_q_sys.append(q_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost])
        all_a_sys.append(a_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost])
        all_t.append(t)
        all_div.append(div[cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost])

        nt += 1  # Increment the time step counter

        # Print every 20 time steps
        if nt % 20 == 0:
            print(f"Step: {nt}, Time: {t:.2f}")


    print(f"Finished after {time.time() - start:.5f} seconds")
    
    sv.save_q_sys(q_sys,t)
    sv.save_a_sys(a_sys,t)
    sv.save_div(div,t)

    # Convert lists to arrays
    all_q_sys_array = np.array(all_q_sys)
    all_a_sys_array = np.array(all_a_sys)
    all_t_array = np.array(all_t)
    all_div_array = np.array(all_div)

    # Save arrays to a compressed .npz file
    filename = f'simulation_data/result_{cfg.nx1}_by_{cfg.nx2}.npz'
    np.savez(filename, all_q_sys=all_q_sys_array, all_a_sys=all_a_sys_array, all_t=all_t_array, all_div=all_div_array)
    print('Done!')

if __name__ == "__main__":
    main()
