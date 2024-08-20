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
import os
import time
import numpy as np

# User Defined Libraries (not all needed, but all defined here)
import boundary_conditions as bc  # Update Boundary Conditions
import configuration as cfg       # Input Parameters
import cons2prim as c2p           # Convert Conserved to Primitive Variables
import eigenvalues as ev          # Compute Eigenvalues
import get_flux as gf             # Compute Flux
import ghost as ght               # Add Ghost Cells
import init as ic                 # Initialize Test Problem
import lf_flux as lf              # Compute Lax-Friedrichs Flux Vector Splitting
import plot_all as pa             # Plots Saved .npz Files in Results Folder
import plotting as eplt           # Plotting Solution During Simulation
import save as sv                 # Save Data
import set_rght_eigEntropy as rev # Compute Right Eigenvectors
import set_rght_eigEntropy as lev # Compute Left Eigenvectors
import ssp_rk as rk               # Strong Stability Preserving Runge-Kutta (SSP-RK3 and SSP-RK4)
import time_step as ts            # Compute Time Step
import w_half as wh               # Compute w_{i+1/2} (or w_{i-1/2})
import weno as wn                 # Compute WENO Reconstruction (old version)
import weno5 as wn5               # Compute WENO Reconstruction

def main():

    start = time.time()

    # Initialize the Test Problem
    q_sys = ic.initial_condition()

    t = cfg.ti

    nt = 0  # Initialize the time step counter

    ################################################################################################
    # Saving Results
    ################################################################################################

    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join('results', 'animations'), exist_ok=True) 
    os.makedirs(os.path.join('results', 'simulation_data'), exist_ok=True)
    os.makedirs(os.path.join('results', 'txt_files'), exist_ok=True)

    # All solutions for making movies and saving entire simulations
    all_q_sys = []
    all_t = []
    all_q_sys.append(q_sys[:,cfg.nghost:-cfg.nghost])
    all_t.append(t)
    
    # Save initial condition as text files in results/txt_files folder
    sv.save_q_sys(q_sys,t)

    ################################################################################################
    # Time Integration
    ################################################################################################

    # Plot the Initial Condition
    eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],t)

    while t < (cfg.tf):
        
        # Update Boundary Conditions 
        q_sys = bc.boundary_conditions(q_sys)
        
        # Compute Time Step ∆t from CFL Condition
        dt = ts.time_step(q_sys,t)

        # SSP-RK Time Integration Scheme
        q_sys = rk.rk3(q_sys, dt)
         
        # Update Time Step
        t += dt

        # All solutions for making movies and saving entire simulations
        all_q_sys.append(q_sys[:,cfg.nghost:-cfg.nghost])
        all_t.append(t)

        nt += 1  # Increment the time step counter

        # Print every 20 time steps
        if nt % 20 == 0:
            print(f"Step: {nt}, Time: {t:.2f}")


    print(f"Finished after {time.time() - start:.5f} seconds")

    # Plot the Final Solution
    eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],t)

    # Save final condition as text files in results/txt_files folder
    sv.save_q_sys(q_sys,t)

    # Convert lists to arrays
    all_q_sys_array = np.array(all_q_sys)
    all_t_array = np.array(all_t)

    # Save arrays to a compressed .npz file
    filename = f'results/simulation_data/result_{cfg.nx1}.npz'
    np.savez(filename, all_q_sys=all_q_sys_array, all_t=all_t_array)
    print('Simulation Saved')
    print('Done!')

if __name__ == "__main__":
    main()
