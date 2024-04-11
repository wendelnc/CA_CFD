'''
####################################################################################################
1D Euler Equations
Carolyn Wendeln
04/09/2024
####################################################################################################

We have the equations
∂rho/∂t     + ∂(rho*u)/∂x      = 0
∂(rho*u)/∂t + ∂(rho*u*u +p)/∂x = 0
∂(E)/∂t     + ∂(u(E + p))/∂x   = 0

These equations are the conservation of mass, momentum, and energy

Or, written slightly different we have

∂q/∂t + ∇·f = 0

where

    |  rho  |
q = | rho*u |
    |   E   |

    |    rho*u    |
f = | rho*u^2 + p |
    |   u*(E+p)   |

The equations are closed by the ideal gas law

p = (γ-1)(E - 0.5*rho*u^2)

'''

# Standard Python Libraries
import time
import numpy as np

# User Defined Libraries
import configuration as cfg      # Input Parameters
import init as ic                # Initialize Test Problem
import plotting as eplt          # Plotting Solution
import ghost as ght              # Add Ghost Cells
import boundary_conditions as bc # Update Boundary Conditions
import time_step as ts           # Compute Time Step
import cons2prim as c2p          # Convert Conserved to Primitive Variables
import get_flux as gf            # Compute Flux
import w_half as wh              # Compute w_{i+1/2} (or w_{i-1/2})
import eigenvectors as ev        # Compute Right and Left Eigenvectors
import weno as wn                # Compute WENO Reconstruction
import lf_flux as lf             # Compute Lax-Friedrichs Flux Vector Splitting
import RHS as rhs                # Compute Right Hand Side

def main():

    start = time.time()

    # Initialize the Test Problem
    q_sys = ic.initial_condition()
 
    # Add Ghost Cells
    q_sys = ght.add_ghost_cells(q_sys,cfg.nx1,cfg.nghost)

    q_sys_new = q_sys.copy()

    # Plot Initial Condition
    # eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],cfg.ti)

    t = cfg.ti

    while t < (cfg.tf):
        
        # Update Boundary Conditions 
        q_sys = bc.boundary_conditions(q_sys,cfg.nghost,cfg.bndc)

        # Compute Time Step ∆t from CFL Condition
        dt, alpha = ts.time_step(q_sys,t)

        for i in range(cfg.nghost,cfg.nx1+cfg.nghost):

            # Compute Lax-Friedrichs Flux Vector Splitting
            f_l = lf.lf_flux(np.array([q_sys[:,i-3],q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2]]),alpha)
            f_r = lf.lf_flux(np.array([q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2],q_sys[:,i+3]]),alpha)
        
            # Update Solution
            q_sys_new[:,i] = q_sys[:,i] - ((dt)/(cfg.dx))*(f_r - f_l)  

        # q_sys_new =  rhs.RHS(q_sys, q_sys_new, alpha, dt)
        q_sys = np.copy(q_sys_new)

        # Update Time Step
        t += dt

    print(f"Finished after {time.time() - start:.5f} seconds")

    eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],cfg.ti)





if __name__ == "__main__":
    main()
