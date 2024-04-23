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
import ssp_rk as rk              # Strong Stability Preserving Runge-Kutta (SSP-RK3 and SSP-RK4)

def main():

    start = time.time()

    # Initialize the Test Problem
    q_sys = ic.initial_condition()
 
    # Add Ghost Cells
    q_sys = ght.add_ghost_cells(q_sys,cfg.nx1,cfg.nghost)

    # Plot Initial Condition
    # eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],cfg.ti)

    t = cfg.ti

    while t < (cfg.tf):
        
        # Update Boundary Conditions 
        q_sys = bc.boundary_conditions(q_sys,cfg.nghost,cfg.bndc)

        # Compute Time Step ∆t from CFL Condition
        dt, alpha = ts.time_step(q_sys,t)

        # SSP-RK4 Time Integration Scheme
        q_sys = rk.rk4(q_sys, alpha, dt)

        # Update Time Step
        t += dt

    print(f"Finished after {time.time() - start:.5f} seconds")

    eplt.plot_solution(q_sys[:, cfg.nghost:-cfg.nghost],t)


if __name__ == "__main__":
    main()
