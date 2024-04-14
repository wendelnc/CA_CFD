'''
####################################################################################################
2D Euler Equations
Carolyn Wendeln
04/10/2024
####################################################################################################

We have the equations
∂ρ/∂t     + ∂(ρ*u)/∂x         + ∂(ρ*v)/∂y         = 0
∂(ρ*u)/∂t + ∂(ρ*u^2 +p)/∂x    + ∂(ρ*u*v)/∂y       = 0
∂(ρ*v)/∂t + ∂(ρ*u*v)/∂x       + ∂(ρ*v^2 + p)/∂y   = 0
∂(E)/∂t   + ∂(u(E + p))/∂x    + ∂(v(E + p))/∂y    = 0

These equations are the conservation of mass, momentum, and energy

Or, written slightly different we have

∂Q/∂t + ∂F/∂x + ∂G/∂y = 0

where

    |  ρ  |
Q = | ρ*u |
    | ρ*v |
    | ρ*E |

    |    ρ*u    |
F = | ρ*u^2 + p |
    |   ρ*u*v   |
    |  u*(E+p)  |

    |    ρ*v    |
G = |   ρ*u*v   |
    | ρ*v^2 + p |
    |  v*(E+p)  |

The equations are closed by the ideal gas law

p = (γ-1)(E - 0.5*ρ*(u^2+v^2))

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
import w_half as wh              # Compute w_{i+1/2} (or w_{i-1/2}
import right_eigenvectors as rev # Compute Right Eigenvectors
import left_eigenvectors as lev  # Compute Left Eigenvectors
import weno as wn                # Compute WENO Reconstruction
import lf_flux as lf             # Compute Lax-Friedrichs Flux Vector Splitting
import RHS as rhs                # Compute Right Hand Side

def main():

    start = time.time()
    
    print(f"Started at {time.strftime('%H:%M')}")

    # Initialize the Test Problem
    q_sys = ic.initial_condition()
 
    # # Add Ghost Cells
    q_sys = ght.add_ghost_cells(q_sys)

    q_sys_new = q_sys.copy()

    # Plot Initial Condition
    # eplt.plot_prim(q_sys[:,cfg.nghost:-cfg.nghost,cfg.nghost:-cfg.nghost],cfg.ti)

    t = cfg.ti

    all_solns = []
    all_t = []
    all_solns.append(q_sys[:,cfg.nghost:-cfg.nghost, cfg.nghost:-cfg.nghost])
    all_t.append(t)

    while t < (cfg.tf):
        
        # Update Boundary Conditions 
        q_sys = bc.boundary_conditions(q_sys)

        # Compute Time Step ∆t from CFL Condition
        dt, alpha_x, alpha_y = ts.time_step(q_sys,t)
        
        for i in range(cfg.nghost,cfg.nx1+cfg.nghost):
            for j in range(cfg.nghost,cfg.nx2+cfg.nghost):

                f_l = lf.lf_flux(np.array([q_sys[:,j,i-3],q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2]]),alpha_x,1,0)
                f_r = lf.lf_flux(np.array([q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2],q_sys[:,j,i+3]]),alpha_x,1,0)
                g_l = lf.lf_flux(np.array([q_sys[:,j-3,i],q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i]]),alpha_y,0,1)
                g_r = lf.lf_flux(np.array([q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i],q_sys[:,j+3,i]]),alpha_y,0,1)

                q_sys_new[:,j,i] = q_sys[:,j,i] - ((dt)/(cfg.dx))*(f_r - f_l)  -  ((dt)/(cfg.dy))*(g_r - g_l) 

        q_sys = np.copy(q_sys_new)


        # Update Time Step
        t += dt

        all_solns.append(q_sys[:,cfg.nghost:-cfg.nghost, cfg.nghost:-cfg.nghost])
        all_t.append(t)

    print(f"Finished after {time.time() - start:.5f} seconds")

    print("let's make some movies!")

    eplt.movie_maker(all_solns,all_t)
    eplt.movie_maker_den(all_solns,all_t)

    # 2D Riemann Problems
    # if cfg.case == 0 or cfg.case == 1 or cfg.case == 2 or cfg.case == 7 or cfg.case == 8:
    eplt.plot_prim(q_sys[:, cfg.nghost:-cfg.nghost, cfg.nghost:-cfg.nghost], t)
    eplt.plot_den_contour(q_sys[:, cfg.nghost:-cfg.nghost, cfg.nghost:-cfg.nghost], t)
    
    # 1D Sod Shock Tube
    # if cfg.case == 3 or cfg.case == 5: 
    #     eplt.plot_1D(q_sys[:, cfg.nx2//2, cfg.nghost:-cfg.nghost], t)
    #     eplt.plot_prim(q_sys[:, cfg.nghost:-cfg.nghost, cfg.nghost:-cfg.nghost], t)

    # elif cfg.case == 4 or cfg.case == 6:
    #     eplt.plot_1D(q_sys[:, cfg.nghost:-cfg.nghost, cfg.nx1//2], t)
    #     eplt.plot_prim(q_sys[:, cfg.nghost:-cfg.nghost, cfg.nghost:-cfg.nghost], t)


if __name__ == "__main__":
    main()
