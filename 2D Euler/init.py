# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

def initial_condition():
    '''
    Function Name:      initial_condition
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 03-24-2024

    Definition:         loads initial conditions for classic 2D Riemann Problems

    Inputs:             case: various tests defined below

    Outputs:            q_sys = (rho, rho*u, rho*v, E)^T

    Dependencies:       none

    '''
    
    q_sys = np.zeros((4,cfg.nx1,cfg.nx2))

    if cfg.case == 0:
        # Setup corresponding to configuration 3 of the paper
        # Carsten W. Schulz-Rinne, James P. Collins, and Harland M. Glaz.
        # Numerical solution of the riemann problem for two-dimensional gas dynamics.
        # SIAM Journal on Scientific Computing, 14(6):1394-1414, 1993.
        # Top Right, Top Left, Bottom Left, Bottom Right
        rho = [1.5, 0.5323, 0.138, 0.5323]
        u   = [0.0,  1.206, 1.206, 0.0   ]
        v   = [0.0,    0.0, 1.206, 1.206 ]
        p   = [1.5,    0.3, 0.029, 0.3   ]

    if cfg.case == 1:
        # Fig 8. from the following paper
        # https://hal.science/hal-02100764/document
        rho = [  1.0,  2.0,   1.0,  3.0 ]
        u   = [ 0.75, 0.75, -0.75, -0.75]
        v   = [ -0.5,  0.5,   0.5, -0.5 ]
        p   = [  1.0,  1.0,   1.0,  1.0 ]

    if cfg.case == 2:
        # Fig 8. from the following paper
        # https://hal.science/hal-02100764/document
        rho = [  1.0,   2.0,  1.0,  3.0 ]
        u   = [-0.75, -0.75, 0.75,  0.75]
        v   = [ -0.5,   0.5,  0.5, -0.5 ]
        p   = [  1.0,   1.0,  1.0,  1.0 ]

    if cfg.case ==3:
        # 1D Sod Shock Tube
        rho = [  0.125,   1.0,  1.0,  0.125 ]
        u   = [0,0,0,0]
        v   = [ 0,0,0,0 ]
        p   = [  0.1,   1.0,  1.0,  0.1 ] 

    if cfg.case ==4:
        # Rotated 1D Sod Shock Tube
        rho = [  1.0,   1.0,  0.125,  0.125 ]
        u   = [0,0,0,0]
        v   = [ 0,0,0,0 ]
        p   = [  1.0,   1.0,  0.1,  0.1 ] 

    if cfg.case == 5:
        # Reverse 1D Sod Shock Tube
        rho = [  1.0,   0.125,  0.125,  1.0 ]
        u   = [0,0,0,0]
        v   = [ 0,0,0,0 ]
        p   = [  1.0,   0.1,  0.1,  1.0 ] 
    
    if cfg.case == 6:
        # Rotated Reverse 1D Sod Shock Tube
        rho = [  0.125,   0.125,  1.0,  1.0 ]
        u   = [0,0,0,0]
        v   = [ 0,0,0,0 ]
        p   = [  0.1,   0.1,  1.0,  1.0 ] 

    if cfg.case == 7:
        # Reverse Carsten W. Schulz-Rinne 2D Riemann Problem
        # Top Right, Top Left, Bottom Left, Bottom Right
        rho = [0.138, 0.5323, 1.5, 0.5323]
        u   = [1.206,  1.206, 0.0, 0.0   ]
        v   = [1.206,    0.0, 0.0, 1.206 ]
        p   = [0.029,    0.3, 1.5, 0.3   ]        

    # Define the discontinuity location
    xdist = 0.5
    ydist = 0.5
    
    # Calculate the quadrant indices for each point
    top_right = (cfg.X > xdist) & (cfg.Y > ydist)
    top_left = (cfg.X < xdist) & (cfg.Y > ydist)
    bottom_left = (cfg.X < xdist) & (cfg.Y < ydist)
    bottom_right = (cfg.X > xdist) & (cfg.Y < ydist)

    # Assign values based on quadrant indices
    q_sys[0, top_right] = rho[0]
    q_sys[1, top_right] = rho[0] * u[0]
    q_sys[2, top_right] = rho[0] * v[0]
    q_sys[3, top_right] = (p[0] / (cfg.gamma - 1.)) + (0.5 * rho[0] * (u[0]**2. + v[0]**2.))

    q_sys[0, top_left] = rho[1]
    q_sys[1, top_left] = rho[1] * u[1]
    q_sys[2, top_left] = rho[1] * v[1]
    q_sys[3, top_left] = (p[1] / (cfg.gamma - 1.)) + (0.5 * rho[1] * (u[1]**2. + v[1]**2.))

    q_sys[0, bottom_left] = rho[2]
    q_sys[1, bottom_left] = rho[2] * u[2]
    q_sys[2, bottom_left] = rho[2] * v[2]
    q_sys[3, bottom_left] = (p[2] / (cfg.gamma - 1.)) + (0.5 * rho[2] * (u[2]**2. + v[2]**2.))

    q_sys[0, bottom_right] = rho[3]
    q_sys[1, bottom_right] = rho[3] * u[3]
    q_sys[2, bottom_right] = rho[3] * v[3]
    q_sys[3, bottom_right] = (p[3] / (cfg.gamma - 1.)) + (0.5 * rho[3] * (u[3]**2. + v[3]**2.))    

    return q_sys
