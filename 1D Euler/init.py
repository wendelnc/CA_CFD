# Standard Python Libraries
import numpy as np

# User Defined Libraries
import configuration as cfg # configuration file

def initial_condition():
    '''
    Function Name:      initial_condition
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 03-27-2024

    Definition:         loads initial conditions for classic 1D Riemann Problems

    Inputs:             case: various tests defined below

    Outputs:            q_sys = (rho, rho*u, E)^T

    Dependencies:       none

    '''

    q_sys = np.zeros((3,cfg.nx1))

    # q_l and q_r are arrays of
    # density, velocity, and pressure

    if cfg.case == 0:
        # Sod Shock Tube
        q_l = np.array([1.0  , 0.0, 1.0])
        q_r = np.array([0.125, 0.0, 0.1])

    elif cfg.case == 1:
        # Mach = 3 test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997
        q_l = np.array([3.857, 0.92, 10.333])
        q_r = np.array([1.0,   3.55, 1.0])

    elif cfg.case == 2:
        # Shocktube problem of G.A. Sod, JCP 27:1, 1978
        q_l = np.array([1.0,   0.75, 1.0])
        q_r = np.array([0.125,  0.0, 0.1])

    elif cfg.case == 3:
        # Lax test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997
        q_l = np.array([0.445, 0.698, 3.528])
        q_r = np.array([0.5,     0.0, 0.571])

    elif cfg.case == 4:
        # Shocktube problem with supersonic zone
        q_l = np.array([1.0,  0.0,  1.0])
        q_r = np.array([0.02, 0.0, 0.02])

    elif cfg.case == 5:
        # Stationary shock
        q_l = np.array([1.0,   -2.0, 1.0])
        q_r = np.array([0.125, -2.0, 0.1])

    elif cfg.case == 6:
        # Double Expansion
        q_l = np.array([1.0, -2.0, 40.0])
        q_r = np.array([2.5,  2.0, 40.0])

    elif cfg.case == 7:
        # Reverse Sod Shock Tube
        q_l = np.array([0.125, 0.0, 0.1])
        q_r = np.array([1.0  , 0.0, 1.0])

    # define where the Riemann problem begins on our grid
    xdiscont = 0.5

    left = (cfg.grid < xdiscont) 
    right = (cfg.grid >= xdiscont)

    # Left States
    q_sys[0,left] = q_l[0]
    q_sys[1,left] = q_l[0] * q_l[1]
    q_sys[2,left] = 0.5 * q_l[0] * q_l[1]**2. + (q_l[2] / ((cfg.gamma-1.)))

    # Right States
    q_sys[0,right]  = q_r[0]
    q_sys[1,right] = q_r[0] * q_r[1]
    q_sys[2,right] = 0.5 * q_r[0] * q_r[1]**2. + (q_r[2] / ((cfg.gamma-1.)))

    return q_sys

