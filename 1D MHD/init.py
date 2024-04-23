# Standard Python Libraries
import numpy as np

# User Defined Libraries
import configuration as cfg # configuration file

def initial_condition():
    '''
    Function Name:      initial_condition
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-13-2024

    Definition:         loads initial conditions for classic 1.5D Riemann Problems

    Inputs:             case: various tests defined below

    Outputs:            q_sys = ( ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z )^T

    Dependencies:       none

    '''

    q_sys = np.zeros((8,cfg.nx1))

    if cfg.case == 0:
        # Brio-Wu Shock Tube (Qi Tang)
        #             ([  ρ, u_x, u_y, u_z,   p,  B_x,  B_y, B_z])
        q_l = np.array([1.0, -0.4, 0.0, 0.0, 1.0, 0.75,  1.0, 0.0])
        q_r = np.array([0.2, -0.4, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0])

    if cfg.case == 1:
        # Brio-Wu Shock Tube
        #             ([    ρ, u_x, u_y, u_z,   p,  B_x,  B_y, B_z])
        q_l = np.array([  1.0, 0.0, 0.0, 0.0, 1.0, 0.75,  1.0, 0.0])
        q_r = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0])
    
    if cfg.case == 2:
        # Brio-Wu Shock Tube (Qi Tang)
        #             ([  ρ, u_x, u_y, u_z,   p,  B_x,  B_y, B_z])
        q_l = np.array([1.0, 0.4, 0.0, 0.0, 1.0, 0.75,  1.0, 0.0])
        q_r = np.array([0.2, 0.4, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0])
    
    # define where the Riemann problem begins on our grid
    xdiscont = 0.0

    left = (cfg.xgrid <= xdiscont) 
    right = (cfg.xgrid > xdiscont)

    # Left States
    q_sys[0,left] = q_l[0]
    q_sys[1,left] = q_l[0] * q_l[1]
    q_sys[2,left] = q_l[0] * q_l[2]
    q_sys[3,left] = q_l[0] * q_l[3]
    q_sys[4,left] = (q_l[4] / ((cfg.gamma-1.))) + 0.5 * q_l[0] * (q_l[1]**2 + q_l[2]**2 + q_l[3]**2) + 0.5 * (q_l[5]**2 + q_l[6]**2 + q_l[7]**2)
    q_sys[5,left] = q_l[5]
    q_sys[6,left] = q_l[6]
    q_sys[7,left] = q_l[7]

    # Right States
    q_sys[0,right] = q_r[0]
    q_sys[1,right] = q_r[0] * q_r[1]
    q_sys[2,right] = q_r[0] * q_r[2]
    q_sys[3,right] = q_r[0] * q_r[3]
    q_sys[4,right] = (q_r[4] / ((cfg.gamma-1.))) + 0.5 * q_r[0] * (q_r[1]**2 + q_r[2]**2 + q_r[3]**2) + 0.5 * (q_r[5]**2 + q_r[6]**2 + q_r[7]**2)
    q_sys[5,right] = q_r[5]
    q_sys[6,right] = q_r[6]
    q_sys[7,right] = q_r[7]

    return q_sys

