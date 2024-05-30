# Standard Python Libraries
import numpy as np

# User Defined Libraries
import configuration as cfg # configuration file

def initial_condition():
    '''
    Function Name:      initial_condition
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-18-2024

    Definition:         loads initial conditions for 2D MHD Problems

    Inputs:             case: various tests defined below

    Outputs:            q_sys = ( ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z )^T

    Dependencies:       none

    '''

    q_sys = np.zeros((8,cfg.nx2,cfg.nx1))

    # Orszag-Tang Vortex
    if cfg.case == 0:

        rho = (cfg.gamma)**2 / (4 * np.pi) 
        pre =  cfg.gamma     / (4 * np.pi)

        vex = - np.sin(2 * np.pi * cfg.Y) 
        vey =   np.sin(2 * np.pi * cfg.X)
        vez = 0

        B0 = 1 / np.sqrt(4 * np.pi)
        Bx = - B0 * np.sin(2 * np.pi * cfg.Y)
        By =   B0 * np.sin(4 * np.pi * cfg.X)
        Bz = 0

        q_sys[0,:,:] = rho
        q_sys[1,:,:] = rho * vex
        q_sys[2,:,:] = rho * vey
        q_sys[3,:,:] = rho * vez
        q_sys[4,:,:] = (pre / ((cfg.gamma-1.))) + 0.5 * rho * (vex**2 + vey**2 + vez**2) + 0.5 * (Bx**2 + By**2 + Bz**2)
        q_sys[5,:,:] = Bx
        q_sys[6,:,:] = By
        q_sys[7,:,:] = Bz

    # Orszag-Tang Vortex (Qi)
    if cfg.case == 1:

        rho = (cfg.gamma)**2 
        pre =  cfg.gamma     

        vex = - np.sin(cfg.Y) 
        vey =   np.sin(cfg.X)
        vez = 0

        Bx = - np.sin(cfg.Y) + 2
        By =   np.sin(2 * cfg.X) + 2
        Bz = 0

        q_sys[0,:,:] = rho
        q_sys[1,:,:] = rho * vex
        q_sys[2,:,:] = rho * vey
        q_sys[3,:,:] = rho * vez
        q_sys[4,:,:] = (pre / ((cfg.gamma-1.))) + 0.5 * rho * (vex**2 + vey**2 + vez**2) + 0.5 * (Bx**2 + By**2 + Bz**2)
        q_sys[5,:,:] = Bx
        q_sys[6,:,:] = By
        q_sys[7,:,:] = Bz

    # Orszag-Tang Vortex (Princeton)
    if cfg.case == 2:

        rho = 25 / (36 * np.pi)
        pre =  5 / (12 * np.pi)     

        vex = - np.sin(2 * np.pi * cfg.Y) 
        vey =   np.sin(2 * np.pi * cfg.X)
        vez = 0

        B0 = 1 / np.sqrt(4 * np.pi)
        Bx = - B0 * np.sin(2 * np.pi * cfg.Y)
        By =   B0 * np.sin(4 * np.pi * cfg.X)
        Bz = 0

        q_sys[0,:,:] = rho
        q_sys[1,:,:] = rho * vex
        q_sys[2,:,:] = rho * vey
        q_sys[3,:,:] = rho * vez
        q_sys[4,:,:] = (pre / ((cfg.gamma-1.))) + 0.5 * rho * (vex**2 + vey**2 + vez**2) + 0.5 * (Bx**2 + By**2 + Bz**2)
        q_sys[5,:,:] = Bx
        q_sys[6,:,:] = By
        q_sys[7,:,:] = Bz
        
    # 1D Brio-Wu Shock Tube (Qi Tang)
    if cfg.case == 3:
        # Top Right, Top Left, Bottom Left, Bottom Right
        rho = [ 0.2,   1.0,  1.0,  0.2 ]
        vex = [ -0.4, -0.4, -0.4, -0.4 ]
        vey = [  0.0,  0.0,  0.0,  0.0 ]
        vez = [  0.0,  0.0,  0.0,  0.0 ]
        pre = [  0.1,  1.0,  1.0,  0.1 ]
        b_x = [ 0.75, 0.75, 0.75, 0.75 ]
        b_y = [ -1.0,  1.0,  1.0, -1.0 ]
        b_z = [  0.0,  0.0,  0.0,  0.0 ]

        # Define the discontinuity location
        xdist = 0.0
        ydist = 0.0

        # Calculate the quadrant indices for each point
        top_right = (cfg.X >= xdist) & (cfg.Y >= ydist)
        top_left = (cfg.X < xdist) & (cfg.Y >= ydist)
        bottom_left = (cfg.X < xdist) & (cfg.Y < ydist)
        bottom_right = (cfg.X >= xdist) & (cfg.Y < ydist)

        # Assign values based on quadrant indices
        q_sys[0, top_right] = rho[0]
        q_sys[1, top_right] = rho[0] * vex[0]
        q_sys[2, top_right] = rho[0] * vey[0]
        q_sys[3, top_right] = rho[0] * vez[0]
        q_sys[4, top_right] = (pre[0] / ((cfg.gamma-1.))) + 0.5 * rho[0] * (vex[0]**2 + vey[0]**2 + vez[0]**2) + 0.5 * (b_x[0]**2 + b_y[0]**2 + b_z[0]**2)
        q_sys[5, top_right] = b_x[0]
        q_sys[6, top_right] = b_y[0]
        q_sys[7, top_right] = b_z[0]

        q_sys[0, top_left] = rho[1]
        q_sys[1, top_left] = rho[1] * vex[1]
        q_sys[2, top_left] = rho[1] * vey[1]
        q_sys[3, top_left] = rho[1] * vez[1]
        q_sys[4, top_left] = (pre[1] / ((cfg.gamma-1.))) + 0.5 * rho[1] * (vex[1]**2 + vey[1]**2 + vez[1]**2) + 0.5 * (b_x[1]**2 + b_y[1]**2 + b_z[1]**2)
        q_sys[5, top_left] = b_x[1]
        q_sys[6, top_left] = b_y[1]
        q_sys[7, top_left] = b_z[1]

        q_sys[0, bottom_left] = rho[2]
        q_sys[1, bottom_left] = rho[2] * vex[2]
        q_sys[2, bottom_left] = rho[2] * vey[2]
        q_sys[3, bottom_left] = rho[2] * vez[2]
        q_sys[4, bottom_left] = (pre[2] / ((cfg.gamma-1.))) + 0.5 * rho[2] * (vex[2]**2 + vey[2]**2 + vez[2]**2) + 0.5 * (b_x[2]**2 + b_y[2]**2 + b_z[2]**2)
        q_sys[5, bottom_left] = b_x[2]
        q_sys[6, bottom_left] = b_y[2]
        q_sys[7, bottom_left] = b_z[2]

        q_sys[0, bottom_right] = rho[3]
        q_sys[1, bottom_right] = rho[3] * vex[3]
        q_sys[2, bottom_right] = rho[3] * vey[3]
        q_sys[3, bottom_right] = rho[3] * vez[3]
        q_sys[4, bottom_right] = (pre[3] / ((cfg.gamma-1.))) + 0.5 * rho[3] * (vex[3]**2 + vey[3]**2 + vez[3]**2) + 0.5 * (b_x[3]**2 + b_y[3]**2 + b_z[3]**2)
        q_sys[5, bottom_right] = b_x[3]
        q_sys[6, bottom_right] = b_y[3]
        q_sys[7, bottom_right] = b_z[3]

    return q_sys

