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
  

    q_sys = np.zeros((4,cfg.nx2,cfg.nx1))

    if cfg.case == 0:
        # Setup corresponding to configuration 3 of the paper
        # Carsten W. Schulz-Rinne, James P. Collins, and Harland M. Glaz.
        # Numerical solution of the riemann problem for two-dimensional gas dynamics.
        # SIAM Journal on Scientific Computing, 14(6):1394-1414, 1993.
        # Top Right, Top Left, Bottom Left, Bottom Right
        rho = [ 1.5, 0.5323, 0.138, 0.5323 ]
        u   = [ 0.0,  1.206, 1.206, 0.0    ]
        v   = [ 0.0,    0.0, 1.206, 1.206  ]
        p   = [ 1.5,    0.3, 0.029, 0.3    ]
        
    if cfg.case == 1:
        # Configuration 1 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 1.0,  0.5197,  0.1072,  0.2579 ]
        u   = [ 0.0, -0.7259, -0.7259,     0.0 ]
        v   = [ 0.0,     0.0, -1.4045, -1.4045 ]
        p   = [ 1.0,     0.4,  0.0439,    0.15 ]

    if cfg.case == 2:
        # Configuration 2 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 1.0,  0.5197,     1.0,  0.5197 ]
        u   = [ 0.0, -0.7259, -0.7259,     0.0 ]
        v   = [ 0.0,     0.0, -0.7259, -0.7259 ]
        p   = [  1.0,    0.4,     1.0,     0.4 ]

    if cfg.case == 3:
        # Configuration 3 from the following paper (same as case 0, but kept for reference)
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 1.5, 0.5323, 0.138, 0.5323 ]
        u   = [ 0.0,  1.206, 1.206, 0.0    ]
        v   = [ 0.0,    0.0, 1.206, 1.206  ]
        p   = [ 1.5,    0.3, 0.029, 0.3    ]

    if cfg.case == 4:
        # Configuration 4 from the following paper 
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 1.1, 0.5065,    1.1, 0.5065 ]
        u   = [ 0.0, 0.8939, 0.8939,    0.0 ]
        v   = [ 0.0,    0.0, 0.8939, 0.8939 ]
        p   = [ 1.1,   0.35,    1.1,   0.35 ]

    if cfg.case == 5:
        # Configuration 5 from the following paper 
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [   1.0,   2.0,  1.0,  3.0 ]
        u   = [ -0.75, -0.75, 0.75, 0.75 ]
        v   = [  -0.5,   0.5,  0.5, -0.5 ]
        p   = [   1.0,   1.0,  1.0,  1.0 ]

    if cfg.case == 6:
        # Configuration 6 from the following paper 
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [  1.0,  2.0,   1.0,   3.0 ]
        u   = [ 0.75, 0.75, -0.75, -0.75 ]
        v   = [ -0.5,  0.5,   0.5,  -0.5 ]
        p   = [  1.0,  1.0,   1.0,   1.0 ]

    if cfg.case == 7:
        # Configuration 7 from the following paper 
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 1.0,  0.5197, 0.8,  0.5197 ]
        u   = [ 0.1, -0.6259, 0.1,     0.1 ]
        v   = [ 0.1,     0.1, 0.1, -0.6259 ]
        p   = [ 1.0,     0.4, 0.4,     0.4 ]

    
    if cfg.case == 8:
        # Configuration 8 from the following paper 
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 0.5197,     1.0, 0.8,     1.0 ]
        u   = [    0.1, -0.6259, 0.1,     0.1 ]
        v   = [    0.1,     0.1, 0.1, -0.6259 ]
        p   = [    0.4,     1.0, 1.0,     1.0 ]

    if cfg.case == 9:
        # Configuration 9 from the following paper 
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 1.0,  2.0,   1.039,  0.5197 ]
        u   = [ 0.0,  0.0,     0.0,     0.0 ]
        v   = [ 0.3, -0.3, -0.8133, -0.4259 ]
        p   = [ 1.0,  1.0,     0.4,     0.4 ]

    if cfg.case == 10:
        # Configuration 10 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [    1.0,    0.5,  0.2281,   0.4562 ]
        u   = [    0.0,    0.0,     0.0,     0.0 ]
        v   = [ 0.4297, 0.6076, -0.6076, -0.4297 ]
        p   = [    1.0,    1.0,  0.3333,  0.3333 ]
    
    if cfg.case == 11:
        # Configuration 11 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 1.0, 0.5313, 0.8, 0.5313 ]
        u   = [ 0.1, 0.8276, 0.1,    0.1 ]
        v   = [ 0.0,    0.0, 0.0, 0.7276 ]
        p   = [ 1.0,    0.4, 0.4,    0.4 ]
    
    if cfg.case == 12:
        # Configuration 12 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [ 0.5313,    1.0, 0.8,    1.0 ]
        u   = [    0.0, 0.7276, 0.0,    0.0 ]
        v   = [    0.0,    0.0, 0.0, 0.7276 ]
        p   = [    0.4,    1.0, 1.0,    1.0 ]


    if cfg.case == 13:
        # Configuration 13 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [  1.0, 2.0, 1.0625, 0.5313 ]
        u   = [  0.0, 0.0,    0.0,    0.0 ]
        v   = [ -0.3, 0.3, 0.8145, 0.4276 ]
        p   = [  1.0, 1.0,    0.4,    0.4 ]

    if cfg.case == 14:
        # Configuration 14 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [     2.0,     1.0, 0.4736, 0.9474 ]
        u   = [     0.0,     0.0,    0.0,    0.0 ]
        v   = [ -0.5606, -1.2172, 1.2172, 1.1606 ]
        p   = [     8.0,     8.0, 2.6667, 2.6667 ]

    if cfg.case == 15:
        # Configuration 15 from the following paper
        # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
        rho = [  1.0,  0.5197,  0.8, 0.5313 ]
        u   = [  0.1, -0.6259,  0.1,    0.1 ]
        v   = [ -0.3,    -0.3, -0.3, 0.4276 ]
        p   = [  1.0,     0.4,  0.4,    0.4 ]

    # if cfg.case == :
    #     # Configuration  from the following paper
    #     # https://www.math.umd.edu/~tadmor/pub/central-schemes/Kurganov-Tadmor.NMPDEs02.pdf
    #     rho = []
    #     u   = []
    #     v   = []
    #     p   = []
    

    # if cfg.case ==3:
    #     # 1D Sod Shock Tube
    #     rho = [ 0.125, 1.0, 1.0, 0.125 ]
    #     u   = [   0.0, 0.0, 0.0,   0.0 ]
    #     v   = [   0.0, 0.0, 0.0,   0.0 ]
    #     p   = [   0.1, 1.0, 1.0,   0.1 ] 

    # if cfg.case ==4:
    #     # Rotated 1D Sod Shock Tube
    #     rho = [ 0.125, 0.125, 1.0, 1.0 ]
    #     u   = [   0.0,   0.0, 0.0, 0.0 ]
    #     v   = [   0.0,   0.0, 0.0, 0.0 ]
    #     p   = [   0.1,  0.1,  1.0, 1.0 ] 

    # if cfg.case == 5:
    #     # Reverse 1D Sod Shock Tube
    #     rho = [ 1.0, 0.125, 0.125, 1.0 ]
    #     u   = [ 0.0,   0.0,   0.0, 0.0 ]
    #     v   = [ 0.0,   0.0,   0.0, 0.0 ]
    #     p   = [ 1.0,   0.1,  0.1,  1.0 ] 

    # if cfg.case == 6:
    #     # Rotated Reverse 1D Sod Shock Tube
    #     rho = [ 1.0, 1.0, 0.125, 0.125 ]
    #     u   = [ 0.0, 0.0,   0.0,   0.0 ]
    #     v   = [ 0.0, 0.0,   0.0,   0.0 ]
    #     p   = [ 1.0, 1.0,   0.1,   0.1 ] 

    # if cfg.case == 7:
    #     # Reverse Carsten W. Schulz-Rinne 2D Riemann Problem
    #     rho = [ 0.138, 0.5323, 1.5, 0.5323 ]
    #     u   = [ 1.206,    0.0, 0.0,  1.206 ]
    #     v   = [ 1.206,  1.206, 0.0,    0.0 ]
    #     p   = [ 0.029,    0.3, 1.5,    0.3 ] 

    # if cfg.case == 8:
    #     # Custom Test Problem
    #     rho = [ 3, 2, 1, 4 ]
    #     u   = [ 0, 0, 0, 0 ]
    #     v   = [ 0, 0, 0, 0 ]
    #     p   = [ 1, 1, 1, 1 ]
       

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
