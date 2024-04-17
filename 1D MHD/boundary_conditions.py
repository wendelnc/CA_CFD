# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def boundary_conditions(q_sys):
    '''
    Function Name:      boundary_conditions
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-10-2024

    Definition:         Updates boundary conditions based on the type
                        bc = 0: "periodic" 
                        bc = 1: "reflective"

    Inputs:             q_sys: initial condition

    Outputs:            updated q_sys based on boundary conditions

    Dependencies:       none
    '''

    # If we have the domain of grid cells denoted by |-----|, |~~~~~| signifies ghost cells

	#  [0]   [1]   [2]   [3]         [.]         [N]  [N+1] [N+2] [N+3]
	#|~~~~~|~~~~~|-----|-----|-----|-----|-----|-----|-----|~~~~~|~~~~~|

    # Our swap for periodic boundary conditions is as follows:
    # a[0] = a[N] 
    # a[1] = a[N+1]
    # a[N+2] = a[2]
    # a[N+3] = a[3]

    # Our swap for reflective boundary conditions is as follows:
    # a[0] = a[2]
    # a[1] = a[3]
    # a[N+2] = a[N]
    # a[N+3] = a[N+1]

    n_ghost = cfg.nghost
    bndc = cfg.bndc

    # Periodic Boundary Conditions
    if bndc == 0:
        for i in range(n_ghost):
            q_sys[:,i] = q_sys[:,-2*n_ghost+i]
            q_sys[:,-1*n_ghost+i] = q_sys[:,n_ghost+i]

    # Reflective Boundary Conditions
    elif bndc == 1:
        for i in range(n_ghost):
            q_sys[:,i] = q_sys[:,n_ghost+i]
            q_sys[:,-1*n_ghost+i] = q_sys[:,-2*n_ghost+i]

    # Dirichlet Boundary Conditions (for Brio-Wu Shock Tube)
    elif bndc == 2:
        if cfg.case == 0:
            q_l = np.array([1.0, -0.4, 0.0, 0.0, 1.0, 0.75,  1.0, 0.0])
            q_r = np.array([0.2, -0.4, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0])
        elif cfg.case == 1: 
            q_l = np.array([  1.0, 0.0, 0.0, 0.0, 1.0, 0.75,  1.0, 0.0])
            q_r = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0])
        for i in range(n_ghost):
            # Left States
            q_sys[0,i] = q_l[0]
            q_sys[1,i] = q_l[0] * q_l[1]
            q_sys[2,i] = q_l[0] * q_l[2]
            q_sys[3,i] = q_l[0] * q_l[3]
            q_sys[4,i] = (q_l[4] / ((cfg.gamma-1.))) + 0.5 * q_l[0] * (q_l[1]**2 + q_l[2]**2 + q_l[3]**2) + 0.5 * (q_l[5]**2 + q_l[6]**2 + q_l[7]**2)
            q_sys[5,i] = q_l[5]
            q_sys[6,i] = q_l[6]
            q_sys[7,i] = q_l[7]
            # Right States
            q_sys[0,-1*n_ghost+i] = q_r[0]
            q_sys[1,-1*n_ghost+i] = q_r[0] * q_r[1]
            q_sys[2,-1*n_ghost+i] = q_r[0] * q_r[2]
            q_sys[3,-1*n_ghost+i] = q_r[0] * q_r[3]
            q_sys[4,-1*n_ghost+i] = (q_r[4] / ((cfg.gamma-1.))) + 0.5 * q_r[0] * (q_r[1]**2 + q_r[2]**2 + q_r[3]**2) + 0.5 * (q_r[5]**2 + q_r[6]**2 + q_r[7]**2)
            q_sys[5,-1*n_ghost+i] = q_r[5]
            q_sys[6,-1*n_ghost+i] = q_r[6]
            q_sys[7,-1*n_ghost+i] = q_r[7]

    # Outflow Boundary Conditions
    elif bndc == 3:
        for i in range(n_ghost):
            q_sys[:,i] = q_sys[:,n_ghost]
            q_sys[:,-1*n_ghost+i] = q_sys[:,-2*n_ghost]

    return q_sys