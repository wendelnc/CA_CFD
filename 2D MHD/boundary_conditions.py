# Standard Python Libraries
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

# @njit
def boundary_conditions(q_sys):
    '''
    Function Name:      boundary_conditions
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-02-2024

    Definition:         Updates boundary conditions based on the type
                        Type 1: "periodic" 
                        Type 2: "reflective"

    Inputs:             q_sys: initial condition

    Outputs:            updated q_sys based on boundary conditions

    Dependencies:       None
    '''

    # If we have the domain as follows, where |~~~~~| signifies ghost cells

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

    # Periodic Boundary Conditions
    if cfg.bndc == 0:
        q_sys[:,:, :cfg.nghost] = q_sys[:,:, -2*cfg.nghost:-cfg.nghost]
        q_sys[:,:, -cfg.nghost:] = q_sys[:,:, cfg.nghost:2*cfg.nghost]
        q_sys[:,:cfg.nghost, :] = q_sys[:,-2*cfg.nghost:-cfg.nghost, :]
        q_sys[:,-cfg.nghost:, :] = q_sys[:,cfg.nghost:2*cfg.nghost, :] 
    
    # Reflective Boundary Conditions
    elif cfg.bndc == 1:
        q_sys[:,:, :cfg.nghost] = q_sys[:,:, cfg.nghost:2*cfg.nghost]
        q_sys[:,:, -cfg.nghost:] = q_sys[:,:, -2*cfg.nghost:-cfg.nghost]
        q_sys[:,:cfg.nghost, :] = q_sys[:,cfg.nghost:2*cfg.nghost, :]
        q_sys[:,-cfg.nghost:, :] = q_sys[:,-2*cfg.nghost:-cfg.nghost, :]
        
    return q_sys