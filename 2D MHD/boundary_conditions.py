# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters


def periodic_bc(q_sys):
    '''
    If we have the domain as follows, where |~~~~~| signifies ghost cells

	  [0]   [1]   [2]   [3]         [.]         [N]  [N+1] [N+2] [N+3]
	|~~~~~|~~~~~|-----|-----|-----|-----|-----|-----|-----|~~~~~|~~~~~|

    Our swap for periodic boundary conditions is as follows:
    a[0] = a[N] 
    a[1] = a[N+1]
    a[N+2] = a[2]
    a[N+3] = a[3]
    '''

    # X-Direction
    q_sys[:,:cfg.nghost, :] = q_sys[:,-2*cfg.nghost:-cfg.nghost, :]
    q_sys[:,-cfg.nghost:, :] = q_sys[:,cfg.nghost:2*cfg.nghost, :] 
    # Y-Direction
    q_sys[:,:, :cfg.nghost] = q_sys[:,:, -2*cfg.nghost:-cfg.nghost]
    q_sys[:,:, -cfg.nghost:] = q_sys[:,:, cfg.nghost:2*cfg.nghost]

    return q_sys

def ct_periodic_bc(a_sys):
    '''
    If we have the domain as follows, where |~~~~~| signifies ghost cells

	  [0]   [1]   [2]   [3]         [.]         [N]  [N+1] [N+2] [N+3]
	|~~~~~|~~~~~|-----|-----|-----|-----|-----|-----|-----|~~~~~|~~~~~|

    Our swap for periodic boundary conditions is as follows:
    a[0] = a[N] 
    a[1] = a[N+1]
    a[N+2] = a[2]
    a[N+3] = a[3]
    '''

    if cfg.case == 0:
        theta = 0.463647609000806
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        B1 = 1.0
        b1av = B1*ctheta
        b2av = B1*stheta
    else:
        b1av = 0.0
        b2av = 0.0

    # Subtract off non-periodic part of Magnetic Potential
    a_sys[2] = a_sys[2] - (b1av*cfg.Y - b2av*cfg.X)

    # X-Direction
    a_sys[:,:cfg.nghost, :]  = a_sys[:,-2*cfg.nghost:-cfg.nghost, :]
    a_sys[:,-cfg.nghost:, :] = a_sys[:,cfg.nghost:2*cfg.nghost, :] 
    # Y-Direction
    a_sys[:,:, :cfg.nghost]  = a_sys[:,:, -2*cfg.nghost:-cfg.nghost]
    a_sys[:,:, -cfg.nghost:] = a_sys[:,:, cfg.nghost:2*cfg.nghost]

    # Add non-periodic part back-in to Magnetic Potential
    a_sys[2] = a_sys[2] + (b1av*cfg.Y - b2av*cfg.X)

    return a_sys
    
def reflective_bc(q_sys):
    '''
    If we have the domain as follows, where |~~~~~| signifies ghost cells

	  [0]   [1]   [2]   [3]         [.]         [N]  [N+1] [N+2] [N+3]
	|~~~~~|~~~~~|-----|-----|-----|-----|-----|-----|-----|~~~~~|~~~~~|

    Our swap for reflective boundary conditions is as follows:
    a[0] = a[2]
    a[1] = a[3]
    a[N+2] = a[N]
    a[N+3] = a[N+1]
    '''
    
    # X-Direction
    q_sys[:,:cfg.nghost, :] = q_sys[:,cfg.nghost:2*cfg.nghost, :]
    q_sys[:,-cfg.nghost:, :] = q_sys[:,-2*cfg.nghost:-cfg.nghost, :]
    # Y-Direction
    q_sys[:,:, :cfg.nghost] = q_sys[:,:, cfg.nghost:2*cfg.nghost]
    q_sys[:,:, -cfg.nghost:] = q_sys[:,:, -2*cfg.nghost:-cfg.nghost]
    
    return q_sys

def boundary_conditions(q_sys, a_sys):
    '''
    Function Name:      boundary_conditions
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-02-2024

    Definition:         Updates boundary conditions based on the type

    Inputs:             q_sys, a_sys

    Outputs:            updated q_sys & a_sys based on boundary conditions

    Dependencies:       None
    '''

    if cfg.case == 1:
        q_sys = periodic_bc(q_sys)
        a_sys = periodic_bc(a_sys)
    
        
    return q_sys, a_sys