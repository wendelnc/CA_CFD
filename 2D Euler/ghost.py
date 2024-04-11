# Standard Python Libraries
import numpy as np

# User Defined Libraries
import configuration as cfg      # Input Parameters

def add_ghost_cells(q_sys):
    '''
    Function Name:      add_ghost_cells
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 03-24-2023

    Definition:         adds 'nghost' number of ghost cells to the 'nx1 x nx2' array 'q_sys'

    Inputs:             q_sys: initial condition

    Outputs:            q_sys_with_ghost: A 2D nx2+2*(nghost) x nx1+2*(nghost) array where 
                        the ghost cells have a value of zero and 
                        the interior values are those of the input array q_sys

                        e.g.,
                        Our nx1 sized array for q_sys:
								  #   #   #   #   #   #   #
						      |---|---|---|---|---|---|---|

                     	Our nx1+2*(nghost) array for q_with_ghost

                        0   0   #   #   #   #   #   #   #   0   0
                      |---|---|---|---|---|---|---|---|---|---|---|

    Dependencies:       None
    '''

    q_sys_with_ghost = np.zeros((4,cfg.nx2+(2*cfg.nghost),cfg.nx1+(2*cfg.nghost)))
    q_sys_with_ghost[:,(cfg.nghost):-1*(cfg.nghost),(cfg.nghost):-1*(cfg.nghost)] = q_sys[:]

    return q_sys_with_ghost
