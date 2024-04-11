# Standard Python Libraries
import numpy as np

# User Defined Libraries

def add_ghost_cells(q_sys,n_points,n_ghost):
    '''
    Function Name:      add_ghost_cells
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-17-2023

    Definition:         adds 'n_ghost' number of ghost cells to the 'n_points' q arrays

    Inputs:             q_arr: initial condition
                        n_points: The number of points in the domain
                        n_ghost: The number of ghost cells required

    Outputs:            a_with_ghost: A 1D N+2*(n_ghost) array where the ghost cells have a
    					          value of zero, and the interior values are those of the input array a

                        Our n_points sized array for q:
								                #   #   #   #   #   #   #
							                |---|---|---|---|---|---|---|

                     	Our n_points+2*(n_ghost) array for q_with_ghost

                        0   0   #   #   #   #   #   #   #   0   0
                      |---|---|---|---|---|---|---|---|---|---|---|

    Dependencies:       None
    '''

    q_sys_with_ghost = np.zeros((3,n_points+(2*n_ghost)))
    q_sys_with_ghost[:,(n_ghost):-1*(n_ghost)] = q_sys[:]

    return q_sys_with_ghost