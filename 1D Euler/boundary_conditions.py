
# Standard Python Libraries
# from numba import njit

# User Defined Libraries

# @njit
def boundary_conditions(q_sys,n_ghost,bndc):
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

    Dependencies:       None
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


    return q_sys