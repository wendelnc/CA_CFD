# Standard Python Libraries
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def cons2prim(q_sys):
    '''
    Function Name:      cons2prim
    Creator:            Carolyn Wendeln
    Date Created:       03-26-2024
    Date Last Modified: 04-15-2024

    Definition:         cons2prim converts the conserved variables to primative variables

    Inputs:             q_sys = [ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z]^T

    Outputs:            w_sys = [ρ, u_x, u_y, u_z, p, B_x, B_y, B_z]^T

    Dependencies:       none
    '''

    rho = q_sys[0]
    vex = q_sys[1] / q_sys[0]
    vey = q_sys[2] / q_sys[0]
    vez = q_sys[3] / q_sys[0]
    Eng = q_sys[4]
    Bx  = q_sys[5]
    By  = q_sys[6]
    Bz  = q_sys[7]
    pre = (cfg.gamma-1.)*(Eng - 0.5*rho*(vex**2 + vey**2 + vez**2) - 0.5*(Bx**2 + By**2 + Bz**2))
    
    return rho, vex, vey, vez, pre, Bx, By, Bz