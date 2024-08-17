# Standard Python Libraries
import numpy as np
import matplotlib.pyplot as plt

# User Defined Libraries
import configuration as cfg # configuration file

def initial_condition():
    '''
    Function Name:      initial_condition
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-18-2024

    Definition:         loads initial conditions for 3D MHD Problems

    Inputs:             case: various tests defined below

    Outputs:            q_sys = ( ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z )^T
                        a_sys = ( a_x, a_y, a_z )^T

    Dependencies:       none

    '''

    q_sys = np.zeros((8,cfg.nx3+(2*cfg.nghost),cfg.nx2+(2*cfg.nghost),cfg.nx1+(2*cfg.nghost)))

    a_sys = np.zeros((3,cfg.nx3+(2*cfg.nghost),cfg.nx2+(2*cfg.nghost),cfg.nx1+(2*cfg.nghost)))

    # Smooth Alfven Waves 
    if cfg.case == 0:

        pass
    
    # Orszag-Tang Vortex
    if cfg.case == 1:
        '''
        
        6.3.1. Orszag-Tang Vortex from the paper

        An Unstaggered Constrained Transport Method for the 3D Ideal Magnetohydrodynamic Equations

        by Christiane Helzel, James A. Rossmanith, and Bertram Taetz

        '''

        ep = 0.2

        rho = (cfg.gamma)**2 
        pre =  cfg.gamma     

        vex = - np.sin(cfg.Y) * (1 + ep * np.sin(cfg.Z))
        vey =   np.sin(cfg.X) * (1 + ep * np.sin(cfg.Z))
        vez = ep * np.sin(cfg.Z)

        Bx = - np.sin(cfg.Y) 
        By =   np.sin(2 * cfg.X) 
        Bz = 0

        q_sys[0,:,:,:] = rho
        q_sys[1,:,:,:] = rho * vex
        q_sys[2,:,:,:] = rho * vey
        q_sys[3,:,:,:] = rho * vez
        q_sys[4,:,:,:] = (pre / ((cfg.gamma-1.))) + 0.5 * rho * (vex**2 + vey**2 + vez**2) + 0.5 * (Bx**2 + By**2 + Bz**2)
        q_sys[5,:,:,:] = Bx
        q_sys[6,:,:,:] = By
        q_sys[7,:,:,:] = Bz

        a_sys[2,:,:,:] = 0.5 * np.cos(2*cfg.X) + np.cos(cfg.Y)

    # 3D Field Loop
    if cfg.case == 2:
        '''
        4.4. 3D Field Loop from the paper

        A High Order Finite Difference Weighted Essentially NonOscillatory Scheme 
        with a Kernel-Based Constrained Transport Method for Ideal Magnetohydrodynamics

        by Firat Cakir, Andrew Christlieb, and Yan Jiang

        '''

        rho = 1.0
        vex = 2 / np.sqrt(6)
        vey = 1 / np.sqrt(6)
        vez = 1 / np.sqrt(6)
        pre = 1.0

        R = 0.3

        for i in range(cfg.nx1+(2*cfg.nghost)):
            for j in range(cfg.nx2+(2*cfg.nghost)):
                for k in range(cfg.nx3+(2*cfg.nghost)):
                    r = np.sqrt(cfg.X[i,j,k]**2 + cfg.Y[i,j,k]**2)
                    if r <= R:
                        a_sys[2,i,j,k] = 0.001*(R-r)

        for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
            for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
                for k in range(cfg.nghost,(cfg.nx3)+cfg.nghost):

                    q_sys[5,i,j,k] = (1/(12.0*cfg.dy))*(a_sys[2,i,j-2,k] - 8.0*a_sys[2,i,j-1,k] + 8.0*a_sys[2,i,j+1,k] - a_sys[2,i,j+2,k]) \
                                - (1/(12.0*cfg.dz))*(a_sys[1,i,j,k-2] - 8.0*a_sys[1,i,j,k-1] + 8.0*a_sys[1,i,j,k+1] - a_sys[1,i,j,k+2])

                    q_sys[6,i,j,k] = (1/(12.0*cfg.dz))*(a_sys[0,i,j,k-2] - 8.0*a_sys[0,i,j,k-1] + 8.0*a_sys[0,i,j,k+1] - a_sys[0,i,j,k+2]) \
                                - (1/(12.0*cfg.dx))*(a_sys[2,i-2,j,k] - 8.0*a_sys[2,i-1,j,k] + 8.0*a_sys[2,i+1,j,k] - a_sys[2,i+2,j,k])  

                    q_sys[7,i,j,k] = (1/(12.0*cfg.dx))*(a_sys[1,i-2,j,k] - 8.0*a_sys[1,i-1,j,k] + 8.0*a_sys[1,i+1,j,k] - a_sys[1,i+2,j,k]) \
                                - (1/(12.0*cfg.dy))*(a_sys[0,i,j-2,k] - 8.0*a_sys[0,i,j-1,k] + 8.0*a_sys[0,i,j+1,k] - a_sys[0,i,j+2,k])
  
        q_sys[0,:,:,:] = rho
        q_sys[1,:,:,:] = rho * vex
        q_sys[2,:,:,:] = rho * vey
        q_sys[3,:,:,:] = rho * vez
        q_sys[4,:,:,:] = (pre / ((cfg.gamma-1.))) + 0.5 * rho * (vex**2 + vey**2 + vez**2) + 0.5 * ((q_sys[5,:,:,:])**2 + (q_sys[6,:,:,:])**2 + (q_sys[7,:,:,:])**2)

    return q_sys, a_sys

