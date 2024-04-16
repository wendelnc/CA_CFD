# Standard Python Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# User Defined Libraries
import configuration as cfg

def plot_solution(q_sys,t):

    '''
    Function Name:      plot_solution
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 05-01-2023

    Definition:         plot_solution plots the density, pressure, x velocity
                        y velocity, B_y and E for the MHD Equaitons

    Inputs:             q_sys: conserved variables
                        t: current time step

    Outputs:            image of the solution at given time step

    Dependencies:       none
    '''

    rho = q_sys[0,:]
    vex = q_sys[1,:]/q_sys[0,:]
    vey = q_sys[2,:]/q_sys[0,:]
    vez = q_sys[3,:]/q_sys[0,:]
    E = q_sys[4,:]
    Bx = q_sys[5,:]
    By = q_sys[6,:]
    Bz = q_sys[7,:]
    pre = (cfg.gamma-1.)*(E - 0.5*rho*(vex**2 + vey**2 + vez**2) - 0.5*(Bx**2 + By**2 + Bz**2))

    fig, plots = plt.subplots(2, 3, figsize=(18, 8))

    fig.suptitle('Solution at t = %.3f' % t)

    # Top Left Plot
    plots[0][0].plot(cfg.xgrid,rho,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    # plots[0][0].set_xlabel('x')
    plots[0][0].set_ylabel('Density')
    plots[0][0].set_xlim(-1, 1)
    # plots[0][0].legend()

    # Bottom Left Plot
    plots[1][0].plot(cfg.xgrid,pre,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('Pressure')
    plots[1][0].set_xlim(-1, 1)
    # plots[1][0].legend()

    # Top Middle Plot
    plots[0][1].plot(cfg.xgrid,vex,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    # plots[0][1].set_xlabel('x')
    plots[0][1].set_ylabel('X Velocity')
    plots[0][1].set_xlim(-1, 1)
    # plots[0][1].legend()

    # Bottom Middle Plot
    plots[1][1].plot(cfg.xgrid,vey,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_ylabel('Y Velocity')
    plots[1][1].set_xlim(-1, 1)
    # plots[1][1].legend()

    # Top Right Plot
    plots[0][2].plot(cfg.xgrid,By,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    # plots[0][2].set_xlabel('x')
    plots[0][2].set_ylabel('Y Magnetic Field')
    plots[0][2].set_xlim(-1, 1)
    # plots[0][2].legend()

    # Bottom Right Plot
    plots[1][2].plot(cfg.xgrid,E,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    plots[0][2].set_xlabel('x')
    plots[1][2].set_ylabel('Total Energy')
    plots[1][2].set_xlim(-1, 1)

    # plots[1][2].legend()

    plt.show()


