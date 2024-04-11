# Standard Python Libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# User Defined Libraries
import configuration as cfg      # Input Parameters

def plot_prim(q_sys,t):
    '''
    Function Name:      plot_prim
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 03-25-2024

    Definition:         plot_prim plots the primative variables for the Euler Equaitons

    Inputs:             q_sys: conserved variables
                        t: current time step

    Outputs:            image of all primative variables

    Dependencies:       none
    '''

    rho = q_sys[0]
    u   = q_sys[1] / q_sys[0]
    v   = q_sys[2] / q_sys[0]
    p   = (cfg.gamma - 1.) * (q_sys[3] - (0.5 * q_sys[0] * ((u)**2. + (v)**2.)))
    
    fig, plots = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.0})

    fig.suptitle('Solution at t = %.3f' % t)

    im1 = plots[0][0].imshow(rho, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][0].set_ylabel('y')
    plots[0][0].set_title('Density')
    cbar1 = fig.colorbar(im1, ax=plots[0][0])
    cbar1.set_label(' ')

    im2 = plots[0][1].imshow(p, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][1].set_yticklabels([])
    plots[0][1].set_title('Pressure')
    cbar2 = fig.colorbar(im2, ax=plots[0][1])
    cbar2.set_label(' ')

    im3 = plots[1][0].imshow(u, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('y')
    plots[1][0].set_title('X Velocity')
    cbar3 = fig.colorbar(im3, ax=plots[1][0])
    cbar3.set_label(' ')

    im4 = plots[1][1].imshow(v, cmap='viridis',extent=[0,1,0,1],origin='lower')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_yticklabels([])
    plots[1][1].set_title('Y Velocity')
    cbar4 = fig.colorbar(im4, ax=plots[1][1])
    cbar4.set_label(' ')

    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    plt.show()

def plot_den_contour(q_sys,t):

    rho = q_sys[0]
    fig = plt.figure(figsize=(7,7), dpi=100)
    plt.imshow(rho,extent=[0,1,0,1],origin='lower')
    plt.title('Density at t = %.3f' % t)
    contour = plt.contour(rho,extent=[0,1,0,1], cmap='turbo', levels=10)
    plt.clabel(contour, inline=False, fontsize=12, colors = 'black')

    plt.show()