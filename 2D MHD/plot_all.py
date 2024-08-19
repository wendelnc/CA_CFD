# Standard Python Libraries
import os
import numpy as np
import pandas as pd
import datetime
from time import time, sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit


# User Defined Libraries
import configuration as cfg       # Input Parameters
import cons2prim as c2p           # Convert Conserved to Primitive Variables


def main():
    
    nx1 = 32;     nx2 = 32;   
    
    filename = f'simulation_data/result_{nx1}_by_{nx2}.npz'
    data = np.load(filename)
    
    all_q_sys = data['all_q_sys']
    all_a_sys = data['all_a_sys']
    all_t = data['all_t']
    all_div = data['all_div']

    plot_all_prim(all_q_sys[-1],all_t[-1])

def plot_all_prim(q_sys,t):
    '''
    Function Name:      plot_soln
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-22-2024

    Definition:         plot_soln plots the solution for the MHD Equaitons

    Inputs:             q_sys: conserved variables
                        t: current time step

    Outputs:            image of different variables

    Dependencies:       none
    '''

    rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)

    fig, plots = plt.subplots(2, 4, gridspec_kw={'wspace': 0.25}, figsize=(16, 8))

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(' ')
        return cbar

    fig.suptitle('Primative Solution at t = %.3f' % t)

    im1 = plots[0][0].imshow(rho, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][0].set_ylabel('y')
    plots[0][0].set_title('Density')
    cbar1 = add_colorbar(im1, plots[0][0])
    cbar1.set_label(' ')

    im2 = plots[0][1].imshow(u, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][1].set_yticklabels([])
    plots[0][1].set_title('X Velocity')
    cbar2 = add_colorbar(im2, plots[0][1])
    cbar2.set_label(' ')

    im3 = plots[0][2].imshow(v, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][2].set_yticklabels([])
    plots[0][2].set_title('Y Velocity')
    cbar3 = add_colorbar(im3, plots[0][2])   
    cbar3.set_label(' ')

    im4 = plots[0][3].imshow(w, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][3].set_yticklabels([])
    plots[0][3].set_title('Z Velocity')
    cbar4 = add_colorbar(im4, plots[0][3])
    cbar4.set_label(' ')

    im5 = plots[1][0].imshow(p, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('y')
    plots[1][0].set_title('Total Pressure')
    cbar5 = add_colorbar(im5, plots[1][0])
    cbar5.set_label(' ')

    im6 = plots[1][1].imshow(Bx, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_yticklabels([])
    plots[1][1].set_title('Bx')
    cbar6 = add_colorbar(im6, plots[1][1])
    cbar6.set_label(' ')

    im7 = plots[1][2].imshow(By, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][2].set_xlabel('x')
    plots[1][2].set_yticklabels([])
    plots[1][2].set_title('By')
    cbar7 = add_colorbar(im7, plots[1][2])
    cbar7.set_label(' ')

    im8 = plots[1][3].imshow(Bz, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][3].set_xlabel('x')
    plots[1][3].set_yticklabels([])
    plots[1][3].set_title('Bz')
    cbar8 = add_colorbar(im8, plots[1][3])
    cbar8.set_label(' ')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.075)

    plt.show()



if __name__ == "__main__":
    main()