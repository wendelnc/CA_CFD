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

    slice = cfg.nx3 // 2

    im1 = plots[0][0].imshow(rho[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][0].set_ylabel('y')
    plots[0][0].set_title('Density')
    cbar1 = add_colorbar(im1, plots[0][0])
    cbar1.set_label(' ')

    im2 = plots[0][1].imshow(u[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][1].set_yticklabels([])
    plots[0][1].set_title('X Velocity')
    cbar2 = add_colorbar(im2, plots[0][1])
    cbar2.set_label(' ')

    im3 = plots[0][2].imshow(v[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][2].set_yticklabels([])
    plots[0][2].set_title('Y Velocity')
    cbar3 = add_colorbar(im3, plots[0][2])   
    cbar3.set_label(' ')

    im4 = plots[0][3].imshow(w[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][3].set_yticklabels([])
    plots[0][3].set_title('Z Velocity')
    cbar4 = add_colorbar(im4, plots[0][3])
    cbar4.set_label(' ')

    im5 = plots[1][0].imshow(p[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('y')
    plots[1][0].set_title('Total Pressure')
    cbar5 = add_colorbar(im5, plots[1][0])
    cbar5.set_label(' ')

    im6 = plots[1][1].imshow(Bx[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_yticklabels([])
    plots[1][1].set_title('Bx')
    cbar6 = add_colorbar(im6, plots[1][1])
    cbar6.set_label(' ')

    im7 = plots[1][2].imshow(By[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][2].set_xlabel('x')
    plots[1][2].set_yticklabels([])
    plots[1][2].set_title('By')
    cbar7 = add_colorbar(im7, plots[1][2])
    cbar7.set_label(' ')

    im8 = plots[1][3].imshow(Bz[:,:,slice], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][3].set_xlabel('x')
    plots[1][3].set_yticklabels([])
    plots[1][3].set_title('Bz')
    cbar8 = add_colorbar(im8, plots[1][3])
    cbar8.set_label(' ')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.075)

    plt.show()

def movie_maker_mag(all_q_sys,all_a_sys,all_t):
    '''
    Function Name:      movie_maker_mag
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-24-2023

    Definition:         movie_maker_mag plots the magnitude of B and A

    Inputs:             all_q_sys: list of the solution at every time step
                        all_a_sys: list of the vector potential at every time step
                        all_t: list of all the time steps

    Outputs:            movie labeled movie_title

    Dependencies:       none
    '''

    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Convert the datetime object to a string
    movie_title = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    movie_title = movie_title + ".mp4"

    # Create a figure and axis for plotting
    fig, plots = plt.subplots(1, 2, gridspec_kw={'wspace': 0.25}, figsize=(16, 8))

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(' ')
        return cbar
    
    n_steps = len(all_t)

    q_sys = all_q_sys[0]
    a_sys = all_a_sys[0]

    rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)


    slice = cfg.nx3 // 2

    B = np.sqrt(Bx**2 + By**2 + Bz**2)

    A = np.sqrt(a_sys[0]**2 + a_sys[1]**2 + a_sys[2]**2)

    fig.suptitle(r'$\mathbf{B}$ and $\mathbf{A}$ at time $t = %.1f$' % all_t[0])

    im1 = plots[0].imshow(B[:,:,slice], cmap='viridis', extent=[-0.5, 0.5, -0.5, 0.5],origin='lower')
    cbar1 = add_colorbar(im1, plots[0])
    cbar1.set_label(' ')

    im2 = plots[1].imshow(A[:,:,slice], cmap='viridis', extent=[-0.5, 0.5, -0.5, 0.5],origin='lower')
    cbar2 = add_colorbar(im2, plots[1])
    cbar2.set_label(' ')

    def animate(i):
        ''' function to update the plot for each frame  '''
        
        slice = cfg.nx3 // 2

        q_sys = all_q_sys[i]
        a_sys = all_a_sys[i]

        rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)
   
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        A = np.sqrt(a_sys[0]**2 + a_sys[1]**2 + a_sys[2]**2)

        fig.suptitle(r'$\mathbf{B}$ and $\mathbf{A}$ at time $t = %.1f$' % all_t[i])

        im1.set_array(B[:,:,slice])
        im1.set_clim(vmin=B.min(), vmax=B.max())

        im2.set_array(A[:,:,slice])
        im2.set_clim(vmin=A.min(), vmax=A.max())


    # Create the animation using matplotlib's FuncAnimation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100)

    folder_path = os.path.join(os.getcwd(), "animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the B Plot...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)