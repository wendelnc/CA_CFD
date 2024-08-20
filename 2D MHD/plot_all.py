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
    
    filename = f'results/simulation_data/result_{nx1}_by_{nx2}.npz'
    data = np.load(filename)
    
    all_q_sys = data['all_q_sys']
    all_a_sys = data['all_a_sys']
    all_t = data['all_t']
    all_div = data['all_div']

    # plot_all_prim(all_q_sys[-1],all_t[-1])
    movie_maker_all_prim(all_q_sys,all_t)

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

def movie_maker_all_prim(all_solns,all_t):
    '''
    Function Name:      movie_maker
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-24-2023

    Definition:         movie_maker plots the primative variables for the Euler Equaitons

    Inputs:             all_solns: list of the solution at every time step
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
    fig, plots = plt.subplots(2, 4, gridspec_kw={'wspace': 0.25}, figsize=(16, 8))

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(' ')
        return cbar

    n_steps = len(all_t)

    q_sys = all_solns[0]

    rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)
   
    fig.suptitle('Primative Solution at t = %.3f' % all_t[0])

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
 
    def animate(i):
        ''' function to update the plot for each frame  '''

        q_sys = all_solns[i]

        rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)
   
        fig.suptitle('Primative Solution at t = %.3f' % all_t[i])

        im1.set_array(rho)
        im1.set_clim(vmin=rho.min(), vmax=rho.max())

        im2.set_array(u)
        im2.set_clim(vmin=u.min(), vmax=u.max())

        im3.set_array(v)
        im3.set_clim(vmin=v.min(), vmax=v.max())

        im4.set_array(w)
        im4.set_clim(vmin=w.min(), vmax=w.max())

        im5.set_array(p)
        im5.set_clim(vmin=p.min(), vmax=p.max())

        im6.set_array(Bx)
        im6.set_clim(vmin=Bx.min(), vmax=Bx.max())

        im7.set_array(By)
        im7.set_clim(vmin=By.min(), vmax=By.max())

        im8.set_array(Bz)
        im8.set_clim(vmin=Bz.min(), vmax=Bz.max())

        fig.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.075)

    # Create the animation using matplotlib's FuncAnimation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100)

    folder_path = os.path.join(os.getcwd(), "results/animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the Primative Movie...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)

if __name__ == "__main__":
    main()