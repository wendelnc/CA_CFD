# Standard Python Libraries
import os
import numpy as np
import pandas as pd
import datetime
from time import time, sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_all_cons(q_sys,t):
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

    fig, plots = plt.subplots(2, 4, gridspec_kw={'wspace': 0.25}, figsize=(16, 8))

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(' ')
        return cbar

    fig.suptitle('Conserved Solution at t = %.3f' % t)

    im1 = plots[0][0].imshow(q_sys[0], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][0].set_ylabel('y')
    plots[0][0].set_title('Density')
    cbar1 = add_colorbar(im1, plots[0][0])
    cbar1.set_label(' ')

    im2 = plots[0][1].imshow(q_sys[1], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][1].set_yticklabels([])
    plots[0][1].set_title('X Momentum')
    cbar2 = add_colorbar(im2, plots[0][1])
    cbar2.set_label(' ')

    im3 = plots[0][2].imshow(q_sys[2], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][2].set_yticklabels([])
    plots[0][2].set_title('Y Momentum')
    cbar3 = add_colorbar(im3, plots[0][2])   
    cbar3.set_label(' ')

    im4 = plots[0][3].imshow(q_sys[3], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][3].set_yticklabels([])
    plots[0][3].set_title('Z Momentum')
    cbar4 = add_colorbar(im4, plots[0][3])
    cbar4.set_label(' ')

    im5 = plots[1][0].imshow(q_sys[4], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('y')
    plots[1][0].set_title('Total Energy')
    cbar5 = add_colorbar(im5, plots[1][0])
    cbar5.set_label(' ')

    im6 = plots[1][1].imshow(q_sys[5], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_yticklabels([])
    plots[1][1].set_title('Bx')
    cbar6 = add_colorbar(im6, plots[1][1])
    cbar6.set_label(' ')

    im7 = plots[1][2].imshow(q_sys[6], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][2].set_xlabel('x')
    plots[1][2].set_yticklabels([])
    plots[1][2].set_title('By')
    cbar7 = add_colorbar(im7, plots[1][2])
    cbar7.set_label(' ')

    im8 = plots[1][3].imshow(q_sys[7], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][3].set_xlabel('x')
    plots[1][3].set_yticklabels([])
    plots[1][3].set_title('Bz')
    cbar8 = add_colorbar(im8, plots[1][3])
    cbar8.set_label(' ')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.075)

    plt.show()

def plot_mag_pot(a_sys,t):
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

    fig, plots = plt.subplots(1, 3, gridspec_kw={'wspace': 0.25}, figsize=(10, 5))

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(' ')
        return cbar

    fig.suptitle('Magnetic Potential Solution at t = %.3f' % t)

    im1 = plots[0].imshow(a_sys[0], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0].set_ylabel('y')
    plots[0].set_title('A_x')
    cbar1 = add_colorbar(im1, plots[0])
    cbar1.set_label(' ')

    im2 = plots[1].imshow(a_sys[1], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1].set_yticklabels([])
    plots[1].set_title('A_y')
    cbar2 = add_colorbar(im2, plots[1])
    cbar2.set_label(' ')

    im3 = plots[2].imshow(a_sys[2], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[2].set_yticklabels([])
    plots[2].set_title('A_z')
    cbar3 = add_colorbar(im3, plots[2])   
    cbar3.set_label(' ')

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

    folder_path = os.path.join(os.getcwd(), "animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the Primative Plot...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)

def movie_maker_all_cons(all_solns,all_t):
    '''
    Function Name:      movie_maker
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-24-2023

    Definition:         movie_maker plots the conserved variables for the MHD Equaitons

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
   
    fig.suptitle('Solution at t = %.3f' % all_t[0])

    im1 = plots[0][0].imshow(q_sys[0], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][0].set_ylabel('y')
    plots[0][0].set_title('Density')
    cbar1 = add_colorbar(im1, plots[0][0])
    cbar1.set_label(' ')

    im2 = plots[0][1].imshow(q_sys[1], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][1].set_yticklabels([])
    plots[0][1].set_title('X Momentum')
    cbar2 = add_colorbar(im2, plots[0][1])
    cbar2.set_label(' ')

    im3 = plots[0][2].imshow(q_sys[2], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][2].set_yticklabels([])
    plots[0][2].set_title('Y Momentum')
    cbar3 = add_colorbar(im3, plots[0][2])   
    cbar3.set_label(' ')

    im4 = plots[0][3].imshow(q_sys[3], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][3].set_yticklabels([])
    plots[0][3].set_title('Z Momentum')
    cbar4 = add_colorbar(im4, plots[0][3])
    cbar4.set_label(' ')

    im5 = plots[1][0].imshow(q_sys[4], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('y')
    plots[1][0].set_title('Total Energy')
    cbar5 = add_colorbar(im5, plots[1][0])
    cbar5.set_label(' ')

    im6 = plots[1][1].imshow(q_sys[5], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_yticklabels([])
    plots[1][1].set_title('Bx')
    cbar6 = add_colorbar(im6, plots[1][1])
    cbar6.set_label(' ')

    im7 = plots[1][2].imshow(q_sys[6], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][2].set_xlabel('x')
    plots[1][2].set_yticklabels([])
    plots[1][2].set_title('By')
    cbar7 = add_colorbar(im7, plots[1][2])
    cbar7.set_label(' ')

    im8 = plots[1][3].imshow(q_sys[7], cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][3].set_xlabel('x')
    plots[1][3].set_yticklabels([])
    plots[1][3].set_title('Bz')
    cbar8 = add_colorbar(im8, plots[1][3])
    cbar8.set_label(' ')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.075)
 
    def animate(i):
        ''' function to update the plot for each frame  '''

        q_sys = all_solns[i]
   
        fig.suptitle('Solution at t = %.3f' % all_t[i])

        im1.set_array(q_sys[0])
        im1.set_clim(vmin=q_sys[0].min(), vmax=q_sys[0].max())

        im2.set_array(q_sys[1])
        im2.set_clim(vmin=q_sys[1].min(), vmax=q_sys[1].max())

        im3.set_array(q_sys[2])
        im3.set_clim(vmin=q_sys[2].min(), vmax=q_sys[2].max())

        im4.set_array(q_sys[3])
        im4.set_clim(vmin=q_sys[3].min(), vmax=q_sys[3].max())

        im5.set_array(q_sys[4])
        im5.set_clim(vmin=q_sys[4].min(), vmax=q_sys[4].max())

        im6.set_array(q_sys[5])
        im6.set_clim(vmin=q_sys[5].min(), vmax=q_sys[5].max())

        im7.set_array(q_sys[6])
        im7.set_clim(vmin=q_sys[6].min(), vmax=q_sys[6].max())

        im8.set_array(q_sys[7])
        im8.set_clim(vmin=q_sys[7].min(), vmax=q_sys[7].max())

        fig.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.075)

    # Create the animation using matplotlib's FuncAnimation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100)

    folder_path = os.path.join(os.getcwd(), "animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the Conserved Plot...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)

def plot_soln(q_sys,t):
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
    pgas = (cfg.gamma-1.)*rho*(u**2 + v**2 + w**2) 
    pmag = 0.5*(Bx**2 + By**2 + Bz**2)
    
    fig, plots = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.0}, figsize=(10, 10))

    fig.suptitle('Solution at t = %.3f' % t)

    im1 = plots[0][0].imshow(rho, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][0].set_ylabel('y')
    plots[0][0].set_title('Density')
    cbar1 = fig.colorbar(im1, ax=plots[0][0])
    cbar1.set_label(' ')

    im2 = plots[0][1].imshow(p, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][1].set_yticklabels([])
    plots[0][1].set_title('Total Pressure')
    cbar2 = fig.colorbar(im2, ax=plots[0][1])
    cbar2.set_label(' ')

    im3 = plots[1][0].imshow(pgas, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('y')
    plots[1][0].set_title('Thermal Pressure')
    cbar3 = fig.colorbar(im3, ax=plots[1][0])
    cbar3.set_label(' ')

    im4 = plots[1][1].imshow(pmag, cmap='viridis',extent=[0,1,0,1],origin='lower')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_yticklabels([])
    plots[1][1].set_title('Magnetic Pressure')
    cbar4 = fig.colorbar(im4, ax=plots[1][1])
    cbar4.set_label(' ')

    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    plt.show()

def plot_den_contour(q_sys,t):

    rho = q_sys
    fig = plt.figure(figsize=(7,7), dpi=100)
    plt.imshow(rho,extent=[0,1,0,1],origin='lower')
    plt.title('Density at t = %.3f' % t)
    contour = plt.contour(rho,extent=[0,1,0,1], cmap='turbo', levels=25)
    plt.clabel(contour, inline=False, fontsize=12, colors = 'black')

    plt.show()

def load_exact(case):
    '''
    Function Name:      load_exact
    Creator:            Carolyn Wendeln
    Date Created:       04-15-2023
    Date Last Modified: 04-16-2024

    Definition:         loads 'exact' solutions to 1D Riemann Problems
                        credit goes to Qi Tang for the exact solutions
                        https://www.linkedin.com/in/qi-tang-28a64123/

    Inputs:             case: various test problems

    Outputs:            final solution in primitive variables

    Dependencies:       none

    '''

    folder_path = os.path.join(os.getcwd(), 'exact_solutions')

    if case == 3: # Brio-Wu Shock Tube (Qi Tang)
        file_path = os.path.join(folder_path, 'shock_tube.8')
    # if case == 1: # Brio-Wu Shock Tube
    #     file_path = os.path.join(folder_path, 'brio_wu.8')
    # if case == 2: # Figure 2a
    #     file_path = os.path.join(folder_path, 'fig_2a.8')
    # if case == 3: # Reversed Brio-Wu Shock Tube (Qi Tang)
    #     file_path = os.path.join(folder_path, 'reverse_shock_tube.8')

    df = pd.read_csv(file_path, sep="\s+", header=None)  

    den = df[0].values
    vex = df[2].values
    vey = df[3].values
    vez = df[6].values
    pre = df[1].values
    Bx  = df[4].values
    By  = df[5].values
    Bz  = df[7].values

    grid = np.linspace(-1, 1, len(den))

    return den, vex, vey, vez, pre, Bx, By, Bz, grid

def plot_1D(q_sys,t):
    '''
    Function Name:      plot_1D
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 05-01-2023

    Definition:         plot_1D plots the pressure, density, velocity, and
                        total energy for the Euler Equaitons

    Inputs:             q_sys: conserved variables
                        t: current time step

    Outputs:            image of the solution at given time step

    Dependencies:       none
    '''

    exact_den, exact_vex, exact_vey, exact_vez, exact_pre, exact_Bx, exact_By, exact_Bz, exact_grid = load_exact(cfg.case)
    exact_E = exact_pre / (cfg.gamma - 1.) + 0.5 * exact_den * (exact_vex**2 + exact_vey**2 + exact_vez**2) + 0.5 * (exact_Bx**2 + exact_By**2 + exact_Bz**2)
    
    rho = q_sys[0]
    vex = q_sys[1]/q_sys[0]
    vey = q_sys[2]/q_sys[0]
    vez = q_sys[3]/q_sys[0]
    E = q_sys[4]
    Bx = q_sys[5]
    By = q_sys[6]
    Bz = q_sys[7]
    pre = (cfg.gamma-1.)*(E - 0.5*rho*(vex**2 + vey**2 + vez**2) - 0.5*(Bx**2 + By**2 + Bz**2))

    fig, plots = plt.subplots(2, 3, figsize=(18, 8))

    fig.suptitle('Solution at t = %.3f' % t)

    # Top Left Plot
    plots[0][0].plot(exact_grid,exact_den,color = 'gold', linestyle='-',linewidth=2.5,label='Qi Soln.')
    plots[0][0].plot(cfg.xgrid,rho,color = 'crimson', linestyle='-',linewidth=1.0,label='Approx.')
    plots[0][0].set_ylabel('Density')
    plots[0][0].legend()

    # Bottom Left Plot
    plots[1][0].plot(exact_grid,exact_pre,color = 'gold', linestyle='-',linewidth=2.5,label='Qi Soln.')
    plots[1][0].plot(cfg.xgrid,pre,color = 'crimson', linestyle='-',linewidth=1.0,label='Approx.')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('Pressure')
    plots[1][0].legend()

    # Top Middle Plot
    plots[0][1].plot(exact_grid,exact_vex,color = 'gold', linestyle='-',linewidth=2.5,label='Qi Soln.')
    plots[0][1].plot(cfg.xgrid,vex,color = 'crimson', linestyle='-',linewidth=1.0,label='Approx.')
    plots[0][1].set_ylabel('X Velocity')
    plots[0][1].legend()

    # Bottom Middle Plot
    plots[1][1].plot(exact_grid,exact_vey,color = 'gold', linestyle='-',linewidth=2.5,label='Qi Soln.')
    plots[1][1].plot(cfg.xgrid,vey,color = 'crimson', linestyle='-',linewidth=1.0,label='Approx.')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_ylabel('Y Velocity')
    plots[1][1].legend()

    # Top Right Plot
    plots[0][2].plot(exact_grid,exact_By,color = 'gold', linestyle='-',linewidth=2.5,label='Qi Soln.')
    plots[0][2].plot(cfg.xgrid,By,color = 'crimson', linestyle='-',linewidth=1.0,label='Approx.')
    plots[0][2].set_ylabel('Y Magnetic Field')
    plots[0][2].legend()

    # Bottom Right Plot
    plots[1][2].plot(exact_grid,exact_E,color = 'gold', linestyle='-',linewidth=2.5,label='Qi Soln.')
    plots[1][2].plot(cfg.xgrid,E,color = 'crimson', linestyle='-',linewidth=1.0,label='Approx.')
    plots[1][2].set_xlabel('x')
    plots[1][2].set_ylabel('Total Energy')
    plots[1][2].legend()

    plt.show()



    plt.show()

def movie_maker(all_solns,all_t):
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
    fig, plots = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.0})

    n_steps = len(all_t)

    q_sys = all_solns[0]

    rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)
    pgas = (cfg.gamma-1.)*rho*(u**2 + v**2 + w**2) 
    pmag = 0.5*(Bx**2 + By**2 + Bz**2)
   
    fig.suptitle('Solution at t = %.3f' % all_t[0])

    im1 = plots[0][0].imshow(rho, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][0].set_ylabel('y')
    plots[0][0].set_title('Density')
    cbar1 = fig.colorbar(im1, ax=plots[0][0])
    cbar1.set_label(' ')

    im2 = plots[0][1].imshow(p, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[0][1].set_yticklabels([])
    plots[0][1].set_title('Total Pressure')
    cbar2 = fig.colorbar(im2, ax=plots[0][1])
    cbar2.set_label(' ')

    im3 = plots[1][0].imshow(pgas, cmap='viridis', extent=[0, 1, 0, 1],origin='lower')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('y')
    plots[1][0].set_title('Thermal Pressure')
    cbar3 = fig.colorbar(im3, ax=plots[1][0])
    cbar3.set_label(' ')

    im4 = plots[1][1].imshow(pmag, cmap='viridis',extent=[0,1,0,1],origin='lower')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_yticklabels([])
    plots[1][1].set_title('Magnetic Pressure')
    cbar4 = fig.colorbar(im4, ax=plots[1][1])
    cbar4.set_label(' ')

    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    def animate(i):
        ''' function to update the plot for each frame  '''

        q_sys = all_solns[i]

        rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)
        pgas = (cfg.gamma-1.)*rho*(u**2 + v**2 + w**2) 
        pmag = 0.5*(Bx**2 + By**2 + Bz**2)
   
        fig.suptitle('Solution at t = %.3f' % all_t[i])

        im1.set_array(rho)
        im1.set_clim(vmin=rho.min(), vmax=rho.max())

        im2.set_array(p)
        im2.set_clim(vmin=p.min(), vmax=p.max())

        im3.set_array(pgas)
        im3.set_clim(vmin=u.min(), vmax=u.max())

        im4.set_array(pmag)
        im4.set_clim(vmin=v.min(), vmax=v.max())

        fig.subplots_adjust(wspace=0.2, hspace=0.3)

    # Create the animation using matplotlib's FuncAnimation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100)

    folder_path = os.path.join(os.getcwd(), "animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the Primative Plot...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)

def movie_maker_den(all_solns,all_t):
    '''
    Function Name:      movie_maker_den
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 04-24-2023

    Definition:         movie_maker_den plots contour lines for the density for the Euler Equaitons

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

    # fig, ax = plt.figure(figsize=(8,8), dpi=100)
    # fig, ax = plt.subplots(figsize=(8,8), dpi=100)

    n_steps = len(all_t)

    q_sys = all_solns[0]

    rho = q_sys[0]

    fig, ax = plt.subplots()
    im1 = ax.imshow(rho,origin='lower')

    im1.axes.set_xticklabels([])
    im1.axes.set_yticklabels([])

    fig.colorbar(im1, ax=ax)

    contour_lines = ax.contour(rho,cmap='turbo', levels=25)

    contour_labels = plt.clabel(contour_lines, inline=False, fontsize=12, colors = 'black')

    for coll in contour_lines.collections:
        coll.remove()

    for coll in contour_labels:
        coll.remove()
        
    def animate(i):
        ''' Function to update the plot for each frame '''
        global contour_lines
        global contour_labels
        
        q_sys = all_solns[i]

        rho = q_sys[0]
        
        # Update the image
        im1.set_array(rho)
        im1.set_clim(vmin=rho.min(), vmax=rho.max())

        im1.axes.set_xticklabels([])
        im1.axes.set_yticklabels([])
        
        if 'contour_lines' not in globals():
            contour_lines = ax.contour(rho,cmap='turbo', levels=15)
        else:
            # Update the contour lines
            for coll in contour_lines.collections:
                coll.remove()
            contour_lines = ax.contour(rho,cmap='turbo', levels=15)

        if 'contour_labels' not in globals():
            contour_labels = plt.clabel(contour_lines, inline=False, fontsize=12, colors = 'black')
        else:
            for coll in contour_labels:
                coll.remove()
            contour_labels = plt.clabel(contour_lines, inline=False, fontsize=12, colors = 'black')

        # Update the title
        ax.set_title('Solution at t = %.3f' % all_t[i])

        return contour_lines

    # Create the animation using matplotlib's FuncAnimation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100)

    folder_path = os.path.join(os.getcwd(), "animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the Contour Plot...")
    # Save the animation as a video file
    ani.save(file_path,writer='ffmpeg', dpi=500)




