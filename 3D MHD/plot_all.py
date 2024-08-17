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
    # nx1 = 16;     nx2 = 16;    nx3 = 16
    nx1 = 32;     nx2 = 32;    nx3 = 32
    
    filename = f'simulation_data/CFL_0.1_tf_1.0__result_{nx1}_by_{nx2}_by_{nx3}.npz'
    data = np.load(filename)
    
    all_q_sys = data['all_q_sys']
    all_a_sys = data['all_a_sys']
    all_t = data['all_t']
    all_div = data['all_div']

    # movie_maker_mag(all_q_sys,all_a_sys,all_t)
    # movie_maker_div(all_div,all_t)
    # plot_slice(all_q_sys[-1],all_a_sys[-1])
    plot_slice2(all_q_sys[0],all_q_sys[-1])
    plot_div(all_div[-1])
    print('Done!')

def plot_slice2(q1,q2):

    rho1, u1, v1, w1, p1, Bx1, By1, Bz1 = c2p.cons2prim(q1)
    rho2, u2, v2, w2, p2, Bx2, By2, Bz2 = c2p.cons2prim(q2)

    slice = 16

    B1 = np.sqrt(Bx1**2 + By1**2 + Bz1**2)
    B2 = np.sqrt(Bx2**2 + By2**2 + Bz2**2)

    extent = [-0.5, 0.5, -0.5, 0.5]

    # Create a figure with two subplots next to each other
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(B1[:,:,slice], cmap='viridis',origin='lower', extent=extent)
    contour = ax[0].contour(B1[:,:,slice], levels=15, colors='white',extent = extent,linewidths=0.5) 
    ax[0].set_title('$\|\mathbf{B}\|$ at t = 0.0')
    # ax[0].axis('off')  # Turn off axis labels
    ax[1].imshow(B2[:,:,slice], cmap='viridis',origin='lower', extent=extent)
    contour = ax[1].contour(B2[:,:,slice], levels=15, colors='white',extent = extent,linewidths=0.5)  
    ax[1].set_title('$\|\mathbf{B}\|$ at t = 1.0')
    # ax[1].axis('off')  # Turn off axis labels
    plt.tight_layout()
    plt.savefig('images/compare.png',dpi=300)

    plt.show()

def plot_div(div):
    

    # Assuming div is your 3D array and slice and extent are defined
    extent = [-0.5, 0.5, -0.5, 0.5]
    slice = 16

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(div[:, :, slice], cmap='viridis', origin='lower', extent=extent)
    ax.set_title(r'$\nabla \cdot \mathbf{B}$ at t = 1.0')
    # Add contour lines
    contour = ax.contour(div[:, :, slice], levels=15, colors='white', extent=extent, linewidths=0.5)

    # Add colorbar and adjust its height
    cbar = plt.colorbar(im, ax=ax)

    # Get the position of the plot and adjust the colorbar to match the height
    pos = ax.get_position()
    cbar.ax.set_position([pos.x1 + 0.02, pos.y0, 0.03, pos.height])

    # cbar.set_label('Colorbar Label')  # Optional: add a label to the colorbar
    plt.savefig('images/error.png',dpi=300)

    # Show the plot
    plt.show()

def plot_slice(q_sys,a_sys):

    rho, u, v, w, p, Bx, By, Bz = c2p.cons2prim(q_sys)

    slice = 16

    B = np.sqrt(Bx**2 + By**2 + Bz**2)

    A = np.sqrt(a_sys[0]**2 + a_sys[1]**2 + a_sys[2]**2)

    extent = [-0.5, 0.5, -0.5, 0.5]

    # Create a figure with two subplots next to each other
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(B[:,:,slice], cmap='viridis')
    # ax[0].set_title('Image 1')
    # ax[0].axis('off')  # Turn off axis labels
    # ax[1].imshow(A[:,:,slice], cmap='plasma')
    # ax[1].set_title('Image 2')
    # ax[1].axis('off')  # Turn off axis labels
    # plt.tight_layout()


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(B[:,:,slice], cmap='viridis',origin='lower', extent=extent)
    contour = ax.contour(B[:,:,slice], levels=15, colors='white',extent = extent,linewidths=0.5)  
    # ax.set_title('Image')
    # ax.axis('off')  # Turn off axis labels


    # Show the plot
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

    fig.suptitle(r'$\mathbf{B}$ and $\mathbf{A}$ at time $t = %.3f$' % all_t[0])

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

        fig.suptitle(r'$\mathbf{B}$ and $\mathbf{A}$ at time $t = %.3f$' % all_t[i])

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

    print("Saving the B & A Plot...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)

def movie_maker_div(all_div,all_t):
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
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the size as needed

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(' ')
        return cbar

    n_steps = len(all_t)

    div = all_div[0]

    slice = cfg.nx3 // 2
   
    fig.suptitle(r'$\nabla \cdot \mathbf{B}$ at time $t = %.3f$' % all_t[0])

    im1 = ax.imshow(div[:,:,slice], cmap='viridis', extent=[-0.5, 0.5, -0.5, 0.5], origin='lower')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    # ax.set_title('Div B')
    cbar1 = add_colorbar(im1, ax)
    cbar1.set_label(' ')
 
    def animate(i):
        ''' function to update the plot for each frame  '''

        div = all_div[i]
   
        fig.suptitle(r'$\nabla \cdot \mathbf{B}$ at time $t = %.3f$' % all_t[i])

        im1.set_array(div[:,:,slice])
        im1.set_clim(vmin=div.min(), vmax=div.max())

    # Create the animation using matplotlib's FuncAnimation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100)

    folder_path = os.path.join(os.getcwd(), "animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the Div B Plot...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)

if __name__ == "__main__":
    main()