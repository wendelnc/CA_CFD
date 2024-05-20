# Standard Python Libraries
import os
import datetime
import numpy as np
import pandas as pd
from time import time, sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# User Defined Libraries
import configuration as cfg

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

    if case == 0: # Brio-Wu Shock Tube (Qi Tang)
        file_path = os.path.join(folder_path, 'shock_tube.8')
    if case == 1: # Brio-Wu Shock Tube
        file_path = os.path.join(folder_path, 'brio_wu.8')
    if case == 2: # Figure 2a
        file_path = os.path.join(folder_path, 'fig_2a.8')
    if case == 3: # Reversed Brio-Wu Shock Tube (Qi Tang)
        file_path = os.path.join(folder_path, 'reverse_shock_tube.8')

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

    exact_den, exact_vex, exact_vey, exact_vez, exact_pre, exact_Bx, exact_By, exact_Bz, exact_grid = load_exact(cfg.case)
    exact_E = exact_pre / (cfg.gamma - 1.) + 0.5 * exact_den * (exact_vex**2 + exact_vey**2 + exact_vez**2) + 0.5 * (exact_Bx**2 + exact_By**2 + exact_Bz**2)
    
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

    n_steps = len(all_t)

    q_sys = all_solns[0]

    exact_den, exact_vex, exact_vey, exact_vez, exact_pre, exact_Bx, exact_By, exact_Bz, exact_grid = load_exact(cfg.case)
    exact_E = exact_pre / (cfg.gamma - 1.) + 0.5 * exact_den * (exact_vex**2 + exact_vey**2 + exact_vez**2) + 0.5 * (exact_Bx**2 + exact_By**2 + exact_Bz**2)
    
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

    fig.suptitle('Solution at t = %.3f' % all_t[0])

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

    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    def animate(i):
        ''' function to update the plot for each frame  '''


        plots[0][0].clear()
        plots[1][0].clear()
        plots[0][1].clear()
        plots[1][1].clear()
        plots[0][2].clear()
        plots[1][2].clear()

        q_sys = all_solns[i]

        rho = q_sys[0,:]
        vex = q_sys[1,:]/q_sys[0,:]
        vey = q_sys[2,:]/q_sys[0,:]
        vez = q_sys[3,:]/q_sys[0,:]
        E = q_sys[4,:]
        Bx = q_sys[5,:]
        By = q_sys[6,:]
        Bz = q_sys[7,:]
        pre = (cfg.gamma-1.)*(E - 0.5*rho*(vex**2 + vey**2 + vez**2) - 0.5*(Bx**2 + By**2 + Bz**2))

        fig.suptitle('Solution at t = %.3f' % all_t[i])

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

        fig.subplots_adjust(wspace=0.2, hspace=0.3)

    # Create the animation using matplotlib's FuncAnimation
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100)

    folder_path = os.path.join(os.getcwd(), "animations")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, movie_title)

    print("Saving the animation...")
    # Save the animation as a video file
    
    ani.save(file_path,writer='ffmpeg', dpi=500)

