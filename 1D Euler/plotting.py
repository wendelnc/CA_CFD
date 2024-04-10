# Standard Python Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# User Defined Libraries
import configuration as cfg

def load_exact(case):
    '''
    Function Name:      load_exact
    Creator:            Carolyn Wendeln
    Date Created:       04-15-2023
    Date Last Modified: 05-01-2023

    Definition:         loads exact solution to 1D Riemann Problems
                        credit goes to Manuel Diaz
                        https://github.com/wme7/MUSCL/tree/master/1D_MUSCL

    Inputs:             case: various test problems

    Outputs:            final solution for density, velocity, pressure, and total energy

    Dependencies:       none

    '''

    folder_path = os.path.join(os.getcwd(), 'exact_solutions')

    if case == 0: # Sod Shock Tube
        file_path = os.path.join(folder_path, 'sod_shock_tube.txt')
    if case == 1: # Mach = 3 test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997
        file_path = os.path.join(folder_path, 'mach_3.txt')
    if case == 2: # Shocktube problem of G.A. Sod, JCP 27:1, 1978
        file_path = os.path.join(folder_path, 'g_a_sod.txt')
    if case == 3: # Lax test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997
        file_path = os.path.join(folder_path, 'lax_test_case.txt')
    if case == 4: # Shocktube problem with supersonic zone
        file_path = os.path.join(folder_path, 'supersonic.txt')
    if case == 5: # Stationary shock
        file_path = os.path.join(folder_path, 'stationary_shock.txt')
    if case == 6: # Double Expansion
        file_path = os.path.join(folder_path, 'double_expansion.txt')
    if case == 7: # Reverse Sod Shock Tube
        file_path = os.path.join(folder_path, 'sod_shock_tube.txt')


    euler_exact = pd.read_csv(file_path, header = 0, delimiter='\t')
    grid = euler_exact['xe']
    rho = euler_exact['re']
    u = euler_exact['ue']
    p = euler_exact['pe']
    E = euler_exact['Ee']

    if case == 7:
        rho = np.flip(rho)
        # flip u in both x and y
        u = -1 * np.flip(u)
        p = np.flip(p)
        E = np.flip(E)

    return grid, rho, u, p, E

def plot_solution(q_sys,t):

    '''
    Function Name:      plot_solution
    Creator:            Carolyn Wendeln
    Date Created:       02-15-2023
    Date Last Modified: 05-01-2023

    Definition:         plot_solution plots the pressure, density, velocity, and
                        total energy for the Euler Equaitons

    Inputs:             q_sys: conserved variables
                        t: current time step

    Outputs:            image of the solution at given time step

    Dependencies:       none
    '''

    exact_grid, exact_rho, exact_u, exact_p, exact_E = load_exact(cfg.case)

    # print("exact_grid = np.array({})".format(exact_grid.tolist()))
    # print("exact_rho = np.array({})".format(exact_rho.tolist()))
    # print("exact_u = np.array({})".format(exact_u.tolist()))
    # print("exact_p = np.array({})".format(exact_p.tolist()))
    # print("exact_E = np.array({})".format(exact_E.tolist()))
    
    rho = q_sys[0,:]
    u = q_sys[1,:]/q_sys[0,:]
    E = q_sys[2,:]
    p = (cfg.gamma-1.)*(E - 0.5*rho*u**2.)

    fig, plots = plt.subplots(2, 2, figsize=(18, 6))

    fig.suptitle('Solution at t = %.3f' % t)

    plots[0][0].plot(exact_grid,exact_rho,color = 'gold', linestyle='-',linewidth=2.5,label='Exact Soln.')
    plots[0][0].plot(cfg.grid,rho,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    plots[0][0].set_xlabel('x')
    plots[0][0].set_ylabel('Density')
    plots[0][0].legend()

    plots[0][1].plot(exact_grid,exact_u,color = 'gold', linestyle='-',linewidth=2.5,label='Exact Soln.')
    plots[0][1].plot(cfg.grid,u,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    plots[0][1].set_xlabel('x')
    plots[0][1].set_ylabel('Velocity')
    plots[0][1].legend()

    plots[1][0].plot(exact_grid,exact_p,color = 'gold', linestyle='-',linewidth=2.5,label='Exact Soln.')
    plots[1][0].plot(cfg.grid,p,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    plots[1][0].set_xlabel('x')
    plots[1][0].set_ylabel('Pressure')
    plots[1][0].legend()

    plots[1][1].plot(exact_grid,exact_E*exact_rho,color = 'gold', linestyle='-',linewidth=2.5,label='Exact Soln.')
    plots[1][1].plot(cfg.grid,E,color = 'crimson', linestyle='-',linewidth=1.5,label='Approx.')
    plots[1][1].set_xlabel('x')
    plots[1][1].set_ylabel('Total Energy')
    plots[1][1].legend()

    plt.show()


