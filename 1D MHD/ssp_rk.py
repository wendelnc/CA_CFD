# Standard Python Libraries
import numpy as np
# from numba import njit

# User Defined Libraries
import configuration as cfg       # Input Parameters
import boundary_conditions as bc  # Update Boundary Conditions
import eigenvalues as ev          # Compute Eigenvalues
import lf_flux as lf              # Compute Lax-Friedrichs Flux Vector Splitting
import qi_lf_flux as qlf            # Compute Lax-Friedrichs Flux Vector Splitting

def mathcal_L(q_sys):
    '''
    Our approximation of the spatial derivative
    '''

    q_new = np.zeros_like(q_sys)  

    q_sys = bc.boundary_conditions(q_sys)

    alpha = ev.eigenvalues(q_sys)

    for i in range(cfg.nghost,(cfg.nx1+1)+cfg.nghost):
        f_l = qlf.qi_lf_flux(np.array([q_sys[:,i-3],q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2]]),alpha, 1, 0, 0)
        f_r = qlf.qi_lf_flux(np.array([q_sys[:,i-2],q_sys[:,i-1],q_sys[:,i],q_sys[:,i+1],q_sys[:,i+2],q_sys[:,i+3]]),alpha, 1, 0, 0)
        q_new[:,i] = (f_r - f_l) / cfg.dx  

    return -1*q_new

def rk3(q_sys, dt):
    '''
    Equation (2.18) from the following paper:
    "Efficient Implementation of Essentially Non-Oscillatory Shock Capturing Schemes"
    By C.-W. Shu and S. Osher
    '''

    q1 = q_sys + (dt * mathcal_L(q_sys))
    q2 = ((3/4)*q_sys) + ((1/4)*q1) + ((1/4)*dt*mathcal_L(q1))
    q_new = ((1/3)*q_sys) + ((2/3)*q2) + ((2/3)*dt*mathcal_L(q2))

    return q_new

def rk4(q_sys, dt):
    '''
    Pseudocode 3 from the following paper:
    "HIGHLY EFFICIENT STRONG STABILITY-PRESERVING RUNGE-KUTTA METHODS WITH LOW-STORAGE IMPLEMENTATIONS" 
    By David I. Ketcheson 
    '''

    # Initialize q1 and q2
    q1 = q_sys.copy()
    q2 = q_sys.copy()

    # Loop from i = 1 to 5
    for i in range(1, 6):
        q1 = q1 + (dt * mathcal_L(q1) / 6)

    q2 = (1/25 * q2) + (9/25 * q1)
    q1 = (15 * q2) - (5 * q1)

    # Loop from i = 6 to 9
    for i in range(6, 10):
        q1 = q1 + dt * mathcal_L(q1) / 6

    q1 = q2 + (3/5 * q1) + (1/10 * dt * mathcal_L(q1))

    q_sys = np.copy(q1)

    return q_sys

    