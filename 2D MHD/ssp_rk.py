# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg       # Input Parameters
import boundary_conditions as bc  # Update Boundary Conditions
import eigenvalues as ev          # Compute Eigenvalues
import lf_flux as lf              # Compute Lax-Friedrichs Flux Vector Splitting
import hj_flux as hj              # Compute Hamilton-Jacobi Lax-Friedrichs Flux Vector Splitting

# @njit
def one_step(q_sys,a_sys,dt):
    '''
    Our approximation of the spatial derivative
    '''
    q_star = np.zeros_like(q_sys)
    a_new = np.zeros_like(a_sys)

    q_sys = bc.boundary_conditions(q_sys)
    a_sys = bc.boundary_conditions(a_sys)

    alphax, alphay = ev.eigenvalues(q_sys)
    max_vex = alphax[0]
    max_vey = alphay[0]

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            
            f_l = lf.lf_flux(np.array([q_sys[:,j,i-3],q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2]]),alphax,1,0,0)
            f_r = lf.lf_flux(np.array([q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2],q_sys[:,j,i+3]]),alphax,1,0,0)
            g_l = lf.lf_flux(np.array([q_sys[:,j-3,i],q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i]]),alphay,0,1,0)
            g_r = lf.lf_flux(np.array([q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i],q_sys[:,j+3,i]]),alphay,0,1,0)
            
            # Q^{*}_{MHD}
            q_star[:,j,i] = q_sys[:,j,i] - ((dt)/(cfg.dx))*(f_r - f_l)  -  ((dt)/(cfg.dy))*(g_r - g_l) 

            Axp, Axm = hj.hj_flux(a_sys[2,j,i-3],a_sys[2,j,i-2],a_sys[2,j,i-1],a_sys[2,j,i],a_sys[2,j,i+1],a_sys[2,j,i+2],a_sys[2,j,i+3],cfg.dx)
            Ayp, Aym = hj.hj_flux(a_sys[2,j-3,i],a_sys[2,j-2,i],a_sys[2,j-1,i],a_sys[2,j,i],a_sys[2,j+1,i],a_sys[2,j+2,i],a_sys[2,j+3,i],cfg.dy)

            # Q^{n+1}_{A}
            a_new[2,j,i] = a_sys[2,j,i] - dt*(q_sys[1,j,i]/q_sys[0,j,i])*(Axp + Axm)*0.5 - dt*(q_sys[2,j,i]/q_sys[0,j,i])*(Ayp + Aym)*0.5 \
                + dt*max_vex*(Axp-Axm)*0.5 + dt*max_vey*(Ayp-Aym)*0.5
    
    a_new = bc.boundary_conditions(a_new)
    q_new = np.copy(q_star)

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            # Q^{n+1}_{MHD}
            q_new[5,j,i] = (1/(12.0*cfg.dy))*(a_new[2,j-2,i] - 8.0*a_new[2,j-1,i] + 8.0*a_new[2,j+1,i] - a_new[2,j+2,i]) 
            q_new[6,j,i] = (1/(12.0*cfg.dx))*(a_new[2,j,i+2] - 8.0*a_new[2,j,i+1] + 8.0*a_new[2,j,i-1] - a_new[2,j,i-2])

    return q_new, a_new


def mathcal_L(q_sys):
    
    q_new = np.zeros_like(q_sys)
    
    q_sys = bc.boundary_conditions(q_sys)

    alphax, alphay = ev.eigenvalues(q_sys)

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            
            f_l = lf.lf_flux(np.array([q_sys[:,j,i-3],q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2]]),alphax,1,0,0)
            f_r = lf.lf_flux(np.array([q_sys[:,j,i-2],q_sys[:,j,i-1],q_sys[:,j,i],q_sys[:,j,i+1],q_sys[:,j,i+2],q_sys[:,j,i+3]]),alphax,1,0,0)
            g_l = lf.lf_flux(np.array([q_sys[:,j-3,i],q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i]]),alphay,0,1,0)
            g_r = lf.lf_flux(np.array([q_sys[:,j-2,i],q_sys[:,j-1,i],q_sys[:,j,i],q_sys[:,j+1,i],q_sys[:,j+2,i],q_sys[:,j+3,i]]),alphay,0,1,0)
            
            # Q^{*}_{MHD}
            q_new[:,j,i] =  (f_r - f_l)/cfg.dx  +  (g_r - g_l)/cfg.dy

    return -1*q_new

def mathcal_H(a_sys,q_sys):

    a_new = np.zeros_like(a_sys)

    a_sys = bc.boundary_conditions(a_sys)
    q_sys = bc.boundary_conditions(q_sys)

    max_vex = np.max(q_sys[1,:,:]) / np.max(q_sys[0,:,:])
    max_vey = np.max(q_sys[2,:,:]) / np.max(q_sys[0,:,:])

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
    
            Axp, Axm = hj.hj_flux(a_sys[2,j,i-3],a_sys[2,j,i-2],a_sys[2,j,i-1],a_sys[2,j,i],a_sys[2,j,i+1],a_sys[2,j,i+2],a_sys[2,j,i+3],cfg.dx)
            Ayp, Aym = hj.hj_flux(a_sys[2,j-3,i],a_sys[2,j-2,i],a_sys[2,j-1,i],a_sys[2,j,i],a_sys[2,j+1,i],a_sys[2,j+2,i],a_sys[2,j+3,i],cfg.dy)

            # Q^{n+1}_{A}
            a_new[2,j,i] = (q_sys[1,j,i]/q_sys[0,j,i])*(Axp + Axm)*0.5 + (q_sys[2,j,i]/q_sys[0,j,i])*(Ayp + Aym)*0.5 \
                - max_vex*(Axp-Axm)*0.5 - max_vey*(Ayp-Aym)*0.5
    
    return -1*a_new

def afterstep(q_sys,a_sys):

    a_sys = bc.boundary_conditions(a_sys)

    q_new = np.copy(q_sys)
    
    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            q_new[5,j,i] = (1/(12.0*cfg.dy))*(a_sys[2,j-2,i] - 8.0*a_sys[2,j-1,i] + 8.0*a_sys[2,j+1,i] - a_sys[2,j+2,i]) 
            q_new[6,j,i] = (1/(12.0*cfg.dx))*(a_sys[2,j,i+2] - 8.0*a_sys[2,j,i+1] + 8.0*a_sys[2,j,i-1] - a_sys[2,j,i-2])
    
    return q_new


def fe(q_sys,a_sys,dt):

    q_new, a_new = one_step(q_sys,a_sys,dt)

    return q_new, a_new


def rk3(q_sys,a_sys,dt):
    '''
    Equation (2.18) from the following paper:
    "Efficient Implementation of Essentially Non-Oscillatory Shock Capturing Schemes"
    By C.-W. Shu and S. Osher
    '''
    # q1 = q_sys + dt*mathcal_L(q_sys)
    q1_star = q_sys + dt*mathcal_L(q_sys)
    a1 = a_sys + dt*mathcal_H(a_sys,q_sys)
    q1 = afterstep(q1_star,a1)

    # q2 = ((3/4)*q_sys) + ((1/4)*q1) + ((1/4)*dt*mathcal_L(q1))
    q2_star = ((3/4)*q_sys) + ((1/4)*q1) + ((1/4)*dt*mathcal_L(q1))
    a2 = ((3/4)*a_sys) + ((1/4)*a1) + ((1/4)*dt*mathcal_H(a1,q1))
    q2 = afterstep(q2_star,a2)

    # q_new = ((1/3)*q_sys) + ((2/3)*q2) + ((2/3)*dt*mathcal_L(q2))
    q_new_star = ((1/3)*q_sys) + ((2/3)*q2) + ((2/3)*dt*mathcal_L(q2))
    a_new = ((1/3)*a_sys) + ((2/3)*a2) + ((2/3)*dt*mathcal_H(a2,q2))
    q_new = afterstep(q_new_star,a_new)

    return q_new, a_new


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

    