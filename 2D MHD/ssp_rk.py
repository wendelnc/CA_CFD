# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg       # Input Parameters
import boundary_conditions as bc  # Update Boundary Conditions
import eigenvalues as ev          # Compute Eigenvalues
import lf_flux as lf              # Compute Lax-Friedrichs Flux Vector Splitting
import hj_flux as hj              # Compute Hamilton-Jacobi Lax-Friedrichs Flux Vector Splitting

def mathcal_L(q_sys):
    
    q_new = np.zeros_like(q_sys)
    
    q_sys = bc.periodic_bc(q_sys)

    alphax, alphay = ev.eigenvalues(q_sys)

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            
            f_l = lf.lf_flux(np.array([q_sys[:,i-3,j],q_sys[:,i-2,j],q_sys[:,i-1,j],q_sys[:,i,j],q_sys[:,i+1,j],q_sys[:,i+2,j]]),alphax,1,0,0)
            f_r = lf.lf_flux(np.array([q_sys[:,i-2,j],q_sys[:,i-1,j],q_sys[:,i,j],q_sys[:,i+1,j],q_sys[:,i+2,j],q_sys[:,i+3,j]]),alphax,1,0,0)
            
            g_l = lf.lf_flux(np.array([q_sys[:,i,j-3],q_sys[:,i,j-2],q_sys[:,i,j-1],q_sys[:,i,j],q_sys[:,i,j+1],q_sys[:,i,j+2]]),alphay,0,1,0)
            g_r = lf.lf_flux(np.array([q_sys[:,i,j-2],q_sys[:,i,j-1],q_sys[:,i,j],q_sys[:,i,j+1],q_sys[:,i,j+2],q_sys[:,i,j+3]]),alphay,0,1,0)
            
            # Q^{*}_{MHD}
            q_new[:,i,j] =  (f_r - f_l)/cfg.dx  +  (g_r - g_l)/cfg.dy

    return -1*q_new

def mathcal_H(a_sys,q_sys):

    a_new = np.zeros_like(a_sys)

    q_sys = bc.periodic_bc(q_sys)
    a_sys = bc.ct_periodic_bc(a_sys)

    max_vex = np.max(q_sys[1,:,:]) / np.max(q_sys[0,:,:]) # alpha^1
    max_vey = np.max(q_sys[2,:,:]) / np.max(q_sys[0,:,:]) # alpha^2

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
    
            Axp, Axm = hj.hj_flux(a_sys[2,i-3,j],a_sys[2,i-2,j],a_sys[2,i-1,j],a_sys[2,i,j],a_sys[2,i+1,j],a_sys[2,i+2,j],a_sys[2,i+3,j],cfg.dx)
            Ayp, Aym = hj.hj_flux(a_sys[2,i,j-3],a_sys[2,i,j-2],a_sys[2,i,j-1],a_sys[2,i,j],a_sys[2,i,j+1],a_sys[2,i,j+2],a_sys[2,i,j+3],cfg.dy)

            # Q^{n+1}_{A}
            a_new[2,i,j] = (q_sys[1,i,j]/q_sys[0,i,j])*(Axp + Axm)*0.5 + (q_sys[2,i,j]/q_sys[0,i,j])*(Ayp + Aym)*0.5 \
                - max_vex*(Axp-Axm)*0.5 - max_vey*(Ayp-Aym)*0.5
    
    return -1*a_new

def afterstep(q_sys,a_sys):

    q_sys = bc.periodic_bc(q_sys)
    a_sys = bc.ct_periodic_bc(a_sys)
    
    q_new = np.copy(q_sys)
    
    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            q_new[5,i,j] = (1/(12.0*cfg.dy))*(a_sys[2,i,j-2] - 8.0*a_sys[2,i,j-1] + 8.0*a_sys[2,i,j+1] - a_sys[2,i,j+2]) 
            q_new[6,i,j] = (1/(12.0*cfg.dx))*(a_sys[2,i+2,j] - 8.0*a_sys[2,i+1,j] + 8.0*a_sys[2,i-1,j] - a_sys[2,i-2,j])
    
    return q_new


def ct_fe(q_sys,a_sys,dt):
    
    # 0. Start with Q^{n}_{MHD} and Q^{n}_{A}

    # 1. Build the right hand sides of both semi-discrete systems
        #
        #   Q^{*}_{MHD} = Q^{n}_{MHD} + ∆t \mathcal{L}(Q^{n}_{MHD})
        #   Q^{n+1}_{A} = Q^{n}_{A}   + ∆t \mathcal{H}(Q^{n}_{A}, \textbf{u}^{n})
        #
        #   where Q^{*}_{MHD} = (ρ^{n+1}, ρ\textbf{u}^{n+1}, E^{*}, B^{*})
    q_star = q_sys + dt*mathcal_L(q_sys)
    a_new = a_sys + dt*mathcal_H(a_sys,q_sys)
    
    # 2. Correct B^{*} by the magnetic potential Q^{n+1}_{A} at new stage by a discrete curl operator
    q_new = afterstep(q_star,a_new)

    # 3. Set the total energy density E^{n+1} via Option 1
    #    where E^{n+1} = E^{*}

    return q_new, a_new

def rk3(q_sys,a_sys,dt):
    '''
    Equation (2.18) from the following paper:
    "Efficient Implementation of Essentially Non-Oscillatory Shock Capturing Schemes"
    By C.-W. Shu and S. Osher
    '''
    q1 = q_sys + dt*mathcal_L(q_sys)
    q2 = ((3/4)*q_sys) + ((1/4)*q1) + ((1/4)*dt*mathcal_L(q1))
    q_new = ((1/3)*q_sys) + ((2/3)*q2) + ((2/3)*dt*mathcal_L(q2))

    a_new = np.copy(a_sys)

    return q_new, a_new

def ct_rk3(q_sys,a_sys,dt):
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

    