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

    alphax, alphay, alphaz = ev.eigenvalues(q_sys)

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            for k in range(cfg.nghost,(cfg.nx3)+cfg.nghost):
    
                f_l = lf.lf_flux(np.array([q_sys[:,i-3,j,k],q_sys[:,i-2,j,k],q_sys[:,i-1,j,k],q_sys[:,i,j,k],q_sys[:,i+1,j,k],q_sys[:,i+2,j,k]]),alphax,1,0,0)
                f_r = lf.lf_flux(np.array([q_sys[:,i-2,j,k],q_sys[:,i-1,j,k],q_sys[:,i,j,k],q_sys[:,i+1,j,k],q_sys[:,i+2,j,k],q_sys[:,i+3,j,k]]),alphax,1,0,0)

                g_l = lf.lf_flux(np.array([q_sys[:,i,j-3,k],q_sys[:,i,j-2,k],q_sys[:,i,j-1,k],q_sys[:,i,j,k],q_sys[:,i,j+1,k],q_sys[:,i,j+2,k]]),alphay,0,1,0)
                g_r = lf.lf_flux(np.array([q_sys[:,i,j-2,k],q_sys[:,i,j-1,k],q_sys[:,i,j,k],q_sys[:,i,j+1,k],q_sys[:,i,j+2,k],q_sys[:,i,j+3,k]]),alphay,0,1,0)

                h_l = lf.lf_flux(np.array([q_sys[:,i,j,k-3],q_sys[:,i,j,k-2],q_sys[:,i,j,k-1],q_sys[:,i,j,k],q_sys[:,i,j,k+1],q_sys[:,i,j,k+2]]),alphaz,0,0,1)
                h_r = lf.lf_flux(np.array([q_sys[:,i,j,k-2],q_sys[:,i,j,k-1],q_sys[:,i,j,k],q_sys[:,i,j,k+1],q_sys[:,i,j,k+2],q_sys[:,i,j,k+3]]),alphaz,0,0,1)

                # Q^{*}_{MHD}
                q_new[:,i,j,k] =  (f_r - f_l)/cfg.dx  +  (g_r - g_l)/cfg.dy + (h_r - h_l)/cfg.dz

    return -1*q_new

def mathcal_H(a_sys,q_sys,dt):

    a_new = np.zeros_like(a_sys)

    q_sys, a_sys = bc.boundary_conditions(q_sys, a_sys)

    max_vex = np.max(q_sys[1,:,:,:]) / np.max(q_sys[0,:,:,:]) # alpha^1
    max_vey = np.max(q_sys[2,:,:,:]) / np.max(q_sys[0,:,:,:]) # alpha^2
    max_vez = np.max(q_sys[3,:,:,:]) / np.max(q_sys[0,:,:,:]) # alpha^3

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            for k in range(cfg.nghost,(cfg.nx3)+cfg.nghost):
    
                epsilon = 1.0e-8
                nu = 0.1
                delta = 0.0

                # u^{x}_{ijk}, u^{y}_{ijk}, u^{z}_{ijk}
                vex = q_sys[1,i,j,k]/q_sys[0,i,j,k]
                vey = q_sys[2,i,j,k]/q_sys[0,i,j,k]
                vez = q_sys[3,i,j,k]/q_sys[0,i,j,k]

                # A^{x+}_{xijk} and A^{x-}_{xijk}
                Ax_xp, Ax_xm = hj.hj_flux(a_sys[0,i-3,j,k],a_sys[0,i-2,j,k],a_sys[0,i-1,j,k],a_sys[0,i,j,k],a_sys[0,i+1,j,k],a_sys[0,i+2,j,k],a_sys[0,i+3,j,k],cfg.dx)
                # A^{y+}_{xijk} and A^{y-}_{xijk}
                Ay_xp, Ay_xm = hj.hj_flux(a_sys[1,i-3,j,k],a_sys[1,i-2,j,k],a_sys[1,i-1,j,k],a_sys[1,i,j,k],a_sys[1,i+1,j,k],a_sys[1,i+2,j,k],a_sys[1,i+3,j,k],cfg.dx)
                # A^{z+}_{xijk} and A^{z-}_{xijk}
                Az_xp, Az_xm = hj.hj_flux(a_sys[2,i-3,j,k],a_sys[2,i-2,j,k],a_sys[2,i-1,j,k],a_sys[2,i,j,k],a_sys[2,i+1,j,k],a_sys[2,i+2,j,k],a_sys[2,i+3,j,k],cfg.dx)

                # A^{x+}_{yijk} and A^{x-}_{yijk}
                Ax_yp, Ax_ym = hj.hj_flux(a_sys[0,i,j-3,k],a_sys[0,i,j-2,k],a_sys[0,i,j-1,k],a_sys[0,i,j,k],a_sys[0,i,j+1,k],a_sys[0,i,j+2,k],a_sys[0,i,j+3,k],cfg.dy)
                # A^{y+}_{yijk} and A^{y-}_{yijk}
                Ay_yp, Ay_ym = hj.hj_flux(a_sys[1,i,j-3,k],a_sys[1,i,j-2,k],a_sys[1,i,j-1,k],a_sys[1,i,j,k],a_sys[1,i,j+1,k],a_sys[1,i,j+2,k],a_sys[1,i,j+3,k],cfg.dy)
                # A^{z+}_{yijk} and A^{z-}_{yijk}
                Az_yp, Az_ym = hj.hj_flux(a_sys[2,i,j-3,k],a_sys[2,i,j-2,k],a_sys[2,i,j-1,k],a_sys[2,i,j,k],a_sys[2,i,j+1,k],a_sys[2,i,j+2,k],a_sys[2,i,j+3,k],cfg.dy)

                # A^{x+}_{zijk} and A^{x-}_{zijk}
                Ax_zp, Ax_zm = hj.hj_flux(a_sys[0,i,j,k-3],a_sys[0,i,j,k-2],a_sys[0,i,j,k-1],a_sys[0,i,j,k],a_sys[0,i,j,k+1],a_sys[0,i,j,k+2],a_sys[0,i,j,k+3],cfg.dz)
                # A^{y+}_{zijk} and A^{y-}_{zijk}
                Ay_zp, Ay_zm = hj.hj_flux(a_sys[1,i,j,k-3],a_sys[1,i,j,k-2],a_sys[1,i,j,k-1],a_sys[1,i,j,k],a_sys[1,i,j,k+1],a_sys[1,i,j,k+2],a_sys[1,i,j,k+3],cfg.dz)
                # A^{z+}_{zijk} and A^{z-}_{zijk}
                Az_zp, Az_zm = hj.hj_flux(a_sys[2,i,j,k-3],a_sys[2,i,j,k-2],a_sys[2,i,j,k-1],a_sys[2,i,j,k],a_sys[2,i,j,k+1],a_sys[2,i,j,k+2],a_sys[2,i,j,k+3],cfg.dz)

                axm = (epsilon + (cfg.dx * Ax_xm)**2.0 )**(-2.0)
                axp = (epsilon + (cfg.dx * Ax_xp)**2.0 )**(-2.0)
                gamma_1 = np.abs( (axm / (axm + axp)) - 0.5 )

                aym = (epsilon + (cfg.dy * Ay_ym)**2.0 )**(-2.0)
                ayp = (epsilon + (cfg.dy * Ay_yp)**2.0 )**(-2.0)
                gamma_2 = np.abs( (aym / (aym + ayp)) - 0.5 )

                azm = (epsilon + (cfg.dz * Az_zm)**2.0 )**(-2.0)
                azp = (epsilon + (cfg.dz * Az_zp)**2.0 )**(-2.0)
                gamma_3 = np.abs( (azm / (azm + azp)) - 0.5 )

                a_new[0,i,j,k] = vey * (Ay_xm + Ay_xp) * 0.5 + vez * (Az_xm + Az_xp) * 0.5 \
                    + (2 * nu * gamma_1)*((a_sys[0,i-1,j,k] - 2*a_sys[0,i,j,k] + a_sys[0,i+1,j,k])/(delta+dt)) \
                    - vey * (Ax_ym + Ax_yp) * 0.5 - vez * (Ax_zm + Ax_zp) * 0.5 \
                    + max_vey*(Ax_yp - Ax_ym)*0.5 + max_vez*(Ax_zp - Ax_zm)*0.5
                
                a_new[1,i,j,k] = vex * (Ax_ym + Ax_yp) * 0.5 + vez * (Az_ym + Az_yp) * 0.5 \
                    + (2 * nu * gamma_2)*((a_sys[1,i,j-1,k] - 2*a_sys[1,i,j,k] + a_sys[1,i,j+1,k])/(delta+dt)) \
                    - vex * (Ay_xm + Ay_xp) * 0.5 - vez * (Ay_zm + Ay_zp) * 0.5 \
                    + max_vex*(Ay_xp - Ay_xm)*0.5 + max_vez*(Ay_zp - Ay_zm)*0.5
                
                a_new[2,i,j,k] = vex * (Ax_zm + Ax_zp) * 0.5 + vey * (Ay_zm + Ay_zp) * 0.5 \
                    + (2 * nu * gamma_3)*((a_sys[2,i,j,k-1] - 2*a_sys[2,i,j,k] + a_sys[2,i,j,k+1])/(delta+dt)) \
                    - vex * (Az_xm + Az_xp) * 0.5 - vey * (Az_ym + Az_yp) * 0.5 \
                    + max_vex*(Az_xp - Az_xm)*0.5 + max_vey*(Az_yp - Az_ym)*0.5
               
    return a_new

def afterstep(q_sys,a_sys):

    q_sys, a_sys = bc.boundary_conditions(q_sys, a_sys)

    q_new = np.copy(q_sys)
    
    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            for k in range(cfg.nghost,(cfg.nx3)+cfg.nghost):

                q_new[5,i,j,k] = (1/(12.0*cfg.dy))*(a_sys[2,i,j-2,k] - 8.0*a_sys[2,i,j-1,k] + 8.0*a_sys[2,i,j+1,k] - a_sys[2,i,j+2,k]) \
                               - (1/(12.0*cfg.dz))*(a_sys[1,i,j,k-2] - 8.0*a_sys[1,i,j,k-1] + 8.0*a_sys[1,i,j,k+1] - a_sys[1,i,j,k+2])

                q_new[6,i,j,k] = (1/(12.0*cfg.dz))*(a_sys[0,i,j,k-2] - 8.0*a_sys[0,i,j,k-1] + 8.0*a_sys[0,i,j,k+1] - a_sys[0,i,j,k+2]) \
                               - (1/(12.0*cfg.dx))*(a_sys[2,i-2,j,k] - 8.0*a_sys[2,i-1,j,k] + 8.0*a_sys[2,i+1,j,k] - a_sys[2,i+2,j,k])  

                q_new[7,i,j,k] = (1/(12.0*cfg.dx))*(a_sys[1,i-2,j,k] - 8.0*a_sys[1,i-1,j,k] + 8.0*a_sys[1,i+1,j,k] - a_sys[1,i+2,j,k]) \
                               - (1/(12.0*cfg.dy))*(a_sys[0,i,j-2,k] - 8.0*a_sys[0,i,j-1,k] + 8.0*a_sys[0,i,j+1,k] - a_sys[0,i,j+2,k])
    return q_new

def fe(q_sys,a_sys,dt):
    
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

def ct_rk3(q_sys,a_z,dt):
    '''
    Equation (2.18) from the following paper:
    "Efficient Implementation of Essentially Non-Oscillatory Shock Capturing Schemes"
    By C.-W. Shu and S. Osher
    '''
    
    # q1 = q_sys + dt*mathcal_L(q_sys)
    q1_star = q_sys + dt*mathcal_L(q_sys)
    a1 = a_z + dt*mathcal_H(a_z,q_sys,dt)
    q1 = afterstep(q1_star,a1)

    # q2 = ((3/4)*q_sys) + ((1/4)*q1) + ((1/4)*dt*mathcal_L(q1))
    q2_star = ((3/4)*q_sys) + ((1/4)*q1) + ((1/4)*dt*mathcal_L(q1))
    a2 = ((3/4)*a_z) + ((1/4)*a1) + ((1/4)*dt*mathcal_H(a1,q1,dt))
    q2 = afterstep(q2_star,a2)

    # q_new = ((1/3)*q_sys) + ((2/3)*q2) + ((2/3)*dt*mathcal_L(q2))
    q_new_star = ((1/3)*q_sys) + ((2/3)*q2) + ((2/3)*dt*mathcal_L(q2))
    a_new = ((1/3)*a_z) + ((2/3)*a2) + ((2/3)*dt*mathcal_H(a2,q2,dt))
    q_new = afterstep(q_new_star,a_new)

    return q_new, a_new

# NOT UPDATED YET
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

    