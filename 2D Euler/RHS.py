# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import lf_flux as lf             # Compute Lax-Friedrichs Flux Vector Splitting

@njit
def RHS(q_sys,q_sys_new,dt):

    for i in range(cfg.nghost,cfg.nx2+cfg.nghost):
            for j in range(cfg.nghost,cfg.nx1+cfg.nghost):

                # Euler Equations
                f_l = lf.lf_flux(q_sys[:,i-3,j],q_sys[:,i-2,j],q_sys[:,i-1,j],q_sys[:,i,j],q_sys[:,i+1,j],q_sys[:,i+2,j],0,1)
                f_r = lf.lf_flux(q_sys[:,i-2,j],q_sys[:,i-1,j],q_sys[:,i,j],q_sys[:,i+1,j],q_sys[:,i+2,j],q_sys[:,i+3,j],0,1)
                g_l = lf.lf_flux(q_sys[:,i,j-3],q_sys[:,i,j-2],q_sys[:,i,j-1],q_sys[:,i,j],q_sys[:,i,j+1],q_sys[:,i,j+2],1,0)
                g_r = lf.lf_flux(q_sys[:,i,j-2],q_sys[:,i,j-1],q_sys[:,i,j],q_sys[:,i,j+1],q_sys[:,i,j+2],q_sys[:,i,j+3],1,0)

                q_sys_new[:,i,j] = q_sys[:,i,j] - ((dt)/(cfg.dx))*(f_r - f_l)  -  ((dt)/(cfg.dy))*(g_r - g_l) 

    return q_sys_new
