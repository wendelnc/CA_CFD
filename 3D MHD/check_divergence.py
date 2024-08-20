# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

# @njit
def check_divergence(q_sys):

    div = np.zeros((cfg.nx1+(2*cfg.nghost),cfg.nx2+(2*cfg.nghost),cfg.nx3+(2*cfg.nghost)))

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
            for k in range(cfg.nghost,(cfg.nx3)+cfg.nghost):
                dbx = (1/(12.0*cfg.dx))*(q_sys[5,i-2,j,k] - 8.0*q_sys[5,i-1,j,k] + 8.0*q_sys[5,i+1,j,k] - q_sys[5,i+2,j,k])
                dby = (1/(12.0*cfg.dy))*(q_sys[6,i,j-2,k] - 8.0*q_sys[6,i,j-1,k] + 8.0*q_sys[6,i,j+1,k] - q_sys[6,i,j+2,k])
                dbz = (1/(12.0*cfg.dz))*(q_sys[7,i,j,k-2] - 8.0*q_sys[7,i,j,k-1] + 8.0*q_sys[7,i,j,k+1] - q_sys[7,i,j,k+2])
                div[i,j,k] = dbx + dby + dbz

    return div