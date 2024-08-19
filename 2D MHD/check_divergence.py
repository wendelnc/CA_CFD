# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

def check_divergence(q_sys):

    div = np.zeros((cfg.nx1+(2*cfg.nghost),cfg.nx2+(2*cfg.nghost)))

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):

            dbx = (1/(12.0*cfg.dx))*(q_sys[5,i-2,j] - 8.0*q_sys[5,i-1,j] + 8.0*q_sys[5,i+1,j] - q_sys[5,i+2,j])
            dby = (1/(12.0*cfg.dy))*(q_sys[6,i,j-2] - 8.0*q_sys[6,i,j-1] + 8.0*q_sys[6,i,j+1] - q_sys[6,i,j+2])
            div[i,j] = dbx + dby

    return div