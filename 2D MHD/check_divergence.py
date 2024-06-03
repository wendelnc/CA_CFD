# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def check_divergence(q_sys):

    div = np.zeros((cfg.nx2+(2*cfg.nghost),cfg.nx1+(2*cfg.nghost)))

    for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
        for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):

            dbx = (1/(12.0*cfg.dx))*(q_sys[5,j,i-2] - 8.0*q_sys[5,j,i-1] + 8.0*q_sys[5,j,i+1] - q_sys[5,j,i+2])
            dby = (1/(12.0*cfg.dy))*(q_sys[6,j-2,i] - 8.0*q_sys[6,j-1,i] + 8.0*q_sys[6,j+1,i] - q_sys[6,j+2,i])
            div[j,i] = dbx + dby

    return div