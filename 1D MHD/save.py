# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables

def save_q_sys(q_sys,t):

    rho, vex, vey, vez, pre, Bx, By, Bz = c2p.cons2prim(q_sys)

    format_spec = "{:13.16e} "

    filename = f"results/txt_files/q_sys_at_time_{t}_for_{cfg.nx1}.txt"
    
    with open(filename, "w") as file:
        for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
            file.write(format_spec.format(rho[i]))
            file.write(format_spec.format(vex[i]))
            file.write(format_spec.format(vey[i]))
            file.write(format_spec.format(vez[i]))
            file.write(format_spec.format(pre[i]))
            file.write(format_spec.format(Bx[i]))
            file.write(format_spec.format(By[i]))
            file.write(format_spec.format(Bz[i]))
            file.write("\n")
    
    print("File created successfully.")

