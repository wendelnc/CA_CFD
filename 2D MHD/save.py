# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables


def save_q_sys(q_sys,t):

    rho, vex, vey, vez, pre, Bx, By, Bz = c2p.cons2prim(q_sys)

    format_spec = "{:13.16e} "

    filename = f"results/q_sys_at_time_{t}_for_{cfg.nx1}_by_{cfg.nx2}.txt"
    
    with open(filename, "w") as file:
        for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
            for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
                file.write(format_spec.format(rho[i,j]))
                file.write(format_spec.format(vex[i,j]))
                file.write(format_spec.format(vey[i,j]))
                file.write(format_spec.format(vez[i,j]))
                file.write(format_spec.format(pre[i,j]))
                file.write(format_spec.format(Bx[i,j]))
                file.write(format_spec.format(By[i,j]))
                file.write(format_spec.format(Bz[i,j]))
                file.write("\n")
    
    print("File created successfully.")

def save_a_sys(a_sys,t):

    format_spec = "{:13.16e}"

    filename = f"results/a_sys_at_time_{t}_for_{cfg.nx1}_by_{cfg.nx2}.txt"
    with open(filename, "w") as file:
        for i in range(cfg.nghost,(cfg.nx1)+cfg.nghost):
            for j in range(cfg.nghost,(cfg.nx2)+cfg.nghost):
                file.write(format_spec.format(a_sys[2,i,j]))
                file.write("\n")

    print("File created successfully.")

def save_div(div,t):

    format_spec = "{:13.16e}"

    filename = f"results/div_at_time_{t}_for_{cfg.nx1}_by_{cfg.nx2}.txt"
    with open(filename, "w") as file:
        for i in range(cfg.nx1):
            for j in range(cfg.nx2):
                file.write(format_spec.format(div[i,j]))
                file.write("\n")

    print("File created successfully.")