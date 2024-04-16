# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters

@njit
def time_step(q_sys,t):

    # 1) Convert our Conserved Variables q_sys = [ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z]^T 
    # to our Primitive Variables w_sys = [ρ, u_x, u_y, u_z, p, B_x, B_y, B_z]^T
    all_rho = q_sys[0, :]
    all_vex = q_sys[1, :] / q_sys[0, :]
    all_vey = q_sys[2, :] / q_sys[0, :]
    all_vez = q_sys[3, :] / q_sys[0, :]
    all_E   = q_sys[4, :]
    all_Bx  = q_sys[5, :]
    all_By  = q_sys[6, :]
    all_Bz  = q_sys[7, :]
    all_pre = (cfg.gamma-1.)*(all_E - 0.5*all_rho*(all_vex**2 + all_vey**2 + all_vez**2) - 0.5*(all_Bx**2 + all_By**2 + all_Bz**2))

    # 2) Compute Sound Speed
    a = np.sqrt(cfg.gamma * all_pre / all_rho)

    # 3) Compute Alfven Wave Speed
    ca = np.sqrt((all_Bx**2 + all_By**2 + all_Bz**2) / all_rho)
    cax = np.sqrt(all_Bx**2 / all_rho)

    # 4) Compute Fast Magnetosonic Wave Speed
    cf = np.sqrt(0.5 * ( a**2 + ca**2 + np.sqrt((a**2 + ca**2)**2 - (4 * a**2 * cax**2))))

    # 5) Compute Slow Magnetosonic Wave Speed
    cs = np.sqrt(0.5 * ( a**2 + ca**2 - np.sqrt((a**2 + ca**2)**2 - (4 * a**2 * cax**2))))

    # 3) Find Maximum Wave Speed
    max_eigen_1 = np.max(np.abs(all_vex-cf))
    max_eigen_2 = np.max(np.abs(all_vex-cax))
    max_eigen_3 = np.max(np.abs(all_vex-cs))
    max_eigen_4 = np.max(np.abs(all_vex))
    max_eigen_5 = np.max(np.abs(all_vex))
    max_eigen_6 = np.max(np.abs(all_vex+cs))
    max_eigen_7 = np.max(np.abs(all_vex+cax))
    max_eigen_8 = np.max(np.abs(all_vex+cf))

    alpha = np.array([max_eigen_1, max_eigen_2 , max_eigen_3, max_eigen_4, max_eigen_5, max_eigen_6, max_eigen_7, max_eigen_8])

    max_speed = np.max(alpha)

    # 4) Compute Time Step
    dt = cfg.CFL * cfg.dx / max_speed

    # Stops our dt from going past tf
    if t + dt > cfg.tf:
        dt = cfg.tf - t

    return dt, alpha