# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg       # Input Parameters

# @njit
def eigenvalues(q_sys):

    em = 1.0e-15

    # 1) Convert our Conserved Variables q_sys = [ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z]^T 
    # to our Primitive Variables w_sys = [ρ, u_x, u_y, u_z, p, B_x, B_y, B_z]^T
    all_rho = q_sys[0]
    all_vex = q_sys[1] / q_sys[0]
    all_vey = q_sys[2] / q_sys[0]
    all_vez = q_sys[3] / q_sys[0]
    all_E   = q_sys[4]
    all_Bx  = q_sys[5]
    all_By  = q_sys[6]
    all_Bz  = q_sys[7]
    all_pre = (cfg.gamma-1.)*(all_E - 0.5*all_rho*(all_vex**2 + all_vey**2 + all_vez**2) - 0.5*(all_Bx**2 + all_By**2 + all_Bz**2))

    # 2) Compute Sound Speed
    a = np.sqrt(cfg.gamma * all_pre / all_rho)

    # 3) Compute Alfven Wave Speed
    ca  = np.sqrt((all_Bx**2 + all_By**2 + all_Bz**2) / all_rho)
    
    ############################################################################################################
    # X Direction
    ############################################################################################################

    cax = np.abs(np.sqrt(all_Bx**2 / all_rho))

    # 4) Compute Fast Magnetosonic Wave Speed
    cfx = np.sqrt(0.5 * np.abs( a**2 + ca**2 + np.sqrt((a**2 + ca**2)**2 - (4 * a**2 * cax**2))))

    # 5) Compute Slow Magnetosonic Wave Speed
    csx = np.sqrt(0.5 * np.abs( a**2 + ca**2 - np.sqrt((a**2 + ca**2)**2 - (4 * a**2 * cax**2))))

    # 3) Find Maximum Wave Speed
    max_eigen_1_x = max(em, np.max(np.abs(all_vex      )))
    max_eigen_2_x = max(em, np.max(np.abs(all_vex      )))
    max_eigen_3_x = max(em, np.max(np.abs(all_vex + cax)))
    max_eigen_4_x = max(em, np.max(np.abs(all_vex - cax)))
    max_eigen_5_x = max(em, np.max(np.abs(all_vex + cfx)))
    max_eigen_6_x = max(em, np.max(np.abs(all_vex - cfx)))
    max_eigen_7_x = max(em, np.max(np.abs(all_vex + csx)))
    max_eigen_8_x = max(em, np.max(np.abs(all_vex - csx)))

    alphax = np.array([max_eigen_1_x, max_eigen_2_x, max_eigen_3_x, max_eigen_4_x, max_eigen_5_x, max_eigen_6_x, max_eigen_7_x, max_eigen_8_x])

    ############################################################################################################
    # Y Direction
    ############################################################################################################

    cay = np.abs(np.sqrt(all_By**2 / all_rho))

    # 4) Compute Fast Magnetosonic Wave Speed
    cfy = np.sqrt(0.5 * np.abs( a**2 + ca**2 + np.sqrt((a**2 + ca**2)**2 - (4 * a**2 * cay**2))))

    # 5) Compute Slow Magnetosonic Wave Speed
    csy = np.sqrt(0.5 * np.abs( a**2 + ca**2 - np.sqrt((a**2 + ca**2)**2 - (4 * a**2 * cay**2))))

    # 3) Find Maximum Wave Speed
    max_eigen_1_y = max(em, np.max(np.abs(all_vey      )))
    max_eigen_2_y = max(em, np.max(np.abs(all_vey      )))
    max_eigen_3_y = max(em, np.max(np.abs(all_vey + cay)))
    max_eigen_4_y = max(em, np.max(np.abs(all_vey - cay)))
    max_eigen_5_y = max(em, np.max(np.abs(all_vey + cfy)))
    max_eigen_6_y = max(em, np.max(np.abs(all_vey - cfy)))
    max_eigen_7_y = max(em, np.max(np.abs(all_vey + csy)))
    max_eigen_8_y = max(em, np.max(np.abs(all_vey - csy)))

    alphay = np.array([max_eigen_1_y, max_eigen_2_y, max_eigen_3_y, max_eigen_4_y, max_eigen_5_y, max_eigen_6_y, max_eigen_7_y, max_eigen_8_y])

    return alphax, alphay