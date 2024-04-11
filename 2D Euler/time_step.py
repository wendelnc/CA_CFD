# Standard Python Libraries
import numpy as np

# User Defined Libraries
import configuration as cfg      # Input Parameters

def time_step(q_sys,t):

    # 1) Convert q_sys = [rho, rho*u, rho*v, E]^T to Primitive Variables w_sys = [rho, u, v, p]^T
    all_rho = q_sys[0,:,:]
    all_vex = q_sys[1,:,:] / q_sys[0,:,:]
    all_vey = q_sys[2,:,:] / q_sys[0,:,:]
    all_pre = (cfg.gamma - 1.) * (q_sys[3,:,:] - (0.5 * q_sys[0,:,:] * ((q_sys[1,:,:]/q_sys[0,:,:])**2. + (q_sys[2,:,:]/q_sys[0,:,:])**2.)))

    # 2) Compute Speed of Sound
    a = np.sqrt(cfg.gamma * all_pre / all_rho)

    # 3) Find Maximum Wave Speed
    max_u1 = np.max(np.abs(all_vex - a))
    max_u2 = np.max(np.abs(all_vex))
    max_u3 = np.max(np.abs(all_vex + a))

    alpha_x = np.array([max_u1,max_u2,max_u2,max_u3])

    max_x = np.max(alpha_x)

    max_v1 = np.max(np.abs(all_vey - a))
    max_v2 = np.max(np.abs(all_vey))
    max_v3 = np.max(np.abs(all_vey + a))

    alpha_y = np.array([max_v1,max_v2,max_v2,max_v3])

    max_y = np.max(alpha_y)

    # max_speed = np.max(np.array([max_u1,max_u2,max_u3,max_v1,max_v2,max_v3]))
    
    # 4) Compute Time Step
    dt = cfg.CFL * (cfg.dx / max_x + cfg.dy / max_y)

    # Stops our td from going past t1
    if t + dt > cfg.tf:
        dt = cfg.tf - t

    return dt, alpha_x, alpha_y