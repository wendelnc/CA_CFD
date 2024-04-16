import numpy as np

def set_rght_eigPN(rho, u1, u2, u3, p, B1, B2, B3, gamma):
    
    rr = np.zeros((8, 8))
    rhosq = np.sqrt(rho)
    a2 = (gamma * p / rho)
    a = np.sqrt(a2)
    d = a2 + (B1**2 + B2**2 + B3**2) / rho

    ca = np.sqrt(B1**2 / rho)
    cf = np.sqrt(0.5 * abs(d + np.sqrt(d**2 - 4 * a2 * B1 * B1 / rho)))
    cs = np.sqrt(0.5 * abs(d - np.sqrt(d**2 - 4 * a2 * B1 * B1 / rho)))

    beta1 = np.sign(1.0, B1 * 1.0)

    if (abs(B2) + abs(B3)) <= 1e-14:
        beta2 = 1.0 / np.sqrt(2.0)
        beta3 = 1.0 / np.sqrt(2.0)
    else:
        beta2 = B2 / np.sqrt(B2 * B2 + B3 * B3)
        beta3 = B3 / np.sqrt(B2 * B2 + B3 * B3)

    if (abs(B2) + abs(B3)) <= 1e-14 and abs(a2 - B1 * B1 / rho) < 1e-14:
        alphaf = 1.0
        alphas = 1.0
    else:
        alphaf = np.sqrt(abs(cf * cf - ca * ca)) / np.sqrt(abs(cf * cf - cs * cs))
        alphas = np.sqrt(abs(cf * cf - a2)) / np.sqrt(abs(cf * cf - cs * cs))

    g1 = gamma - 1.0
    g2 = gamma - 2.0

    hmf = (alphaf * cf * cf / g1) - alphaf * cf * u1 + alphas * ca * beta1 * (beta2 * u2 + beta3 * u3) + (g2 / g1) * alphaf * (cf * cf - a2)
    hpf = (alphaf * cf * cf / g1) + alphaf * cf * u1 - alphas * ca * beta1 * (beta2 * u2 + beta3 * u3) + (g2 / g1) * alphaf * (cf * cf - a2)
    hms = (alphas * cs * cs / g1) - alphas * cs * u1 - alphaf * a * beta1 * (beta2 * u2 + beta3 * u3) + (g2 / g1) * alphas * (cs * cs - a2)
    hps = (alphas * cs * cs / g1) + alphas * cs * u1 + alphaf * a * beta1 * (beta2 * u2 + beta3 * u3) + (g2 / g1) * alphas * (cs * cs - a2)

    # 1 - right eigenvector
    rr[0, 0] = alphaf
    rr[1, 0] = alphaf * (u1 - cf)
    rr[2, 0] = alphaf * u2 + alphas * beta2 * ca * beta1
    rr[3, 0] = alphaf * u3 + alphas * beta3 * ca * beta1
    rr[4, 0] = 0.5 * alphaf * (u1**2 + u2**2 + u3**2) + hmf
    rr[5, 0] = 0.0
    rr[6, 0] = alphas * beta2 * cf / rhosq
    rr[7, 0] = alphas * beta3 * cf / rhosq

    # 2 - right eigenvector
    rr[0, 1] = 0.0
    rr[1, 1] = 0.0
    rr[2, 1] = beta3 * beta1
    rr[3, 1] = -beta2 * beta1
    rr[4, 1] = (beta3 * u2 - beta2 * u3) * beta1
    rr[5, 1] = 0.0
    rr[6, 1] = beta3 / rhosq
    rr[7, 1] = -beta2 / rhosq

    # 3 - right eigenvector
    rr[0, 2] = alphas
    rr[1, 2] = alphas * (u1 - cs)
    rr[2, 2] = alphas * u2 - alphaf * beta2 * a * beta1
    rr[3, 2] = alphas * u3 - alphaf * beta3 * a * beta1
    rr[4, 2] = 0.5 * alphas * (u1**2 + u2**2 + u3**2) + hms
    rr[5, 2] = 0.0
    rr[6, 2] = -alphaf * beta2 * a2 / (cf * rhosq)
    rr[7, 2] = -alphaf * beta3 * a2 / (cf * rhosq)

    # 4 - right eigenvector
    rr[0, 3] = 1.0
    rr[1, 3] = u1
    rr[2, 3] = u2
    rr[3, 3] = u3
    rr[4, 3] = 0.5 * (u1**2 + u2**2 + u3**2)
    rr[5, 3] = 0.0
    rr[6, 3] = 0.0
    rr[7, 3] = 0.0

    # 5 - left eigenvector
    rr[0, 4] = 0.0
    rr[1, 4] = 0.0
    rr[2, 4] = 0.0
    rr[3, 4] = 0.0
    rr[4, 4] = 0.0
    rr[5, 4] = 1.0
    rr[6, 4] = 0.0
    rr[7, 4] = 0.0

    # 6 - right eigenvector
    rr[0, 5] = alphas
    rr[1, 5] = alphas * (u1 + cs)
    rr[2, 5] = alphas * u2 + alphaf * beta2 * a * beta1
    rr[3, 5] = alphas * u3 + alphaf * beta3 * a * beta1
    rr[4, 5] = 0.5 * alphas * (u1**2 + u2**2 + u3**2) + hps
    rr[5, 5] = 0.0
    rr[6, 5] = -alphaf * beta2 * a2 / (cf * rhosq)
    rr[7, 5] = -alphaf * beta3 * a2 / (cf * rhosq)

    # 7 - right eigenvector
    rr[0, 6] = 0.0
    rr[1, 6] = 0.0
    rr[2, 6] = -beta3 * beta1
    rr[3, 6] = beta2 * beta1
    rr[4, 6] = -(beta3 * u2 - beta2 * u3) * beta1
    rr[5, 6] = 0.0
    rr[6, 6] = beta3 / rhosq
    rr[7, 6] = -beta2 / rhosq

    # 8 - right eigenvector
    rr[0, 7] = alphaf
    rr[1, 7] = alphaf * (u1 + cf)
    rr[2, 7] = alphaf * u2 - alphas * beta2 * ca * beta1
    rr[3, 7] = alphaf * u3 - alphas * beta3 * ca * beta1
    rr[4, 7] = 0.5 * alphaf * (u1**2 + u2**2 + u3**2) + hpf
    rr[5, 7] = 0.0
    rr[6, 7] = alphas * beta2 * cf / rhosq
    rr[7, 7] = alphas * beta3 * cf / rhosq

    return rr

