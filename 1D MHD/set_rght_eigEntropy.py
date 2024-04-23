# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables

# @njit
def set_rght_eigEntropy(rho, u1, u2, u3, p, B1, B2, B3, n1, n2, n3, t1, t2, t3):
    '''
    See Section 1.5 (Scaling Theorem Example: Magnetohydrodynamic Equations) of:
    "Numerical Methods for Gas dynamic Systems on Unstructured Meshes"
    By Timothy J. Barth
    In particular, Entropy Scaled Eigenvectors ofthe Modified MHD Equations
    Equations (54) - (57)
    '''

    rr = np.zeros((8, 8))
    ru = np.zeros((8, 8))

    gamma = cfg.gamma

    rhosq = np.sqrt(rho)
    g1 = gamma - 1.0
    a2 = gamma * p / rho
    a = np.sqrt(a2)
    sqg2 = np.sqrt(1.0 / (2.0 * gamma))
    sq12 = np.sqrt(0.5)
    sqpr = np.sqrt(p) / rho
    sqpor = np.sqrt(p / rho)
    sq1og = np.sqrt(1.0 / gamma)
    b1s = B1 / rhosq
    b2s = B2 / rhosq
    b3s = B3 / rhosq
    bN = b1s * n1 + b2s * n2 + b3s * n3
    d = a2 + (b1s ** 2 + b2s ** 2 + b3s ** 2)
    cf = np.sqrt(0.5 * abs(d + np.sqrt(d ** 2 - 4.0 * a2 * (bN) ** 2)))
    cs = np.sqrt(0.5 * abs(d - np.sqrt(d ** 2 - 4.0 * a2 * (bN) ** 2)))
    # beta1 = np.sign((b1s * n1 + b2s * n2 + b3s * n3), np.array([1.0]))
    beta1 = np.sign(bN * 1.0)

    if np.abs(cf * cf - cs * cs) <= 1.0e-12:
        alphaf = np.sin(np.arctan(1.0) / 2.0)
        alphas = np.cos(np.arctan(1.0) / 2.0)
    else:
        alphaf = np.sqrt(np.abs(a2 - cs * cs)) / np.sqrt(np.abs(cf * cf - cs * cs))
        alphas = np.sqrt(np.abs(cf * cf - a2)) / np.sqrt(np.abs(cf * cf - cs * cs))

    TxN1 = n3 * t2 - n2 * t3
    TxN2 = n1 * t3 - n3 * t1
    TxN3 = n2 * t1 - n1 * t2

    # 1 - right eigenvector Entropy Wave
    ru[0, 0] = np.sqrt(g1 / gamma) * rhosq
    ru[1, 0] = 0.0
    ru[2, 0] = 0.0
    ru[3, 0] = 0.0
    ru[4, 0] = 0.0
    ru[5, 0] = 0.0
    ru[6, 0] = 0.0
    ru[7, 0] = 0.0

    # 2 - right eigenvector Divergence Wave
    ru[0, 1] = 0.0
    ru[1, 1] = 0.0
    ru[2, 1] = 0.0
    ru[3, 1] = 0.0
    ru[4, 1] = 0.0
    ru[5, 1] = sq1og * a * n1
    ru[6, 1] = sq1og * a * n2
    ru[7, 1] = sq1og * a * n3

    # 3 - right eigenvector Alfven Wave
    ru[0, 2] = 0.0
    ru[1, 2] = -sq12 * (sqpr * TxN1)
    ru[2, 2] = -sq12 * (sqpr * TxN2)
    ru[3, 2] = -sq12 * (sqpr * TxN3)
    ru[4, 2] = 0.0
    ru[5, 2] = sq12 * sqpor * TxN1
    ru[6, 2] = sq12 * sqpor * TxN2
    ru[7, 2] = sq12 * sqpor * TxN3

    # 4 - right eigenvector Alfven Wave
    ru[0, 3] =  ru[0, 2]
    ru[1, 3] = -ru[1, 2]
    ru[2, 3] = -ru[2, 2]
    ru[3, 3] = -ru[3, 2]
    ru[4, 3] =  ru[4, 2]
    ru[5, 3] =  ru[5, 2]
    ru[6, 3] =  ru[6, 2]
    ru[7, 3] =  ru[7, 2]

    # 5 - right eigenvector
    bst = b1s * t1 + b2s * t2 + b3s * t3

    ru[0, 4] = sqg2 * alphaf * rhosq
    ru[1, 4] = sqg2 * ((alphaf * a2 * n1 + alphas * a * ((bst) * n1 - (bN) * t1))) / (rhosq * cf)
    ru[2, 4] = sqg2 * ((alphaf * a2 * n2 + alphas * a * ((bst) * n2 - (bN) * t2))) / (rhosq * cf)
    ru[3, 4] = sqg2 * ((alphaf * a2 * n3 + alphas * a * ((bst) * n3 - (bN) * t3))) / (rhosq * cf)
    ru[4, 4] = sqg2 * alphaf * rhosq * a2
    ru[5, 4] = sqg2 * alphas * a * t1
    ru[6, 4] = sqg2 * alphas * a * t2
    ru[7, 4] = sqg2 * alphas * a * t3

    # 6 - right eigenvector
    ru[0, 5] =  ru[0, 4]
    ru[1, 5] = -ru[1, 4]
    ru[2, 5] = -ru[2, 4]
    ru[3, 5] = -ru[3, 4]
    ru[4, 5] =  ru[4, 4]
    ru[5, 5] =  ru[5, 4]
    ru[6, 5] =  ru[6, 4]
    ru[7, 5] =  ru[7, 4]

    # 7 - right eigenvector
    ru[0, 6] = sqg2 * alphas * rhosq
    ru[1, 6] = beta1 * sqg2 * (alphaf * cf**2 * t1 + a * n1 * alphas * (bN)) / (rhosq * cf)
    ru[2, 6] = beta1 * sqg2 * (alphaf * cf**2 * t2 + alphas * a * (bN) * n2) / (rhosq * cf)
    ru[3, 6] = beta1 * sqg2 * (alphaf * cf**2 * t3 + alphas * a * (bN) * n3) / (rhosq * cf)
    ru[4, 6] = a**2 * sqg2 * alphas * rhosq
    ru[5, 6] = -sqg2 * alphaf * a * t1
    ru[6, 6] = -sqg2 * alphaf * a * t2
    ru[7, 6] = -sqg2 * alphaf * a * t3

    # 8 - right eigenvector
    ru[0, 7] =  ru[0, 6]
    ru[1, 7] = -ru[1, 6]
    ru[2, 7] = -ru[2, 6]
    ru[3, 7] = -ru[3, 6]
    ru[4, 7] =  ru[4, 6]
    ru[5, 7] =  ru[5, 6]
    ru[6, 7] =  ru[6, 6]
    ru[7, 7] =  ru[7, 6]

    for m in range(8):
        rr[0, m] = ru[0, m] / g1
        rr[1, m] = (ru[0, m] * u1 + ru[1, m] * rho) / g1
        rr[2, m] = (ru[0, m] * u2 + ru[2, m] * rho) / g1
        rr[3, m] = (ru[0, m] * u3 + ru[3, m] * rho) / g1
        rr[4, m] = (ru[4, m] / g1 + B1 * ru[5, m] + B2 * ru[6, m] + B3 * ru[7, m] +
                    0.5 * ru[0, m] * (u1**2 + u2**2 + u3**2) +
                    ru[1, m] * u1 * rho + ru[2, m] * u2 * rho + ru[3, m] * u3 * rho) / g1
        rr[5, m] = ru[5, m] / g1
        rr[6, m] = ru[6, m] / g1
        rr[7, m] = ru[7, m] / g1

    return rr
