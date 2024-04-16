# Standard Python Libraries
import numpy as np
from numba import njit

# User Defined Libraries
import configuration as cfg      # Input Parameters
import cons2prim as c2p          # Convert Conserved to Primitive Variables

# @njit
def set_left_eigEntropy(rho, u1, u2, u3, p, B1, B2, B3, n1, n2, n3, t1, t2, t3):

    lu = np.zeros((8, 8))
    lq = np.zeros((8, 8))

    gamma = cfg.gamma

    rhosq = np.sqrt(rho)
    g1 = gamma - 1.0
    a2 = (gamma * p / rho)
    a = np.sqrt(a2)

    b1s = B1 / rhosq
    b2s = B2 / rhosq
    b3s = B3 / rhosq

    BNs = b1s * n1 + b2s * n2 + b3s * n3

    BN = (B1 * n1 + B2 * n2 + B3 * n3)

    d = a2 + (b1s**2 + b2s**2 + b3s**2)
    cf = np.sqrt(0.5 * np.abs(d + np.sqrt(d**2 - 4.0 * a2 * (BNs)**2)))
    cs = np.sqrt(0.5 * np.abs(d - np.sqrt(d**2 - 4.0 * a2 * (BNs)**2)))

    if np.abs(cf**2 - cs**2) <= 1.0e-12:
        alphaf = np.sin(np.arctan(1.0) / 2.0)
        alphas = np.cos(np.arctan(1.0) / 2.0)
    else:
        alphaf = np.sqrt(np.abs(a2 - cs**2)) / np.sqrt(np.abs(cf**2 - cs**2))
        alphas = np.sqrt(np.abs(cf**2 - a2)) / np.sqrt(np.abs(cf**2 - cs**2))

    psq = np.sqrt(p)
    twosq = np.sqrt(2.0)
    um2 = 0.5 * (u1**2 + u2**2 + u3**2)

    beta1 = np.sign(BNs, np.array([1.0]))

    sqgam = np.sqrt(g1 / gamma)

    TxN1 = n3 * t2 - n2 * t3
    TxN2 = n1 * t3 - n3 * t1
    TxN3 = n2 * t1 - n1 * t2

    BT = B1 * t1 + B2 * t2 + B3 * t3

    gu1sq = np.sqrt(1.0 / gamma)

    # 1 - left eigenvector
    lu[0, 0] = 1.0 / (sqgam * rhosq)
    lu[0, 1] = 0.0
    lu[0, 2] = 0.0
    lu[0, 3] = 0.0
    lu[0, 4] = -1.0 / (a2 * sqgam * rhosq)
    lu[0, 5] = 0.0
    lu[0, 6] = 0.0
    lu[0, 7] = 0.0

    # 2 - left eigenvector
    nen = (n3**2 * (t1**2 + t2**2) - 2.0 * n1 * n3 * t1 * t3 - 2.0 * n2 * t2 *
           (n1 * t1 + n3 * t3) + n2**2 * (t1**2 + t3**2) + n1**2 * (t2**2 + t3**2))

    nen2 = a * gu1sq * nen

    lu[1, 0] = 0.0
    lu[1, 1] = 0.0
    lu[1, 2] = 0.0
    lu[1, 3] = 0.0
    lu[1, 4] = 0.0
    lu[1, 5] = (-n2 * t1 * t2 - n3 * t1 * t3 + n1 * (t2**2 + t3**2)) / nen2
    lu[1, 6] = (-t2 * (n1 * t1 + n3 * t3) + n2 * (t1**2 + t3**2)) / nen2
    lu[1, 7] = (n3 * (t1**2 + t2**2) - (n1 * t1 + n2 * t2) * t3) / nen2

    # 3 - left eigenvector
    nen3 = np.sqrt(2.0) * np.sqrt(p) * nen
    nen31 = nen3 / np.sqrt(rho)

    lu[2, 0] = 0.0
    lu[2, 1] = rho * (-TxN1) / nen3
    lu[2, 2] = rho * (-TxN2) / nen3
    lu[2, 3] = rho * (-TxN3) / nen3
    lu[2, 4] = 0.0
    lu[2, 5] = TxN1 / nen31
    lu[2, 6] = TxN2 / nen31
    lu[2, 7] = TxN3 / nen31

    # 4 - left eigenvector
    lu[3, 0] = lu[2, 0]
    lu[3, 1] = -lu[2, 1]
    lu[3, 2] = -lu[2, 2]
    lu[3, 3] = -lu[2, 3]
    lu[3, 4] = lu[2, 4]
    lu[3, 5] = lu[2, 5]
    lu[3, 6] = lu[2, 6]
    lu[3, 7] = lu[2, 7]

    # 5 - left eigenvector
    Term51 = rho * cf * ((n2 * t1 * t2 + n3 * t1 * t3 - n1 * (t2**2 + t3**2)) * rhosq * cf**2 *
                         alphaf - a * BN * alphas * (-n2**2 * t1 + n1 * n2 * t2 + n3 * t2 * n3 * TxN2))

    Term52 = rho * cf * ((-t2 * (n1 * t1 + n3 * t3) + n2 * (t1**2 + t3**2)) * rhosq * cf**2 *
                         alphaf - a * BN * alphas * (-n1 * n2 * t1 + n1**2 * t2 + n3 * TxN1))

    Term53 = rho * cf * ((n3 * (t1**2 + t2**2) - (n1 * t1 + n2 * t2) * t3) * rhosq * cf**2 *
                         alphaf - a * (-n1 * n3 * t1 + n1**2 * t3 + n2 * (-n3 * t2 + n2 * t3)) * BN * alphas)

    Term54 = alphaf / (twosq * a**2 * gu1sq * rhosq * (alphaf**2 + alphas**2))

    Term55 = (n2**2 * t1 - n1 * n2 * t2 + n3 * (n3 * t1 - n1 * t3)) * alphas

    Term56 = alphas * (-n1 * n2 * t1 + n1**2 * t2 + n3 * TxN1)
    Term57 = (-n1 * n3 * t1 + n1**2 * t3 + n2 * (-n3 * t2 + n2 * t3)) * alphas

    nen51 = twosq * a * nen * gu1sq * (a * BN**2 * alphas**2 + rhosq * cf**2 * alphaf *
                                       (a * rhosq * alphaf + BT * alphas))

    nen52 = twosq * a * gu1sq * (alphaf**2 + alphas**2) * nen

    lu[4, 0] = 0.0
    lu[4, 1] = -Term51 / nen51
    lu[4, 2] = Term52 / nen51
    lu[4, 3] = Term53 / nen51
    lu[4, 4] = Term54
    lu[4, 5] = Term55 / nen52
    lu[4, 6] = Term56 / nen52
    lu[4, 7] = Term57 / nen52

    # 6 - left eigenvector
    lu[5, 0] = lu[4, 0]
    lu[5, 1] = -lu[4, 1]
    lu[5, 2] = -lu[4, 2]
    lu[5, 3] = -lu[4, 3]
    lu[5, 4] = lu[4, 4]
    lu[5, 5] = lu[4, 5]
    lu[5, 6] = lu[4, 6]
    lu[5, 7] = lu[4, 7]

    # 7 - left eigenvector
    Term71 = rho * cf * (a * (n2**2 * t1 - n1 * n2 * t2 + n3 * (n3 * t1 - n1 * t3)) * rhosq *
                         alphaf + ((B3 * n2 * t1 - B2 * n3 * t1 - B3 * n1 * t2 + B2 * n1 * t3) *
                         (-n3 * t2 + n2 * t3) + B1 * (n2**2 * t1**2 + n3**2 * t1**2 -
                         2.0 * n1 * n2 * t1 * t2 - 2.0 * n1 * n3 * t1 * t3 + n1**2 * (t2**2 + t3**2))) * alphas)

    Term72 = rho * cf * (((n3 * t1 - n1 * t3) * (B3 * n2 * t1 - B3 * n1 * t2 + B1 * n3 * t2 - B1 * n2 * t3) +
                         B2 * ((n1**2 + n3**2) * t2**2 - 2.0 * n2 * t2 * (n1 * t1 + n3 * t3) +
                         n2**2 * (t1**2 + t3**2))) * alphas + a * rhosq * alphaf * (-n1 * n2 * t1 +
                         n1**2 * t2 + n3 * TxN1))

    Term73 = rho * cf * (a * (-n1 * n3 * t1 + n1**2 * t3 + n2 * (-n3 * t2 + n2 * t3)) * rhosq *
                         alphaf + alphas * (B3 * (n3**2 * (t1**2 + t2**2) - 2.0 * n3 *
                         (n1 * t1 + n2 * t2) * t3 + (n1**2 + n2**2) * t3**2) +
                         B2 * (n3 * t1 - n1 * t3) * TxN3 + B1 * (-n3 * t2 + n2 * t3) * TxN3))

    Term74 = alphas / (twosq * a**2 * gu1sq * rhosq * (alphaf**2 + alphas**2))

    Term75 = -alphaf * (n2**2 * t1 - n1 * n2 * t2 + n3 * (n3 * t1 - n1 * t3))

    Term76 = -alphaf * (-n1 * n2 * t1 + n1**2 * t2 + n3 * TxN1)

    Term77 = -alphaf * (-n1 * n3 * t1 + n1**2 * t3 + n2 * (-n3 * t2 + n2 * t3))

    nen71 = twosq * beta1 * nen * gu1sq * (a * BN**2 * alphas**2 + rhosq * cf**2 * alphaf *
                                           (a * rhosq * alphaf + BT * alphas))

    nen72 = nen52

    lu[6, 0] = 0.0
    lu[6, 1] = Term71 / nen71
    lu[6, 2] = Term72 / nen71
    lu[6, 3] = Term73 / nen71
    lu[6, 4] = Term74
    lu[6, 5] = Term75 / nen72
    lu[6, 6] = Term76 / nen72
    lu[6, 7] = Term77 / nen72

    # 8 - left eigenvector
    lu[7, 0] = lu[6, 0]
    lu[7, 1] = -lu[6, 1]
    lu[7, 2] = -lu[6, 2]
    lu[7, 3] = -lu[6, 3]
    lu[7, 4] = lu[6, 4]
    lu[7, 5] = lu[6, 5]
    lu[7, 6] = lu[6, 6]
    lu[7, 7] = lu[6, 7]

    # CONSERVATIVE E-VECTORS
    lq = np.zeros_like(lu)

    for m in range(8):
        lq[m, 0] = lu[m, 0] * g1 - lu[m, 1] * u1 * g1 / rho - lu[m, 2] * u2 * g1 / rho - lu[m, 3] * u3 * g1 / rho + lu[m, 4] * g1**2 * (u1**2 + u2**2 + u3**2)**0.5
        lq[m, 1] = -lu[m, 4] * u1 * g1**2 + lu[m, 1] * g1 / rho
        lq[m, 2] = -lu[m, 4] * u2 * g1**2 + lu[m, 2] * g1 / rho
        lq[m, 3] = -lu[m, 4] * u3 * g1**2 + lu[m, 3] * g1 / rho
        lq[m, 4] = lu[m, 4] * g1**2
        lq[m, 5] = lu[m, 5] * g1 - B1 * lu[m, 4] * g1**2
        lq[m, 6] = lu[m, 6] * g1 - B2 * lu[m, 4] * g1**2
        lq[m, 7] = lu[m, 7] * g1 - B3 * lu[m, 4] * g1**2

    # Return the updated 'lq' matrix
    return lq