# Standard Python Libraries
from numba import njit

# User Defined Libraries

@njit
def weno_hj(a,b,c,d):

    epsilon = 1E-6

    IS0 = 13*(a-b)**2 + 3*(a-3*b)**2
    IS1 = 13*(b-c)**2 + 3*(b+c)**2
    IS2 = 13*(c-d)**2 + 3*(3*c-d)**2

    alpha0 = 1.0 / (epsilon + IS0)**2
    alpha1 = 6.0 / (epsilon + IS1)**2
    alpha2 = 3.0 / (epsilon + IS2)**2

    omega0 = alpha0 / (alpha0 + alpha1 + alpha2)
    omega2 = alpha2 / (alpha0 + alpha1 + alpha2)

    phi = (1.0/3.0) * omega0 * (a - 2*b + c) + (1.0/6.0) * (omega2 - 0.5) * (b - 2*c + d)

    return phi

@njit
def hj_flux(q0,q1,q2,q3,q4,q5,q6,dx):
    '''
            |           |    q1     |           |    q3     |           |    q5     |           |
            |    q0     |___________|           |___________|    q4     |___________|           |
            |___________|           |    q2     |           |___________|           |    q6     |               
            |           |           |___________|           |           |           |___________|
            |           |           |           |           |           |           |           |
    ...-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----...
            |    i-3    |    i-2    |    i-1    |     i     |    i+1    |    i+2    |    i+3    |
            |+         -|+         -|+         -|+         -|+         -|+         -|+         -|
            i-7/2       i-5/2       i-3/2       i-1/2       i+1/2       i+3/2       i+5/2       i+7/2
    '''

    # Compute the First Derivatives
    der1 = (q2 - q1) / dx # i-1 - i-2
    der2 = (q3 - q2) / dx # i   - i-1
    der3 = (q4 - q3) / dx # i+1 - i
    der4 = (q5 - q4) / dx # i+2 - i+1

    # Compute the Common Term in Equation (2.6)
    common = (-der1 + 7 * der2 + 7 * der3 - der4) / (12.0)

    # Compute the Second Derivatives
    secder1 = (q6 - 2 * q5 + q4) / dx # i+3 - 2i+2 + i+1
    secder2 = (q5 - 2 * q4 + q3) / dx # i+2 - 2i+1 + i
    secder3 = (q4 - 2 * q3 + q2) / dx # i+1 - 2i   + i-1
    secder4 = (q3 - 2 * q2 + q1) / dx # i   - 2i-1 + i-2
    secder5 = (q2 - 2 * q1 + q0) / dx # i-1 - 2i-2 + i-3

    # Compute the WENO Reconstruction
    weno_plus_flux = weno_hj(secder1, secder2, secder3, secder4)
    u_x_plus = common + weno_plus_flux # Equation (2.9)
    weno_minus_flux = weno_hj(secder5, secder4, secder3, secder2)
    u_x_minus = common - weno_minus_flux # Equation (2.6)
    
    return u_x_plus, u_x_minus


    