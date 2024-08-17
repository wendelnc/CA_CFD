# Standard Python Libraries
from numba import njit

# User Defined Libraries

@njit
def weno5(vmm,vm,v,vp,vpp):
    
    epsilon = 1E-6

    beta_0 = (13.0/12.0)*(vmm - 2*vm +   v)**2. + (1.0/4.0)*(vmm - 4*vm + 3*v)**2.
    beta_1 = (13.0/12.0)*(vm  - 2*v  +  vp)**2. + (1.0/4.0)*(vm  - vp)**2.
    beta_2 = (13.0/12.0)*(v   - 2*vp + vpp)**2. + (1.0/4.0)*(vpp - 4*vp + 3*v)**2.

    alpha_0 = 0.1 / (epsilon + beta_0)**2.    
    alpha_1 = 0.6 / (epsilon + beta_1)**2.    
    alpha_2 = 0.3 / (epsilon + beta_2)**2.

    alpha_sum = alpha_0 + alpha_1 + alpha_2

    weight_0 = alpha_0 / alpha_sum
    weight_1 = alpha_1 / alpha_sum
    weight_2 = alpha_2 / alpha_sum

    wn  = (1.0/6.0)*( weight_0*(2*vmm - 7*vm + 11*v) \
                    + weight_1*(-vm   + 5*v  + 2*vp) \
                    + weight_2*(2*v   + 5*vp - vpp) )
    
    return wn