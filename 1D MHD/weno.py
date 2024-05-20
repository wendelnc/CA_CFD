# Standard Python Libraries
from numba import njit

# User Defined Libraries

@njit
def weno(vmm,vm,v,vp,vpp,umm,um,u,up,upp):

    '''
    Function Name:      weno
    Creator:            Carolyn Wendeln
    Date Created:       04-01-2023
    Date Last Modified: 04-10-2024

    Definition:         weno computes the 5th order WENO Lax-Friedrichs flux vector splitting reconstruction
                        following Procedure 2.5 from the following paper:
                        "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws"
                        By: Chi-Wang Shu
                        https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf

    Inputs:             vmm, ... , vpp: positive fluxes for w_{i+1/2}^{-}
                        umm, ... , upp: negative fluxes for w_{i+1/2}^{+}

    Outputs:            wn + wp: result of the WENO reconstruction for q_{i+1/2}

    Dependencies:       none

    ############################################################################################################
    # Define Our Grid
    ############################################################################################################

                |           |  q(i-2)   |           |   q(i)    |           |   q(i+2)  |           |
                |  q(i-3)   |___________|           |___________|   q(i+1)  |___________|           |
                |___________|           |   q(i-1)  |           |___________|           |   q(i+3)  |               
                |           |           |___________|           |           |           |___________|
                |           |           |           |           |           |           |           |
        ...-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----...
                |    i-3    |    i-2    |    i-1    |     i     |    i+1    |    i+2    |    i+3    |
                |+         -|+         -|+         -|+         -|+         -|+         -|+         -|
                i-7/2       i-5/2       i-3/2       i-1/2       i+1/2       i+3/2       i+5/2       i+7/2

    ############################################################################################################
    # Use f^{+} (vmm,...,vpp) to obtain w_{i+1/2}^{-}
    ############################################################################################################

                                                    |_____v__________vp__________vpp____|     

                                        |____vm___________v__________vp_____|

                            |____vmm_________vm___________v_____|

        ...-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----...
                |    i-3    |    i-2    |    i-1    |     i    -|    i+1    |    i+2    |    i+3    |
                                                                i+1/2

    ############################################################################################################
    # Use f^{-} (umm,...,upp) to obtain w_{i+1/2}^{+}     
    ############################################################################################################                                                  
                                                            
                                                                |_____u__________up__________upp____|     

                                                    |____um___________u__________up_____|

                                        |____umm_________um___________u_____|

        ...-----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----...
                |    i-3    |    i-2    |    i-1    |     i     |+   i+1    |    i+2    |    i+3    |
                                                                i+1/2                                                        

    ############################################################################################################
    # Because I always forget...
    ############################################################################################################                                             
         --------------
        |              |
        |      flux -->|              when α > 0 we want to use f^{+} = f(u) + αu to compute w_{i+1/2}^{-}
        |              |
        |              | <-- flux     when α < 0 we want to use f^{-} = f(u) - αu to compute w_{i+1/2}^{+}
        |              |
         --------------
                                       
    '''
    
    ################################################################################################
    # Lax-Friedrichs splitting
    # f^{±}(u) = 0.5 * (f(u) ± αu)
    # α = maxu |f′(u)|
    ################################################################################################
    
    # WENO-JS
    epsilon = 1E-6
    
    # WENO-Z
    # epsilon = 1E-40

    ################################################################################################
    # Choose the positive fluxes: v_i = f^{+}(u_i) = 0.5 * (f(u) + αu)
    # to obtain the cell boundary values : v_{i+1/2}^{-} = f_{i+1/2}^{+}
    ################################################################################################

    beta_0n = (13.0/12.0)*(vmm - 2*vm +   v)**2. + (1.0/4.0)*(vmm - 4*vm + 3*v)**2.
    beta_1n = (13.0/12.0)*(vm  - 2*v  +  vp)**2. + (1.0/4.0)*(vm  - vp)**2.
    beta_2n = (13.0/12.0)*(v   - 2*vp + vpp)**2. + (1.0/4.0)*(vpp - 4*vp + 3*v)**2.

    d_0n = 1.0/10.0
    d_1n = 6.0/10.0
    d_2n = 3.0/10.0

    # WENO-JS
    alpha_0n = d_0n / (epsilon + beta_0n)**2.    
    alpha_1n = d_1n / (epsilon + beta_1n)**2.    
    alpha_2n = d_2n / (epsilon + beta_2n)**2.

    # WENO-Z    
    # tau5n = np.abs(beta_0n-beta_2n)   
    # alpha_0n = d_0n * (1 + (tau5n / (epsilon + beta_0n)))    
    # alpha_1n = d_1n * (1 + (tau5n / (epsilon + beta_1n)))   
    # alpha_2n = d_2n * (1 + (tau5n / (epsilon + beta_2n)))   

    alpha_sumn = alpha_0n + alpha_1n + alpha_2n

    weight_0n = alpha_0n / alpha_sumn
    weight_1n = alpha_1n / alpha_sumn
    weight_2n = alpha_2n / alpha_sumn

    wn  = (1.0/6.0)*( weight_0n*(2*vmm - 7*vm + 11*v) \
                    + weight_1n*(-vm  + 5*v  + 2*vp) \
                    + weight_2n*(2*v   + 5*vp - vpp) )
    
    ################################################################################################
    # Choose the negative fluxes: v_i = f^{-}(u_i) = 0.5 * (f(u) - αu)
    # to obtain the cell boundary values : v_{i+1/2}^{+} = f_{i+1/2}^{-}
    ################################################################################################

    beta_0p = (13.0/12.0)*(umm - 2*um +   u)**2. + (1.0/4.0)*(umm - 4*um + 3*u)**2.
    beta_1p = (13.0/12.0)*(um  - 2*u  +  up)**2. + (1.0/4.0)*(um  - up)**2.
    beta_2p = (13.0/12.0)*(u   - 2*up + upp)**2. + (1.0/4.0)*(upp - 4*up + 3*u)**2.

    d_0p = 3.0/10.0
    d_1p = 6.0/10.0
    d_2p = 1.0/10.0

    # WENO-JS
    alpha_0p = d_0p / (epsilon + beta_0p)**2.    
    alpha_1p = d_1p / (epsilon + beta_1p)**2.    
    alpha_2p = d_2p / (epsilon + beta_2p)**2.

    # WENO-Z    
    # tau5p = np.abs(beta_0p-beta_2p)   
    # alpha_0p = d_0p * (1 + (tau5p / (epsilon + beta_0p)))    
    # alpha_1p = d_1p * (1 + (tau5p / (epsilon + beta_1p)))   
    # alpha_2p = d_2p * (1 + (tau5p / (epsilon + beta_2p)))   

    alpha_sump = alpha_0p + alpha_1p + alpha_2p

    weight_0p = alpha_0p / alpha_sump
    weight_1p = alpha_1p / alpha_sump
    weight_2p = alpha_2p / alpha_sump

    wp  = (1.0/6.0)*( weight_0p*(-umm + 5*um +   2*u) \
                    + weight_1p*(2*um + 5*u  -    up) \
                    + weight_2p*(11*u - 7*up + 2*upp) )
  
    ################################################################################################
    # Numerical flux
    # f_{i+1/2} = f_{i+1/2}^{+} + f_{i+1/2}^{-}
    ################################################################################################
    
    return wn + wp
    