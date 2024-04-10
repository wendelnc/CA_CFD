# Standard Python Libraries
import numpy as np

# User Defined Libraries

####################################################################################################
# Input Data
####################################################################################################

# Number of Spatial Itterations
nx1 = 128

# Number of Ghost Cells
nghost = 3

# Spatial Domain
x1min = 0.0;    x1max = 1.0

# Step Size ∆x
dx = (x1max-x1min)/(nx1)

# Spatial Domain
grid = np.linspace(x1min,x1max-dx,nx1)

# Time Domain
ti = 0.0;   tf = 0.15

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 0.1

# Heat Capacity Ratio
gamma = 1.4

####################################################################################################
# Define Our Test Problem
####################################################################################################

case = 0 # Sod's Shock Tube
# case = 1 # Mach = 3 test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997
# case = 2 # Shocktube problem of G.A. Sod, JCP 27:1, 1978
# case = 3 # Lax test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997
# case = 4 # Shocktube problem with supersonic zone
# case = 5 # Stationary shock 
# case = 6 # Double Expansion (does not work)
# case = 7 # Reverse Sod Shock Tube

####################################################################################################
# Define Our Boundary Conditions
####################################################################################################

# bndc = 0 # Periodic Boundary Conditions
bndc = 1 # Reflective Boundary Conditions