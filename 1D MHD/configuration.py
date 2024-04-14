# Standard Python Libraries
import numpy as np

# User Defined Libraries

####################################################################################################
# Input Data
####################################################################################################

# Number of Spatial Itterations
#x-direction; y-direction; z-direction
nx1 = 32;     nx2 = 32;    nx3 = 32

# Number of Ghost Cells
nghost = 3

# Spatial Domain
x1min = -1.0;    x1max = 1.0
x2min = -1.0;    x2max = 1.0
x3min = -1.0;    x3max = 1.0

# Step Size ∆x, ∆y, ∆z
dx = (x1max-x1min)/(nx1)
dy = (x2max-x2min)/(nx2)
dz = (x3max-x3min)/(nx3)

# Spatial Domain Grid
xgrid = np.linspace(x1min, x1max, nx1)
ygrid = np.linspace(x2min, x2max, nx2)
zgrid = np.linspace(x3min, x3max, nx3)

# Initial Time 
ti = 0.0; tf = 0.15

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 0.1

# Heat Capacity Ratio
gamma = 1.4

####################################################################################################
# Define Our Test Problem
####################################################################################################

case = 0  # Brio & Wu Shock Tube 

####################################################################################################
# Define Our Boundary Conditions
####################################################################################################

# bndc = 0 # Periodic Boundary Conditions
bndc = 1 # Reflective Boundary Conditions