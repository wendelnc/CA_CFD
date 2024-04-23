# Standard Python Libraries
import numpy as np

# User Defined Libraries

####################################################################################################
# Input Data
####################################################################################################

# Number of Spatial Itterations
#x-direction; y-direction; z-direction
nx1 = 128;     nx2 = 32;    nx3 = 32

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
ti = 0.0; tf = 0.2

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 3.0

# Heat Capacity Ratio
gamma = 1.666666666666666666667
# gamma = 5.0/3.0
# gamma = 2.0

####################################################################################################
# Define Our Test Problem
####################################################################################################

case = 0  # Brio & Wu Shock Tube (Qi Tang)
# case = 1  # Brio & Wu Shock Tube (Default)
# case = 2

####################################################################################################
# Define Our Boundary Conditions
####################################################################################################

# bndc = 0 # Periodic Boundary Conditions
# bndc = 1 # Reflective Boundary Conditions
# bndc = 2 # Dirichlet Boundary Conditions (for Brio-Wu Shock Tube)
bndc = 3 # Outflow Boundary Conditions