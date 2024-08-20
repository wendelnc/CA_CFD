# Standard Python Libraries
import numpy as np

# User Defined Libraries

####################################################################################################
# Define Our Test Problem
####################################################################################################

case = 0  # Brio & Wu Shock Tube (Qi Tang)
# case = 1  # Brio & Wu Shock Tube (Default)
# case = 2  # Figure 2a 
# case = 3  # Reversed Brio & Wu Shock Tube (Qi Tang)

####################################################################################################
# Input Data
####################################################################################################

# Initial Time 
ti = 0.0; tf = 0.2

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 1.0

# Heat Capacity Ratio
gamma = 5.0/3.0
# gamma = 2.0

# Number of Spatial Itterations
#x-direction; y-direction; z-direction
nx1 = 512;     nx2 = 32;    nx3 = 32

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
xgrid = np.linspace(x1min - nghost * dx, x1max - dx + nghost * dx, nx1 + 2 * nghost)
ygrid = np.linspace(x2min - nghost * dy, x2max - dy + nghost * dy, nx2 + 2 * nghost)
zgrid = np.linspace(x3min - nghost * dz, x3max - dz + nghost * dz, nx3 + 2 * nghost)
