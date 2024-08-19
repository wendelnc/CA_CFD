# Standard Python Libraries
import numpy as np

# User Defined Libraries

####################################################################################################
# Define Our Test Problem
####################################################################################################

# 2D MHD Test Problems
case = 0 # Smooth Alfven Wave
# case = 1 # Orszag-Tang Vortex 
# case = 2 # Orszag-Tang Vortex (Princeton) 

# 1D MHD Test Problems
# case = 3 # Brio & Wu Shock Tube (Qi Tang)

####################################################################################################
# Input Data
####################################################################################################

# Initial Time 
ti = 0.0; tf = 1.0

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 0.1

# Heat Capacity Ratio
gamma = 5.0/3.0

# Number of Spatial Itterations
#x-direction; y-direction; z-direction
# nx1 = 16;     nx2 = 32;    nx3 = 32
# nx1 = 32;     nx2 = 64;    nx3 = 64
nx1 = 64;     nx2 = 128;   nx3 = 128
# nx1 = 128;    nx2 = 256;   nx3 = 256

# Number of Ghost Cells
nghost = 3

# Spatial Domain
if case == 0:
    theta = 0.463647609000806
    x1min = 0.0;    x1max = 1.0/np.cos(theta)
    x2min = 0.0;    x2max = 2.0 * x1max
    x3min = 0.0;    x3max = 2.0 * x1max
if case == 1 or case == 2:
    x1min = 0.0;    x1max = 2.0 * np.pi
    x2min = 0.0;    x2max = 2.0 * np.pi
    x3min = 0.0;    x3max = 2.0 * np.pi

# Step Size ∆x, ∆y, ∆z
dx = (x1max-x1min)/(nx1)
dy = (x2max-x2min)/(nx2)
dz = (x3max-x3min)/(nx3)

# Spatial Domain Grid
xgrid = np.linspace(x1min - nghost * dx, x1max - dx + nghost * dx, nx1 + 2 * nghost)
ygrid = np.linspace(x2min - nghost * dy, x2max - dy + nghost * dy, nx2 + 2 * nghost)
zgrid = np.linspace(x3min - nghost * dz, x3max - dz + nghost * dz, nx3 + 2 * nghost)

X, Y = np.meshgrid(xgrid, ygrid, indexing='ij')
