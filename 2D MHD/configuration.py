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
x1min = 0.0;    x1max = 2.0 * np.pi
x2min = 0.0;    x2max = 2.0 * np.pi
x3min = 0.0;    x3max = 2.0 * np.pi

# Step Size ∆x, ∆y, ∆z
dx = (x1max-x1min)/(nx1)
dy = (x2max-x2min)/(nx2)
dz = (x3max-x3min)/(nx3)

# Spatial Domain Grid
xgrid = np.linspace(x1min,x1max-dx,nx1)
ygrid = np.linspace(x2min,x2max-dy,nx2)
zgrid = np.linspace(x3min,x3max-dz,nx3)

X, Y = np.meshgrid(xgrid, ygrid)

# Initial Time 
ti = 0.0; tf = 1.0

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 0.5

# Heat Capacity Ratio
gamma = 5.0/3.0

####################################################################################################
# Define Our Test Problem
####################################################################################################

# 2D MHD Test Problems

# case = 0 # Orszag-Tang Vortex
case = 1 # Orszag-Tang Vortex (Qi)
# case = 2 # Orszag-Tang Vortex (Princeton) 

# 1D MHD Test Problems
# case = 3 # Brio & Wu Shock Tube (Qi Tang)

####################################################################################################
# Define Our Boundary Conditions
####################################################################################################

bndc = 0 # Periodic Boundary Conditions
# bndc = 1 # Reflective Boundary Conditions
