# Standard Python Libraries
import numpy as np

# User Defined Libraries

####################################################################################################
# Input Data
####################################################################################################

# Number of Spatial Itterations
# x-direction; y-direction
nx1 = 64;      nx2 = 64

# Number of Ghost Cells
nghost = 3

# Spatial Domain
x1min = 0.0;    x1max = 1.0
x2min = 0.0;    x2max = 1.0

# Step Size ∆y, ∆x
dx = (x1max-x1min)/(nx1)
dy = (x2max-x2min)/(nx2)

# Spatial Domain Grid
xgrid = np.linspace(0, 1, nx1)
ygrid = np.linspace(0, 1, nx2)

# Create the meshgrid
X, Y = np.meshgrid(xgrid, ygrid)

# Time Domain
ti = 0.0;   tf = 0.15

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 0.1

# Heat Capacity Ratio
gamma = 1.4

####################################################################################################
# Define Our Test Problem
####################################################################################################

# case = 0 # Carsten W. Schulz-Rinne 2D Riemann Problem
# case = 1 # Deng, X. Fig 8 2D Riemann Problem
# case = 2 # Deng, X. Fig 9 2D Riemann Problem
# case = 3 # 1D Sod Shock Tube
case = 4 # Rotated 1D Sod Shock Tube
# case = 5 # Reverse 1D Sod Shock Tube
# case = 6 # Rotated Reverse 1D Sod Shock Tube
# case = 7 # Reverse Carsten W. Schulz-Rinne 2D Riemann Problem

####################################################################################################
# Define Our Boundary Conditions
####################################################################################################

# bndc = 0 # Periodic Boundary Conditions
bndc = 1 # Reflective Boundary Conditions