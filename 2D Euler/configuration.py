# Standard Python Libraries
import numpy as np

# User Defined Libraries

####################################################################################################
# Input Data
####################################################################################################

# Number of Spatial Itterations
# x-direction; y-direction
nx1 = 512;      nx2 = 512
# nx1 = 32;      nx2 = 32

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

# Initial Time 
ti = 0.0

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 0.1

# Heat Capacity Ratio
gamma = 1.4

####################################################################################################
# Define Our Test Problem
####################################################################################################

# 2D Riemann Problems
case = 0  # Carsten W. Schulz-Rinne 
# case = 1  # Kurganov & Tadmor Configuration 1 
# case = 2  # Kurganov & Tadmor Configuration 2
# case = 3  # Kurganov & Tadmor Configuration 3
# case = 4  # Kurganov & Tadmor Configuration 4
# case = 5  # Kurganov & Tadmor Configuration 5
# case = 6  # Kurganov & Tadmor Configuration 6
# case = 7  # Kurganov & Tadmor Configuration 7
# case = 8  # Kurganov & Tadmor Configuration 8
# case = 9  # Kurganov & Tadmor Configuration 9
# case = 10 # Kurganov & Tadmor Configuration 10
# case = 11 # Kurganov & Tadmor Configuration 11
# case = 12 # Kurganov & Tadmor Configuration 12
# case = 13 # Kurganov & Tadmor Configuration 13
# case = 14 # Kurganov & Tadmor Configuration 14
# case = 15 # Kurganov & Tadmor Configuration 15

# 1D Riemann Problems
# case = 3 # 1D Sod Shock Tube
# case = 4 # Rotated 1D Sod Shock Tube
# case = 5 # Reverse 1D Sod Shock Tube
# case = 6 # Rotated Reverse 1D Sod Shock Tube
# case = 7 # Reverse Carsten W. Schulz-Rinne 2D Riemann Problem
# case = 8 # Custom Test Problem

####################################################################################################
# Set Final Time
####################################################################################################

if case == 0 or case == 3 or case == 6 or case == 9 or case == 11 or case == 13:
    tf = 0.30  
elif case == 1 or case == 2 or case == 15:
    tf = 0.20
elif case == 4 or case == 7 or case == 8 or case == 12:
    tf = 0.25
elif case == 5:
    tf = 0.23
elif case == 10:
    tf = 0.15
elif case == 14:
    tf = 0.10
# elif case == 3 or case == 4 or case == 5 or case == 6 or case == 8:
    # tf = 0.15

####################################################################################################
# Define Our Boundary Conditions
####################################################################################################

# bndc = 0 # Periodic Boundary Conditions
bndc = 1 # Reflective Boundary Conditions