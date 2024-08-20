# CA_CFD
A collection of my computational fluid dynamic (CFD) codes written in Python


**How To Run:** From each folder (e.g., 2D MHD) run python main.py from the terminal. This will generate the folder 'results', with three subfolders: animations, simulation data, and txt files. main.py is setup to save the initial condition and final solution in txt files, and it will save the entire simulation (.npz file) in simulation data. plotting.py is for making plots while running the simulation, plot all.py is for plotting data from the saved .npz file. You should run python plot all.py from the main folder. It will generate plots/movies etc of the saved .npz file in the animations subfolder. All movies, txt files, and .npz simulation results have generic names and should be renamed to avoid being written over. 

configuration.py controls which test problem you will run, the grid size, CFL, etc. 