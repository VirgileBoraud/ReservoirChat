import numpy as np
from reservoirpy import Reservoir

# Load Mackey-Glass time series
X = np.load('mackey_glass.npy')

# Create a reservoir with 100 nodes and a spectral radius of 0.9
reservoir = Reservoir(n_nodes=100, spectral_radius=0.9)

# Run the reservoir over the entire timeseries
states = reservoir.run(X)

# Use the last state as the output
y_pred = states[-1]
