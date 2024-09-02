import reservoirpy as rpy

# Define the reservoir
reservoir = rpy.nodes.Reservoir(100, sr=0.99, input_scaling=0.1,
                                bias=False)

# Define the input
input_data = rpy.datasets.mackey_glass(1000)

# Define the readout
readout = rpy.nodes.Ridge(0.001, 1000)

# Create the LSM
lsm = rpy.nodes.EsnNode(reservoir, readout=readout, input_dim=1,
                          output_dim=1, name="LSM")

# Train the LSM
lsm.fit(input_data, input_data, verbose=True)
