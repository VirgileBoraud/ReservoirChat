from reservoirpy.nodes import Reservoir, Ridge
import numpy as np
import matplotlib.pyplot as plt

# Create a reservoir with 100 units
reservoir = Reservoir(100)

# Create a ridge readout with a ridge parameter of 1e-7
readout = Ridge(ridge=1e-7)

# Define a simple input signal
X = np.sin(np.arange(0, 10, 0.1))

# Run the input signal through the reservoir
states = reservoir.run(X, reset=True)

# Train the readout on the reservoir states
readout = readout.fit(states, X)

# Run the input signal through the reservoir and readout to get the output
Y = readout.run(states)

# Plot the input and output signals
plt.figure(figsize=(10, 3))
plt.title("Input and output signals")
plt.xlabel("Time")
plt.plot(X, label="Input", color="blue")
plt.plot(Y, label="Output", color="red")
plt.legend()
plt.show()
