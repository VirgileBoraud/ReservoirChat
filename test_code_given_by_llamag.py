import numpy as np
from reservoirpy.nodes import Reservoir, Ridge

# Parameters
n_inputs = 10
n_hidden = 100
n_outputs = 1
leak_rate = 0.2
spectral_radius = 0.9
learning_rate = 0.01

# Create the reservoir
reservoir = Reservoir(n_inputs=n_inputs, n_units=n_hidden, leak_rate=leak_rate,
                      spectral_radius=spectral_radius)

# Create the readout
readout = Ridge(n_in=n_hidden, n_out=n_outputs, learning_rate=learning_rate)

# Train the network
X_train = np.random.rand(1000, n_inputs)
y_train = np.random.rand(1000, n_outputs)
reservoir.fit(X_train, y_train, epochs=100)
y_pred = reservoir.run(X_train)

print("Root Mean Squared Error:", np.mean((y_pred - y_train) ** 2))