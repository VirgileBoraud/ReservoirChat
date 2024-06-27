from reservoirpy.nodes import Reservoir, Ridge
import numpy as np

# Set up the Mackey-Glass node
node = Ridge(delta=0.1, alpha=0.2, beta=0.3, tau_min=5, tau_max=15)

# Generate some input data (e.g., a Mackey-Glass sequence)
x = [0.9]
for i in range(1000):
    x.append(0.4 * x[-1] + 0.3 * x[-2] - 0.1 * x[-3] + 0.6)

# Run the input through the reservoir
y = []
for t, xi in enumerate(x):
    y.append(node(xi))

# Visualize the output
import matplotlib.pyplot as plt
plt.plot(y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.show()