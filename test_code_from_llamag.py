from reservoirpy.nodes import Reservoir, Ridge
import numpy as np

# Create the reservoir
reservoir = Reservoir(
    units=500,
    sr=0.9,
    lr=0.1,
)

# Create the readout node (in this case a linear regression model)
readout = Ridge(ridge=1e-6)

# Create the model by connecting the reservoir to the readout
model = reservoir >> readout

# Initialize and run the model on some data
X = np.sin(np.linspace(0, 6*np.pi, 1000)).reshape(-1, 1)

X_train = X[:500]
y_train = X[1:501]
y_pred = model.fit(X_train, y_train, warmup=100)