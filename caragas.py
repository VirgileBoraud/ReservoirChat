import reservoirpy as rpy
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse
from reservoirpy.nodes import Reservoir, Ridge

rpy.set_seed(0)

# Load the Mackey-Glass dataset
X = mackey_glass(2500)

# Split dataset for training
X_train, Y_train = X[:2000], X[10:2010]
X_test, Y_test = X[2000:-10], X[2010:]

# Create a reservoir node
reservoir = Reservoir(100, lr=0.3, sr=0.9)

# Create a readout node (ridge linear regression)
readout = Ridge(ridge=1e-6)

# Create an Echo State Network (ESN)
esn = reservoir >> readout

# Train and run the ESN
Y_pred = esn.fit(X_train, Y_train).run(X_test)

# Print the Root Mean Squared Error
print("Root Mean Squared Error:", rmse(Y_test, Y_pred))
