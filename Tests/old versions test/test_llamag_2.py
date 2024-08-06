import reservoirpy as rpy
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse
from reservoirpy.nodes import Reservoir, Ridge

rpy.set_seed(0)

# Load the Mackey-Glass dataset
X = mackey_glass(2500)

# Split the dataset for training and testing
X_train, Y_train = X[:2000], X[10:2010]
X_test, Y_test = X[2000:-10], X[2010:]

# Create a Reservoir node with 100 neurons, a custom leak rate of 0.3, and a spectral radius of 0.9
reservoir = Reservoir(100, lr=0.3, sr=0.9)

# Create a Readout node using Ridge linear regression with a regularization parameter of 1e-6
readout = Ridge(ridge=1e-6)

# Create an ESN by connecting the Reservoir node to the Readout node
esn = reservoir >> readout

# Train and run the ESN on the test data
Y_pred = esn.fit(X_train, Y_train).run(X_test)

# Print the Root Mean Squared Error
print("Root Mean Squared Error:", rmse(Y_test, Y_pred))