import reservoirpy as rpy
from reservoirpy.datasets import mackey_glass
from reservoirpy.nodes import Reservoir, Ridge

# Next, let's create the Mackey-Glass dataset:

X = mackey_glass(n_timesteps=2000)

# Now, let's create a reservoir and a readout node:

reservoir = Reservoir(units=100, lr=0.3, sr=1.25)
readout = Ridge(ridge=1e-6)

# Finally, let's create an Echo State Network (ESN) by connecting the reservoir to the readout:

esn = reservoir >> readout




# Ce que j'ai du rajouter
from reservoirpy.observables import rmse
# Split the dataset for training and testing
X_train, Y_train = X[:2000], X[10:2010]
X_test, Y_test = X[2000:-10], X[2010:]

# Train and run the ESN on the test data
Y_pred = esn.fit(X_train, Y_train).run(X_test)

# Print the Root Mean Squared Error
print("Root Mean Squared Error:", rmse(Y_test, Y_pred))