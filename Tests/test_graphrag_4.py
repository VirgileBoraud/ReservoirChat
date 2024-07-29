from reservoirpy.datasets import mackey_glass
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse

# Generate Mackey-Glass timeseries
X = mackey_glass(n_timesteps=2000)

# Create a reservoir with 100 units, leak rate of 0.3, and spectral radius of 1.25
reservoir = Reservoir(units=100, lr=0.3, sr=1.25)

# Create a readout using Ridge regression
readout = Ridge(ridge=1e-6)

# Create an Echo State Network by connecting the reservoir to the readout
esn = reservoir >> readout

# Train the ESN on the first 1500 timesteps and predict the next 500 timesteps
Y_pred = esn.fit(X[:1500], X[1:1501]).run(X[1501:])

# Print the RMSE of the prediction
print("Root Mean Squared Error:", rmse(X[1501:], Y_pred))
