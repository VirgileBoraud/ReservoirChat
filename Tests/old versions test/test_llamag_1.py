import reservoirpy as rpy
from reservoirpy.datasets import japanese_vowels
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.metrics import accuracy_score

# Load the Japanese Vowels dataset
X_train, y_train, X_test, y_test = japanese_vowels()

# Create a Reservoir node with 100 neurons, a custom leak rate of 0.3, and a spectral radius of 0.9
reservoir = Reservoir(100, lr=0.3, sr=0.9)

# Create a Readout node using Ridge linear regression with a regularization parameter of 1e-6
readout = Ridge(ridge=1e-6)

# Create an ESN by connecting the Reservoir node to the Readout node
esn = reservoir >> readout

# Train the ESN on the training data
esn.fit(X_train, y_train)

# Use the ESN to predict the labels of the test data
y_pred = esn.run(X_test)

# Print the accuracy score of the predictions
print("Accuracy score:", accuracy_score(y_test, y_pred))