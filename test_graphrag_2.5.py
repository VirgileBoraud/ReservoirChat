# Trying to correct the error : index 1 is out of bounds for axis 0 with size 1

from reservoirpy.datasets import japanese_vowels
from reservoirpy.nodes import Reservoir, Ridge, Input

# Load the dataset
X_train, Y_train, X_test, Y_test = japanese_vowels()

# Define the nodes
source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

# Create the model
model = source >> reservoir >> readout

# Train the model
model.fit(X_train, Y_train)

# Make a prediction
prediction = model.run(X_test[0].reshape(1, -1))

# Print the prediction
print("Prediction:", prediction)

# This code will load the Japanese Vowels dataset, define a reservoir with 500 units, a spectral radius of 0.9, and a learning rate of 0.1, and a readout node with a ridge parameter of 1e-6. It will then train the model on the training data and make a prediction on the first test sample. Finally, it will print the prediction.

# The error "index 1 is out of bounds for axis 0 with size 1" usually occurs when you try to access an element at an index that is out of range for the size of the array. In this case, it might be happening because the code is trying to reshape a single sample from the test set into a 2D array, but the reshape function expects the input to be at least 2D. To fix this, you can add a check to see if the input is 1D and, if so, reshape it into a 2D array with one row. Here's an updated version of the code that includes this check:

# Make a prediction
if X_test[0].ndim == 1:
    prediction = model.run(X_test[0].reshape(1, -1))
else:
    prediction = model.run(X_test[0])

# Print the prediction
print("Prediction:", prediction)