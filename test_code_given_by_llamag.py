import numpy as np
from reservoirpy.nodes import Reservoir, Ridge, ScikitLearnNode

# Create a reservoir with 100 units, spectral radius of 0.95 and input scaling of 1.0
reservoir = Reservoir(n_units=100, sr=0.95, input_scaling=1.0)

# Define the training data and labels
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))

# Create a readout node with a logistic regression classifier
readout = Ridge(reservoir=reservoir, model=ScikitLearnNode('logistic_regression'))

# Train the reservoir and readout on the training data
reservoir.train(X_train)
readout.train(X_train, y_train)

# Make predictions on the training data
y_pred_train = readout.run(X_train)

# Evaluate the performance of the model on the training data
accuracy_train = np.mean(y_pred_train == y_train)
print(f'Training accuracy: {accuracy_train:.2f}')

# Create a reservoir with 100 units, spectral radius of 0.95 and input scaling of 1.0
reservoir_test = Reservoir(n_units=100, sr=0.95, input_scaling=1.0)

# Define the test data
X_test = np.random.rand(50, 10)

# Make predictions on the test data
y_pred_test = readout.run(X_test)

# Evaluate the performance of the model on the test data
accuracy_test = np.mean(y_pred_test == y_train[:50])
print(f'Test accuracy: {accuracy_test:.2f}')
