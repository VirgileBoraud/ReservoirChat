import numpy as np
from reservoirpy import nodes
from sklearn import linear_model

# Create a Reservoir node
reservoir = nodes.Reservoir(10, random_state=42)

# Create a ScikitLearnNode for classification
node = nodes.SciketLearnNode(linear_model.Perceptron())

# Connect the Reservoir to the ScikitLearnNode
reservoir >> node

# Generate some sample data
X = np.random.rand(100, 5)
y = np.random.choice([-1, 1], size=100)

# Run the data through the reservoir and fit the classifier
states = reservoir.run(X)
node.fit(states, y)