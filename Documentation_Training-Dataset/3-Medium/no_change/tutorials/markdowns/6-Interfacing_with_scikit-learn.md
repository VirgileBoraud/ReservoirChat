# Interfacing with scikit-learn

The `ScikitLearnNode` allows you to use any model from the scikit-learn library as a regular offline readout node (similar to the Ridge node). As scikit-learn implements most major methods for classification and regression, you can easily implement and experiment different methods in your ReservoirPy models.



In the following tutorial, you will learn how to use the `ScikitLearnNode` on both regression and classification tasks.



For more information about scikit-learn models, please visit the [API reference](https://scikit-learn.org/stable/modules/classes.html).



## Summary



- <a href="#chapter1">Chapter 1: `ScikitLearnNode` basic usage</a>

- <a href="#chapter2">Chapter 2: Using `ScikitLearnNode` for classification</a>

```python
import numpy as np

import matplotlib.pyplot as plt



import reservoirpy

from reservoirpy.observables import nrmse, rsquare

reservoirpy.set_seed(42)

reservoirpy.verbosity(0)
```

## Chapter 1: `ScikitLearnNode` basic usage <span id="chapter1"/>



### Instantiate a node

To create a node, simply instantiate a `ScikitLearnNode` by passing the model class as an argument. Here, we will use [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).



You can use any model that implements the `.fit` and `.predict` methods !

```python
from sklearn import linear_model



from reservoirpy.nodes import ScikitLearnNode

import reservoirpy

reservoirpy.verbosity(0)

reservoirpy.set_seed(42)



readout = ScikitLearnNode(linear_model.Lasso)
```

If you want to specify parameters to the model, you can pass them into the `model_hypers` parameter as a `dict`.

```python
readout = ScikitLearnNode(

    model = linear_model.Lasso, 

    model_hypers = {"alpha": 1e-3},

    name = "Lasso"

)
```

### Node usage



The `ScikitLearnNode` follows the same syntax as any other offline readout nodes : you can call the `.fit` method to train its parameters, and `.run` to get predictions !



Under the hood, ReservoirPy handles the respective calls to the model's `.fit` and `.predict` methods.



Let's test our model on the Mackey-Glass task seen in the previous tutorials. 

```python
# create the model

reservoir = reservoirpy.nodes.Reservoir(

    units = 500,

    lr = 0.3,

    sr = 0.9,

)



model = reservoir >> readout
```

```python
# create the dataset to train our model on

from reservoirpy.datasets import mackey_glass, to_forecasting



mg = mackey_glass(n_timesteps=10_000, tau=17)

# rescale between -1 and 1

mg = 2 * (mg - mg.min()) / mg.ptp() - 1



X_train, X_test, y_train, y_test = to_forecasting(mg, forecast=10, test_size=0.2)
```

```python
model.fit(X_train, y_train, warmup=100)
```

### Evaluate the model

```python
def plot_results(y_pred, y_test, sample=500):



    fig = plt.figure(figsize=(15, 7))

    plt.subplot(211)

    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")

    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")

    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")



    plt.legend()

    plt.show()
```

```python
y_pred = model.run(X_test)
```

```python
plot_results(y_pred, y_test)

rsquare(y_test, y_pred), nrmse(y_test, y_pred)
```

### Node internals



The instance of the scikit-learn model can be accessed using its `instances` param.

```python
node = ScikitLearnNode(linear_model.PassiveAggressiveRegressor)

node.initialize(x=np.ones((10, 3)), y=np.ones((10, 1)))

str(node.instances)
```

Most scikit-learn models only handles one output feature. In that case, multiple instances of the model are created under the hood, and each output features are dispatched among the models. And `node.instances` is a list containing the model instances.

```python
node = ScikitLearnNode(linear_model.PassiveAggressiveRegressor)

# we now have 2 output features !

node.initialize(x=np.ones((10, 3)), y=np.ones((10, 2)))

node.instances
```

## Chapter 2: Using `ScikitLearnNode` for classification <span id="chapter2"/>

In the previous tutorial, we have seen a trick to handle a classification task only using ridge regression, with the `Ridge` node.



Although the results were pretty satisfying, regressions methods are not designed for classification tasks. One issue is that outlier data can significantly shift the decision boundary, as described and illustrated [here](https://stats.stackexchange.com/questions/22381/why-not-approach-classification-through-regression).



Fortunately, `scikit-learn` implements many different classifiers, such as `RidgeClassifier`, `LogisticRegression` or `Perceptron`. Let's try them !



In this chapter, we will take the same dataset as in Tutorial 5. You can check it to get some data visualization.

```python
import numpy as np

from reservoirpy.datasets import japanese_vowels



# repeat_target ensure that we obtain one label per timestep, and not one label per utterance.

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)



# Y_train and Y_test are one-hot encoded, but we want qualitative values here.

Y_train = [np.argmax(sample, 1, keepdims=True) for sample in Y_train]

Y_test = [np.argmax(sample, 1, keepdims=True) for sample in Y_test]



X_train[0].shape, Y_train[0].shape
```

```python
from reservoirpy.nodes import Reservoir, ScikitLearnNode

from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron



reservoir = Reservoir(500, sr=0.9, lr=0.1)

sk_ridge = ScikitLearnNode(RidgeClassifier, name="RidgeClassifier")

sk_logistic = ScikitLearnNode(LogisticRegression, name="LogisticRegression")

sk_perceptron = ScikitLearnNode(Perceptron, name="Perceptron")



# One reservoir for 3 readout. That's the magic of reservoir computing!

model = reservoir >> [sk_ridge, sk_logistic, sk_perceptron]
```

```python
model.fit(X_train, Y_train, stateful=False, warmup=2)

Y_pred = model.run(X_test, stateful=False)
```

```python
from sklearn.metrics import accuracy_score



speaker = np.concatenate(Y_test, dtype=np.float64)



for model, pred in Y_pred.items():

    model_pred = np.concatenate(pred)

    score = accuracy_score(speaker, model_pred)

    print(f"{model} score: {score * 100:.3f} %")
```

