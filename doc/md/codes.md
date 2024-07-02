# Extracted Code from Notebooks

## Getting Started
```python
import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everything reproducible!
```

## Create a Reservoir
```python
from reservoirpy.nodes import Reservoir

reservoir = Reservoir(100, lr=0.5, sr=0.9)
```

## Initialize and run the reservoir
```python
import numpy as np
import matplotlib.pyplot as plt

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

plt.figure(figsize=(10, 3))
plt.title("A sine wave.")
plt.ylabel("$sin(t)$")
plt.xlabel("$t$")
plt.plot(X)
plt.show()
```

## Call on a single timestep
```python
s = reservoir(X[0].reshape(1, -1))

print("New state vector shape: ", s.shape)
```

```python
s = reservoir.state()
```

```python
states = np.empty((len(X), reservoir.output_dim))
for i in range(len(X)):
    states[i] = reservoir(X[i].reshape(1, -1))
```

```python
plt.figure(figsize=(10, 3))
plt.title("Activation of 20 reservoir neurons.")
plt.ylabel("$reservoir(sin(t))$")
plt.xlabel("$t$")
plt.plot(states[:, :20])
plt.show()
```

## Run over a whole timeseries
```python
states = reservoir.run(X)
```

## Reset or modify reservoir state
```python
reservoir = reservoir.reset()
```

```python
states_from_null = reservoir.run(X, reset=True)
```

```python
a_state_vector = np.random.uniform(-1, 1, size=(1, reservoir.output_dim))

states_from_a_starting_state = reservoir.run(X, from_state=a_state_vector)
```

```python
previous_states = reservoir.run(X)

with reservoir.with_state(reset=True):
    states_from_null = reservoir.run(X)
    
# as if the with_state never happened!
states_from_previous = reservoir.run(X) 
```

## Create a readout
```python
from reservoirpy.nodes import Ridge

readout = Ridge(ridge=1e-7)
```

## Define a training task
```python
X_train = X[:50]
Y_train = X[1:51]

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(X_train, label="sin(t)", color="blue")
plt.plot(Y_train, label="sin(t+1)", color="red")
plt.legend()
plt.show()
```

## Train the readout
```python
train_states = reservoir.run(X_train, reset=True)
```

```python
readout = readout.fit(train_states, Y_train, warmup=10)
```

```python
test_states = reservoir.run(X[50:])
Y_pred = readout.run(test_states)

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Predicted sin(t)", color="blue")
plt.plot(X[51:], label="Real sin(t+1)", color="red")
plt.legend()
plt.show()
```

## Create the ESN model
```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge
```

## Train the ESN
```python
esn_model = esn_model.fit(X_train, Y_train, warmup=10)
```

```python
print(reservoir.is_initialized, readout.is_initialized, readout.fitted)
```

## Run the ESN
```python
Y_pred = esn_model.run(X[50:])

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Predicted sin(t+1)", color="blue")
plt.plot(X[51:], label="Real sin(t+1)", color="red")
plt.legend()
plt.show()
```

## Advanced features
```python
import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everything reproducible!
```

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

X_train = X[:50]
Y_train = X[1:51]

plt.figure(figsize=(10, 3))
plt.title("A sine wave.")
plt.ylabel("$sin(t)$")
plt.xlabel("$t$")
plt.plot(X)
plt.show()
```

## Input-to-readout connections
```python
from reservoirpy.nodes import Reservoir, Ridge, Input

data = Input()
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)

esn_model = data >> reservoir >> readout & data >> readout
```

```python
esn_model = [data, data >> reservoir] >> readout
```

```python
esn_model.node_names
```

```python
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat

data = Input()
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)
concatenate = Concat()

esn_model = [data, data >> reservoir] >> concatenate >> readout
```

```python
esn_model.node_names
```

## Feedback connections
```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)

reservoir <<= readout

esn_model = reservoir >> readout
```

```python
esn_model = esn_model.fit(X_train, Y_train)
```

```python
esn_model(X[0].reshape(1, -1))

print("Feedback received (reservoir):", reservoir.feedback())
print("State sent: (readout):", readout.state())
```

## Forced feedbacks
```python
esn_model = esn_model.fit(X_train, Y_train, force_teachers=True)  # by default
```

```python
pred = esn_model.run(X_train, forced_feedbacks=Y_train, shift_fb=True)
```

```python
random_feedback = np.random.normal(0, 1, size=(1, readout.output_dim))

with reservoir.with_feedback(random_feedback):
    reservoir(X[0].reshape(1, -1))
```

## Generation and long term forecasting
```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge
```

```python
esn_model = esn_model.fit(X_train, Y_train, warmup=10)
```

```python
warmup_y = esn_model.run(X_train[-10:], reset=True)
```

```python
Y_pred = np.empty((100, 1))
x = warmup_y[-1].reshape(1, -1)

for i in range(100):
    x = esn_model(x)
    Y_pred[i] = x
```

```python
plt.figure(figsize=(10, 3))
plt.title("100 timesteps of a sine wave.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Generated sin(t)")
plt.legend()
plt.show()
```

## From custom initializer functions
```python
from reservoirpy.nodes import Reservoir

def normal_w(n, m, **kwargs):
    return np.random.normal(0, 1, size=(n, m))

reservoir = Reservoir(100, W=normal_w)

reservoir(X[0].reshape(1, -1))

plt.figure(figsize=(5, 5))
plt.title("Weights distribution in $W$")
plt.hist(reservoir.W.ravel(), bins=50)
plt.show()
```

## From reservoirpy.mat_gen module
```python
from reservoirpy.mat_gen import random_sparse

# Random sparse matrix initializer from uniform distribution,
# with spectral radius to 0.9 and connectivity of 0.1.

# Matrix creation can be delayed...
initializer = random_sparse(
    dist="uniform", sr=0.9, connectivity=0.1)
matrix = initializer(100, 100)

# ...or can be performed right away.
matrix = random_sparse(
    100, 100, dist="uniform", sr=0.9, connectivity=0.1)

from reservoirpy.mat_gen import normal

# Dense matrix from Gaussian distribution,
# with mean of 0 and variance of 0.5
matrix = normal(50, 100, loc=0, scale=0.5)

from reservoirpy.mat_gen import uniform

# Sparse matrix from uniform distribution in [-0.5, 0.5],
# with connectivity of 0.9 and input_scaling of 0.3.
matrix = uniform(
    200, 60, low=0.5, high=0.5, 
    connectivity=0.9, input_scaling=0.3)

from reservoirpy.mat_gen import bernoulli

# Sparse matrix from a Bernoulli random variable
# giving 1 with probability p and -1 with probability 1-p,
# with p=0.5 (by default) with connectivity of 0.2
# and fixed seed, in Numpy format.
matrix = bernoulli(
    10, 60, connectivity=0.2, sparsity_type="dense")
```

## From Numpy arrays or Scipy sparse matrices
```python
from reservoirpy.nodes import Reservoir

W_matrix = np.random.normal(0, 1, size=(100, 100))
bias_vector = np.ones((100, 1))

reservoir = Reservoir(W=W_matrix, bias=bias_vector)

states = reservoir(X[0].reshape(1, -1))
```

## Parallelization of ESN training/running
```python
X = np.array([[np.sin(np.linspace(0, 12*np.pi, 1000)) 
               for j in range(50)] 
              for i in range(500)]).reshape(-1, 1000, 50)

Y = np.array([[np.sin(np.linspace(0, 12*np.pi, 1000))
               for j in range(40)] 
              for i in range(500)]).reshape(-1, 1000, 40)

print(X.shape, Y.shape)
```

```python
from reservoirpy.nodes import Reservoir, Ridge, ESN
import time

reservoir = Reservoir(100, lr=0.3, sr=1.0)
readout = Ridge(ridge=1e-6)

esn = ESN(reservoir=reservoir, readout=readout, workers=-1)

start = time.time()
esn = esn.fit(X, Y)
print("Parallel (multiprocessing):", 
      "{:.2f}".format(time.time() - start), "seconds")

esn = ESN(reservoir=reservoir, readout=readout, backend="sequential")

start = time.time()
esn = esn.fit(X, Y)
print("Sequential:", 
      "{:.2f}".format(time.time() - start), "seconds")
```

## Example 1 - Hierarchical ESN
```python
from reservoirpy.nodes import Reservoir, Ridge, Input


reservoir1 = Reservoir(100, name="res1-1")
reservoir2 = Reservoir(100, name="res2-1")

readout1 = Ridge(ridge=1e-5, name="readout1-1")
readout2 = Ridge(ridge=1e-5, name="readout2-1")

model = reservoir1 >> readout1 >> reservoir2 >> readout2
```

```python
model = model.fit(X_train, {"readout1-1": Y_train, "readout2-1": Y_train})
```

## Example 2 - Deep ESN
```python
from reservoirpy.nodes import Reservoir, Ridge, Input


reservoir1 = Reservoir(100, name="res1-2")
reservoir2 = Reservoir(100, name="res2-2")
reservoir3 = Reservoir(100, name="res3-2")

readout = Ridge(name="readout-2")

model = reservoir1 >> reservoir2 >> reservoir3 & \
        data >> [reservoir1, reservoir2, reservoir3] >> readout
```

## Example 3 - Multi-inputs
```python
from reservoirpy.nodes import Reservoir, Ridge, Input


reservoir1 = Reservoir(100, name="res1-3")
reservoir2 = Reservoir(100, name="res2-3")
reservoir3 = Reservoir(100, name="res3-3")

readout1 = Ridge(name="readout2")
readout2 = Ridge(name="readout1")

model = [reservoir1, reservoir2] >> readout1 & \
        [reservoir2, reservoir3] >> readout2
```

## Summary
```python
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import reservoirpy as rpy

# just a little tweak to center the plots, nothing to worry about
from IPython.core.display import HTML
HTML("""
<style>
.img-center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    }
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
    }
</style>
""")

rpy.set_seed(42)
```

## Chapter 1 : Reservoir Computing for chaotic timeseries forecasting <span id="chapter1"/>
```python
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import nrmse, rsquare

timesteps = 2510
tau = 17
X = mackey_glass(timesteps, tau=tau)

# rescale between -1 and 1
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1
```

```python
def plot_mackey_glass(X, sample, tau):

    fig = plt.figure(figsize=(13, 5))
    N = sample

    ax = plt.subplot((121))
    t = np.linspace(0, N, N)
    for i in range(N-1):
        ax.plot(t[i:i+2], X[i:i+2], color=plt.cm.magma(255*i//N), lw=1.0)

    plt.title(f"Timeseries - {N} timesteps")
    plt.xlabel("$t$")
    plt.ylabel("$P(t)$")

    ax2 = plt.subplot((122))
    ax2.margins(0.05)
    for i in range(N-1):
        ax2.plot(X[i:i+2], X[i+tau:i+tau+2], color=plt.cm.magma(255*i//N), lw=1.0)

    plt.title(f"Phase diagram: $P(t) = f(P(t-\\tau))$")
    plt.xlabel("$P(t-\\tau)$")
    plt.ylabel("$P(t)$")

    plt.tight_layout()
    plt.show()
```

```python
plot_mackey_glass(X, 500, tau)
```

## Data preprocessing
```python
def plot_train_test(X_train, y_train, X_test, y_test):
    sample = 500
    test_len = X_test.shape[0]
    fig = plt.figure(figsize=(15, 5))
    plt.plot(np.arange(0, 500), X_train[-sample:], label="Training data")
    plt.plot(np.arange(0, 500), y_train[-sample:], label="Training ground truth")
    plt.plot(np.arange(500, 500+test_len), X_test, label="Testing data")
    plt.plot(np.arange(500, 500+test_len), y_test, label="Testing ground truth")
    plt.legend()
    plt.show()
```

```python
from reservoirpy.datasets import to_forecasting

x, y = to_forecasting(X, forecast=10)
X_train1, y_train1 = x[:2000], y[:2000]
X_test1, y_test1 = x[2000:], y[2000:]

plot_train_test(X_train1, y_train1, X_test1, y_test1)
```

## Build your first Echo State Network
```python
units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
regularization = 1e-8
seed = 1234
```

```python
def reset_esn():
    from reservoirpy.nodes import Reservoir, Ridge

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed)
    readout   = Ridge(1, ridge=regularization)

    return reservoir >> readout
```

```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout
```

```python
y = esn(X[0])  # initialisation
reservoir.Win is not None, reservoir.W is not None, readout.Wout is not None
```

```python
np.all(readout.Wout == 0.0)
```

## ESN training
```python
esn = esn.fit(X_train1, y_train1)
```

```python
def plot_readout(readout):
    Wout = readout.Wout
    bias = readout.bias
    Wout = np.r_[bias, Wout]

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(111)
    ax.grid(axis="y")
    ax.set_ylabel("Coefs. of $W_{out}$")
    ax.set_xlabel("reservoir neurons index")
    ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])

    plt.show()
```

```python
plot_readout(readout)
```

## ESN test
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
y_pred1 = esn.run(X_test1)
```

```python
plot_results(y_pred1, y_test1)
```

```python
rsquare(y_test1, y_pred1), nrmse(y_test1, y_pred1)
```

## 1.2 Make the task harder
```python
x, y = to_forecasting(X, forecast=100)
X_train2, y_train2 = x[:2000], y[:2000]
X_test2, y_test2 = x[2000:], y[2000:]

plot_train_test(X_train2, y_train2, X_test2, y_test2)
```

```python
y_pred2 = esn.fit(X_train2, y_train2).run(X_test2)
```

```python
plot_results(y_pred2, y_test2, sample=400)
```

```python
rsquare(y_test2, y_pred2), nrmse(y_test2, y_pred2)
```

## Chapter 2 : Use generative mode <span id="chapter2"/>
```python
units = 500
leak_rate = 0.3
spectral_radius = 0.99
input_scaling = 1.0
connectivity = 0.1      # - density of reservoir internal matrix
input_connectivity = 0.2  # and of reservoir input matrix
regularization = 1e-4
seed = 1234             # for reproducibility
```

```python
def plot_generation(X_gen, X_t, nb_generations, warming_out=None, warming_inputs=None, seed_timesteps=0):

    plt.figure(figsize=(15, 5))
    if warming_out is not None:
        plt.plot(np.vstack([warming_out, X_gen]), label="Generated timeseries")
    else:
        plt.plot(X_gen, label="Generated timeseries")

    plt.plot(np.arange(nb_generations)+seed_timesteps, X_t, linestyle="--", label="Real timeseries")

    if warming_inputs is not None:
        plt.plot(np.arange(seed_timesteps), warming_inputs, linestyle="--", label="Warmup")

    plt.plot(np.arange(nb_generations)+seed_timesteps, np.abs(X_t - X_gen),
             label="Absolute deviation")

    if seed_timesteps > 0:
        plt.fill_between([0, seed_timesteps], *plt.ylim(), facecolor='lightgray', alpha=0.5, label="Warmup")

    plt.plot([], [], ' ', label=f"$R^2 = {round(rsquare(X_t, X_gen), 4)}$")
    plt.plot([], [], ' ', label=f"$NRMSE = {round(nrmse(X_t, X_gen), 4)}$")
    plt.legend()
    plt.show()
```

## Training for one-timestep-ahead forecast
```python
esn = reset_esn()

x, y = to_forecasting(X, forecast=1)
X_train3, y_train3 = x[:2000], y[:2000]
X_test3, y_test3 = x[2000:], y[2000:]

esn = esn.fit(X_train3, y_train3)
```

## Generative mode
```python
seed_timesteps = 100

warming_inputs = X_test3[:seed_timesteps]

warming_out = esn.run(warming_inputs, reset=True)  # warmup
```

```python
nb_generations = 400

X_gen = np.zeros((nb_generations, 1))
y = warming_out[-1]
for t in range(nb_generations):  # generation
    y = esn(y)
    X_gen[t, :] = y
```

```python
X_t = X_test3[seed_timesteps: nb_generations+seed_timesteps]
plot_generation(X_gen, X_t, nb_generations, warming_out=warming_out,
                warming_inputs=warming_inputs, seed_timesteps=seed_timesteps)
```

## Chapter 3 : Online learning <span id="chapter3"/>
```python
units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
seed = 1234
```

```python
from reservoirpy.nodes import FORCE

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)


esn_online = reservoir >> readout
```

## Step by step training
```python
outputs_pre = np.zeros(X_train1.shape)
for t, (x, y) in enumerate(zip(X_train1, y_train1)): # for each timestep of training data:
    outputs_pre[t, :] = esn_online.train(x, y)
```

```python
plot_results(outputs_pre, y_train1, sample=100)
```

```python
plot_results(outputs_pre, y_train1, sample=500)
```

## Training on a whole timeseries
```python
reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)


esn_online = reservoir >> readout
```

```python
esn_online.train(X_train1, y_train1)

pred_online = esn_online.run(X_test1)  # Wout est maintenant figée
```

```python
plot_results(pred_online, y_test1, sample=500)
```

```python
rsquare(y_test1, pred_online), nrmse(y_test1, pred_online)
```

## Loading and data pre-processing
```python
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from joblib import delayed, Parallel
from tqdm import tqdm
```

```python
features = ['com_x', 'com_y', 'com_z', 'trunk_pitch', 'trunk_roll', 'left_x', 'left_y',
            'right_x', 'right_y', 'left_ankle_pitch', 'left_ankle_roll', 'left_hip_pitch',
            'left_hip_roll', 'left_hip_yaw', 'left_knee', 'right_ankle_pitch',
            'right_ankle_roll', 'right_hip_pitch', 'right_hip_roll',
            'right_hip_yaw', 'right_knee']

prediction = ['fallen']
force = ['force_orientation', 'force_magnitude']
```

```python
files = glob.glob("./r4-data/experiments/*")
dfs = []

with Parallel(n_jobs=-1) as parallel:
    dfs = parallel(delayed(pd.read_csv)(f, compression="gzip", header=0, sep=",") for f in tqdm(files))
```

```python
X = []
Y = []
F = []
for i, df in enumerate(dfs):
    X.append(df[features].values)
    Y.append(df[prediction].values)
    F.append(df["force_magnitude"].values)
```

```python
Y_train = []
for y in Y:
    y_shift = np.roll(y, -500)
    y_shift[-500:] = y[-500:]
    Y_train.append(y_shift)
```

```python
def plot_robot(Y, Y_train, F):
    plt.figure(figsize=(10, 7))
    plt.plot(Y_train[1], label="Objective")
    plt.plot(Y[1], label="Fall indicator")
    plt.plot(F[1], label="Applied force")
    plt.legend()
    plt.show()
```

```python
plot_robot(Y, Y_train, F)
```

## Training the ESN
```python
X_train, X_test, y_train, y_test = train_test_split(X, Y_train, test_size=0.2, random_state=42)
```

```python
from reservoirpy.nodes import ESN

reservoir = Reservoir(300, lr=0.5, sr=0.99, input_bias=False)
readout   = Ridge(1, ridge=1e-3)
esn = ESN(reservoir=reservoir, readout=readout, workers=-1)  # version distribuée
```

```python
esn = esn.fit(X_train, y_train)
```

```python
res = esn.run(X_test)
```

```python
from reservoirpy.observables import rmse
scores = []
for y_t, y_p in zip(y_test, res):
    score = rmse(y_t, y_p)
    scores.append(score)


filt_scores = []
for y_t, y_p in zip(y_test, res):
    y_f = y_p.copy()
    y_f[y_f > 0.5] = 1.0
    y_f[y_f <= 0.5] = 0.0
    score = rmse(y_t, y_f)
    filt_scores.append(score)
```

```python
def plot_robot_results(y_test, y_pred):
    for y_t, y_p in zip(y_test, y_pred):
        if y_t.max() > 0.5:
            y_shift = np.roll(y, 500)
            y_shift[:500] = 0.0

            plt.figure(figsize=(7, 5))
            plt.plot(y_t, label="Objective")
            plt.plot(y_shift, label="Fall")
            plt.plot(y_p, label="Prediction")
            plt.legend()
            plt.show()
            break
```

```python
plot_robot_results(y_test, res)
```

```python
print("Averaged RMSE :", f"{np.mean(scores):.4f}", "±", f"{np.std(scores):.5f}")
print("Averaged RMSE (with threshold) :", f"{np.mean(filt_scores):.4f}", "±", f"{np.std(filt_scores):.5f}")
```

## Chapter 5: use case in the wild: canary song decoding <span id="chapter5"/>
```python
from IPython.display import Audio

audio = Audio(filename="./static/song.wav")
```

```python
display(audio)
```

```python
im = plt.imread("./static/canary_outputs.png")
plt.figure(figsize=(15, 15)); plt.imshow(im); plt.axis('off'); plt.show()
```

## Loading and data preprocessing
```python
import os
import glob
import math
import pandas as pd
import librosa as lbr

from tqdm import tqdm
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder

win_length = 1024
n_fft = 2048
hop_length = 512
fmin = 500
fmax = 8000
lifter = 40
n_mfcc = 13


def load_data(directory, max_songs=450):
    audios = sorted(glob.glob(directory + "/**/*.wav", recursive=True))
    annotations = sorted(glob.glob(directory + "/**/*.csv", recursive=True))


    X = []
    Y = []
    vocab = set()

    max_songs = min(len(audios), max_songs)

    for audio, annotation, _ in tqdm(zip(audios, annotations, range(max_songs)), total=max_songs):
        df = pd.read_csv(annotation)
        wav, rate = lbr.load(audio, sr=None)
        x = lbr.feature.mfcc(y=wav, sr=rate,
                              win_length=win_length, hop_length=hop_length,
                              n_fft=n_fft, fmin=fmin, fmax=fmax, lifter=lifter,
                              n_mfcc=n_mfcc)
        delta = lbr.feature.delta(x, mode="wrap")
        delta2 = lbr.feature.delta(x, order=2, mode="wrap")

        X.append(np.vstack([x, delta, delta2]).T)

        y = [["SIL"]] * x.shape[1]

        for annot in df.itertuples():
            start = max(0, round(annot.start * rate / hop_length))
            end = min(x.shape[1], round(annot.end * rate / hop_length))
            y[start:end] = [[annot.syll]] * (end - start)
            vocab.add(annot.syll)

        Y.append(y)

    return X, Y, list(vocab)

X, Y, vocab = load_data("./canary-data")
```

## One-hot encoding of phrase labels
```python
one_hot = OneHotEncoder(categories=[vocab], sparse_output=False)

Y = [one_hot.fit_transform(np.array(y)) for y in Y]
```

```python
X_train, y_train = X[:-10], Y[:-10]
X_test, y_test = X[-10:], Y[-10:]
```

## ESN training
```python
from reservoirpy.nodes import ESN

units = 1000
leak_rate = 0.05
spectral_radius = 0.5
inputs_scaling = 0.001
connectivity = 0.1
input_connectivity = 0.1
regularization = 1e-5
seed = 1234


reservoir = Reservoir(units, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout = Ridge(ridge=regularization)


esn = ESN(reservoir=reservoir, readout=readout, workers=-1)
```

```python
esn = esn.fit(X_train, y_train)
```

```python
outputs = esn.run(X_test)
```

```python
from sklearn.metrics import accuracy_score

scores = []
for y_t, y_p in zip(y_test, outputs):
    targets = np.vstack(one_hot.inverse_transform(y_t)).flatten()

    top_1 = np.argmax(y_p, axis=1)
    top_1 = np.array([vocab[t] for t in top_1])

    accuracy = accuracy_score(targets, top_1)

    scores.append(accuracy)
```

```python
scores  # for each song in the testing set
```

```python
print("Average accuracy :", f"{np.mean(scores):.4f}", "±", f"{np.std(scores):.5f}")

```

## Understand ESN hyperparameters
```python
UNITS = 100               # - number of neurons
LEAK_RATE = 0.3           # - leaking rate
SPECTRAL_RADIUS = 1.25    # - spectral radius of W
INPUT_SCALING = 1.0       # - input scaling
RC_CONNECTIVITY = 0.1     # - density of reservoir internal matrix
INPUT_CONNECTIVITY = 0.2  # and of reservoir input matrix
REGULARIZATION = 1e-8     # - regularization coefficient for ridge regression
SEED = 1234               # for reproductibility
```

```python
import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass

import reservoirpy as rpy
rpy.verbosity(0)

X = mackey_glass(2000)

# rescale between -1 and 1
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1
```

```python
plt.figure()
plt.xlabel("$t$")
plt.title("Mackey-Glass timeseries")
plt.plot(X[:500])
plt.show()
```

## Spectral radius
```python
states = []
spectral_radii = [0.1, 1.25, 10.0]
for spectral_radius in spectral_radii:
    reservoir = Reservoir(
        units=UNITS, 
        sr=spectral_radius, 
        input_scaling=INPUT_SCALING, 
        lr=LEAK_RATE, 
        rc_connectivity=RC_CONNECTIVITY,
        input_connectivity=INPUT_CONNECTIVITY,
        seed=SEED,
    )

    s = reservoir.run(X[:500])
    states.append(s)
```

```python
UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))
for i, s in enumerate(states):
    plt.subplot(len(spectral_radii), 1, i+1)
    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)
    plt.ylabel(f"$sr={spectral_radii[i]}$")
plt.xlabel(f"Activations ({UNITS_SHOWN} neurons)")
plt.show()
```

## Input scaling
```python
states = []
input_scalings = [0.1, 1.0, 10.]
for input_scaling in input_scalings:
    reservoir = Reservoir(
        units=UNITS, 
        sr=SPECTRAL_RADIUS, 
        input_scaling=input_scaling, 
        lr=LEAK_RATE,
        rc_connectivity=RC_CONNECTIVITY, 
        input_connectivity=INPUT_CONNECTIVITY, 
        seed=SEED,
    )

    s = reservoir.run(X[:500])
    states.append(s)
```

```python
UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))
for i, s in enumerate(states):
    plt.subplot(len(input_scalings), 1, i+1)
    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)
    plt.ylabel(f"$iss={input_scalings[i]}$")
plt.xlabel(f"Activations ({UNITS_SHOWN} neurons)")
plt.show()
```

```python
def correlation(states, inputs):
    correlations = [np.corrcoef(states[:, i].flatten(), inputs.flatten())[0, 1] for i in range(states.shape[1])]
    return np.mean(np.abs(correlations))
```

```python
print("input_scaling    correlation")
for i, s in enumerate(states):
    corr = correlation(states[i], X[:500])
    print(f"{input_scalings[i]: <13}    {corr}")
```

## Leaking rate
```python
states = []
leaking_rates = [0.02, 0.3, 1.0]
for leaking_rate in leaking_rates:
    reservoir = Reservoir(
        units=UNITS, 
        sr=SPECTRAL_RADIUS, 
        input_scaling=INPUT_SCALING, 
        lr=leaking_rate,
        rc_connectivity=RC_CONNECTIVITY, 
        input_connectivity=INPUT_CONNECTIVITY, 
        seed=SEED
    )

    s = reservoir.run(X[:500])
    states.append(s)
```

```python
UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))
for i, s in enumerate(states):
    plt.subplot(len(leaking_rates), 1, i+1)
    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)
    plt.ylabel(f"$lr={leaking_rates[i]}$")
plt.xlabel(f"States ({UNITS_SHOWN} neurons)")
plt.show()
```

## Optimize hyperparameters
```python
from reservoirpy.datasets import doublescroll

timesteps = 2000
x0 = [0.37926545, 0.058339, -0.08167691]
X = doublescroll(timesteps, x0=x0, method="RK23")
```

```python
fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("Double scroll attractor (1998)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.grid(False)

for i in range(timesteps-1):
    ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color=plt.cm.cividis(255*i//timesteps), lw=1.0)

plt.show()
```

## Step 1: define the objective
```python
from reservoirpy.observables import nrmse, rsquare
```

```python
# Objective functions accepted by ReservoirPy must respect some conventions:
#  - dataset and config arguments are mandatory, like the empty '*' expression.
#  - all parameters that will be used during the search must be placed after the *.
#  - the function must return a dict with at least a 'loss' key containing the result of the loss function.
# You can add any additional metrics or information with other keys in the dict. See hyperopt documentation for more informations.
def objective(dataset, config, *, input_scaling, N, sr, lr, ridge, seed):
    # This step may vary depending on what you put inside 'dataset'
    x_train, y_train, x_test, y_test = dataset
    
    # You can access anything you put in the config file from the 'config' parameter.
    instances = config["instances_per_trial"]
    
    # The seed should be changed across the instances to be sure there is no bias in the results due to initialization.
    variable_seed = seed 
    
    losses = []; r2s = [];
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(
            units=N, 
            sr=sr, 
            lr=lr, 
            input_scaling=input_scaling, 
            seed=variable_seed
        )

        readout = Ridge(ridge=ridge)

        model = reservoir >> readout


        # Train your model and test your model.
        predictions = model.fit(x_train, y_train) \
                           .run(x_test)
        
        loss = nrmse(y_test, predictions, norm_value=np.ptp(x_train))
        r2 = rsquare(y_test, predictions)
        
        # Change the seed between instances
        variable_seed += 1
        
        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}
```

## Step 2: define the research space
```python
import json

hyperopt_config = {
    "exp": "hyperopt-multiscroll",    # the experimentation name
    "hp_max_evals": 200,              # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
    "hp_space": {                     # what are the ranges of parameters explored
        "N": ["choice", 500],             # the number of neurons is fixed to 500
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "lr": ["loguniform", 1e-3, 1],    # idem with the leaking rate, from 1e-3 to 1
        "input_scaling": ["choice", 1.0], # the input scaling is fixed
        "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)
```

## Step 3: prepare the data
```python
train_len = 1000
forecast = 2

X_train = X[:train_len]
Y_train = X[forecast : train_len + forecast]

X_test = X[train_len : -forecast]
Y_test = X[train_len + forecast:]

dataset = (X_train, Y_train, X_test, Y_test)
```

```python
from reservoirpy.datasets import to_forecasting

X_train, X_test, Y_train, Y_test = to_forecasting(X, forecast=forecast, test_size=train_len-forecast)
```

## Step 4: launch *hyperopt*
```python
from reservoirpy.hyper import research
best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
```

## Step 5: choose parameters
```python
from reservoirpy.hyper import plot_hyperopt_report
fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr", "ridge"), metric="r2")
```

## Classification with Reservoir Computing
```python
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from reservoirpy.datasets import japanese_vowels
from reservoirpy import set_seed, verbosity
from reservoirpy.observables import nrmse, rsquare

from sklearn.metrics import accuracy_score

set_seed(42)
verbosity(0)
```

## References
```python
X_train, Y_train, X_test, Y_test = japanese_vowels()
```

```python
plt.figure()
plt.imshow(X_train[0].T, vmin=-1.2, vmax=2)
plt.title(f"A sample vowel of speaker {np.argmax(Y_train[0]) +1}")
plt.xlabel("Timesteps")
plt.ylabel("LPC (cepstra)")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(X_train[50].T, vmin=-1.2, vmax=2)
plt.title(f"A sample vowel of speaker {np.argmax(Y_train[50]) +1}")
plt.xlabel("Timesteps")
plt.ylabel("LPC (cepstra)")
plt.colorbar()
plt.show()
```

```python
sample_per_speaker = 30
n_speaker = 9
X_train_per_speaker = []

for i in range(n_speaker):
    X_speaker = X_train[i*sample_per_speaker: (i+1)*sample_per_speaker]
    X_train_per_speaker.append(np.concatenate(X_speaker).flatten())

plt.boxplot(X_train_per_speaker)
plt.xlabel("Speaker")
plt.ylabel("LPC (cepstra)")
plt.show()
```

## Transduction (sequence-to-sequence model)
```python
# repeat_target ensure that we obtain one label per timestep, and not one label per utterance.
X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)
```

## Train a simple Echo State Network to solve this task:
```python
from reservoirpy.nodes import Reservoir, Ridge, Input
```

```python
source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

model = [source >> reservoir, source] >> readout
```

```python
Y_pred = model.fit(X_train, Y_train, stateful=False, warmup=2).run(X_test, stateful=False)
```

```python
Y_pred_class = [np.argmax(y_p, axis=1) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t, axis=1) for y_t in Y_test]

score = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(Y_pred_class, axis=0))

print("Accuracy: ", f"{score * 100:.3f} %")
```

## Classification (sequence-to-vector model)
```python
X_train, Y_train, X_test, Y_test = japanese_vowels()
```

```python
from reservoirpy.nodes import Reservoir, Ridge, Input
```

```python
source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

model = source >> reservoir >> readout
```

```python
states_train = []
for x in X_train:
    states = reservoir.run(x, reset=True)
    states_train.append(states[-1, np.newaxis])
```

```python
readout.fit(states_train, Y_train)
```

```python
Y_pred = []
for x in X_test:
    states = reservoir.run(x, reset=True)
    y = readout.run(states[-1, np.newaxis])
    Y_pred.append(y)
```

```python
Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t) for y_t in Y_test]

score = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score * 100:.3f} %")
```

## Summary
```python
import numpy as np
import matplotlib.pyplot as plt

import reservoirpy
from reservoirpy.observables import nrmse, rsquare
reservoirpy.set_seed(42)
reservoirpy.verbosity(0)
```

## Instantiate a node
```python
from sklearn import linear_model

from reservoirpy.nodes import ScikitLearnNode
import reservoirpy
reservoirpy.verbosity(0)
reservoirpy.set_seed(42)

readout = ScikitLearnNode(linear_model.Lasso)
```

```python
readout = ScikitLearnNode(
    model = linear_model.Lasso, 
    model_hypers = {"alpha": 1e-3},
    name = "Lasso"
)
```

## Node usage
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

## Evaluate the model
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

## Node internals
```python
node = ScikitLearnNode(linear_model.PassiveAggressiveRegressor)
node.initialize(x=np.ones((10, 3)), y=np.ones((10, 1)))
str(node.instances)
```

```python
node = ScikitLearnNode(linear_model.PassiveAggressiveRegressor)
# we now have 2 output features !
node.initialize(x=np.ones((10, 3)), y=np.ones((10, 2)))
node.instances
```

## Chapter 2: Using `ScikitLearnNode` for classification <span id="chapter2"/>
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
