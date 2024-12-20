# General Introduction to Reservoir Computing

## Summary



- <a href="#chapter1">Chapter 1 : A simple task</a>

- <a href="#chapter2">Chapter 2 : Generative models</a>

- <a href="#chapter3">Chapter 3 : Online learning</a>

- <a href="#chapter4">Chapter 4:  use case in the wild: robot falling</a>

- <a href="#chapter5">Chapter 5: use case in the wild: canary song decoding</a>

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

**Mackey-Glass timeseries**



Mackey-Glass equation are a set of delayed differential equations

describing the temporal behavior of different physiological signal,

for example, the relative quantity of mature blood cells over time.

The equations are defined as:



$$

\frac{dP(t)}{dt} = \frac{a P(t - \tau)}{1 + P(t - \tau)^n} - bP(t)

$$



where $a = 0.2$, $b = 0.1$, $n = 10$, and the time delay $\tau = 17$.

$\tau$ controls the chaotic behavior of the equations (the higher it is,

the more chaotic the timeseries becomes.

$\tau=17$ already gives good chaotic results.)

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

- Not completely unpredictable... (not random)

- ...but not easily predictable (not periodic)

- Similar to ECG rhythms, stocks, weather...

### 1.1. Task 1: 10 timesteps ahead forecast

Predict $P(t + 10)$ given $P(t)$.

#### Data preprocessing

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

### Build your first Echo State Network

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

#### ESN training



Training is performed *offline*: it happens only once on the whole dataset.

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

#### ESN test

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

$R^2$ and NRMSE :

```python
rsquare(y_test1, y_pred1), nrmse(y_test1, y_pred1)
```

### 1.2 Make the task harder



Now, let's have a forecasting horizon of 100 timesteps.

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

$R^2$ and NRMSE:

```python
rsquare(y_test2, y_pred2), nrmse(y_test2, y_pred2)
```

## Chapter 2 : Use generative mode <span id="chapter2"/>



- Train ESN on a one-timestep-ahead forecasting task.

- Run the ESN on its own predictions (closed loop generative mode)

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

#### Training for one-timestep-ahead forecast

```python
esn = reset_esn()



x, y = to_forecasting(X, forecast=1)

X_train3, y_train3 = x[:2000], y[:2000]

X_test3, y_test3 = x[2000:], y[2000:]



esn = esn.fit(X_train3, y_train3)
```

#### Generative mode



- 100 steps of the real timeseries used as warmup.

- 300 steps generated by the reservoir, without external inputs.

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



Some learning rules allow to update the readout parameters at every timestep of input series.

Using **FORCE** algorithm *(Sussillo and Abott, 2009)*

<div>

    <img src="./static/online.png" width="700">

</div>

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

#### Step by step training

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

#### Training on a whole timeseries

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

$R^2$ and NRMSE:

```python
rsquare(y_test1, pred_online), nrmse(y_test1, pred_online)
```

## Other timeseries



Try out the other chaotic timeseries included in ReservoirPy: Lorenz chaotic attractor, Hénon map, Logistic map, Double scroll attractor...

## Chapter 4:  use case in the wild: robot falling <span id="chapter4"/>



Data for this use case can be found on Zenodo: https://zenodo.org/record/5900966

<div>

    <img src="./static/sigmaban.gif" width="500">

</div>

#### Loading and data pre-processing

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

#### Training the ESN

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



Data for this use case can be found on Zenodo :

https://zenodo.org/record/4736597

<div>

    <img src="./static/canary.png" width="500">

</div>

```python
from IPython.display import Audio



audio = Audio(filename="./static/song.wav")
```

```python
display(audio)
```

Several temporal motifs to classify: the *phrases*





- There is one label per phrase type.

- A *SIL* label denotes silence. Silence also needs to be detected to segment songs properly.

```python
im = plt.imread("./static/canary_outputs.png")

plt.figure(figsize=(15, 15)); plt.imshow(im); plt.axis('off'); plt.show()
```

#### Loading and data preprocessing

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

#### One-hot encoding of phrase labels

```python
one_hot = OneHotEncoder(categories=[vocab], sparse_output=False)



Y = [one_hot.fit_transform(np.array(y)) for y in Y]
```

We will conduct a first preliminary trial on 100 songs (90 for training, 10 for testing).



The dataset contains 459 songs in total. You may improve your results by adding more data and performing cross validation.

```python
X_train, y_train = X[:-10], Y[:-10]

X_test, y_test = X[-10:], Y[-10:]
```

#### ESN training



We use the special node `ESN` to train our model. This node allows parallelization of states computations, which will speed up the training on this large dataset.

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

