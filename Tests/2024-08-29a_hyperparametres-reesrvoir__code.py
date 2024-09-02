from reservoirpy.datasets import doublescroll
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.hyper import research
import numpy as np
import json

# Step 1: Préparer les données
timesteps = 2000
x0 = [0.37926545, 0.058339, -0.08167691]
X = doublescroll(timesteps, x0=x0, method="RK23")
train_len = 1000
forecast = 2
X_train = X[:train_len]
Y_train = X[forecast:train_len+forecast]
X_test = X[train_len:-forecast]
Y_test = X[train_len+forecast:]
dataset = (X_train, Y_train, X_test, Y_test)

# Step 2: Définir la fonction objectif
def objective():
    reservoir = Reservoir(reservoir_size = 100, input_dimension = 1, random_state=42)
    readout = Ridge(ridge=1e-5)
    reservoir.fit(X_train)
    Y_pred = reservoir.run(X_test)
    return np.mean((Y_pred - Y_test)**2)

# Step 3: Définir la configuration Hyperopt
hyperopt_config = {
    "exp": "esn_optimization",
    "hp_max_evals": 100,
    "hp_method": "random",
    "seed": 42,
    "instances_per_trial": 5,
    "hp_space": {
        "N": ["choice", 500],
        "sr": ["loguniform", 1e-2, 10],
        "lr": ["loguniform", 1e-3, 1],
        "input_scaling": ["choice", 1.0],
        "ridge": ["loguniform", 1e-8, 1e1],
        "seed": ["choice", 1234]
    }
}
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

# Step 4: Lancer Hyperopt
best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
