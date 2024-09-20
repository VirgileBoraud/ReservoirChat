import numpy as np
import jax
from jax.experimental import sparse
import matplotlib.pyplot as plt
from functools import partial


class ESN:
    def _bernoulli(self, key, shape, dtype):
        boolean = jax.random.bernoulli(key=key, p=0.5, shape=shape)
        return 2.0 * jax.numpy.array(boolean, dtype=dtype) - 1.0

    def __init__(
        self,
        units,
        connectivity=0.1,
        input_connectivity=0.1,
        weight_scale=0.1,
        lr=0.9,
        input_scaling=1.0,
        ridge=1e-10,
        seed=2504,
        input_dim=1,
        output_dim=1,
    ):
        # Création du modèle

        self.key = jax.random.PRNGKey(seed=seed)
        (
            self.key,
            W_key,
            Win_key,
        ) = jax.random.split(key=self.key, num=3)
        # Création du réservoir
        unique_indices = False
        self.W = (
            sparse.random_bcoo(
                key=W_key,
                shape=(units, units),
                dtype=np.float32,
                indices_dtype=int,
                nse=connectivity,
                generator=jax.random.normal,
                unique_indices=unique_indices,
            )
            * weight_scale
        )

        self.Win = sparse.random_bcoo(
            key=Win_key,
            shape=(units, input_dim),
            dtype=np.float32,
            indices_dtype=int,
            nse=input_connectivity,
            generator=self._bernoulli,
        )

        # état actuel
        self.x = jax.numpy.zeros((units, 1))
        self.Wout = jax.numpy.zeros((output_dim, units + 1))

        self.lr = lr
        self.units = units
        self.ridge = ridge

    def _step_reservoir(x, u, W, Win, lr):
        # print("_step_reservoir", "x", x.shape, "u", u.shape, "W", W.shape, "Win", Win.shape)
        u = u.reshape(-1, 1)
        new_x = lr * jax.numpy.tanh(W @ x + Win @ u) + (1 - lr) * x
        return new_x, new_x[:, 0]

    def _run_reservoir(W, Win, lr, x, U):
        # print("_run_reservoir", "W", W.shape, "Win", Win.shape, "x", x.shape, "U", U.shape)
        step_ = partial(ESN._step_reservoir, W=W, Win=Win, lr=lr)
        new_x, states = jax.lax.scan(step_, x, U)
        return new_x, states

    def _ridge_regression(ridge, X, Y):
        # print("_ridge_regression", "X", X.shape, "Y", Y.shape)
        XXT = X.T @ X
        YXT = Y.T @ X
        n = XXT.shape[0]
        I_n = jax.numpy.eye(n)
        Wout = jax.scipy.linalg.solve(XXT + ridge * I_n, YXT.T, assume_a="sym")

        return Wout.T

    def _fit(W, Win, lr, ridge, x, U, Y):
        # print("_fit", "W", W.shape, "Win", Win.shape, "x", x.shape, "U", U.shape, "Y", Y.shape)
        new_x, X = ESN._run_reservoir(W, Win, lr, x, U)
        Wout = ESN._ridge_regression(ridge, X, Y)

        return new_x, Wout

    @jax.jit
    def _step(x, u, W, Win, Wout, lr):
        # print("_step", "W", W.shape, "Win", Win.shape, "Wout", Wout.shape, "x", x.shape, "u", u.shape)
        new_x, new_state = ESN._step_reservoir(x=x, u=u, W=W, Win=Win, lr=lr)
        # print("_step after reservoir", "Wout", Wout.shape, "new_x", new_x.shape, "new_state", new_state.shape)
        y = Wout @ new_x
        return new_x, y.reshape(-1)

    def _run(x, U, W, Win, Wout, lr):
        # print("_run", "W", W.shape, "Win", Win.shape, "Wout", Wout.shape, "x", x.shape, "U", U.shape)
        step_ = partial(ESN._step, W=W, Win=Win, Wout=Wout, lr=lr)
        new_x, Y = jax.lax.scan(step_, x, U)

        return new_x, Y

    def fit(self, U, Y):
        # print("fit", "U", U.shape, "Y", Y.shape)
        self.x, self.Wout = ESN._fit(
            W=self.W,
            Win=self.Win,
            lr=self.lr,
            ridge=self.ridge,
            x=self.x,
            U=U,
            Y=Y,
        )
        _ = self.x.block_until_ready()
        _ = self.Wout.block_until_ready()

    def run(self, U):
        # print("run", "U", U.shape)
        new_x, Y = ESN._run(
            W=self.W, Win=self.Win, lr=self.lr, Wout=self.Wout, x=self.x, U=U
        )
        _ = new_x.block_until_ready()
        _ = Y.block_until_ready()
        return Y

    def plot_Ypred(self, U_train, U_test, Y_train, Y_test, input_noise=False):
        # print("plot_Ypred", "U_train", U_train.shape, "U_test", U_test.shape, "Y_train", Y_train.shape, "Y_test", Y_test.shape)
        if input_noise:
            T_train = U_train.shape[0]
            noise_train = 0.2 * jax.random.bernoulli(
                key=jax.random.key(0), p=0.5, shape=(T_train, 1)
            )
            U_train = jax.numpy.concatenate((U_train, noise_train), axis=1)

            T_test = U_test.shape[0]
            noise_test = 0.2 * jax.random.bernoulli(
                key=jax.random.key(1), p=0.5, shape=(T_test, 1)
            )
            U_test = jax.numpy.concatenate((U_test, noise_test), axis=1)

        self.fit(U_train, Y_train)
        Y_pred = self.run(U_test)
        rmse = jax.numpy.sqrt(jax.numpy.mean(jax.numpy.square(Y_test - Y_pred)))
        print(rmse)

        plt.figure()
        plt.plot(Y_test, color="black", label="Y_test")
        plt.plot(Y_pred, color="red", label="Y_pred")
        plt.legend()
        plt.show()

        return Y_pred
