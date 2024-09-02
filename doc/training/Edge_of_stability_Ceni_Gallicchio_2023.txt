Edge of Stability Echo State Network

Ceni, A., & Gallicchio, C. (2023). Edge of stability echo state
networks. arXiv preprint arXiv:2308.02902.

    from functools import partial

    import tqdm



    import matplotlib.pyplot as plt

    import numpy as np



    from reservoirpy.activationsfunc import tanh

    from reservoirpy.mat_gen import normal, uniform

    from reservoirpy.node import Node

    from reservoirpy.nodes import Ridge, Reservoir

    from reservoirpy.datasets import mackey_glass, to_forecasting



    import reservoirpy

    reservoirpy.set_seed(260_418)

    reservoirpy.verbosity(0)

Authors abstract

Echo State Networks (ESNs) are time-series processing models working
under the Echo State Property (ESP)

principle. The ESP is a notion of stability that imposes an asymptotic
fading of the memory of the input. On

the other hand, the resulting inherent architectural bias of ESNs may
lead to an excessive loss of information,

which in turn harms the performance in certain tasks with long
short-term memory requirements. With the

goal of bringing together the fading memory property and the ability to
retain as much memory as possible,

in this paper we introduce a new ESN architecture, called the Edge of
Stability Echo State Network (ES2N).

The introduced ES2N model is based on defining the reservoir layer as a
convex combination of a nonlinear

reservoir (as in the standard ESN), and a linear reservoir that
implements an orthogonal transformation. We

provide a thorough mathematical analysis of the introduced model,
proving that the whole eigenspectrum

of the Jacobian of the ES2N map can be contained in an annular
neighbourhood of a complex circle of

controllable radius, and exploit this property to demonstrate that the
ES2N’s forward dynamics evolves

close to the edge-of-chaos regime by design. Remarkably, our
experimental analysis shows that the newly

introduced reservoir model is able to reach the theoretical maximum
short-term memory capacity. At the

same time, in comparison to standard ESN, ES2N is shown to offer an
excellent trade-off between memory

and nonlinearity, as well as a significant improvement of performance in
autoregressive nonlinear modeling.

Implementation of the ES²N model using ReservoirPy

This notebook is provided as a demo of custom node creation in
ReservoirPy, by implementing the ES^(N) reservoir model proposed in Edge
of stability echo state networks by Andrea Ceni and Claudio Gallicchio.

-   The original basic reservoir equation in a leaky echo state network
    (without noise and feedback) is the following:

$$x[t+1] = ( x[t]

    + \mathbf{W}_{in} \cdot \mathbf{u}[t+1])

-   (1 - ) * x[t]$$

Where α ∈ [0; 1] is the leak rate. A complete description of the
reservoir equation can be found in the documentation.

-   The ES²N model proposed in this paper has the following equation:

$$x[t+1] = (

    \mathbf{W} \cdot x[t]

    + \mathbf{W}_{in} \cdot \mathbf{u}[t+1])

-   (1 - ) * x[t]$$

where O is a random orthogonal matrix, and β ∈ [0; 1] is an
hyper-parameter called proximity. While it has a similar position than
in the Leaky Echo State Network, it serves a very different role.

    # Random orthogonal matrix generation

    # We generate a random matrix and we apply a QR factorization

    def random_orthogonal(units, seed = None):

        D = uniform(units, units,

            sparsity_type = "dense",

            seed=seed,

        )

        Q, _ = np.linalg.qr(D)



        return Q

Node creation

    def forward(node, x):

        f = node.activation

        b = node.proximity

        state = node.state().T



        nonlinear = f(node.W @ state + node.Win @ x.T)

        orthogonal = node.O @ state



        out = b * nonlinear + (1 - b) * orthogonal

        return out.T



    def initialize(

        node,

        x,

        y = None,

        sr = None,

        input_scaling = None,

        input_dim = None,

        seed = None,

    ):

        node.set_input_dim(x.shape[-1])

        node.set_output_dim(node.units)



        # W

        if node.params["W"] is None:

            W = normal(

                node.units, node.units,

                loc = 0.,

                scale = 1,

                sr = sr,

                connectivity = 1.,

                sparsity_type = "dense",

                seed = seed,

            )

            node.set_param("W", W)

        

        # Win

        if node.params["Win"] is None:

            Win = uniform(

                node.units,

                x.shape[-1],

                low = -1,

                high = 1,

                input_scaling = input_scaling,

                connectivity = 1.,

                sparsity_type = "dense",

                seed = seed,

            )

            node.set_param("Win", Win)

        

        # O

        if node.params["O"] is None:

            O = random_orthogonal(

                node.units,

                seed=seed,

            )

            node.set_param("O", O)





    class ES2N(Node):

        def __init__(self,

            units = None,

            sr = 1.,            # ~ rho

            input_scaling = 1., # omega

            proximity = 0.5,    # beta

            activation = tanh,  # phi

            W = None,

            Win = None,

            O = None,

            input_dim = None,

            seed = None,

            name = None,

        ):

            super(ES2N, self).__init__(

                forward = forward,

                initializer = partial(

                    initialize,

                    sr = sr,

                    input_scaling = input_scaling,

                    input_dim = input_dim,

                ),

                params = {

                    "W": W,

                    "Win": Win,

                    "O": O,

                    "internal_state": None,

                },

                hypers = {

                    "units": units,

                    "sr": sr,

                    "input_scaling": input_scaling,

                    "proximity": proximity,

                    "activation": activation,

                },

                name = name,

            )

Quick evaluation

    es2n = ES2N(500)

    readout = Ridge(ridge=1e-4)



    model = es2n >> readout

    mg = mackey_glass(n_timesteps=2000, tau=17)

    X_train, X_test, Y_train, Y_test = to_forecasting(mg, forecast=20, test_size=400)



    model.fit(X_train, Y_train)



    Y_pred = model.run(X_test)



    plt.figure()

    plt.plot(Y_test, color="black", label="test")

    plt.plot(Y_pred, color="red", label="predicted")

    plt.legend()

    plt.show()

Edge of chaos in ES²N

This sections provides a mathematical analysis of the ES2N reservoir
model.

    # Computing the Jacobian matrix of the ES2N node



    def D(es2n_node, u, x):

        kernel = es2n_node.W @ x.T + es2n_node.Win * u

        # Assuming the activation function phi is tanh

        # Its derivative is 1 - tanh^2

        return np.diagflat(1 - np.tanh(kernel)**2)



    def jacobian(es2n_node, u, x):

        beta = es2n_node.proximity

        D_u_x = D(es2n_node, u, x)

        return beta * D_u_x @ es2n_node.W + (1 - beta) * es2n_node.O

    # Visualization of the eigenvalues

    units = 300

    configs = [

        (10, 2, 0.1 ),

        (10, 0, 0.01),

        (1,  0, 0.1 ),

        (1,  0, 0.5 ),

        (1,  2, 0.5 ),

        (1,  0, 0.9 ),

    ]

    circle_x = np.cos(np.linspace(0, 2*np.pi, 1000))

    circle_y = np.sin(np.linspace(0, 2*np.pi, 1000))



    X = np.ones((2*units, 1))

    x = np.ones((1, 1))



    plt.figure(figsize=(15, 4), dpi=200)



    # Upper row (ES2N)

    plt.subplot(2, 6, 1, aspect="equal")

    plt.ylabel("$ES^2N$", fontsize=10)

    for i, (rho, omega, beta) in enumerate(configs):

        es2n = ES2N(units, input_scaling=omega, proximity=beta, sr=rho)

        es2n.run(X)

        J = jacobian(es2n, x, es2n.state())

        vals, _ = np.linalg.eig(J)



        plt.subplot(2, 6, i+1, aspect="equal")

        plt.title(f"$\\rho = {rho}, \\omega = {omega}, \\beta = {beta}$", fontsize=10)

        plt.scatter(vals.real, vals.imag, s=1, color="red")

        plt.plot(circle_x, circle_y, color="black", linewidth=1)

        plt.xlim(-1.1, 1.1)

        plt.ylim(-1.1, 1.1)

        plt.xticks(fontsize=7)

        plt.yticks(fontsize=7)







    # Lower row (LeakyESN)

    plt.subplot(2, 6, 7, aspect="equal")

    plt.ylabel("$Leaky ESN$", fontsize=10)

    for i, (rho, omega, alpha) in enumerate(configs):

        es2n = ES2N(units, input_scaling=omega, proximity=alpha, sr=rho, O=np.eye(units))

        es2n.run(X)

        J = jacobian(es2n, x, es2n.state())

        vals, _ = np.linalg.eig(J)



        plt.subplot(2, 6, i+7, aspect="equal")

        plt.title(f"$\\alpha = {alpha}$", fontsize=10)

        plt.scatter(vals.real, vals.imag, s=1, color="green")

        plt.plot(circle_x, circle_y, color="black", linewidth=1)

        plt.xlim(-1.1, 1.1)

        plt.ylim(-1.1, 1.1)

        plt.xticks(fontsize=7)

        plt.yticks(fontsize=7)



    plt.show()

Memory capacity

In this section, we evaluate the memory capacity of the ES2N model. The
memory capacity tasks consists in reproducing the input signal u with a
delay of k (u[t − k]).

The memory capacity of a model with the timeseries u and a delay of k is
defined as :

MC_(k) = r(y_(k)[t], u[t − k])²

where r denotes the Pearson correlation coefficient.

The MC score is defined as the sum over all k:

$$MC = \sum_{k=1}^{\infty} MC_k$$

    # Task definition

    rng = np.random.default_rng(seed=2504)

    series = rng.uniform(low=-0.8, high=0.8, size=(6000, 1))

    # Similar to the ReservoirPy method reservoirpy.datasets.to_forecasting, but in the other way: input X is ahead of output Y.

    def to_postcasting(k=1):

        if k == 0:

            return series[:-1000], series[-1000:], series[:-1000], series[-1000:]

        X_train, X_test, shifted_train, shifted_test = to_forecasting(series, forecast=k, test_size=1000)

        return shifted_train, shifted_test, X_train, X_test



    # kth memory capacity (MC_k) as defined in (Jaeger, 2002)

    def kth_memory_capacity(k=1, model=None):

        # Dataset

        X_train, X_test, Y_train, Y_test = to_postcasting(k=k)

        # Model

        if model is None:

            model = ES2N(100, input_scaling=1., proximity=0.1, sr=1.) >> Ridge(ridge=1e-5)

        # Fit and run

        model.fit(X_train, Y_train, warmup=100)

        Y_pred = model.run(X_test)

        # u[t-k] - z_k[t] square correlation

        return np.square(np.corrcoef(Y_pred, Y_test, rowvar=False)[1, 0])

    # Faster method : compute all memory capacities all at once

    from numpy.lib.stride_tricks import sliding_window_view



    def memory_capacity(k=200, model=None):

        # Dataset definition



        # sliding_window_view creates a matrix of the same

        # timeseries with an incremental shift on each column

        dataset = sliding_window_view(series[:, 0], k)[:, ::-1]

        X_train = dataset[:-1000, :1]

        X_test = dataset[-1000:, :1]

        Y_train = dataset[:-1000, 1:]

        Y_test = dataset[-1000:, 1:]

        # Model

        if model is None:

            model = ES2N(100, input_scaling=1., proximity=0.1, sr=1.) >> Ridge(ridge=1e-5)

        # Fit and run

        model.fit(X_train, Y_train, warmup=k)

        Y_pred = model.run(X_test)



        # u[t-k] - z_k[t] square correlation

        capacities = np.square([np.corrcoef(y_pred, y_test, rowvar=False)[1, 0] for y_pred, y_test in zip(Y_pred.T, Y_test.T)])

        return capacities

    from reservoirpy.activationsfunc import identity

    proximity = 0.05



    mc_ES2N = memory_capacity(

        k = 200, 

        model= ES2N(100, sr=.9, proximity=proximity, input_scaling=0.1) >> Ridge(ridge=1e-7)

    )



    mc_ESN = memory_capacity(

        k = 200, 

        model= Reservoir(100, sr=.9, lr=1., input_scaling=0.1) >> Ridge(ridge=1e-7)

    )



    mc_linearESN = memory_capacity(

        k=200, 

        model= Reservoir(100, sr=.9, input_scaling=0.1, input_bias=False, lr=1, activation=identity) >> Ridge(ridge=1e-7)

    )



    mc_orthoESN = memory_capacity(

        k=200, 

        model= Reservoir(W=random_orthogonal(units=100, seed=rng), sr=.9, input_scaling=0.1, input_bias=False, lr=1) >> Ridge(ridge=1e-7)

    )



    def ring_matrix(units):

        return np.roll(np.eye(units), shift=1, axis=0)



    mc_linearSCR = memory_capacity(

        k=200, 

        model= Reservoir(W=0.9*ring_matrix(units=100), input_connectivity=0.01, input_scaling=0.1, input_bias=False, lr=1., activation=identity) >> Ridge(ridge=1e-7)

    )



    plt.figure()

    plt.plot(mc_ES2N, label=f"$ES2N (\\beta={proximity})$")

    plt.plot(mc_ESN, label="$leaky ESN (\\alpha=1)$")

    plt.plot(mc_linearESN, label="$linearESN$")

    plt.plot(mc_orthoESN, label="$orthoESN$")

    plt.plot(mc_linearSCR, label="$linearSCR$")

    plt.xlabel("$k$")

    plt.ylabel("$MC_k$")

    plt.legend()

    plt.show()



    print(f"ES2N memory capacity: {np.sum(mc_ES2N)}")

    print(f"linearESN memory capacity: {np.sum(mc_linearESN)}")

    print(f"orthoESN memory capacity: {np.sum(mc_orthoESN)}")

    print(f"linearSCR memory capacity: {np.sum(mc_linearSCR)}")

    rng = np.random.default_rng(seed=2504)



    a = rng.uniform(0.1, 1, (50, ))

    s = rng.integers(0, 3, 50).astype(np.float64)



    values = np.sort(a * np.power(10, -s))



    esn_mc = np.zeros((50, 10))

    es2n_mc = np.zeros((50, 10))



    for i, value in enumerate(tqdm.tqdm(values)):

        for instance in range(10):

            esn = Reservoir(500, sr=.9, input_scaling=0.1, lr=value) >> Ridge(ridge=1e-5)

            esn_mc[i, instance] = np.sum(memory_capacity(k=200, model=esn))



            es2n = ES2N(100, sr=.9, input_scaling=0.1, proximity=value) >> Ridge(ridge=1e-5)

            es2n_mc[i, instance] = np.sum(memory_capacity(k=200, model=es2n))





    plt.figure()

    plt.plot(values, np.mean(esn_mc, axis=1), ".--", color="green", markersize=10, label="$ESN$")

    plt.plot(values, np.mean(es2n_mc, axis=1), ".--", color="red", markersize=10, label="$ES^2N$")

    plt.xlabel("$\\alpha / \\beta$")

    plt.ylabel("$MC$")

    plt.legend()

    plt.xlim(-0.05, 1.05)

    plt.show()

Memory-nonlinearity trade-off

    # Task definition



    ITER = 100

    rng = np.random.default_rng(seed=2504)

    # Hyper-parameters

    UNITS = 100

    INPUT_SCALINGS = rng.uniform(low=0.2, high=6., size=ITER)

    SPECTRAL_RADII = rng.uniform(low=0.1, high=3., size=ITER)

    a = rng.uniform(0.1, 1, ITER)

    s = rng.integers(0, 2, ITER).astype(np.float64)

    ALPHAS = np.sort(a * np.power(10, -s))



    # Dataset



    def y(u, tau, nu):

        return np.sin(nu * u[MAX_TAU-tau: -tau])



    MAX_TAU = 20

    u = rng.uniform(low=-0.8, high=0.8, size=(6_000+MAX_TAU, 1))

    x_train = u[MAX_TAU : 5_000+MAX_TAU]

    x_test = u[5_000+MAX_TAU : ]



    TAUS = np.linspace(1, MAX_TAU, MAX_TAU).astype(int)

    logNUS = np.linspace(-1.6, 1.6, 33)

    NUS = np.exp(logNUS)



    # Y is a NumPy array of shape (test_timesteps, 20*33).

    # In the following cell, all output dimensions will be trained at the same time.

    Y = np.array([y(u, tau, nu) for tau in TAUS for nu in NUS]).squeeze().T

    y_train = Y[:5_000]

    y_test  = Y[5_000:]

    # leaky ESN



    print("leakyESN")

    best_nrmse_leakyESN = np.full(660, np.Infinity)

    for i in tqdm.tqdm(range(ITER)):

        input_scaling = INPUT_SCALINGS[i]

        spectral_radius = SPECTRAL_RADII[i]

        alpha = ALPHAS[i]



        model = Reservoir(UNITS, 

            sr=spectral_radius, 

            lr=alpha, 

            input_scaling=input_scaling,

        ) >> Ridge(ridge=1e-7)



        model.fit(x_train, y_train, warmup=100)

        y_pred = model.run(x_test)



        rmse = np.sqrt(np.mean(np.square(y_test - y_pred), axis=0))

        nrmse = rmse / y_test.var(axis=0)

        best_nrmse_leakyESN = np.fmin(best_nrmse_leakyESN, nrmse)





    # ES2N



    print("ES2N")

    best_nrmse_ES2N = np.full(660, np.Infinity)

    for i in tqdm.tqdm(range(ITER)):

        input_scaling = INPUT_SCALINGS[i]

        spectral_radius = SPECTRAL_RADII[i]

        alpha = ALPHAS[i]



        model = ES2N(UNITS, 

            sr=spectral_radius, 

            proximity=alpha, 

            input_scaling=input_scaling

        ) >> Ridge(ridge=1e-7)



        model.fit(x_train, y_train, warmup=100)

        y_pred = model.run(x_test)



        rmse = np.sqrt(np.mean(np.square(y_test - y_pred), axis=0))

        nrmse = rmse / y_test.var(axis=0)

        best_nrmse_ES2N = np.fmin(best_nrmse_ES2N, nrmse)





    # linearSCR



    print("linearSCR")

    best_nrmse_linearSCR = np.full(660, np.Infinity)

    for i in tqdm.tqdm(range(ITER)):

        input_scaling = INPUT_SCALINGS[i]

        spectral_radius = SPECTRAL_RADII[i]

        alpha = ALPHAS[i]



        if spectral_radius > 1.:

            continue



        model = Reservoir(

            W=spectral_radius*ring_matrix(units=UNITS),

            input_connectivity=0.1,

            input_scaling=input_scaling,

            input_bias=False,

            lr=1.,

            activation=identity

        ) >> Ridge(ridge=1e-8)



        model.fit(x_train, y_train, warmup=100)

        y_pred = model.run(x_test)



        rmse = np.sqrt(np.mean(np.square(y_test - y_pred), axis=0))

        nrmse = rmse / y_test.var(axis=0)

        best_nrmse_linearSCR = np.fmin(best_nrmse_linearSCR, nrmse)

    from matplotlib.colors import PowerNorm



    plt.figure(figsize=(20, 4), dpi=200)



    plt.subplot(1, 3, 1)

    plt.pcolormesh(

        best_nrmse_leakyESN.reshape(len(TAUS), len(NUS)), 

        norm=PowerNorm(gamma=0.5, vmin=0, vmax=1,),

        edgecolors="#FFFFFF0F",

    )

    plt.colorbar()

    plt.xlabel("Non-linearity strength $\\nu$ (log scale)")

    plt.ylabel("Delay $\\tau$")

    plt.xticks(np.arange(0.5, 33.5, 5), NUS[::5].round(decimals=2))

    plt.yticks(np.arange(0.5, 20.5, 3), TAUS[::3])

    plt.title("NRMSE optimised leaky ESN")



    plt.subplot(1, 3, 2)

    plt.pcolormesh(

        best_nrmse_linearSCR.reshape(len(TAUS), len(NUS)), 

        norm=PowerNorm(gamma=0.5, vmin=0, vmax=1,),

        edgecolors="#FFFFFF0F",

    )

    plt.colorbar()

    plt.xlabel("Non-linearity strength $\\nu$ (log scale)")

    plt.ylabel("Delay $\\tau$")

    plt.xticks(np.arange(0.5, 33.5, 5), NUS[::5].round(decimals=2))

    plt.yticks(np.arange(0.5, 20.5, 3), TAUS[::3])

    plt.title("NRMSE optimised linearSCR")



    plt.subplot(1, 3, 3, )

    plt.pcolormesh(

        best_nrmse_ES2N.reshape(len(TAUS), len(NUS)), 

        norm=PowerNorm(gamma=0.5, vmin=0, vmax=1,),

        edgecolors="#FFFFFF0F",    

    )

    plt.colorbar()

    plt.xlabel("Non-linearity strength $\\nu$ (log scale)")

    plt.ylabel("Delay $\\tau$")

    plt.xticks(np.arange(0.5, 33.5, 5), NUS[::5].round(decimals=2))

    plt.yticks(np.arange(0.5, 20.5, 3), TAUS[::3])

    plt.title("NRMSE optimised ES²N")

    plt.show()

Multiple superimposed oscillators in auto-regressive node

    def mso(timesteps, frequencies, sample_rate=1, normalize=True):

        t = np.arange(timesteps).reshape(timesteps, 1) / sample_rate

        y = np.zeros((timesteps, 1))

        for f in frequencies:

            y += np.sin(f * t)

        

        if normalize:

            return (2 * y - y.min() - y.max()) / (y.max() - y.min())

        else:

            return y





    def mso8(timesteps, sample_rate=1):

        return mso(

            timesteps = timesteps, 

            sample_rate = sample_rate, 

            frequencies = [0.2, 0.311, 0.42, 0.51, 0.63, 0.74, 0.85, 0.97]

        )





    def noisy_tanh(x):

        return np.tanh(x + np.random.normal(loc=0, scale=1e-4))



    TRAINING_STEPS = 6_383

    TEST_STEPS = 50_000



    mso_ts = mso8(timesteps = TRAINING_STEPS+TEST_STEPS)



    x_train, x_test, y_train, y_test = to_forecasting(timeseries=mso_ts, forecast=1, test_size=TEST_STEPS)

    # ES2N



    es2n = ES2N(units=300, proximity=0.03, sr=1., input_scaling=0.11, input_dim=2, activation=noisy_tanh) >> Ridge(ridge = 0)

    es2n.fit(x_train, y_train, warmup=100)



    es2n.hypers["activation"] = np.tanh

    y_pred_es2n = np.zeros((TEST_STEPS+1, 1))

    y_pred_es2n[0] = es2n(x_test[0])



    for step in range(TEST_STEPS):

        y_pred_es2n[step+1] = es2n(y_pred_es2n[step])

    # ESN



    esn = Reservoir(units=3000, lr=0.9, sr=0.99, input_scaling=0.05, activation=noisy_tanh, input_bias=False) >> Ridge(ridge = 0)

    esn.fit(x_train, y_train, warmup=100)



    esn.hypers["activation"] = np.tanh

    y_pred_esn = np.zeros((TEST_STEPS+1, 1))

    y_pred_esn[0] = esn(x_test[0])



    for step in range(TEST_STEPS):

        y_pred_esn[step+1] = esn(y_pred_esn[step])

    plt.figure()

    plt.title("$ESN$ vs $ES^2N$ on the multiple superimposed oscillator task")

    plt.plot(y_test, '--', color="black")

    plt.plot(y_pred_es2n, color="red", alpha=0.5, label="$ES^2N$")

    plt.plot(y_pred_esn, color="green", alpha=0.5, label="$ESN$")

    plt.ylim(-1, 1)

    plt.xlim(0, 300)

    plt.legend()

    plt.show()

    plt.figure()

    plt.title("$ESN$ vs $ES^2N$ on the multiple superimposed oscillator task (after 50 000 timesteps)")

    plt.plot(y_test, '--', color="black")

    plt.plot(y_pred_es2n, color="red", alpha=0.5, label="$ES^2N$")

    plt.plot(y_pred_esn, color="green", alpha=0.5, label="$ESN$")

    plt.ylim(-1, 1)

    plt.xlim(TEST_STEPS-300, TEST_STEPS)

    plt.legend()

    plt.show()

With 5 different seeds

    plt.figure(figsize=(8, 20))

    plt.suptitle("$ESN$ vs $ES^2N$ on the multiple superimposed oscillator task")



    for i in tqdm.tqdm(range(5)):



        es2n = ES2N(units=300, proximity=0.03, sr=1., input_scaling=0.11, input_dim=2, activation=noisy_tanh, seed=i) >> Ridge(ridge = 0)

        es2n.fit(x_train, y_train, warmup=100)



        es2n.hypers["activation"] = np.tanh

        y_pred_es2n = np.zeros((TEST_STEPS+1, 1))

        y_pred_es2n[0] = es2n(x_test[0])



        for step in range(TEST_STEPS):

            y_pred_es2n[step+1] = es2n(y_pred_es2n[step])



        # ESN

        esn = Reservoir(units=3_000, lr=0.9, sr=0.99, input_scaling=0.05, activation=noisy_tanh, input_bias=False, seed=i) >> Ridge(ridge = 0)

        esn.fit(x_train, y_train, warmup=100)



        esn.hypers["activation"] = np.tanh

        y_pred_esn = np.zeros((TEST_STEPS+1, 1))

        y_pred_esn[0] = esn(x_test[0])



        for step in range(TEST_STEPS):

            y_pred_esn[step+1] = esn(y_pred_esn[step])

        

        # Plot line

        plt.subplot(5,2,2*i+1)

        plt.plot(y_test, '--', color="black")

        plt.plot(y_pred_es2n, color="red", alpha=0.5, label="$ES^2N$")

        plt.plot(y_pred_esn, color="green", alpha=0.5, label="$ESN$")

        plt.ylim(-1, 1)

        plt.xlim(0, 300)

        plt.subplot(5,2,2*i+2)

        plt.plot(y_test, '--', color="black")

        plt.plot(y_pred_es2n, color="red", alpha=0.5, label="$ES^2N$")

        plt.plot(y_pred_esn, color="green", alpha=0.5, label="$ESN$")

        if i==0:

            plt.title("after 50 000 timesteps")

            plt.legend()

        plt.ylim(-1, 1)

        plt.xlim(TEST_STEPS-300, TEST_STEPS)



    plt.show()

    plt.figure()

    plt.plot(y_test, '--', color="black")

    plt.plot(y_pred_es2n, color="red", alpha=0.5, label="$ES^2N$")

    plt.plot(y_pred_esn, color="green", alpha=0.5, label="$ESN$")

    if i==0:

        plt.title("after 50 000 timesteps")

        plt.legend()

    # plt.ylim(-1, 1)

    plt.xlim(TEST_STEPS-100, TEST_STEPS)



    plt.show()
