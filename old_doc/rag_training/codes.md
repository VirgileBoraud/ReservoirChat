# File: test_concat.txt
```python
x = [np.ones((1, 5)) for _ in range(3)]

c = Concat()

res = c(x)

assert c.input_dim == (5, 5, 5)
assert_array_equal(res, np.ones((1, 15)))

x = np.ones((1, 5))

c = Concat()

res = c(x)

assert_array_equal(res, np.ones((1, 5)))

reservoirs = [Reservoir(10, name=f"r{i}") for i in range(3)]

model = reservoirs >> Concat()

x = {f"r{i}": np.ones((1, 5)) for i in range(3)}

res = model(x)

assert res.shape == (1, 30)

res_final = Reservoir(20)

model = reservoirs >> res_final

res = model(x)

```
# File: test_nvar.txt
```python
from scipy.special import comb

from math import comb

linear_dim = delay * input_dim
nonlinear_dim = comb(linear_dim + order - 1, order)
return int(linear_dim + nonlinear_dim)

node = NVAR(3, 2)

data = np.ones((1, 10))
res = node(data)

assert node.store is not None
assert node.strides == 1
assert node.delay == 3
assert node.order == 2

data = np.ones((10000, 10))
res = node.run(data)

assert res.shape == (10000, _get_output_dim(10, 3, 2))

node1 = NVAR(3, 1)
node2 = NVAR(3, 2, strides=2)

data = np.ones((1, 10))
res = (node1 >> node2)(data)

```
# File: mat_gen.txt
```python
random_sparse
uniform
normal
bernoulli
zeros
ones
generate_internal_weights
generate_input_weights
fast_spectral_initialization
Initializer

from reservoirpy.mat_gen import random_sparse

initializer = random_sparse(dist="uniform", sr=0.9, connectivity=0.1)
matrix = initializer(100, 100)
print(type(matrix), "\\n", matrix[:5, :5])

matrix = random_sparse(100, 100, dist="uniform", sr=0.9, connectivity=0.1)
print(type(matrix), "\\n", matrix[:5, :5])

from reservoirpy.mat_gen import normal

matrix = normal(7, 10, degree=2, direction="out", loc=0, scale=0.5)
print(type(matrix), "\\n", matrix)

from reservoirpy.mat_gen import normal

matrix = normal(50, 100, loc=0, scale=0.5)
print(type(matrix), "\\n", matrix[:5, :5])

from reservoirpy.mat_gen import uniform

matrix = uniform(200, 60, low=0.5, high=0.5, connectivity=0.9, input_scaling=0.3)
print(type(matrix), "\\n", matrix[:5, :5])

from reservoirpy.mat_gen import bernoulli

matrix = bernoulli(10, 60, connectivity=0.2, sparsity_type="dense")
print(type(matrix), "\\n", matrix[:5, :5])

.. [1] C. Gallicchio, A. Micheli, and L. Pedrelli,
        ‘Fast Spectral Radius Initialization for Recurrent
        Neural Networks’, in Recent Advances in Big Data and
        Deep Learning, Cham, 2020, pp. 380–390,
        doi: 10.1007/978-3-030-16841-4_39.

from typing_extensions import Literal

from typing import Literal

"fast_spectral_initialization",
"generate_internal_weights",
"generate_input_weights",
"random_sparse",
"uniform",
"normal",
"bernoulli",
"zeros",
"ones",

deprecated = {
    "proba": "connectivity",
    "typefloat": "dtype",
    "N": None,
    "dim_input": None,
}

new_kwargs = {}
args = [None, None]
args_order = ["N", "dim_input"]
for depr, repl in deprecated.items():
    if depr in kwargs:
        depr_argument = kwargs.pop(depr)

        msg = f"'{depr}' parameter is deprecated since v0.3.1."
        if repl is not None:
            msg += f" Consider using '{repl}' instead."
            new_kwargs[repl] = depr_argument
        else:
            args[args_order.index(depr)] = depr_argument

        warnings.warn(msg, DeprecationWarning)

args = [a for a in args if a is not None]
kwargs.update(new_kwargs)

return args, kwargs

"""Base class for initializer functions. Allow updating initializer function
parameters several times before calling. May perform spectral radius rescaling
or input scaling as a post-processing to initializer function results.

Parameters
----------
func : callable
    Initializer function. Should have a `shape` argument and return a Numpy array
    or Scipy sparse matrix.
autorize_sr : bool, default to True
    Authorize spectral radius rescaling for this initializer.
autorize_input_scaling : bool, default to True
    Authorize input_scaling for this initializer.
autorize_rescaling : bool, default to True
    Authorize any kind of rescaling (spectral radius or input scaling) for this
    initializer.

Example
-------
>>> from reservoirpy.mat_gen import random_sparse
>>> init_func = random_sparse(dist="uniform")
>>> init_func = init_func(connectivity=0.1)
>>> matrix = init_func(5, 5)  # actually creates the matrix
>>> matrix = random_sparse(5, 5, dist="uniform", connectivity=0.1)  # also creates the matrix
"""

def __init__(
    self,
    func,
    autorize_sr=True,
    autorize_input_scaling=True,
    autorize_rescaling=True,
):
    self._func = func
    self._kwargs = dict()
    self._autorize_sr = autorize_sr
    self._autorize_input_scaling = autorize_input_scaling
    self._autorize_rescaling = autorize_rescaling

    self.__doc__ = func.__doc__
    self.__annotations__ = func.__annotations__
    if self._autorize_sr:
        self.__annotations__.update({"sr": float})
    if self._autorize_input_scaling:
        self.__annotations__.update(
            {"input_scaling": Union[float, Iterable[float]]}
        )

def __repr__(self):
    split = super().__repr__().split(" ")
    return split[0] + f" ({self._func.__name__}) " + " ".join(split[1:])

def __call__(self, *shape, **kwargs):
    if "sr" in kwargs and not self._autorize_sr:
        raise ValueError(
            "Spectral radius rescaling is not supported by this initializer."
        )

    if "input_scaling" in kwargs and not self._autorize_input_scaling:
        raise ValueError("Input scaling is not supported by this initializer.")

    new_shape, kwargs = _filter_deprecated_kwargs(kwargs)

    if len(new_shape) > 1:
        shape = new_shape
    elif len(new_shape) > 0:
        shape = (new_shape[0], new_shape[0])

    init = copy.deepcopy(self)
    init._kwargs.update(kwargs)

    if len(shape) > 0:
        if init._autorize_rescaling:
            return init._func_post_process(*shape, **init._kwargs)
        else:
            return init._func(*shape, **init._kwargs)
    else:
        if len(kwargs) > 0:
            return init
        else:
            return init._func(**init._kwargs)  # should raise, shape is None

def _func_post_process(self, *shape, sr=None, input_scaling=None, **kwargs):
    """Post process initializer with spectral radius or input scaling factors."""
    if sr is not None and input_scaling is not None:
        raise ValueError(
            "'sr' and 'input_scaling' parameters are mutually exclusive for a "
            "given matrix."
        )

    if sr is not None:
        return _scale_spectral_radius(self._func, shape, sr, **kwargs)
    elif input_scaling is not None:
        return _scale_inputs(self._func, shape, input_scaling, **kwargs)
    else:
        return self._func(*shape, **kwargs)

"""Get a scipy.stats random variable generator.

Parameters
----------
dist : str
    A scipy.stats distribution.
random_state : Generator
    A Numpy random generator.

Returns
-------
scipy.stats.rv_continuous or scipy.stats.rv_discrete
    A scipy.stats random variable generator.
"""
if dist == "custom_bernoulli":
    return _bernoulli_discrete_rvs(**kwargs, random_state=random_state)
elif dist in dir(stats):
    distribution = getattr(stats, dist)
    return partial(distribution(**kwargs).rvs, random_state=random_state)
else:
    raise ValueError(
        f"'{dist}' is not a valid distribution name. "
        "See 'scipy.stats' for all available distributions."
    )

p=0.5, value: float = 1.0, random_state: Union[Generator, int] = None

"""Generator of Bernoulli random variables, equal to +value or -value.

Parameters
----------
p : float, default to 0.5
    Probability of single success (+value). Single failure (-value) probability
    is (1-p).
value : float, default to 1.0
    Success value. Failure value is equal to -value.

Returns
-------
callable
    A random variable generator.
"""
rg = rand_generator(random_state)

def rvs(size: int = 1):
    return rg.choice([value, -value], p=[p, 1 - p], replace=True, size=size)

return rvs

"""Change the spectral radius of a matrix created with an
initializer.

Parameters
----------
w_init : Initializer
    An initializer.
shape : tuple of int
    Shape of the matrix.
sr : float
    New spectral radius.
seed: int or Generator
    A random generator or an integer seed.

Returns
-------
Numpy array or Scipy sparse matrix
    Rescaled matrix.
"""
convergence = False

if "seed" in kwargs:
    seed = kwargs.pop("seed")
else:
    seed = None
rg = rand_generator(seed)

w = w_init(*shape, seed=seed, **kwargs)

while not convergence:
    # make sure the eigenvalues are reachable.
    # (maybe find a better way to do this on day)
    try:
        current_sr = spectral_radius(w)
        if -_epsilon < current_sr < _epsilon:
            current_sr = _epsilon  # avoid div by zero exceptions.
        w *= sr / current_sr
        convergence = True
    except ArpackNoConvergence:  # pragma: no cover
        if seed is None:
            seed = rg.integers(1, 9999)
        else:
            seed = rg.integers(1, seed + 1)  # never stuck at 1
        w = w_init(*shape, seed=seed, **kwargs)

return w

"""Rescale a matrix created with an initializer.

Parameters
----------
w_init : Initializer
    An initializer.
shape : tuple of int
    Shape of the matrix.
input_scaling : float
    Scaling parameter.

Returns
-------
Numpy array or Scipy sparse matrix
    Rescaled matrix.
"""
w = w_init(*shape, **kwargs)
if sparse.issparse(w):
    return w.multiply(input_scaling)
else:
    return np.multiply(w, input_scaling)

m: int,
n: int,
degree: int = 10,
direction: Literal["in", "out"] = "out",
format: str = "coo",
dtype: np.dtype = None,
random_state: Union[None, int, np.random.Generator, np.random.RandomState] = None,
data_rvs=None,

"""Generate a sparse matrix of the given shape with randomly distributed values.
- If `direction=out`, each column has `degree` non-zero values.
- If `direction=in`, each line has `degree` non-zero values.

Parameters
----------
m, n : int
    shape of the matrix
degree : int, optional
    in-degree or out-degree of each node of the corresponding graph of the
    generated matrix:
direction : {"in", "out"}, defaults to "out"
    Specify the direction of the `degree` value. Allowed values:
    - "in": `degree` corresponds to in-degrees
    - "out": `degree` corresponds to out-degrees
dtype : dtype, optional
    type of the returned matrix values.
random_state : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional

    If `seed` is None (or `np.random`), the `numpy.random.RandomState`
    singleton is used.
    If `seed` is an int, a new ``RandomState`` instance is used,
    seeded with `seed`.
    If `seed` is already a ``Generator`` or ``RandomState`` instance then
    that instance is used.
    This random state will be used
    for sampling the sparsity structure, but not necessarily for sampling
    the values of the structurally nonzero entries of the matrix.
data_rvs : callable, optional
    Samples a requested number of random values.
    This function should take a single argument specifying the length
    of the ndarray that it will return. The structurally nonzero entries
    of the sparse random matrix will be taken from the array sampled
    by this function. By default, uniform [0, 1) random values will be
    sampled using the same random state as is used for sampling
    the sparsity structure.

Returns
-------
res : sparse matrix

Notes
-----
Only float types are supported for now.

"""
dtype = np.dtype(dtype)

if data_rvs is None:  # pragma: no cover
    if np.issubdtype(dtype, np.complexfloating):

        def data_rvs(n):
            return random_state.uniform(size=n) + random_state.uniform(size=n) * 1j

    else:
        data_rvs = partial(random_state.uniform, 0.0, 1.0)
mn = m * n

tp = np.intc
if mn > np.iinfo(tp).max:  # pragma: no cover
    tp = np.int64

if mn > np.iinfo(tp).max:  # pragma: no cover
    msg = """\

    raise ValueError(msg % np.iinfo(tp).max)

# each column has `degree` non-zero values
if direction == "out":
    if not 0 <= degree <= m:
        raise ValueError(f"'degree'={degree} must be between 0 and m={m}.")

    i = np.zeros((n * degree), dtype=tp)
    j = np.zeros((n * degree), dtype=tp)
    for column in range(n):
        ind = random_state.choice(m, size=degree, replace=False)
        i[column * degree : (column + 1) * degree] = ind
        j[column * degree : (column + 1) * degree] = column

# each line has `degree` non-zero values
elif direction == "in":
    if not 0 <= degree <= n:
        raise ValueError(f"'degree'={degree} must be between 0 and n={n}.")

    i = np.zeros((m * degree), dtype=tp)
    j = np.zeros((m * degree), dtype=tp)
    for line in range(m):
        ind = random_state.choice(n, size=degree, replace=False)
        i[line * degree : (line + 1) * degree] = line
        j[line * degree : (line + 1) * degree] = ind

else:
    raise ValueError(f'\'direction\'={direction} must either be "out" or "in".')

vals = data_rvs(len(i)).astype(dtype, copy=False)
return sparse.coo_matrix((vals, (i, j)), shape=(m, n)).asformat(format, copy=False)

*shape: int,
dist: str,
connectivity: float = 1.0,
dtype: np.dtype = global_dtype,
sparsity_type: str = "csr",
seed: Union[int, np.random.Generator] = None,
degree: Union[int, None] = None,
direction: Literal["in", "out"] = "out",
**kwargs,

"""Create a random matrix.

Parameters
----------
*shape : int, int, ..., optional
    Shape (row, columns, ...) the matrix.
dist: str
    A distribution name from :py:mod:`scipy.stats` module, such as "norm" or
    "uniform". Parameters like `loc` and `scale` can be passed to the distribution
    functions as keyword arguments to this function. Usual distributions for
    internal weights are :py:class:`scipy.stats.norm` with parameters `loc` and
    `scale` to obtain weights following the standard normal distribution,
    or :py:class:`scipy.stats.uniform` with parameters `loc=-1` and `scale=2`
    to obtain weights uniformly distributed between -1 and 1.
    Can also have the value "custom_bernoulli". In that case, weights will be drawn
    from a Bernoulli discrete random variable alternating between -1 and 1 and
    drawing 1 with a probability `p` (default `p` parameter to 0.5).
connectivity: float, default to 1.0
    Also called density of the sparse matrix. By default, creates dense arrays.
sr : float, optional
    If defined, then will rescale the spectral radius of the matrix to this value.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.
sparsity_type : {"csr", "csc", "dense"}, default to "csr"
    If connectivity is inferior to 1 and shape is only 2-dimensional, then the
    function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
    Else, a Numpy array ("dense") will be used.
seed : optional
    Random generator seed. Default to the global value set with
    :py:func:`reservoirpy.set_seed`.
degree: int, default to None
    If not None, override the `connectivity` argument and corresponds to the number
    of non-zero values along the axis specified by `direction`
direction: {"in", "out"}, default to "out"
    If `degree` is not None, specifies the axis along which the `degree` non-zero
    values are distributed.
    - If `direction` is "in", each line will have `degree` non-zero values. In other
    words, each node of the corresponding graph will have `degree` in-degrees
    - If `direction` is "out", each column will have `degree` non-zero values. In
    other words, each node of the corresponding graph will have `degree` out-degrees
**kwargs : optional
    Arguments for the scipy.stats distribution.

Returns
-------
scipy.sparse array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.
"""

rg = rand_generator(seed)
rvs = _get_rvs(dist, **kwargs, random_state=rg)

if degree is not None:
    if len(shape) != 2:
        raise ValueError(
            f"Matrix shape must have 2 dimensions, got {len(shape)}: {shape}"
        )
    m, n = shape
    matrix = _random_degree(
        m=m,
        n=n,
        degree=degree,
        direction=direction,
        format=sparsity_type,
        dtype=dtype,
        random_state=rg,
        data_rvs=rvs,
    )
else:
    if 0 < connectivity > 1.0:
        raise ValueError("'connectivity' must be >0 and <1.")

    if connectivity >= 1.0 or len(shape) != 2:
        matrix = rvs(size=shape).astype(dtype)
        if connectivity < 1.0:
            matrix[rg.random(shape) > connectivity] = 0.0
    else:
        matrix = sparse.random(
            shape[0],
            shape[1],
            density=connectivity,
            format=sparsity_type,
            random_state=rg,
            data_rvs=rvs,
            dtype=dtype,
        )

# sparse.random may return np.matrix if format="dense".
# Only ndarray are supported though, hence the explicit cast.
if type(matrix) is np.matrix:
    matrix = np.asarray(matrix)

return matrix

*shape: int,
low: float = -1.0,
high: float = 1.0,
connectivity: float = 1.0,
dtype: np.dtype = global_dtype,
sparsity_type: str = "csr",
seed: Union[int, np.random.Generator] = None,
degree: Union[int, None] = None,
direction: Literal["in", "out"] = "out",

"""Create an array with uniformly distributed values.

Parameters
----------
*shape : int, int, ..., optional
    Shape (row, columns, ...) of the array.
low, high : float, float, default to -1, 1
    Boundaries of the uniform distribution.
connectivity: float, default to 1.0
    Also called density of the sparse matrix. By default, creates dense arrays.
sr : float, optional
    If defined, then will rescale the spectral radius of the matrix to this value.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.
sparsity_type : {"csr", "csc", "dense"}, default to "csr"
    If connectivity is inferior to 1 and shape is only 2-dimensional, then the
    function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
    Else, a Numpy array ("dense") will be used.
seed : optional
    Random generator seed. Default to the global value set with
    :py:func:`reservoirpy.set_seed`.
degree: int, default to None
    If not None, override the `connectivity` argument and corresponds to the number
    of non-zero values along the axis specified by `direction`
direction: {"in", "out"}, default to "out"
    If `degree` is not None, specifies the axis along which the `degree` non-zero
    values are distributed.
    - If `direction` is "in", each line will have `degree` non-zero values. In other
    words, each node of the corresponding graph will have `degree` in-degrees
    - If `direction` is "out", each column will have `degree` non-zero values. In
    other words, each node of the corresponding graph will have `degree` out-degrees

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.
"""
if high < low:
    raise ValueError("'high' boundary must be > to 'low' boundary.")
return _random_sparse(
    *shape,
    dist="uniform",
    loc=low,
    scale=high - low,
    connectivity=connectivity,
    degree=degree,
    direction=direction,
    dtype=dtype,
    sparsity_type=sparsity_type,
    seed=seed,
)

*shape: int,
loc: float = 0.0,
scale: float = 1.0,
connectivity: float = 1.0,
dtype: np.dtype = global_dtype,
sparsity_type: str = "csr",
seed: Union[int, np.random.Generator] = None,
degree: Union[int, None] = None,
direction: Literal["in", "out"] = "out",

"""Create an array with values distributed following a Gaussian distribution.

Parameters
----------
*shape : int, int, ..., optional
    Shape (row, columns, ...) of the array.
loc, scale : float, float, default to 0, 1
    Mean and scale of the Gaussian distribution.
connectivity: float, default to 1.0
    Also called density of the sparse matrix. By default, creates dense arrays.
sr : float, optional
    If defined, then will rescale the spectral radius of the matrix to this value.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.
sparsity_type : {"csr", "csc", "dense"}, default to "csr"
    If connectivity is inferior to 1 and shape is only 2-dimensional, then the
    function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
    Else, a Numpy array ("dense") will be used.
seed : optional
    Random generator seed. Default to the global value set with
    :py:func:`reservoirpy.set_seed`.
degree: int, default to None
    If not None, override the `connectivity` argument and corresponds to the number
    of non-zero values along the axis specified by `direction`
direction: {"in", "out"}, default to "out"
    If `degree` is not None, specifies the axis along which the `degree` non-zero
    values are distributed.
    - If `direction` is "in", each line will have `degree` non-zero values. In other
    words, each node of the corresponding graph will have `degree` in-degrees
    - If `direction` is "out", each column will have `degree` non-zero values. In
    other words, each node of the corresponding graph will have `degree` out-degrees

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.
"""
return _random_sparse(
    *shape,
    dist="norm",
    loc=loc,
    scale=scale,
    connectivity=connectivity,
    degree=degree,
    direction=direction,
    dtype=dtype,
    sparsity_type=sparsity_type,
    seed=seed,
)

*shape: int,
p: float = 0.5,
connectivity: float = 1.0,
dtype: np.dtype = global_dtype,
sparsity_type: str = "csr",
seed: Union[int, np.random.Generator] = None,
degree: Union[int, None] = None,
direction: Literal["in", "out"] = "out",

"""Create an array with values equal to either 1 or -1. Probability of success
(to obtain 1) is equal to p.

Parameters
----------
*shape : int, int, ..., optional
    Shape (row, columns, ...) of the array.
p : float, default to 0.5
    Probability of success (to obtain 1).
connectivity: float, default to 1.0
    Also called density of the sparse matrix. By default, creates dense arrays.
sr : float, optional
    If defined, then will rescale the spectral radius of the matrix to this value.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.
sparsity_type : {"csr", "csc", "dense"}, default to "csr"
    If connectivity is inferior to 1 and shape is only 2-dimensional, then the
    function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
    Else, a Numpy array ("dense") will be used.
seed : optional
    Random generator seed. Default to the global value set with
    :py:func:`reservoirpy.set_seed`.
degree: int, default to None
    If not None, override the `connectivity` argument and corresponds to the number
    of non-zero values along the axis specified by `direction`
direction: {"in", "out"}, default to "out"
    If `degree` is not None, specifies the axis along which the `degree` non-zero
    values are distributed.
    - If `direction` is "in", each line will have `degree` non-zero values. In other
    words, each node of the corresponding graph will have `degree` in-degrees
    - If `direction` is "out", each column will have `degree` non-zero values. In
    other words, each node of the corresponding graph will have `degree` out-degrees

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.
"""
if 1 < p < 0:
    raise ValueError("'p' must be <= 1 and >= 0.")
return _random_sparse(
    *shape,
    p=p,
    dist="custom_bernoulli",
    connectivity=connectivity,
    dtype=dtype,
    sparsity_type=sparsity_type,
    seed=seed,
    degree=degree,
    direction=direction,
)

"""Create an array filled with 1.

Parameters
----------
*shape : int, int, ..., optional
    Shape (row, columns, ...) of the array.
sr : float, optional
    If defined, then will rescale the spectral radius of the matrix to this value.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.
"""
return np.ones(shape, dtype=dtype)

"""Create an array filled with 0.

Parameters
----------
*shape : int, int, ..., optional
    Shape (row, columns, ...) of the array.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.

Note
----

`sr` parameter is not available for this initializer. The spectral radius of a null
matrix can not be rescaled.
"""
return np.zeros(shape, dtype=dtype)

N: int,
*args,
sr: float = None,
connectivity: float = 1.0,
dtype: np.dtype = global_dtype,
sparsity_type: str = "csr",
seed: Union[int, np.random.Generator] = None,
degree: Union[int, None] = None,
direction: Literal["in", "out"] = "out",

"""Fast spectral radius (FSI) approach for weights
initialization [1]_ of square matrices.

This method is well suited for computation and rescaling of
very large weights matrices, with a number of neurons typically
above 500-1000.

Parameters
----------
N : int, optional
    Shape :math:`N \\times N` of the array.
    This function only builds square matrices.
connectivity: float, default to 1.0
    Also called density of the sparse matrix. By default, creates dense arrays.
sr : float, optional
    If defined, then will rescale the spectral radius of the matrix to this value.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.
sparsity_type : {"csr", "csc", "dense"}, default to "csr"
    If connectivity is inferior to 1 and shape is only 2-dimensional, then the
    function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
    Else, a Numpy array ("dense") will be used.
seed : optional
    Random generator seed. Default to the global value set with
    :py:func:`reservoirpy.set_seed`.
degree: int, default to None
    If not None, override the `connectivity` argument and corresponds to the number
    of non-zero values along the axis specified by `direction`
direction: {"in", "out"}, default to "out"
    If `degree` is not None, specifies the axis along which the `degree` non-zero
    values are distributed.
    - If `direction` is "in", each line will have `degree` non-zero values. In other
    words, each node of the corresponding graph will have `degree` in-degrees
    - If `direction` is "out", each column will have `degree` non-zero values. In
    other words, each node of the corresponding graph will have `degree` out-degrees

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.

Note
----

This function was designed for initialization of a reservoir's internal weights.
In consequence, it can only produce square matrices. If more than one positional
argument of shape are provided, only the first will be used.

References
-----------

.. [1] C. Gallicchio, A. Micheli, and L. Pedrelli,
        ‘Fast Spectral Radius Initialization for Recurrent
        Neural Networks’, in Recent Advances in Big Data and
        Deep Learning, Cham, 2020, pp. 380–390,
        doi: 10.1007/978-3-030-16841-4_39.
"""
if 0 > connectivity < 1.0:
    raise ValueError("'connectivity' must be >0 and <1.")

if sr is None or connectivity <= 0.0:
    a = 1
else:
    a = -(6 * sr) / (np.sqrt(12) * np.sqrt((connectivity * N)))

return _uniform(
    N,
    N,
    low=np.min((a, -a)),
    high=np.max((a, -a)),
    connectivity=connectivity,
    dtype=dtype,
    sparsity_type=sparsity_type,
    seed=seed,
    degree=degree,
    direction=direction,
)

_fast_spectral_initialization,
autorize_input_scaling=False,
autorize_rescaling=False,

N: int,
*args,
dist="norm",
connectivity=0.1,
dtype=global_dtype,
sparsity_type="csr",
seed=None,
degree: Union[int, None] = None,
direction: Literal["in", "out"] = "out",
**kwargs,

"""Generate the weight matrix that will be used for the internal connections of a
    reservoir.

Warning
-------

This function is deprecated since version v0.3.1 and will be removed in future
versions. Please consider using :py:func:`normal`, :py:func:`uniform` or
:py:func:`random_sparse` instead.

Parameters
----------
N : int, optional
    Shape :math:`N \\times N` of the array.
    This function only builds square matrices.
dist: str, default to "norm"
    A distribution name from :py:mod:`scipy.stats` module, such as "norm" or
    "uniform". Parameters like `loc` and `scale` can be passed to the distribution
    functions as keyword arguments to this function. Usual distributions for
    internal weights are :py:class:`scipy.stats.norm` with parameters `loc` and
    `scale` to obtain weights following the standard normal distribution,
    or :py:class:`scipy.stats.uniform` with parameters `loc=-1` and `scale=2`
    to obtain weights uniformly distributed between -1 and 1.
    Can also have the value "custom_bernoulli". In that case, weights will be drawn
    from a Bernoulli discrete random variable alternating between -1 and 1 and
    drawing 1 with a probability `p` (default `p` parameter to 0.5).
connectivity: float, default to 0.1
    Also called density of the sparse matrix.
sr : float, optional
    If defined, then will rescale the spectral radius of the matrix to this value.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.
sparsity_type : {"csr", "csc", "dense"}, default to "csr"
    If connectivity is inferior to 1 and shape is only 2-dimensional, then the
    function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
    Else, a Numpy array ("dense") will be used.
seed : optional
    Random generator seed. Default to the global value set with
    :py:func:`reservoirpy.set_seed`.
degree: int, default to None
    If not None, override the `connectivity` argument and corresponds to the number
    of non-zero values along the axis specified by `direction`
direction: {"in", "out"}, default to "out"
    If `degree` is not None, specifies the axis along which the `degree` non-zero
    values are distributed.
    - If `direction` is "in", each line will have `degree` non-zero values. In other
    words, each node of the corresponding graph will have `degree` in-degrees
    - If `direction` is "out", each column will have `degree` non-zero values. In
    other words, each node of the corresponding graph will have `degree` out-degrees
**kwargs : optional
    Arguments for the scipy.stats distribution.

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.
"""

warnings.warn(
    "'generate_internal_weights' is deprecated since v0.3.1 and will be removed in "
    "future versions. Consider using 'bernoulli' or 'random_sparse'.",
    DeprecationWarning,
)

return _random_sparse(
    N,
    N,
    connectivity=connectivity,
    dtype=dtype,
    dist=dist,
    sparsity_type=sparsity_type,
    seed=seed,
    degree=degree,
    direction=direction,
    **kwargs,
)

_generate_internal_weights, autorize_input_scaling=False

N,
dim_input,
dist="custom_bernoulli",
connectivity=1.0,
dtype=global_dtype,
sparsity_type="csr",
seed=None,
input_bias=False,
degree: Union[int, None] = None,
direction: Literal["in", "out"] = "out",
**kwargs,

"""Generate input or feedback weights for a reservoir.

Weights are drawn by default from a discrete Bernoulli random variable,
i.e. are always equal to 1 or -1. Then, they can be rescaled to a specific constant
using the `input_scaling` parameter.

Warning
-------

This function is deprecated since version v0.3.1 and will be removed in future
versions. Please consider using :py:func:`bernoulli` or :py:func:`random_sparse`
instead.

Parameters
----------
N: int
    Number of units in the connected reservoir.
dim_input: int
    Dimension of the inputs connected to the reservoir.
dist: str, default to "norm"
    A distribution name from :py:mod:`scipy.stats` module, such as "norm" or
    "uniform". Parameters like `loc` and `scale` can be passed to the distribution
    functions as keyword arguments to this function. Usual distributions for
    internal weights are :py:class:`scipy.stats.norm` with parameters `loc` and
    `scale` to obtain weights following the standard normal distribution,
    or :py:class:`scipy.stats.uniform` with parameters `loc=-1` and `scale=2`
    to obtain weights uniformly distributed between -1 and 1.
    Can also have the value "custom_bernoulli". In that case, weights will be drawn
    from a Bernoulli discrete random variable alternating between -1 and 1 and
    drawing 1 with a probability `p` (default `p` parameter to 0.5).
connectivity: float, default to 0.1
    Also called density of the sparse matrix.
input_scaling: float or array, optional
    If defined, then will rescale the matrix using this coefficient or array
    of coefficients.
input_bias: bool, optional
    'input_bias' parameter is deprecated. Bias should be initialized
    separately from the input matrix.
    If True, will add a row to the matrix to take into
    account a constant bias added to the input.
dtype : numpy.dtype, default to numpy.float64
    A Numpy numerical type.
sparsity_type : {"csr", "csc", "dense"}, default to "csr"
    If connectivity is inferior to 1 and shape is only 2-dimensional, then the
    function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
    Else, a Numpy array ("dense") will be used.
seed : optional
    Random generator seed. Default to the global value set with
    :py:func:`reservoirpy.set_seed`.
degree: int, default to None
    If not None, override the `connectivity` argument and corresponds to the number
    of non-zero values along the axis specified by `direction`
direction: {"in", "out"}, default to "out"
    If `degree` is not None, specifies the axis along which the `degree` non-zero
    values are distributed.
    - If `direction` is "in", each line will have `degree` non-zero values. In other
    words, each node of the corresponding graph will have `degree` in-degrees
    - If `direction` is "out", each column will have `degree` non-zero values. In
    other words, each node of the corresponding graph will have `degree` out-degrees
**kwargs : optional
    Arguments for the scipy.stats distribution.

Returns
-------
Numpy array or callable
    If a shape is given to the initializer, then returns a matrix.
    Else, returns a function partially initialized with the given keyword
    parameters, which can be called with a shape and returns a matrix.
"""
warnings.warn(
    "'generate_input_weights' is deprecated since v0.3.1 and will be removed in "
    "future versions. Consider using 'normal', 'uniform' or 'random_sparse'.",
    DeprecationWarning,
)

if input_bias:
    warnings.warn(
        "'input_bias' parameter is deprecated. Bias should be initialized "
        "separately from the input matrix.",
        DeprecationWarning,
    )

    dim_input += 1

return _random_sparse(
    N,
    dim_input,
    connectivity=connectivity,
    dtype=dtype,
    dist=dist,
    sparsity_type=sparsity_type,
    seed=seed,
    degree=degree,
    direction=direction,
    **kwargs,
)

```
# File: test_esn.txt
```python
esn = ESN(units=100, output_dim=1, lr=0.8, sr=0.4, ridge=1e-5, Win_bias=False)

data = np.ones((1, 10))
res = esn(data)

assert esn.reservoir.W.shape == (100, 100)
assert esn.reservoir.Win.shape == (100, 10)
assert esn.reservoir.lr == 0.8
assert esn.reservoir.units == 100

data = np.ones((10000, 10))
res = esn.run(data)

assert res.shape == (10000, 1)

with pytest.raises(ValueError):
    esn = ESN(units=100, output_dim=1, learning_method="foo")

with pytest.raises(ValueError):
    esn = ESN(units=100, output_dim=1, reservoir_method="foo")

res = Reservoir(100, lr=0.8, sr=0.4, input_bias=False)
read = Ridge(1, ridge=1e-5)

esn = ESN(reservoir=res, readout=read)

data = np.ones((1, 10))
res = esn(data)

assert esn.reservoir.W.shape == (100, 100)
assert esn.reservoir.Win.shape == (100, 10)
assert esn.reservoir.lr == 0.8
assert esn.reservoir.units == 100

data = np.ones((10000, 10))
res = esn.run(data)

assert res.shape == (10000, 1)

res = Reservoir(100, lr=0.8, sr=0.4, input_bias=False)
read = Ridge(1, ridge=1e-5)

esn = ESN(reservoir=res, readout=read)

data = np.ones((2, 10, 10))
out = esn.run(data, return_states="all")

assert out["reservoir"][0].shape == (10, 100)
assert out["readout"][0].shape == (10, 1)

out = esn.run(data, return_states=["reservoir"])

assert out["reservoir"][0].shape == (10, 100)

s_reservoir = esn.state()
assert_equal(s_reservoir, res.state())

s_readout = esn.state(which="readout")
assert_equal(s_readout, read.state())

with pytest.raises(ValueError):
    esn.state(which="foo")

esn = ESN(units=100, output_dim=5, lr=0.8, sr=0.4, ridge=1e-5, feedback=True)

data = np.ones((1, 10))
res = esn(data)

assert esn.reservoir.W.shape == (100, 100)
assert esn.reservoir.Win.shape == (100, 10)
assert esn.readout.Wout.shape == (100, 5)
assert res.shape == (1, 5)
assert esn.reservoir.Wfb is not None
assert esn.reservoir.Wfb.shape == (100, 5)

"""Reproducibility of the ESN node across backends.
Results may vary between OSes and NumPy versions.
"""
seed = 1234
rng = np.random.default_rng(seed=seed)
X = list(rng.normal(0, 1, (10, 100, 10)))
Y = [x @ rng.normal(0, 1, size=(10, 5)) for x in X]

set_seed(seed)
base_Wout = (
    ESN(
        units=100,
        ridge=1e-5,
        feedback=True,
        workers=-1,
        backend="sequential",
    )
    .fit(X, Y)
    .readout.Wout
)

for backend in ("loky", "multiprocessing", "threading", "sequential"):

    set_seed(seed)
    esn = ESN(
        units=100,
        ridge=1e-5,
        feedback=True,
        workers=-1,
        backend=backend,
    ).fit(X, Y)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 10)
    assert esn.readout.Wout.shape == (100, 5)

    assert esn.reservoir.Wfb is not None
    assert esn.reservoir.Wfb.shape == (100, 5)
    assert np.abs(np.mean(esn.readout.Wout - base_Wout)) < 1e-14

"""Reproducibility of the ESN node across backends. Results may
vary between OSes and NumPy versions.
"""
seed = 1000
rng = np.random.default_rng(seed=seed)
X = list(rng.normal(0, 1, (10, 100, 10)))
Y = [x @ rng.normal(0, 1, size=(10, 5)) for x in X]

set_seed(seed)
# no feedback here. XXT and YXT sum orders are not deterministic
# which results in small (float precision) differences across fits
# and leads to error accumulation on run with feedback.
set_seed(seed)
esn = ESN(
    units=100,
    ridge=1e-5,
    workers=1,
    backend="sequential",
).fit(X, Y)

base_y_out = esn.run(X[0])

for backend in ("loky", "multiprocessing", "threading", "sequential"):

    set_seed(seed)
    esn = ESN(
        units=100,
        ridge=1e-5,
        workers=-1,
        backend=backend,
    ).fit(X, Y)

    y_out = esn.run(X[0])
    assert np.abs(np.mean(y_out - base_y_out)) < 1e-14

esn1 = ESN(
    units=100,
    lr=0.8,
    sr=0.4,
    ridge=1e-5,
    feedback=True,
    workers=-1,
    backend="loky",
    name="E1",
)

esn2 = ESN(
    units=100,
    lr=0.8,
    sr=0.4,
    ridge=1e-5,
    feedback=True,
    workers=-1,
    backend="loky",
    name="E2",
)

```
# File: sparse_vs_dense.txt
```python



"ReservoirPy: ", rpy.__version__, '\n'

"NumPy:", np.__version__,





t0 = time.time()

mat = mat_gen.normal(

    units, units, 

    connectivity=connectivity, 

    sparsity_type=sparsity_type,

    **kwargs

)

return time.time() - t0

W = mat_gen.normal(

    units, units, 

    connectivity=connectivity, 

    sparsity_type=sparsity_type,

    **kwargs

)

Win = mat_gen.normal(

    units, 1,

    connectivity=connectivity,

    sparsity_type=sparsity_type,

    **kwargs

)

X = np.random.uniform(size=(timesteps, 1))

reservoir = Reservoir(W=W, Win=Win)

reservoir.initialize(X)

t0 = time.time()

reservoir.run(X)

return time.time() - t0





time_run_type[sparse_type] = time_run(

    units=UNITS,

    connectivity=CONNECTIVITY,

    timesteps=TIMESTEPS,

    sparsity_type=sparse_type,

)

print(sparse_type, time_run_type[sparse_type])





for j, sparse_type in enumerate(rpy_sparsity_types):

    time_run_units[i, j] = time_run(

        units=N,

        connectivity=CONNECTIVITY,

        timesteps=TIMESTEPS,

        sparsity_type=sparse_type,

    )

    print(sparse_type, N, time_run_units[i,j])


plt.plot(Ns, time_run_units[:, i], label=s_type)



for j, sparse_type in enumerate(["csc", "csr"]):

    time_run_connectivity[i, j] = time_run(

        units=UNITS,

        connectivity=C,

        timesteps=TIMESTEPS,

        sparsity_type=sparse_type,

    )

    print(sparse_type, C, time_run_connectivity[i,j])


units=UNITS,

connectivity=1.,

timesteps=TIMESTEPS,

sparsity_type="dense",







for j, sparse_type in enumerate(["csc", "csr"]):

    time_run_connectivity2[i, j] = time_run(

        units=UNITS,

        connectivity=C,

        timesteps=TIMESTEPS,

        sparsity_type=sparse_type,

    )

    print(sparse_type, C, time_run_connectivity2[i,j])


units=UNITS,

connectivity=1.,

timesteps=TIMESTEPS,

sparsity_type="dense",





for j, sparse_type in enumerate(["csc", "csr"]):

    time_run_connectivity3[i, j] = time_run(

        units=UNITS,

        connectivity=C,

        timesteps=TIMESTEPS,

        sparsity_type=sparse_type,

    )

    print(sparse_type, C, time_run_connectivity3[i,j])

units=UNITS,

connectivity=1.,

timesteps=TIMESTEPS,

sparsity_type="dense",







dense_instances = [time_run(units, connectivity=1., sparsity_type="dense", timesteps=timesteps) for _ in range(20)]

print(units, timesteps, np.median(dense_instances), np.var(dense_instances))

dense_time = np.median(dense_instances)

# csc_dense_time = time_run(units, connectivity=.99, sparsity_type="csc", timesteps=timesteps)

# if csc_dense_time > dense_time:

#     print(units, 1., csc_dense_time, dense_time)

#     return 1.



min_ = 0.

max_ = 1.

current_time = np.Infinity

iters = 0

while np.abs(dense_time - current_time) > dense_time/50: # 10% de précision

    connectivity = 0.5 * max_ + 0.5 * min_ # 

    current_time = time_run(units, connectivity=connectivity, sparsity_type="csc", timesteps=timesteps)

    if current_time > dense_time:

        max_ = connectivity

    else:

        min_ = connectivity

    iters += 1



print(units, connectivity, current_time)

return connectivity








```
# File: Edge_of_stability_Ceni_Gallicchio_2023.txt
```python
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

(without noise and feedback) is the following:

+ \mathbf{W}_{in} \cdot \mathbf{u}[t+1])

\mathbf{W} \cdot x[t]

+ \mathbf{W}_{in} \cdot \mathbf{u}[t+1])

# Random orthogonal matrix generation

# We generate a random matrix and we apply a QR factorization

def random_orthogonal(units, seed = None):

    D = uniform(units, units,

        sparsity_type = "dense",

        seed=seed,

    )

    Q, _ = np.linalg.qr(D)

    return Q

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

```
# File: lms.txt
```python
_assemble_wout,
_compute_error,
_initialize_readout,
_prepare_inputs_for_learning,
_split_and_save_wout,
readout_forward,

"""Least Mean Squares learning rule."""
# learning rate is a generator to allow scheduling
dw = -next(alpha) * np.outer(e, r)
return dw

"""Train a readout using LMS learning rule."""
x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias, allow_reshape=True)

error, r = _compute_error(node, x, y)

alpha = node._alpha_gen
dw = _lms(alpha, r, error)
wo = _assemble_wout(node.Wout, node.bias, node.input_bias)
wo = wo + dw.T

_split_and_save_wout(node, wo)

readout: "LMS", x=None, y=None, init_func=None, bias_init=None, bias=None

_initialize_readout(readout, x, y, init_func, bias_init, bias)

"""Single layer of neurons learning connections using Least Mean Squares
algorithm.

The learning rules is well described in [1]_.

:py:attr:`LMS.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
``P``              Matrix :math:`\\mathbf{P}` of RLS rule.
================== =================================================================

:py:attr:`LMS.hypers` **list**

================== =================================================================
``alpha``          Learning rate (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by default).
``input_bias``     If True, learn a bias term (True by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
alpha : float or Python generator or iterable, default to 1e-6
    Learning rate. If an iterable or a generator is provided, the learning rate can
    be changed at each timestep of training. A new learning rate will be drawn from
    the iterable or generator at each timestep.
Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

Examples
--------
>>> x = np.random.normal(size=(100, 3))
>>> noise = np.random.normal(scale=0.01, size=(100, 1))
>>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.

>>> from reservoirpy.nodes import LMS
>>> lms_node = LMS(alpha=1e-1)

>>> lms_node.train(x[:50], y[:50])
>>> print(lms_node.Wout.T, lms_node.bias)
[[ 9.156 -0.967   6.411]] [[11.564]]
>>> lms_node.train(x[50:], y[50:])
>>> print(lms_node.Wout.T, lms_node.bias)
[[ 9.998 -0.202  7.001]] [[12.005]]

References
----------

.. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
        Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
        https://doi.org/10.1016/j.neuron.2009.07.018
"""

def __init__(
    self,
    output_dim=None,
    alpha=1e-6,
    Wout=zeros,
    bias=zeros,
    input_bias=True,
    name=None,
):
    if isinstance(alpha, Number):

        def _alpha_gen():
            while True:
                yield alpha

        alpha_gen = _alpha_gen()
    elif isinstance(alpha, Iterable):
        alpha_gen = alpha
    else:
        raise TypeError(
            "'alpha' parameter should be a float or an iterable yielding floats."
        )

```
# File: concat.txt
```python
axis = concat.axis

if not isinstance(data, np.ndarray):
    if len(data) > 1:
        return np.concatenate(data, axis=axis)
    else:
        return np.asarray(data)
else:
    return data

if x is not None:
    if isinstance(x, np.ndarray):
        concat.set_input_dim(x.shape[1])
        concat.set_output_dim(x.shape[1])
    elif isinstance(x, Sequence):
        result = concat_forward(concat, x)
        concat.set_input_dim(tuple([u.shape[1] for u in x]))
        if result.shape[0] > 1:
            concat.set_output_dim(result.shape)
        else:
            concat.set_output_dim(result.shape[1])

"""Concatenate vector of data along feature axis.

This node is automatically created behind the scene when a node receives the input
of more than one node.

For more information on input concatenation, see
:ref:`/user_guide/advanced_demo.ipynb#Input-to-readout-connections`

:py:attr:`Concat.hypers` **list**

============= ======================================================================
``axis``      Concatenation axis.
============= ======================================================================

Examples
--------

>>> x1 = np.arange(0., 10.).reshape(10, 1)
>>> x2 = np.arange(100., 110.).reshape(10, 1)
>>>
>>> from reservoirpy.nodes import Concat
>>> concat_node = Concat()
>>>
>>> out = concat_node.run((x1, x2))
>>> print(out.T)
[[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.]
[100. 101. 102. 103. 104. 105. 106. 107. 108. 109.]]
>>> print(out.shape)
(10, 2)
"""

```
# File: test_activations.txt
```python
"node",
(Tanh(), Softmax(), Softplus(), Sigmoid(), Identity(), ReLU(), Softmax(beta=2.0)),

```
# File: test_utils.txt
```python
_obj_from_kwargs,
progress,
safe_defaultdict_copy,
verbosity,

v = verbosity()
from reservoirpy.utils import VERBOSITY

assert v == VERBOSITY
verbosity(0)
from reservoirpy.utils import VERBOSITY

assert VERBOSITY == 0

verbosity(0)
a = [1, 2, 3]
it = progress(a)
assert id(it) == id(a)

verbosity(1)
it = progress(a)
assert isinstance(it, tqdm)

a = defaultdict(list)

a["a"].extend([1, 2, 3])
a["b"] = 2

b = safe_defaultdict_copy(a)

assert list(b.values()) == [
    [1, 2, 3],
    [
        2,
    ],
]
assert list(b.keys()) == ["a", "b"]

a = dict()

a["a"] = [1, 2, 3]
a["b"] = 2

b = safe_defaultdict_copy(a)

assert list(b.values()) == [
    [1, 2, 3],
    [
        2,
    ],
]
assert list(b.keys()) == ["a", "b"]

class A:
    def __init__(self, a=0, b=2):
        self.a = a
        self.b = b

```
# File: reservoir_jax.txt
```python



mackey_glass = datasets.mackey_glass(timesteps, tau=17)

# rescale between -1 and 1

mackey_glass = 2 * (mackey_glass - mackey_glass.min()) / (mackey_glass.max() - mackey_glass.min()) - 1

return datasets.to_forecasting(

    mackey_glass, 

    forecast=forecast, 

    test_size=0.2

)

timesteps=2000,

forecast=50,

tau=17,







def __init__(

    self, 

    units, 

    connectivity, 

    input_connectivity, 

    weight_scale, 

    lr, 

    input_scaling,

    input_noise, 

    ridge, 

    seed, 

    input_dim, 

    output_dim,

):

    # Création du modèle

    self.key = jax.random.PRNGKey(seed=seed)

    self.key, W_key, Win_key,  = jax.random.split(key=self.key, num=3)

    # Création du réservoir

    self.W = weight_scale * sparse.random_bcoo(

        key = W_key,

        shape = (units, units),

        dtype = np.float32,

        indices_dtype = int,

        nse = connectivity,

        generator = jax.random.normal

    )

    # Création de Win

    self.Win = input_scaling * sparse.random_bcoo(

        key = Win_key,

        shape = (units, input_dim if not input_noise else input_dim+1),

        dtype = np.float32,

        indices_dtype=int,

        nse = input_connectivity,

        generator = ESN._bernoulli

    )

    # état actuel

    self.x = jax.numpy.zeros((units, 1))

    self.Wout = jax.numpy.zeros((output_dim, units+1))

    self.lr = lr

    self.units = units

    self.ridge = ridge

    self.input_noise = input_noise

# Méthodes fonctionnelles (avec underscore): ne prennent pas d'objet en paramètre (dont self)

# et ne modifient pas d'état.

def _bernoulli(key, shape, dtype):

    """ Bernoulli {-1, 1}, p=0.5

    """

    boolean = jax.random.bernoulli(key=key, p=0.5, shape=shape)

    return 2.0 * jax.numpy.array(boolean, dtype=dtype) - 1.0



def _step_reservoir(x, u, W, Win, lr):

    """ 1 pas de temps du réservoir

    """

    u = u.reshape(-1, 1)

    new_x = lr * jax.numpy.tanh(W @ x + Win @ u) + (1 - lr) * x

    return new_x, new_x[:, 0]



def _run_reservoir(W, Win, lr, x, U):

    """ run du réservoir sur une série temporelle

    """

    step_ = partial(ESN._step_reservoir, W=W, Win=Win, lr=lr)

    new_x, states = jax.lax.scan(step_, x, U)

    return new_x, states



def _ridge_regression(ridge, X, Y):

    """ régression ridge entre X et Y

    """

    XXT = X.T @ X

    YXT = Y.T @ X

    n = XXT.shape[0]

    I_n = jax.numpy.eye(n)

    Wout = jax.scipy.linalg.solve(XXT + ridge*I_n, YXT.T, assume_a="sym")

    return Wout.T

def _fit(W, Win, lr, ridge, x, U, Y):

    """ fait tourner le réservoir, et détermine la matrice Wout

    """

    new_x, X = ESN._run_reservoir(W, Win, lr, x, U)

    Wout = ESN._ridge_regression(ridge, X, Y)

    return new_x, Wout



def _step(x, u, W, Win, Wout, lr):

    """ fait tourner un ESN entraîné sur un pas de temps

    """

    new_x, new_state = ESN._step_reservoir(

        x=x, 

        u=u,

        W=W, 

        Win=Win, 

        lr=lr

    )

    y = Wout @ new_x

    return new_x, y.reshape(-1)



def _run(x, U, W, Win, Wout, lr):

    """ fait tourner un ESN entraîné sur une série temporelle

    """

    step_ = partial(ESN._step, W=W, Win=Win, Wout=Wout, lr=lr)

    new_x, Y = jax.lax.scan(step_, x, U)

    return new_x, Y

# Méthodes non-fonctionnelles, pour l'API

def fit(self, U, Y):

    """ Entraîne un ESN

    """

    self.x, self.Wout = ESN._fit(

        W=self.W, 

        Win=self.Win, 

        lr=self.lr, 

        ridge=self.ridge, 

        x=self.x, 

        U=U, 

        Y=Y,

    )

def run(self, U):

    """ Fait tourner un ESN entraîné sur une série temporelle

    """

    new_x, Y = ESN._run(

        W=self.W, 

        Win=self.Win, 

        lr=self.lr, 

        Wout=self.Wout, 

        x=self.x, 

        U=U,

    )

    return Y

def plot_Ypred(self, U_train, U_test, Y_train, Y_test):

    """ Entraîne un ESN, le fait tourner sur U_test, affiche la RMSE, et plot prédiction vs réel

    """

    if self.input_noise:

        T_train = U_train.shape[0]

        noise_train = jax.random.bernoulli(key=jax.random.key(0), p=0.5, shape=(T_train, 1))

        U_train = jax.numpy.concatenate((U_train, noise_train), axis=1)

        T_test = U_test.shape[0]

        noise_test = jax.random.bernoulli(key=jax.random.key(1), p=0.5, shape=(T_test, 1))

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



units = 500,

connectivity = 0.1,

input_connectivity = 0.1,

weight_scale = 0.134,

lr = .9,

input_scaling = 1.,

ridge = 1e-3,

seed = 2341,

input_dim = 1,

output_dim = 1,

input_noise=False,






```
# File: norm.txt
```python
store = node.store
beta = node.beta

new_store = np.roll(store, -1)
new_store[-1] = x

node.set_param("store", new_store)

sigma = np.std(new_store)

if sigma < 1e-8:
    sigma = 1e-8

x_norm = (x - np.mean(new_store)) / sigma

return relu(tanh(x_norm / beta))

if x is not None:
    node.set_input_dim(x.shape[1])
    node.set_output_dim(x.shape[1])

    window = node.window

    node.set_param("store", np.zeros((window, node.output_dim)))

```
# File: type.txt
```python
Any,
Callable,
Dict,
Iterable,
Iterator,
Optional,
Sequence,
Tuple,
TypeVar,
Union,

from typing_extensions import Protocol

from typing import Protocol

"MappedData",
Iterable[np.ndarray],
np.ndarray,
Dict[str, Iterable[np.ndarray]],
Dict[str, np.ndarray],

"""Node base Protocol class for type checking and interface inheritance."""

name: str
params: Dict[str, Any]
hypers: Dict[str, Any]
is_initialized: bool
input_dim: Shape
output_dim: Shape
is_trained_offline: bool
is_trained_online: bool
is_trainable: bool
fitted: bool

def __call__(self, *args, **kwargs) -> np.ndarray:
    ...

def __rshift__(self, other: Union["NodeType", Sequence["NodeType"]]) -> "NodeType":
    ...

def __rrshift__(self, other: Union["NodeType", Sequence["NodeType"]]) -> "NodeType":
    ...

def __and__(self, other: Union["NodeType", Sequence["NodeType"]]) -> "NodeType":
    ...

def get_param(self, name: str) -> Any:
    ...

def initialize(self, x: MappedData = None, y: MappedData = None):
    ...

def reset(self, to_state: np.ndarray = None) -> "NodeType":
    ...

def with_state(
    self, state=None, stateful=False, reset=False
) -> Iterator["NodeType"]:
    ...

def with_feedback(
    self, feedback=None, stateful=False, reset=False
) -> Iterator["NodeType"]:
    ...

```
# File: test_rls.txt
```python
node = RLS(10)

data = np.ones((1, 100))
res = node(data)

assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)
assert node.alpha == 1e-6

data = np.ones((10000, 100))
res = node.run(data)

assert res.shape == (10000, 10)

node = RLS(10)

x = np.ones((5, 2))
y = np.ones((5, 10))

for x, y in zip(x, y):
    res = node.train(x, y)

assert node.Wout.shape == (2, 10)
assert node.bias.shape == (1, 10)
assert node.alpha == 1e-6

data = np.ones((10000, 2))
res = node.run(data)

assert res.shape == (10000, 10)

node = RLS(10)

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

node = RLS(10)

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

node = RLS(10)

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = node.train(x, y)

assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

data = np.ones((1000, 100))
res = node.run(data)

assert res.shape == (1000, 10)

readout = RLS(10)
reservoir = Reservoir(100)

esn = reservoir >> readout

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = esn.train(x, y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)

data = np.ones((1000, 100))
res = esn.run(data)

assert res.shape == (1000, 10)

readout = RLS(10)
reservoir = Reservoir(100)

esn = reservoir >> readout

reservoir <<= readout

X, Y = np.ones((5, 200, 8)), np.ones((5, 200, 10))
for x, y in zip(X, Y):
    res = esn.train(x, y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)
assert reservoir.Wfb.shape == (100, 10)

data = np.ones((1000, 8))
res = esn.run(data)

assert res.shape == (1000, 10)

readout1 = RLS(10, name="r1")
reservoir1 = Reservoir(100)
readout2 = RLS(3, name="r2")
reservoir2 = Reservoir(100)

esn = reservoir1 >> readout1 >> reservoir2 >> readout2

X, Y = np.ones((200, 5)), {"r1": np.ones((200, 10)), "r2": np.ones((200, 3))}
res = esn.train(X, Y)

assert readout1.Wout.shape == (100, 10)
assert readout1.bias.shape == (1, 10)

assert readout2.Wout.shape == (100, 3)
assert readout2.bias.shape == (1, 3)

assert reservoir1.Win.shape == (100, 5)
assert reservoir2.Win.shape == (100, 10)

data = np.ones((10000, 5))
res = esn.run(data)

assert res.shape == (10000, 3)

readout1 = RLS(1, name="r1")
reservoir1 = Reservoir(100)
readout2 = RLS(1, name="r2")
reservoir2 = Reservoir(100)

reservoir1 <<= readout1
reservoir2 <<= readout2

branch1 = reservoir1 >> readout1
branch2 = reservoir2 >> readout2

model = branch1 & branch2

X = np.ones((200, 5))

res = model.train(X, Y={"r1": readout2, "r2": readout1}, force_teachers=True)

assert readout1.Wout.shape == (100, 1)
assert readout1.bias.shape == (1, 1)

assert readout2.Wout.shape == (100, 1)
assert readout2.bias.shape == (1, 1)

```
# File: random.txt
```python
"""Set random state seed globally.

Parameters
----------
    seed : int
"""
global __SEED
global __global_rg
if type(seed) is int:
    __SEED = seed
    __global_rg = default_rng(__SEED)
    np.random.seed(__SEED)
else:
    raise TypeError(f"Random seed must be an integer, not {type(seed)}")

if seed is None:
    return __global_rg
# provided to support legacy RandomState generator
# of Numpy. It is not the best thing to do however
# and recommend the user to keep using integer seeds
# and proper Numpy Generator API.
if isinstance(seed, RandomState):
    mt19937 = MT19937()
    mt19937.state = seed.get_state()
    return Generator(mt19937)

if isinstance(seed, Generator):
    return seed
else:
    return default_rng(seed)

"""Generate noise from a given distribution, and apply a gain factor.

Parameters
----------
    rng : numpy.random.Generator
        A random number generator.
    dist : str, default to 'normal'
        A random variable distribution.
    shape : int or tuple of ints, default to 1
        Shape of the noise vector.
    gain : float, default to 1.0
        Gain factor applied to noise.
    **kwargs
        Any other parameters of the noise distribution.

Returns
-------
    np.ndarray
        A noise vector.

```
# File: observables.txt
```python
spectral_radius
mse
rmse
nrmse
rsquare

from typing_extensions import Literal

from typing import Literal

y_true_array = np.asarray(y_true)
y_pred_array = np.asarray(y_pred)

if not y_true_array.shape == y_pred_array.shape:
    raise ValueError(
        f"Shape mismatch between y_true and y_pred: "
        "{y_true_array.shape} != {y_pred_array.shape}"
    )

return y_true_array, y_pred_array

"""Compute the spectral radius of a matrix `W`.

Spectral radius is defined as the maximum absolute
eigenvalue of `W`.

Parameters
----------
W : array-like (sparse or dense) of shape (N, N)
    Matrix from which the spectral radius will
    be computed.

maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed.
    By default, is equal to `W.shape[0] * 20`.
    See `Scipy documentation <https://docs.scipy.org/
    doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_
    for more informations.

Returns
-------
float
    Spectral radius of `W`.

Raises
------
ArpackNoConvergence
    When computing spectral radius on large
    sparse matrices, it is possible that the
    Fortran ARPACK algorithm used to compute
    eigenvalues don't converge towards precise
    values. To avoid this problem, set the `maxiter`
    parameter to an higher value. Be warned that
    this may drastically increase the computation
    time.

Examples
--------
>>> from reservoirpy.observables import spectral_radius
>>> from reservoirpy.mat_gen import normal
>>> W = normal(1000, 1000, degree=8)
>>> print(spectral_radius(W))
2.8758915077733564
"""
if issparse(W):
    if maxiter is None:
        maxiter = W.shape[0] * 20

    return max(
        abs(
            eigs(
                W,
                k=1,
                which="LM",
                maxiter=maxiter,
                return_eigenvectors=False,
                v0=np.ones(W.shape[0], W.dtype),
            )
        )
    )

return max(abs(linalg.eig(W)[0]))

"""Mean squared error metric:

.. math::

    \\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}

Parameters
----------
y_true : array-like of shape (N, features)
    Ground truth values.
y_pred : array-like of shape (N, features)
    Predicted values.

Returns
-------
float
    Mean squared error.

Examples
--------
>>> from reservoirpy.nodes import ESN
>>> from reservoirpy.datasets import mackey_glass, to_forecasting
>>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
>>> y_pred = ESN(units=100, sr=1).fit(x_train, y_train).run(x_test)

>>> from reservoirpy.observables import mse
>>> print(mse(y_true=y_test, y_pred=y_pred))
0.03962918253990291
"""
y_true_array, y_pred_array = _check_arrays(y_true, y_pred)
return float(np.mean((y_true_array - y_pred_array) ** 2))

"""Root mean squared error metric:

.. math::

    \\sqrt{\\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}}

Parameters
----------
y_true : array-like of shape (N, features)
    Ground truth values.
y_pred : array-like of shape (N, features)
    Predicted values.

Returns
-------
float
    Root mean squared error.

Examples
--------
>>> from reservoirpy.nodes import Reservoir, Ridge
>>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

>>> from reservoirpy.datasets import mackey_glass, to_forecasting
>>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
>>> y_pred = model.fit(x_train, y_train).run(x_test)

>>> from reservoirpy.observables import rmse
>>> print(rmse(y_true=y_test, y_pred=y_pred))
0.00034475744480521534
"""
return np.sqrt(mse(y_true, y_pred))

y_true: np.ndarray,
y_pred: np.ndarray,
norm: Literal["minmax", "var", "mean", "q1q3"] = "minmax",
norm_value: float = None,

"""Normalized mean squared error metric:

.. math::

    \\frac{1}{\\lambda} * \\sqrt{\\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}}

where :math:`\\lambda` may be:
    - :math:`\\max y - \\min y` (Peak-to-peak amplitude) if ``norm="minmax"``;
    - :math:`\\mathrm{Var}(y)` (variance over time) if ``norm="var"``;
    - :math:`\\mathbb{E}[y]` (mean over time) if ``norm="mean"``;
    - :math:`Q_{3}(y) - Q_{1}(y)` (quartiles) if ``norm="q1q3"``;
    - or any value passed to ``norm_value``.

Parameters
----------
y_true : array-like of shape (N, features)
    Ground truth values.
y_pred : array-like of shape (N, features)
    Predicted values.
norm : {"minmax", "var", "mean", "q1q3"}, default to "minmax"
    Normalization method.
norm_value : float, optional
    A normalization factor. If set, will override the ``norm`` parameter.

Returns
-------
float
    Normalized mean squared error.

Examples
--------
>>> from reservoirpy.nodes import Reservoir, Ridge
>>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

>>> from reservoirpy.datasets import mackey_glass, to_forecasting
>>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
>>> y_pred = model.fit(x_train, y_train).run(x_test)

>>> from reservoirpy.observables import nrmse
>>> print(nrmse(y_true=y_test, y_pred=y_pred, norm="var"))
0.007854318015438394
"""
error = rmse(y_true, y_pred)
if norm_value is not None:
    return error / norm_value

else:
    norms = {
        "minmax": lambda y: y.ptp(),
        "var": lambda y: y.var(),
        "mean": lambda y: y.mean(),
        "q1q3": lambda y: np.quantile(y, 0.75) - np.quantile(y, 0.25),
    }

    if norms.get(norm) is None:
        raise ValueError(
            f"Unknown normalization method. "
            f"Available methods are {list(norms.keys())}."
        )
    else:
        return error / norms[norm](np.asarray(y_true))

"""Coefficient of determination :math:`R^2`:

.. math::

    1 - \\frac{\\sum^{N-1}_{i=0} (y - \\hat{y})^2}
    {\\sum^{N-1}_{i=0} (y - \\bar{y})^2}

where :math:`\\bar{y}` is the mean value of ground truth.

Parameters
----------
y_true : array-like of shape (N, features)
    Ground truth values.
y_pred : array-like of shape (N, features)
    Predicted values.

Returns
-------
float
    Coefficient of determination.

Examples
--------
>>> from reservoirpy.nodes import Reservoir, Ridge
>>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

>>> from reservoirpy.datasets import mackey_glass, to_forecasting
>>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
>>> y_pred = model.fit(x_train, y_train).run(x_test)

>>> from reservoirpy.observables import rsquare
>>> print(rsquare(y_true=y_test, y_pred=y_pred))
0.9999972921653904
"""
y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

```
# File: test_lms.txt
```python
node = LMS(10)

data = np.ones((1, 100))
res = node(data)

assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)
assert node.alpha == 1e-6

data = np.ones((1000, 100))
res = node.run(data)

assert res.shape == (1000, 10)

node = LMS(10)

x = np.ones((5, 2))
y = np.ones((5, 10))

for x, y in zip(x, y):
    res = node.train(x, y)

assert node.Wout.shape == (2, 10)
assert node.bias.shape == (1, 10)
assert node.alpha == 1e-6

data = np.ones((1000, 2))
res = node.run(data)

assert res.shape == (1000, 10)

node = LMS(10)

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)

node = LMS(10)

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)

node = LMS(10)

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = node.train(x, y)

assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)

data = np.ones((1000, 100))
res = node.run(data)

assert res.shape == (1000, 10)

readout = LMS(10)
reservoir = Reservoir(100)

esn = reservoir >> readout

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = esn.train(x, y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)

data = np.ones((1000, 100))
res = esn.run(data)

assert res.shape == (1000, 10)

readout = LMS(10)
reservoir = Reservoir(100)

esn = reservoir >> readout

reservoir <<= readout

X, Y = np.ones((5, 200, 8)), np.ones((5, 200, 10))
for x, y in zip(X, Y):
    res = esn.train(x, y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)
assert reservoir.Wfb.shape == (100, 10)

data = np.ones((1000, 8))
res = esn.run(data)

assert res.shape == (1000, 10)

readout1 = LMS(10, name="r1")
reservoir1 = Reservoir(100)
readout2 = LMS(3, name="r2")
reservoir2 = Reservoir(100)

esn = reservoir1 >> readout1 >> reservoir2 >> readout2

X, Y = np.ones((200, 5)), {"r1": np.ones((200, 10)), "r2": np.ones((200, 3))}
res = esn.train(X, Y)

assert readout1.Wout.shape == (100, 10)
assert readout1.bias.shape == (1, 10)

assert readout2.Wout.shape == (100, 3)
assert readout2.bias.shape == (1, 3)

assert reservoir1.Win.shape == (100, 5)
assert reservoir2.Win.shape == (100, 10)

data = np.ones((1000, 5))
res = esn.run(data)

assert res.shape == (1000, 3)

readout1 = LMS(1, name="r1")
reservoir1 = Reservoir(100)
readout2 = LMS(1, name="r2")
reservoir2 = Reservoir(100)

reservoir1 <<= readout1
reservoir2 <<= readout2

branch1 = reservoir1 >> readout1
branch2 = reservoir2 >> readout2

model = branch1 & branch2

X = np.ones((200, 5))

res = model.train(X, Y={"r1": readout2, "r2": readout1}, force_teachers=True)

assert readout1.Wout.shape == (100, 1)
assert readout1.bias.shape == (1, 1)

assert readout2.Wout.shape == (100, 1)
assert readout2.bias.shape == (1, 1)

```
# File: test_random.txt
```python
set_seed(45)
from reservoirpy.utils.random import __SEED

assert __SEED == 45

with pytest.raises(TypeError):
    set_seed("foo")

gen1 = np.random.RandomState(123)
gen2 = rand_generator(gen1)

assert isinstance(gen2, np.random.Generator)

gen1 = rand_generator(123)
gen2 = np.random.default_rng(123)

assert gen1.integers(1000) == gen2.integers(1000)

rng = np.random.default_rng(123)

a = noise(rng, gain=0.0)
assert_equal(a, np.zeros((1,)))

rng = np.random.default_rng(123)

a = noise(rng, dist="uniform", gain=2.0)
b = 2.0 * np.random.default_rng(123).uniform()

assert_equal(a, b)

a = noise(rng, dist="uniform", gain=2.0)
b = noise(rng, dist="uniform", gain=2.0)

```
# File: test_intrinsic_plasticity.txt
```python
res = IPReservoir(100, input_dim=5)

res.initialize()

assert res.W.shape == (100, 100)
assert res.Win.shape == (100, 5)
assert_allclose(res.a, np.ones((100, 1)))
assert_allclose(res.b, np.zeros((100, 1)))

res = IPReservoir(100)
x = np.ones((10, 5))

out = res.run(x)

assert out.shape == (10, 100)
assert res.W.shape == (100, 100)
assert res.Win.shape == (100, 5)
assert_allclose(res.a, np.ones((100, 1)))
assert_allclose(res.b, np.zeros((100, 1)))

with pytest.raises(ValueError):
    res = IPReservoir(100, activation="identity")

x = np.random.normal(size=(100, 5))
X = [x[:10], x[:20]]

res = IPReservoir(100, activation="tanh", epochs=2)

res.fit(x)
res.fit(X)

assert res.a.shape == (100, 1)
assert res.b.shape == (100, 1)

res = IPReservoir(100, activation="sigmoid", epochs=1, mu=0.1)

res.fit(x)
res.fit(X)

assert res.a.shape == (100, 1)
assert res.b.shape == (100, 1)

res.fit(x, warmup=10)
res.fit(X, warmup=5)

assert res.a.shape == (100, 1)
assert res.b.shape == (100, 1)

with pytest.raises(ValueError):
    res.fit(X, warmup=10)

x = np.random.normal(size=(100, 5))
y = np.random.normal(size=(100, 2))
X = [x[:10], x[:20]]
Y = [y[:10], y[:20]]

res = IPReservoir(100, activation="tanh", epochs=2, seed=1234)
readout = Ridge(ridge=1)

model = res >> readout

model.fit(X, Y)

res2 = IPReservoir(100, activation="tanh", epochs=2, seed=1234)
res2.fit(X)

```
# File: randomchoice.txt
```python
choice = node.choice
return x[:, choice.astype(int)]

if x is not None:
    node.set_input_dim(x.shape[1])
    node.set_output_dim(node.n)

    choice = rand_generator(node.seed).choice(
        np.arange(x.shape[1]), node.n, replace=False
    )

    node.set_param("choice", choice)

```
# File: esn.txt
```python
# the 'loky' and 'multiprocessing' backends already deep-copies the ESN. See
# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
_esn = deepcopy(esn)
_esn.reservoir.reset()

seq_len = len(x[list(x)[0]])
states = np.zeros((seq_len, esn.reservoir.output_dim))

for i, (x, forced_feedback, _) in enumerate(dispatch(x, y, shift_fb=True)):
    with _esn.readout.with_feedback(forced_feedback[esn.readout.name]):
        states[i, :] = call(_esn.reservoir, x[esn.reservoir.name])

esn.readout.partial_fit(states, y[esn.readout.name], warmup=warmup, lock=lock)

esn, idx, x, forced_fb, return_states, from_state, stateful, reset, shift_fb

_esn = deepcopy(esn)
X = {_esn.reservoir.name: x[esn.reservoir.name]}

states = _allocate_returned_states(_esn, X, return_states)

with _esn.with_state(from_state, stateful=stateful, reset=reset):
    for i, (x_step, forced_feedback, _) in enumerate(
        dispatch(X, forced_fb, shift_fb=shift_fb)
    ):
        _esn._load_proxys()
        with _esn.readout.with_feedback(forced_feedback):
            state = _esn._call(x_step, return_states=return_states)

        if is_mapping(state):
            for name, value in state.items():
                states[name][i, :] = value
        else:
            states["readout"][i, :] = state

_esn._clean_proxys()
return idx, states

"""Create empty placeholders for model outputs."""
seq_len = inputs[list(inputs.keys())[0]].shape[0]
vulgar_names = {"reservoir": model.reservoir, "readout": model.readout}

# pre-allocate states
if return_states == "all":
    states = {
        name: np.zeros((seq_len, n.output_dim)) for name, n in vulgar_names.items()
    }
elif isinstance(return_states, (list, tuple)):
    states = {
        name: np.zeros((seq_len, n.output_dim))
        for name, n in {name: vulgar_names[name] for name in return_states}.items()
    }
else:
    states = {"readout": np.zeros((seq_len, model.readout.output_dim))}

return states

"""Maintain input order (even with parallelization on)"""
states = sorted(states, key=lambda s: s[0])
states = {n: [s[1][n] for s in states] for n in states[0][1].keys()}

for n, s in states.items():
    if len(s) == 1:
        states[n] = s[0]

if len(states) == 1 and return_states is None:
    states = states["readout"]

return states

"""Echo State Networks as a Node, with parallelization of state update.

This Node is provided as a wrapper for reservoir and readout nodes. Execution
is distributed over several workers when:

- the ``workers`` parameters is equal to `n>1` (using `n` workers) or
    `n<=-1` (using `max_available_workers - n` workers)
- Several independent sequences of inputs are fed to the model at runtime.

When parallelization is enabled, internal states of the reservoir will be reset
to 0 at the beginning of every independent sequence of inputs.

Note
----
    This node can not be connected to other nodes. It is only provided as a
    convenience Node to speed up processing of large datasets with "vanilla"
    Echo State Networks.

:py:attr:`ESN.params` **list**:

================== =================================================================
``reservoir``      A :py:class:`~reservoirpy.nodes.Reservoir` or a :py:class:`~reservoirpy.nodes.NVAR` instance.
``readout``        A :py:class:`~reservoirpy.nodes.Ridge` instance.
================== =================================================================

:py:attr:`ESN.hypers` **list**:

==================== ===============================================================
``workers``          Number of workers for parallelization (1 by default).
``backend``          :py:mod:`joblib` backend to use for parallelization  (``loky`` by default,).
``reservoir_method`` Type of reservoir, may be "reservoir" or "nvar" ("reservoir" by default).
``learning_method``  Type of readout, by default "ridge".
``feedback``         Is readout connected to reservoir through feedback (False by default).
==================== ===============================================================

Parameters
----------
reservoir_method : {"reservoir", "nvar"}, default to "reservoir"
    Type of reservoir, either a :py:class:`~reservoirpy.nodes.Reservoir` or
    a :py:class:`~reservoirpy.nodes.NVAR`.
learning_method : {"ridge"}, default to "ridge"
    Type of readout. The only method supporting parallelization for now is the
    :py:class:`~reservoirpy.nodes.Ridge` readout.
reservoir : Node, optional
    A Node instance to use as a reservoir,
    such as a :py:class:`~reservoirpy.nodes.Reservoir` node.
readout : Node, optional
    A Node instance to use as a readout,
    such as a :py:class:`~reservoirpy.nodes.Ridge` node
    (only this one is supported).
feedback : bool, default to False
    If True, the reservoir is connected to the readout through
    a feedback connection.
Win_bias : bool, default to True
    If True, add an input bias to the reservoir.
Wout_bias : bool, default to True
    If True, add a bias term to the reservoir states entering the readout.
workers : int, default to 1
    Number of workers used for parallelization. If set to -1, all available workers
    (threads or processes) are used.
backend : a :py:mod:`joblib` backend, default to "loky"
    A parallelization backend.
name : str, optional
    Node name.

See Also
--------
Reservoir
Ridge
NVAR

Example
-------
>>> from reservoirpy.nodes import Reservoir, Ridge, ESN
>>> reservoir, readout = Reservoir(100, sr=0.9), Ridge(ridge=1e-6)
>>> model = ESN(reservoir=reservoir, readout=readout, workers=-1)
"""

def __init__(
    self,
    reservoir_method="reservoir",
    learning_method="ridge",
    reservoir: _Node = None,
    readout: _Node = None,
    feedback=False,
    Win_bias=True,
    Wout_bias=True,
    workers=1,
    backend=None,
    name=None,
    use_raw_inputs=False,
    **kwargs,
):

    msg = "'{}' is not a valid method. Available methods for {} are {}."

    if reservoir is None:
        if reservoir_method not in _RES_METHODS:
            raise ValueError(
                msg.format(reservoir_method, "reservoir", list(_RES_METHODS.keys()))
            )
        else:
            klas = _RES_METHODS[reservoir_method]
            kwargs["input_bias"] = Win_bias
            reservoir = _obj_from_kwargs(klas, kwargs)

    if readout is None:
        if learning_method not in _LEARNING_METHODS:
            raise ValueError(
                msg.format(
                    learning_method, "readout", list(_LEARNING_METHODS.keys())
                )
            )
        else:
            klas = _LEARNING_METHODS[learning_method]
            kwargs["input_bias"] = Wout_bias
            readout = _obj_from_kwargs(klas, kwargs)

    if feedback:
        reservoir <<= readout

    if use_raw_inputs:
        source = Input()
        super(ESN, self).__init__(
            nodes=[reservoir, readout, source],
            edges=[(source, reservoir), (reservoir, readout), (source, readout)],
            name=name,
        )
    else:
        super(ESN, self).__init__(
            nodes=[reservoir, readout], edges=[(reservoir, readout)], name=name
        )

    self._hypers.update(
        {
            "workers": workers,
            "backend": backend,
            "reservoir_method": reservoir_method,
            "learning_method": learning_method,
            "feedback": feedback,
        }
    )

    self._params.update({"reservoir": reservoir, "readout": readout})

    self._trainable = True
    self._is_fb_initialized = False

@property
def is_trained_offline(self) -> bool:
    return True

@property
def is_trained_online(self) -> bool:
    return False

@property
def is_fb_initialized(self):
    return self._is_fb_initialized

@property
def has_feedback(self):
    """Always returns False, ESNs are not supposed to receive external
    feedback. Feedback between reservoir and readout must be defined
    at ESN creation."""
    return False

def _call(self, x=None, return_states=None, *args, **kwargs):

    data = x[self.reservoir.name]

    state = call(self.reservoir, data)
    call(self.readout, state)

    state = {}
    if return_states == "all":
        for node in ["reservoir", "readout"]:
            state[node] = getattr(self, node).state()
    elif isinstance(return_states, (list, tuple)):
        for name in return_states:
            state[name] = getattr(self, name).state()
    else:
        state = self.readout.state()

    return state

def state(self, which="reservoir"):
    if which == "reservoir":
        return self.reservoir.state()
    elif which == "readout":
        return self.readout.state()
    else:
        raise ValueError(
            f"'which' parameter of {self.name} "
            f"'state' function must be "
            f"one of 'reservoir' or 'readout'."
        )

def run(
    self,
    X=None,
    forced_feedbacks=None,
    from_state=None,
    stateful=True,
    reset=False,
    shift_fb=True,
    return_states=None,
):

    X, forced_feedbacks = to_data_mapping(self, X, forced_feedbacks)

    self._initialize_on_sequence(X[0], forced_feedbacks[0])

    backend = get_joblib_backend(workers=self.workers, backend=self.backend)

    seq = progress(X, f"Running {self.name}")

    with self.with_state(from_state, reset=reset, stateful=stateful):
        with Parallel(n_jobs=self.workers, backend=backend) as parallel:
            states = parallel(
                delayed(_run_fn)(
                    self,
                    idx,
                    x,
                    y,
                    return_states,
                    from_state,
                    stateful,
                    reset,
                    shift_fb,
                )
                for idx, (x, y) in enumerate(zip(seq, forced_feedbacks))
            )

    return _sort_and_unpack(states, return_states=return_states)

def fit(
    self, X=None, Y=None, warmup=0, from_state=None, stateful=True, reset=False
):

    X, Y = to_data_mapping(self, X, Y)
    self._initialize_on_sequence(X[0], Y[0])

    self.initialize_buffers()

    if (self.workers > 1 or self.workers < 0) and self.backend != "sequential":
        lock = Manager().Lock()
    else:
        lock = None

    backend = get_joblib_backend(workers=self.workers, backend=self.backend)

    seq = progress(X, f"Running {self.name}")
    with self.with_state(from_state, reset=reset, stateful=stateful):
        with Parallel(n_jobs=self.workers, backend=backend) as parallel:
            parallel(
                delayed(_run_partial_fit_fn)(self, x, y, lock, warmup)
                for x, y in zip(seq, Y)
            )

        if verbosity():  # pragma: no cover
            print(f"Fitting node {self.name}...")

        self.readout.fit()

```
# File: rls.txt
```python
_assemble_wout,
_compute_error,
_initialize_readout,
_prepare_inputs_for_learning,
_split_and_save_wout,
readout_forward,

"""Recursive Least Squares learning rule."""
k = np.dot(P, r)
rPr = np.dot(r.T, k).squeeze()
c = float(1.0 / (1.0 + rPr))
P = P - c * np.outer(k, k)

dw = -c * np.outer(e, k)

return dw, P

"""Train a readout using RLS learning rule."""
x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias, allow_reshape=True)

error, r = _compute_error(node, x, y)

P = node.P
dw, P = _rls(P, r, error)
wo = _assemble_wout(node.Wout, node.bias, node.input_bias)
wo = wo + dw.T

_split_and_save_wout(node, wo)

node.set_param("P", P)

readout: "RLS", x=None, y=None, init_func=None, bias_init=None, bias=None

_initialize_readout(readout, x, y, init_func, bias_init, bias)

if x is not None:
    input_dim, alpha = readout.input_dim, readout.alpha

    if readout.input_bias:
        input_dim += 1

    P = np.eye(input_dim) / alpha

    readout.set_param("P", P)

"""Single layer of neurons learning connections using Recursive Least Squares
algorithm.

The learning rules is well described in [1]_.

:py:attr:`RLS.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
``P``              Matrix :math:`\\mathbf{P}` of RLS rule.
================== =================================================================

:py:attr:`RLS.hypers` **list**

================== =================================================================
``alpha``          Diagonal value of matrix P (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by default).
``input_bias``     If True, learn a bias term (True by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
alpha : float or Python generator or iterable, default to 1e-6
    Diagonal value of matrix P.
Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

References
----------

.. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
        Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
        https://doi.org/10.1016/j.neuron.2009.07.018

Examples
--------
>>> x = np.random.normal(size=(100, 3))
>>> noise = np.random.normal(scale=0.1, size=(100, 1))
>>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.

>>> from reservoirpy.nodes import RLS
>>> rls_node = RLS(alpha=1e-1)

>>> rls_node.train(x[:5], y[:5])
>>> print(rls_node.Wout.T, rls_node.bias)
[[ 9.90731641 -0.06884784  6.87944632]] [[12.07802068]]
>>> rls_node.train(x[5:], y[5:])
>>> print(rls_node.Wout.T, rls_node.bias)
[[ 9.99223366 -0.20499636  6.98924066]] [[12.01128622]]
"""

```
# File: test_reservoir.txt
```python
node = Reservoir(100, lr=0.8, input_bias=False)

data = np.ones((1, 10))
res = node(data)

assert node.W.shape == (100, 100)
assert node.Win.shape == (100, 10)
assert node.lr == 0.8
assert node.units == 100

data = np.ones((10000, 10))
res = node.run(data)

assert res.shape == (10000, 100)

with pytest.raises(ValueError):
    Reservoir()

with pytest.raises(ValueError):
    Reservoir(100, equation="foo")

res = Reservoir(100, activation="relu", fb_activation="relu")
assert id(res.activation) == id(relu)
assert id(res.fb_activation) == id(relu)

lr = np.ones((100,)) * 0.5
input_scaling = np.ones((10,)) * 0.8
node = Reservoir(100, lr=lr, input_scaling=input_scaling)

data = np.ones((2, 10))
res = node.run(data)

assert node.W.shape == (100, 100)
assert node.Win.shape == (100, 10)
assert_array_equal(node.lr, np.ones(100) * 0.5)
assert_array_equal(node.input_scaling, np.ones(10) * 0.8)

Win = np.ones((100, 10))

node = Reservoir(100, lr=0.8, Win=Win, input_bias=False)

data = np.ones((1, 10))
res = node(data)

assert node.W.shape == (100, 100)
assert_array_equal(node.Win, Win)
assert node.lr == 0.8
assert node.units == 100

data = np.ones((10000, 10))
res = node.run(data)

assert res.shape == (10000, 100)

Win = np.ones((100, 11))

node = Reservoir(100, lr=0.8, Win=Win, input_bias=True)

data = np.ones((1, 10))
res = node(data)

assert node.W.shape == (100, 100)
assert_array_equal(np.c_[node.bias, node.Win], Win)
assert node.lr == 0.8
assert node.units == 100

data = np.ones((10000, 10))
res = node.run(data)

assert res.shape == (10000, 100)

# Shape override (matrix.shape > units parameter)
data = np.ones((1, 10))
W = np.ones((10, 10))
res = Reservoir(100, W=W)
_ = res(data)
assert res.units == 10
assert res.output_dim == 10

with pytest.raises(ValueError):  # Bad matrix shape
    W = np.ones((10, 11))
    res = Reservoir(W=W)
    res(data)

with pytest.raises(ValueError):  # Bad matrix format
    res = Reservoir(100, W=1.0)
    res(data)

with pytest.raises(ValueError):  # Bias in Win but no bias accepted
    res = Reservoir(100, Win=np.ones((100, 11)), input_bias=False)
    res(data)

with pytest.raises(ValueError):  # Bad Win shape
    res = Reservoir(100, Win=np.ones((100, 20)), input_bias=True)
    res(data)

with pytest.raises(ValueError):  # Bad Win shape
    res = Reservoir(100, Win=np.ones((101, 10)), input_bias=True)
    res(data)

with pytest.raises(ValueError):  # Bad matrix format
    res = Reservoir(100, Win=1.0)
    res(data)

node = Reservoir(100, lr=0.8, input_bias=False)

data = np.ones((1, 10))
res = node(data)

assert node.W.shape == (100, 100)
assert node.Win.shape == (100, 10)
assert node.bias.shape == (100, 1)
assert node.Wfb is None
assert_array_equal(node.bias, np.zeros((100, 1)))
assert node.lr == 0.8
assert node.units == 100

node = Reservoir(100, lr=0.8, input_bias=True)

data = np.ones((1, 10))
res = node(data)

assert node.bias.shape == (100, 1)

bias = np.ones((100, 1))
node = Reservoir(100, bias=bias)
res = node(data)

assert_array_equal(node.bias, bias)

bias = np.ones((100,))
node = Reservoir(100, bias=bias)
res = node(data)

assert_array_equal(node.bias, bias)

with pytest.raises(ValueError):
    bias = np.ones((101, 1))
    node = Reservoir(100, bias=bias)
    node(data)

with pytest.raises(ValueError):
    bias = np.ones((101, 2))
    node = Reservoir(100, bias=bias)
    node(data)

with pytest.raises(ValueError):
    node = Reservoir(100, bias=1.0)
    node(data)

x = np.ones((10, 5))

res = Reservoir(100, equation="internal")
out = res.run(x)
assert out.shape == (10, 100)

res = Reservoir(100, equation="external")
out = res.run(x)
assert out.shape == (10, 100)

node1 = Reservoir(100, lr=0.8, input_bias=False)
node2 = Reservoir(50, lr=1.0, input_bias=False)

data = np.ones((1, 10))
res = (node1 >> node2)(data)

assert node1.W.shape == (100, 100)
assert node1.Win.shape == (100, 10)
assert node2.W.shape == (50, 50)
assert node2.Win.shape == (50, 100)

assert res.shape == (1, 50)

node1 = Reservoir(100, lr=0.8, input_bias=False)
node2 = Reservoir(50, lr=1.0, input_bias=False)

node1 <<= node2

data = np.ones((1, 10))
res = (node1 >> node2)(data)

assert node1.W.shape == (100, 100)
assert node1.Win.shape == (100, 10)
assert node2.W.shape == (50, 50)
assert node2.Win.shape == (50, 100)

assert res.shape == (1, 50)

assert node1.Wfb is not None
assert node1.Wfb.shape == (100, 50)

with pytest.raises(ValueError):
    Wfb = np.ones((100, 51))
    node1 = Reservoir(100, lr=0.8, Wfb=Wfb)
    node2 = Reservoir(50, lr=1.0)
    node1 <<= node2
    data = np.ones((1, 10))
    res = (node1 >> node2)(data)

with pytest.raises(ValueError):
    Wfb = np.ones((101, 50))
    node1 = Reservoir(100, lr=0.8, Wfb=Wfb)
    node2 = Reservoir(50, lr=1.0)
    node1 <<= node2
    data = np.ones((1, 10))
    res = (node1 >> node2)(data)

with pytest.raises(ValueError):
    node1 = Reservoir(100, lr=0.8, Wfb=1.0)
    node2 = Reservoir(50, lr=1.0)
    node1 <<= node2
    data = np.ones((1, 10))
    res = (node1 >> node2)(data)

node1 = Reservoir(100, seed=123, noise_rc=0.1, noise_in=0.5)
node2 = Reservoir(100, seed=123, noise_rc=0.1, noise_in=0.5)

data = np.ones((10, 10))

assert_array_equal(node1.run(data), node2.run(data))

node1 = Reservoir(
    100,
    seed=123,
    noise_rc=0.1,
    noise_in=0.5,
    noise_type="uniform",
    noise_kwargs={"low": -1, "high": 0.5},
)
node2 = Reservoir(
    100,
    seed=123,
    noise_rc=0.1,
    noise_in=0.5,
    noise_type="uniform",
    noise_kwargs={"low": -1, "high": 0.5},
)

data = np.ones((10, 10))

```
# File: force.txt
```python
"""Single layer of neurons learning connections through online learning rules.

Warning
-------

This class is deprecated since v0.3.4 and will be removed in future versions.
Please use :py:class:`~reservoirpy.LMS` or :py:class:`~reservoirpy.RLS` instead.

The learning rules involved are similar to Recursive Least Squares (``rls`` rule)
as described in [1]_ or Least Mean Squares (``lms`` rule, similar to Hebbian
learning) as described in [2]_.

"FORCE" name refers to the training paradigm described in [1]_.

:py:attr:`FORCE.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
``P``              Matrix :math:`\\mathbf{P}` of RLS rule (optional).
================== =================================================================

:py:attr:`FORCE.hypers` **list**

================== =================================================================
``alpha``          Learning rate (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by
                    default).
``input_bias``     If True, learn a bias term (True by default).
``rule``           One of RLS or LMS rule ("rls" by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
alpha : float or Python generator or iterable, default to 1e-6
    Learning rate. If an iterable or a generator is provided and the learning
    rule is "lms", then the learning rate can be changed at each timestep of
    training. A new learning rate will be drawn from the iterable or generator
    at each timestep.
rule : {"rls", "lms"}, default to "rls"
    Learning rule applied for online training.
Wout : callable or array-like of shape (units, targets), default to
    :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to
    :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

References
----------

.. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
        Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
        https://doi.org/10.1016/j.neuron.2009.07.018

.. [2] Hoerzer, G. M., Legenstein, R., & Maass, W. (2014). Emergence of Complex
        Computational Structures From Chaotic Neural Networks Through
        Reward-Modulated Hebbian Learning. Cerebral Cortex, 24(3), 677–690.
        https://doi.org/10.1093/cercor/bhs348
"""

def __init__(
    self,
    output_dim=None,
    alpha=1e-6,
    rule="rls",
    Wout=zeros,
    bias=zeros,
    input_bias=True,
    name=None,
):

    warnings.warn(
        "'FORCE' is deprecated since v0.3.4 and will be removed "
        "in "
        "future versions. Consider using 'RLS' or 'LMS'.",
        DeprecationWarning,
    )

    params = {"Wout": None, "bias": None}

    if rule not in RULES:
        raise ValueError(
            f"Unknown rule for FORCE learning. "
            f"Available rules are {self._rules}."
        )
    else:
        if rule == "lms":
            train = lms_like_train
            initialize = initialize_lms
        else:
            train = rls_like_train
            initialize = initialize_rls
            params["P"] = None

    if isinstance(alpha, Number):

        def _alpha_gen():
            while True:
                yield alpha

        alpha_gen = _alpha_gen()
    elif isinstance(alpha, Iterable):
        alpha_gen = alpha
    else:
        raise TypeError(
            "'alpha' parameter should be a float or an iterable yielding floats."
        )

```
# File: 02-Cupy.txt
```python
import time

import json

from pathlib import Path

from collections import defaultdict

import matplotlib.pyplot as plt

import numpy as np

from scipy import linalg

from scipy import sparse

from tqdm import tqdm, trange

from reservoirpy.compat import ESN

from reservoirpy import mat_gen

from reservoirpy.datasets import mackey_glass, to_forecasting

import cupy as cp

import cupyx as cpx

from cupyx.time import repeat

def nrmse(ytrue, ypred):

    rmse = np.sqrt(np.sum((ytrue - ypred)**2)) / ytrue.shape[0]

    return rmse / (ytrue.max() - ytrue.min())

T = 20001

T_tot = T + 501

X = []

taus = list(range(12, 37, 3))

for tau in taus:

    X.append(mackey_glass(T_tot, tau=tau))

X = np.concatenate(X, axis=1)

X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 1

X, Xtest = X[:-501], X[-501:]

X, y = to_forecasting(X, forecast=1)

Xtest, ytest = to_forecasting(Xtest, forecast=1)

fig, axes = plt.subplots(len(taus), 1)

for i, tau in enumerate(taus):

    _ = axes[i].plot(X[:500, i])

    axes[i].set_ylabel(tau)

N = 1000

W32 = mat_gen.generate_internal_weights(N, sr=1.25, seed=12345).astype(np.float32)

Win32 = mat_gen.generate_input_weights(N, len(taus), input_bias=True, seed=12345).astype(np.float32)

esn32 = ESN(lr=0.3, input_bias=True, W=W32, Win=Win32, typefloat=np.float32)

def esn_kernel(W, Win, s, u, lr):

    xp = cp.get_array_module(s)

    x = s @ W + u @ Win.T

    x = (1 - lr) * s + lr * xp.tanh(x)

    return x

def esn_states_gpu(W, Win, inputs, lr, progress=True):

    states = np.zeros(shape=(len(inputs), W.shape[0]))

    s = cp.zeros(shape=(1, W.shape[0]))

    U = np.hstack([np.ones(shape=(inputs.shape[0], 1)), inputs])

    for i, u in enumerate(tqdm(U, disable=not progress)):

        u = cp.array(u).reshape(1, -1)

        s = esn_kernel(W, Win, s, u, lr)

        states[i, :] = s.get()

    return states

def esn_states_cpu(W, Win, inputs, lr, progress=True):

    states = np.zeros(shape=(len(inputs), W.shape[0]))

    s = np.zeros(shape=(1, W.shape[0]))

    U = np.array(inputs)

    U = np.hstack([np.ones(shape=(U.shape[0], 1)), U])

    for i, u in enumerate(tqdm(U, disable=not progress)):

        s = esn_kernel(W, Win, s, u, lr)

        states[i, :] = s

    return states

states_gpu = esn_states_gpu(cp.array(W32.toarray()), cp.array(Win32), X, 0.3)

states_cpu = esn_states_cpu(W32.toarray(), Win32, X, 0.3)

perf = repeat(esn_states_gpu,

                (cp.array(W32.toarray()), cp.array(Win32), X, 0.3),

                n_repeat=20)

print(perf)

Ns = [100, 300, 500, 800, 1000, 2000, 5000, 10000]

sparse_cpu_times = defaultdict(list)

for n in Ns:

    if n not in sparse_cpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, len(taus), input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_states_cpu(W32, sparse.csr_matrix(Win32), X, 0.3, progress=False)

            sparse_cpu_times[n].append(time.time() - start)

dense_cpu_times = defaultdict(list)

for n in []: # too long, already computed

    if n not in dense_cpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, len(taus), input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_states_cpu(W32.toarray(), Win32, X, 0.3, progress=False)

            dense_cpu_times[n].append(time.time() - start)

dense_gpu_times = defaultdict(list)

for n in Ns:

    if n not in dense_gpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, len(taus), input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_states_gpu(cp.array(W32.toarray()), cp.array(Win32), X, 0.3, progress=False)

            dense_gpu_times[n].append(time.time() - start)

sparse_gpu_times = defaultdict(list)

for n in Ns:

    if n not in sparse_gpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, len(taus), input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_states_gpu(cpx.scipy.sparse.csr_matrix(W32),

                            cpx.scipy.sparse.csr_matrix(sparse.csr_matrix(Win32)), X, 0.3, progress=False)

            sparse_gpu_times[n].append(time.time() - start)

report_nobatch = Path("../resultats/cupy-nobatch")

if not report_nobatch.exists():

    report_nobatch.mkdir(parents=True)

with (report_nobatch / "cpu_sparse.json").open("w+") as fp:

    json.dump(sparse_cpu_times, fp)

with (report_nobatch / "cpu_dense.json").open("w+") as fp:

    json.dump(dense_cpu_times, fp)

with (report_nobatch / "gpu_sparse.json").open("w+") as fp:

    json.dump(sparse_gpu_times, fp)

with (report_nobatch / "gpu_dense.json").open("w+") as fp:

    json.dump(dense_gpu_times, fp)

report_nobatch = Path("../resultats/cupy-nobatch")

with (report_nobatch / "cpu_sparse.json").open("r") as fp:

    sparse_cpu_times = json.load(fp)

with (report_nobatch / "cpu_dense.json").open("r") as fp:

    dense_cpu_times = json.load(fp)

with (report_nobatch / "gpu_sparse.json").open("r") as fp:

    sparse_gpu_times = json.load(fp)

with (report_nobatch / "gpu_dense.json").open("r") as fp:

    dense_gpu_times = json.load(fp)

fig, ax = plt.subplots(1, 1)

mean_cs = np.array([np.mean(v) for v in sparse_cpu_times.values()])

std_cs = np.array([np.std(v) for v in sparse_cpu_times.values()])

mean_cd = np.array([np.mean(v) for v in dense_cpu_times.values()])

std_cd = np.array([np.std(v) for v in dense_cpu_times.values()])

mean_gs = np.array([np.mean(v) for v in sparse_gpu_times.values()])

std_gs = np.array([np.std(v) for v in sparse_gpu_times.values()])

mean_gd = np.array([np.mean(v) for v in dense_gpu_times.values()])

std_gd = np.array([np.std(v) for v in dense_gpu_times.values()])

ax.plot(Ns, mean_cs, label="CPU sparse")

ax.fill_between(Ns, mean_cs + std_cs, mean_cs - std_cs, alpha=0.2)

#ax.plot(Ns, mean_cd, label="CPU dense")

#ax.fill_between(Ns, mean_cd + std_cd, mean_cd - std_cd, alpha=0.2)

ax.plot(Ns, mean_gs, label="GPU sparse")

ax.fill_between(Ns, mean_gs + std_gs, mean_gs - std_gs, alpha=0.2)

ax.plot(Ns, mean_gd, label="GPU dense")

ax.fill_between(Ns, mean_gd + std_gd, mean_gd - std_gd, alpha=0.2)

ax.set_xlabel("N")

ax.set_ylabel("Time (s)")

_ = ax.legend()

def esn_batched_gpu(W, Win, inputs, lr, batch_size=100):

    states = np.zeros(shape=(len(inputs), W.shape[0]))

    s = cp.zeros(shape=(1, W.shape[0]))

    U = np.hstack([np.ones(shape=(inputs.shape[0], 1)), inputs])

    max_length = len(inputs)

    num_batches = int(np.ceil(U.shape[0] / batch_size))

    for i in range(num_batches):

        end = (i+1)*batch_size if (i+1)*batch_size < max_length else max_length

        u_batch = cp.array(U[i*batch_size:end])

        s_batch = cp.empty((u_batch.shape[0], s.shape[1]))

        for j in range(u_batch.shape[0]):

            x = s @ W + cp.dot(u_batch[j, :], Win.T)

            s = (1 - lr) * s + lr * cp.tanh(x)

            s_batch[j, :] = s.reshape(-1)

        states[i*batch_size:end] = s_batch.get()

    return states

states = esn_batched_gpu(cp.array(W32.toarray()), cp.array(Win32), X, 0.3, batch_size=100)

times = []

for i in trange(20):

    start = time.time()

    esn_batched_gpu(cp.array(W32.toarray()), cp.array(Win32), X, 0.3, batch_size=100)

    times.append(time.time() - start)

print(f"Batched (100) GPU time: {np.mean(times)} ± {np.std(times)} "

        f"(min: {np.min(times)}, max: {np.max(times)})")

batches = list(range(100, 1001, 100))

batches.insert(0, 1)

batch_gpu_times = defaultdict(lambda: defaultdict(list))

for n in Ns:

    W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

    Win32 = mat_gen.generate_input_weights(n, len(taus), input_bias=True, seed=12345).astype(np.float32)

    for batch_size in batches:

        for i in trange(20):

            start = time.time()

            esn_batched_gpu(cp.array(W32.toarray()), cp.array(Win32), X, 0.3, batch_size=batch_size)

            batch_gpu_times[n][batch_size].append(time.time() - start)

report_batch = Path("../resultats/cupy-batch")

if not report_batch.exists():

    report_batch.mkdir(parents=True)

with (report_batch / "gpu_batched.json").open("w+") as fp:

    json.dump(batch_gpu_times, fp)

report_batch = Path("../resultats/cupy-batch")

with (report_batch / "gpu_batched.json").open("r") as fp:

    batch_gpu_times = json.load(fp)

import matplotlib as mpl

Ns = [100, 300, 500, 800, 1000, 1500, 2000]

bgt = defaultdict(lambda: defaultdict((list)))

for n, res in batch_gpu_times.items():

    for b, values in res.items():

        bgt[b][n] = values

evenly_spaced_interval = np.linspace(0.5, 1, len(batches))

colors = [mpl.cm.Blues(x) for x in evenly_spaced_interval]

for i, (batch, res) in enumerate(bgt.items()):

    means = np.array([np.mean(v) for v in res.values()])

    stds = np.array([np.std(v) for v in res.values()])

    upper = means + stds

    lower = means - stds

    color = colors[i]

    plt.plot(Ns, means, color=color, label=batch)

    plt.fill_between(Ns, upper, lower, color=color, alpha=0.2)

plt.legend()

def esn_batched_gpu_with_training(W, Win, inputs, teachers, lr, batch_size=500):

    s = cp.zeros(shape=(1, W.shape[0]), dtype=np.float32)

    N = W.shape[0]

    XXT = cp.zeros(shape=(N+1, N+1), dtype=np.float32)

    YXT = cp.zeros(shape=(teachers.shape[1], N+1), dtype=np.float32)

    R = np.eye(N+1, dtype=np.float32) * 10

    U = np.hstack([np.ones(shape=(inputs.shape[0], 1)), inputs])

    max_length = len(inputs)

    num_batches = int(np.ceil(U.shape[0] / batch_size))

    for i in range(num_batches):

        end = (i+1)*batch_size if (i+1)*batch_size < max_length else max_length

        u_batch = cp.array(U[i*batch_size:end]).astype(np.float32)

        t_batch = cp.array(teachers[i*batch_size:end]).astype(np.float32)

        s_batch = cp.empty((u_batch.shape[0], s.shape[1])).astype(np.float32)

        for j in range(u_batch.shape[0]):

            x = s @ W + u_batch[j, :] @ Win.T

            s = (1 - lr) * s + lr * cp.tanh(x)

            s_batch[j, :] = s.reshape(-1)

        s_batch = cp.hstack([cp.ones((s_batch.shape[0], 1)), s_batch])

        XXT += s_batch.T @ s_batch

        YXT += t_batch.T @ s_batch

    Wout = linalg.solve(XXT.get() + R, YXT.T.get(), assume_a="sym")

    return Wout.T

def esn_batched_cpu_with_training(W, Win, inputs, teachers, lr, batch_size=500):

    N = W.shape[0]

    s = np.zeros(shape=(1, N), dtype=np.float32)

    XXT = np.zeros(shape=(N+1, N+1), dtype=np.float32)

    YXT = np.zeros(shape=(teachers.shape[1], N+1), dtype=np.float32)

    R = np.eye(N+1, dtype=np.float32) * 10

    U = np.hstack([np.ones(shape=(inputs.shape[0], 1)), inputs])

    max_length = len(inputs)

    num_batches = int(np.ceil(U.shape[0] / batch_size))

    for i in range(num_batches):

        end = (i+1)*batch_size if (i+1)*batch_size < max_length else max_length

        u_batch = np.array(U[i*batch_size:end]).astype(np.float32)

        t_batch = np.array(teachers[i*batch_size:end]).astype(np.float32)

        s_batch = np.empty((u_batch.shape[0], s.shape[1])).astype(np.float32)

        for j in range(u_batch.shape[0]):

            x = s @ W + u_batch[j, :] @ Win.T

            s = (1 - lr) * s + lr * np.tanh(x)

            s_batch[j, :] = s.reshape(-1)

        s_batch = np.hstack([np.ones((s_batch.shape[0], 1)), s_batch])

        XXT += s_batch.T @ s_batch

        YXT += t_batch.T @ s_batch

    Wout = linalg.solve(XXT + R, YXT.T, assume_a="sym")

    return Wout.T

T = 20001

T_tot = T + 501

X = mackey_glass(T_tot)

X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 1

X, Xtest = X[:-501], X[-501:]

X, y = to_forecasting(X, forecast=1)

Xtest, ytest = to_forecasting(Xtest, forecast=1)

N = 1000

W32 = mat_gen.generate_internal_weights(N, sr=1.25, seed=12345).astype(np.float32)

Win32 = mat_gen.generate_input_weights(N, 1, input_bias=True, seed=12345).astype(np.float32)

Wout_gpu = esn_batched_gpu_with_training(cp.array(W32.toarray()), cp.array(Win32), X, y, 0.3, batch_size=500)

Wout_cpu = esn_batched_cpu_with_training(W32, Win32, X, y, 0.3, batch_size=500)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(Wout_cpu.T, bins=50, label="CPU")

ax2.hist(Wout_gpu.T, bins=50, label="GPU")

_ = ax1.legend()

_ = ax2.legend()

esn32 = ESN(lr=0.3, input_bias=True, W=W32, Win=Win32, typefloat=np.float32)

esn32.Wout = Wout_gpu

outputs_g, _ = esn32.run([Xtest.astype(np.float32)])

esn32.Wout = Wout_cpu

outputs_c, _ = esn32.run([Xtest.astype(np.float32)])

plt.plot(ytest[:500], label="M-G", color="gray", marker="^", markevery=0.1)

plt.plot(outputs_g[0][:500], label="GPU batched")

plt.plot(outputs_c[0][:500], label="CPU batched")

_ = plt.legend()

sparse_cpu_times = defaultdict(list)

dense_cpu_times = defaultdict(list)

dense_gpu_times = defaultdict(list)

sparse_gpu_times = defaultdict(list)

Ns = [100, 300, 500, 800, 1000, 2000, 5000, 10000]

for n in Ns:

    if n not in sparse_cpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, 1, input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_batched_cpu_with_training(W32, sparse.csr_matrix(Win32), X, y, 0.3)

            sparse_cpu_times[n].append(time.time() - start)

for n in []: #too long

    if n not in dense_cpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, 1, input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_batched_cpu_with_training(W32.toarray(), Win32, X, y, 0.3, progress=False)

            dense_cpu_times[n].append(time.time() - start)

for n in Ns:

    if n not in dense_gpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, 1, input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_batched_gpu_with_training(cp.array(W32.toarray()), cp.array(Win32), X, y, 0.3)

            dense_gpu_times[n].append(time.time() - start)

for n in Ns:

    if n not in sparse_gpu_times:

        W32 = mat_gen.generate_internal_weights(n, sr=1.25, seed=12345).astype(np.float32)

        Win32 = mat_gen.generate_input_weights(n, 1, input_bias=True, seed=12345).astype(np.float32)

        for i in trange(20):

            start = time.time()

            esn_batched_gpu_with_training(cpx.scipy.sparse.csr_matrix(W32),

                            cpx.scipy.sparse.csr_matrix(sparse.csr_matrix(Win32)), X, y, 0.3)

            sparse_gpu_times[n].append(time.time() - start)

report_trainbatch = Path("../resultats/cupy-numpy-train-batch")

if not report_trainbatch.exists():

    report_trainbatch.mkdir(parents=True)

with (report_trainbatch / "cpu_sparse.json").open("w+") as fp:

    json.dump(sparse_cpu_times, fp)

with (report_trainbatch / "cpu_dense.json").open("w+") as fp:

    json.dump(dense_cpu_times, fp)

with (report_trainbatch / "gpu_sparse.json").open("w+") as fp:

    json.dump(sparse_gpu_times, fp)

with (report_trainbatch / "gpu_dense.json").open("w+") as fp:

    json.dump(dense_gpu_times, fp)

fig, ax = plt.subplots(1, 1)

mean_cs = np.array([np.mean(v) for v in sparse_cpu_times.values()])

std_cs = np.array([np.std(v) for v in sparse_cpu_times.values()])

mean_cd = np.array([np.mean(v) for v in dense_cpu_times.values()])

std_cd = np.array([np.std(v) for v in dense_cpu_times.values()])

mean_gs = np.array([np.mean(v) for v in sparse_gpu_times.values()])

std_gs = np.array([np.std(v) for v in sparse_gpu_times.values()])

mean_gd = np.array([np.mean(v) for v in dense_gpu_times.values()])

std_gd = np.array([np.std(v) for v in dense_gpu_times.values()])

ax.plot(Ns, mean_cs, label="CPU sparse")

ax.fill_between(Ns, mean_cs + std_cs, mean_cs - std_cs, alpha=0.2)

#ax.plot(Ns, mean_cd, label="CPU dense")

#ax.fill_between(Ns, mean_cd + std_cd, mean_cd - std_cd, alpha=0.2)

ax.plot(Ns, mean_gs, label="GPU sparse")

ax.fill_between(Ns, mean_gs + std_gs, mean_gs - std_gs, alpha=0.2)

ax.plot(Ns, mean_gd, label="GPU dense")

ax.fill_between(Ns, mean_gd + std_gd, mean_gd - std_gd, alpha=0.2)

ax.set_xlabel("N")

ax.set_ylabel("Time (s)")

```
# File: ops.txt
```python
parents, _ = find_parents_and_children(edges)

concatenated = {}
new_nodes = set()
new_edges = set()
for node in nodes:
    indegree = len(parents[node])
    if indegree > 1 and type(node) not in _MULTI_INPUTS_OPS:
        # if parents are already concatenated, use the previously created Concat
        if all([p.name in concatenated for p in parents[node]]):
            concat = concatenated[parents[node][0].name]
        else:
            # add a Concat node
            concat = Concat()

        new_nodes |= {concat, node}
        new_edges |= set([(p, concat) for p in parents[node]] + [(concat, node)])
        # add concatenated nodes to the registry
        concatenated.update({p.name: concat for p in parents[node]})
    else:
        new_nodes |= {node}
        new_edges |= set([(p, node) for p in parents[node]])

return list(new_nodes), list(new_edges)

msg = "Impossible to link nodes: object {} is neither a Node nor a Model."
for nn in nodes:
    if isinstance(nn, Iterable):
        for n in nn:
            if not isinstance(n, _Node):
                raise TypeError(msg.format(n))
    else:
        if not isinstance(nn, _Node):
            raise TypeError(msg.format(nn))

"""Connects two nodes or models. See `link` doc for more info."""
# fetch all nodes in the two subgraphs, if they are models.
all_nodes = []
for node in (node1, node2):
    if isinstance(node, Model) and not isinstance(node, FrozenModel):
        all_nodes += node.nodes
    else:
        all_nodes += [node]

# fetch all edges in the two subgraphs, if they are models.
all_edges = []
for node in (node1, node2):
    if isinstance(node, Model) and not isinstance(node, FrozenModel):
        all_edges += node.edges

# create edges between output nodes of the
# subgraph 1 and input nodes of the subgraph 2.
senders = []
if isinstance(node1, Model) and not isinstance(node, FrozenModel):
    senders += node1.output_nodes
else:
    senders += [node1]

receivers = []
if isinstance(node2, Model) and not isinstance(node, FrozenModel):
    receivers += node2.input_nodes
else:
    receivers += [node2]

new_edges = list(product(senders, receivers))

# maybe nodes are already initialized?
# check if connected dimensions are ok
for sender, receiver in new_edges:
    if (
        sender.is_initialized
        and receiver.is_initialized
        and sender.output_dim != receiver.input_dim
    ):
        raise ValueError(
            f"Dimension mismatch between connected nodes: "
            f"sender node {sender.name} has output dimension "
            f"{sender.output_dim} but receiver node "
            f"{receiver.name} has input dimension "
            f"{receiver.input_dim}."
        )

# all outputs from subgraph 1 are connected to
# all inputs from subgraph 2.
all_edges += new_edges

return all_nodes, all_edges

node1: Union[_Node, Sequence[_Node]],
node2: Union[_Node, Sequence[_Node]],
name: str = None,

"""Link two :py:class:`~.Node` instances to form a :py:class:`~.Model`
instance. `node1` output will be used as input for `node2` in the
created model. This is similar to a function composition operation:

.. math::

    model(x) = (node1 \\circ node2)(x) = node2(node1(x))

You can also perform this operation using the ``>>`` operator::

    model = node1 >> node2

Or using this function::

    model = link(node1, node2)

-`node1` and `node2` can also be :py:class:`~.Model` instances. In this
case, the new :py:class:`~.Model` created will contain all nodes previously
contained in all the models, and link all `node1` outputs to all `node2`
inputs. This allows to chain the  ``>>`` operator::

    step1 = node0 >> node1  # this is a model
    step2 = step1 >> node2  # this is another

-`node1` and `node2` can finally be lists or tuples of nodes. In this
case, all `node1` outputs will be linked to a :py:class:`~.Concat` node to
concatenate them, and the :py:class:`~.Concat` node will be linked to all
`node2` inputs. You can still use the ``>>`` operator in this situation,
except for many-to-many nodes connections::

    # many-to-one
    model = [node1, node2, ..., node] >> node_out
    # one-to-many
    model = node_in >> [node1, node2, ..., node]
    # ! many-to-many requires to use the `link` method explicitly!
    model = link([node1, node2, ..., node], [node1, node2, ..., node])

Parameters
----------
    node1, node2 : _Node or list of _Node
        Nodes or lists of nodes to link.
    name: str, optional
        Name for the chaining Model.

Returns
-------
    Model
        A :py:class:`~.Model` instance chaining the nodes.

Raises
------
    TypeError
        Dimension mismatch between connected nodes: `node1` output
        dimension if different from `node2` input dimension.
        Reinitialize the nodes or create new ones.

Notes
-----

    Be careful to how you link the different nodes: `reservoirpy` does not
    allow to have circular dependencies between them::

        model = node1 >> node2  # fine
        model = node1 >> node2 >> node1  # raises! data would flow in
                                            # circles forever...
"""

_check_all_nodes(node1, node2)

frozens = []
if isinstance(node1, Sequence):
    frozens += [n.name for n in node1 if isinstance(n, FrozenModel)]
else:
    if isinstance(node1, FrozenModel):
        frozens.append(node1)
if isinstance(node2, Sequence):
    frozens += [n.name for n in node2 if isinstance(n, FrozenModel)]
else:
    if isinstance(node2, FrozenModel):
        frozens.append(node2)

if len(frozens) > 0:
    raise TypeError(
        "Impossible to link FrozenModel to other Nodes or "
        f"Model. FrozenModel found: {frozens}."
    )

nodes = set()
edges = set()
if not isinstance(node1, Sequence):
    node1 = [node1]
if not isinstance(node2, Sequence):
    node2 = [node2]

for left in node1:
    for right in node2:
        new_nodes, new_edges = _link_1to1(left, right)
        nodes |= set(new_nodes)
        edges |= set(new_edges)

return Model(nodes=list(nodes), edges=list(edges), name=name)

node: _Node,
feedback: Union[_Node, Sequence[_Node]],
inplace: bool = False,
name: str = None,

"""Create a feedback connection between the `feedback` node and `node`.
Feedbacks nodes will be called at runtime using data from the previous
call.

This is not an in-place operation by default. This function will copy `node`
and then sets the copy `_feedback` attribute as a reference to `feedback`
node. If `inplace` is set to `True`, then `node` is not copied and the
feedback is directly connected to `node`. If `feedback` is a list of nodes
or models, then all nodes in the list are first connected to a
:py:class:`~.Concat` node to create a model gathering all data from all nodes
in a single feedback vector.

    You can also perform this operation using the ``<<`` operator::

    node1 = node1 << node2
    # with feedback from a Model
    node1 = node1 << (fbnode1 >> fbnode2)
    # with feedback from a list of nodes or models
    node1 = node1 << [fbnode1, fbnode2, ...]

Which means that a feedback connection is now created between `node1` and
`node2`. In other words, the forward function of `node1` depends on the
previous output of `node2`:

.. math::
    \\mathrm{node1}(x_t) = \\mathrm{node1}(x_t, \\mathrm{node2}(x_{t - 1}))

You can also use this function to define feedback::

    node1 = link_feedback(node1, node2)
    # without copy (node1 is the same object throughout)
    node1 = link_feedback(node1, node2, inplace=True, name="n1_copy")

Parameters
----------
node : Node
    Node receiving feedback.
feedback : _Node
    Node or Model sending feedback
inplace : bool, defaults to False
    If `True`, then the function returns a copy of `node`.
name : str, optional
    Name of the copy of `node` if `inplace` is `True`.

Returns
-------
    Node
        A node instance taking feedback from `feedback`.

Raises
------
    TypeError
        - If `node` is a :py:class:`~.Model`.
        Models can not receive feedback.

        - If any of the feedback nodes are not :py:class:`~._Node`
        instances.
"""

if isinstance(node, Model):
    raise TypeError(f"{node} is not a Node. Models can't receive feedback.")

msg = (
    "Impossible to receive feedback from {}: "
    "it is not a Node or a Model instance."
)

if isinstance(feedback, Sequence):
    for fb in feedback:
        if not isinstance(fb, _Node):
            raise TypeError(msg.format(fb))

    all_fb = link(feedback, Concat())

elif isinstance(feedback, _Node):
    all_fb = feedback

else:
    raise TypeError(msg.format(feedback))

if inplace:
    node._feedback = DistantFeedback(all_fb, node)
    return node
else:
    # first copy the node, then give it feedback
    # original node is not connected to any feedback then
    new_node = node.copy(name=name)
    new_node._feedback = DistantFeedback(all_fb, new_node)
    return new_node

model: _Node, *models: _Node, inplace: bool = False, name: str = None

"""Merge different :py:class:`~.Model` or :py:class:`~.Node`
instances into a single :py:class:`~.Model` instance.

:py:class:`~.Node` instances contained in the models to merge will be
gathered in a single model, along with all previously defined connections
between them, if they exists.

You can also perform this operation using the ``&`` operator::

    model = (node1 >> node2) & (node1 >> node3))

This is equivalent to::

    model = merge((node1 >> node2), (node1 >> node3))

The in-place operator can also be used::

    model &= other_model

Which is equivalent to::

    model.update_graph(other_model.nodes, other_model.edges)

Parameters
----------
model: Model or Node
    First node or model to merge. The `inplace` parameter takes this
    instance as reference.
*models : Model or Node
    All models to merge.
inplace: bool, default to False
    If `True`, then will update Model `model` in-place. If `model` is not
    a Model instance, this parameter will causes the function to raise
    a `ValueError`.
name: str, optional
    Name of the resulting Model.

Returns
-------
Model
    A new :py:class:`~.Model` instance.

Raises
------
ValueError
    If `inplace` is `True` but `model` is not a Model instance, then the
    operation is impossible. In-place merging can only take place on a
    Model instance.
"""
msg = "Impossible to merge models: object {} is not a Model instance."

if isinstance(model, _Node):
    all_nodes = set()
    all_edges = set()
    for m in models:
        # fuse models nodes and edges (right side argument)
        if isinstance(m, Model) and not isinstance(m, FrozenModel):
            all_nodes |= set(m.nodes)
            all_edges |= set(m.edges)
        elif isinstance(m, _Node):
            all_nodes |= {m}

    if inplace:
        if not isinstance(model, Model) or isinstance(model, FrozenModel):
            raise ValueError(
                f"Impossible to merge models in-place: "
                f"{model} is not a Model instance."
            )
        return model.update_graph(all_nodes, all_edges)

    else:
        # add left side model nodes
        if isinstance(model, Model) and not isinstance(model, FrozenModel):
            all_nodes |= set(model.nodes)
            all_edges |= set(model.edges)
        else:
            all_nodes |= {model}

        return Model(nodes=list(all_nodes), edges=list(all_edges), name=name)

```
# File: _seed.txt
```python
"""Return the current random state seed used for dataset
generation.

Returns
-------
int
    Current seed value.

See also
--------
set_seed: Change the default random seed value for datasets generation.
"""
global _DEFAULT_SEED
return _DEFAULT_SEED

"""Change the default random seed value used for dataset generation.

This will change the behaviour of the Mackey-Glass and NARMA
timeseries generator (see :py:func:`mackey_glass` and :py:func:`narma`).

Parameters
----------
s : int
    A random state generator numerical seed.

```
# File: validation.txt
```python
return isinstance(seq, list) or (isinstance(seq, np.ndarray) and seq.ndim > 2)

return obj is not None and isinstance(obj, np.ndarray) or issparse(obj)

return isinstance(obj, Mapping) or (
    (hasattr(obj, "items") and hasattr(obj, "get"))
    or (
        not (isinstance(obj, list) or isinstance(obj, tuple))
        and hasattr(obj, "__getitem__")
        and not hasattr(obj, "__array__")
    )
)

if isinstance(X, np.ndarray):
    X = np.atleast_2d(X)
    return np.hstack([np.ones((X.shape[0], 1)), X])
elif isinstance(X, list):
    new_X = []
    for x in X:
        x = np.atleast_2d(x)
        new_X.append(np.hstack([np.ones((x.shape[0], 1)), x]))
    return new_X

msg = "."
if caller is not None:
    if hasattr(caller, "name"):
        msg = f" in {caller.name}."
    else:
        msg = f"in {caller}."

if not isinstance(array, np.ndarray):
    # maybe a single number, make it an array
    if isinstance(array, numbers.Number):
        array = np.asarray(array)
    else:
        msg = (
            f"Data type '{type(array)}' not understood. All vectors "
            f"should be Numpy arrays" + msg
        )
        raise TypeError(msg)

if not (np.issubdtype(array.dtype, np.number)):
    msg = f"Impossible to operate on non-numerical data, in array: {array}" + msg
    raise TypeError(msg)

if allow_reshape:
    array = np.atleast_2d(array)

if not allow_timespans:
    if array.shape[0] > 1:
        msg = (
            f"Impossible to operate on multiple timesteps. Data should"
            f" have shape (1, n) but is {array.shape}" + msg
        )
        raise ValueError(msg)

# TODO: choose axis to expand and/or np.atleast_2d

```
# File: parallel.txt
```python
_BACKEND = "sequential"

_BACKEND = "loky"

if backend is not None:
    if sys.version_info < (3, 8):
        return "sequential"
    if backend in _AVAILABLE_BACKENDS:
        return backend
    else:
        raise ValueError(
            f"'{backend}' is not a Joblib backend. Available "
            f"backends are {_AVAILABLE_BACKENDS}."
        )
return _BACKEND if workers > 1 or workers == -1 else "sequential"

global _BACKEND
if backend in _AVAILABLE_BACKENDS:
    _BACKEND = backend
else:
    raise ValueError(
        f"'{backend}' is not a valid joblib "
        f"backend value. Available backends are "
        f"{_AVAILABLE_BACKENDS}."
    )

from .. import _TEMPDIR

global temp_registry

caller = node.name
if data is None:
    if shape is None:
        raise ValueError(
            f"Impossible to create buffer for node {node}: "
            f"neither data nor shape were given."
        )

temp = os.path.join(_TEMPDIR, f"{caller}-{name}-{uuid.uuid4()}")

temp_registry[node].append(temp)

shape = shape if shape is not None else data.shape
dtype = dtype if dtype is not None else global_dtype

memmap = np.memmap(temp, shape=shape, mode=mode, dtype=dtype)

if data is not None:
    memmap[:] = data

return memmap

global temp_registry

```
# File: ridge copy.txt
```python
"""Solve Tikhonov regression."""
return linalg.solve(XXT + ridge, YXT.T, assume_a="sym")

"""Aggregate Xi.Xi^T and Yi.Xi^T matrices from a state sequence i."""
XXT = readout.get_buffer("XXT")
YXT = readout.get_buffer("YXT")
XXT += xxt
YXT += yxt

"""Pre-compute XXt and YXt before final fit."""
X, Y = _prepare_inputs_for_learning(
    X_batch,
    Y_batch,
    bias=readout.input_bias,
    allow_reshape=True,
)

xxt = X.T.dot(X)
yxt = Y.T.dot(X)

if lock is not None:
    # This is not thread-safe using Numpy memmap as buffers
    # ok for parallelization then with a lock (see ESN object)
    with lock:
        _accumulate(readout, xxt, yxt)
else:
    _accumulate(readout, xxt, yxt)

ridge = readout.ridge
XXT = readout.get_buffer("XXT")
YXT = readout.get_buffer("YXT")

input_dim = readout.input_dim
if readout.input_bias:
    input_dim += 1

ridgeid = ridge * np.eye(input_dim, dtype=global_dtype)

Wout_raw = _solve_ridge(XXT, YXT, ridgeid)

if readout.input_bias:
    Wout, bias = Wout_raw[1:, :], Wout_raw[0, :][np.newaxis, :]
    readout.set_param("Wout", Wout)
    readout.set_param("bias", bias)
else:
    readout.set_param("Wout", Wout_raw)

_initialize_readout(
    readout, x, y, bias=readout.input_bias, init_func=Wout_init, bias_init=bias_init
)

"""create memmaped buffers for matrices X.X^T and Y.X^T pre-computed
in parallel for ridge regression
! only memmap can be used ! Impossible to share Numpy arrays with
different processes in r/w mode otherwise (with proper locking)
"""
input_dim = readout.input_dim
output_dim = readout.output_dim

if readout.input_bias:
    input_dim += 1

readout.create_buffer("XXT", (input_dim, input_dim))
readout.create_buffer("YXT", (output_dim, input_dim))

"""A single layer of neurons learning with Tikhonov linear regression.

Output weights of the layer are computed following:

.. math::

    \\hat{\\mathbf{W}}_{out} = \\mathbf{YX}^\\top ~ (\\mathbf{XX}^\\top +
    \\lambda\\mathbf{Id})^{-1}

Outputs :math:`\\mathbf{y}` of the node are the result of:

.. math::

    \\mathbf{y} = \\mathbf{W}_{out}^\\top \\mathbf{x} + \\mathbf{b}

where:
    - :math:`\\mathbf{X}` is the accumulation of all inputs during training;
    - :math:`\\mathbf{Y}` is the accumulation of all targets during training;
    - :math:`\\mathbf{b}` is the first row of :math:`\\hat{\\mathbf{W}}_{out}`;
    - :math:`\\mathbf{W}_{out}` is the rest of :math:`\\hat{\\mathbf{W}}_{out}`.

If ``input_bias`` is True, then :math:`\\mathbf{b}` is non-zero, and a constant
term is added to :math:`\\mathbf{X}` to compute it.

:py:attr:`Ridge.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
================== =================================================================

:py:attr:`Ridge.hypers` **list**

================== =================================================================
``ridge``          Regularization parameter (:math:`\\lambda`) (0.0 by default).
``input_bias``     If True, learn a bias term (True by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
ridge: float, default to 0.0
    L2 regularization parameter.
Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

Example
-------

>>> x = np.random.normal(size=(100, 3))
>>> noise = np.random.normal(scale=0.1, size=(100, 1))
>>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.
>>>
>>> from reservoirpy.nodes import Ridge
>>> ridge_regressor = Ridge(ridge=0.001)
>>>
>>> ridge_regressor.fit(x, y)
>>> ridge_regressor.Wout, ridge_regressor.bias
array([[ 9.992, -0.205,  6.989]]).T, array([[12.011]])
"""

```
# File: test_activationsfunc.txt
```python
"value, expected",
[
    ([1, 2, 3], np.exp([1, 2, 3]) / np.sum(np.exp([1, 2, 3]))),
    (1, np.exp(1) / np.sum(np.exp(1))),
    ([0, 0], [0.5, 0.5]),
],

result = softmax(value)

assert_almost_equal(np.sum(result), 1.0)
assert_array_almost_equal(result, expected)

"value, expected",
[
    (0, np.log(1 + np.exp(0))),
    ([0, 1, 2], np.log(1 + np.exp([0, 1, 2]))),
    ([-2, -1], np.log(1 + np.exp([-2, -1]))),
],

result = softplus(value)

assert np.any(result > 0.0)
assert_array_almost_equal(result, expected)

"value",
[
    ([1, 2, 3]),
    ([1]),
    (0),
    ([0.213565165, 0.1, 1.064598495615132]),
],

result = identity(value)
val = np.asanyarray(value)

assert np.any(result == val)

"value, expected", [([1, 2, 3], np.tanh([1, 2, 3])), (0, np.tanh(0))]

result = tanh(value)

assert_array_almost_equal(result, expected)

"value, expected",
[
    ([1, 2, 3], 1 / (1 + np.exp(-np.array([1, 2, 3])))),
    (0, 1 / (1 + np.exp(0))),
    ([-1000, -2], [0.0, 1.35e-1]),
],

result = sigmoid(value)
assert_array_almost_equal(result, expected, decimal=1)

"value, expected",
[
    ([1, 2, 3], np.array([1, 2, 3])),
    ([-1, -10, 5], np.array([0, 0, 5])),
    (0, np.array(0)),
    ([[1, 2, 3], [-1, 2, 3]], np.array([[1, 2, 3], [0, 2, 3]])),
],

```
# File: test_datasets.txt
```python
"""Disable caching temporarily when running tests"""
datasets._chaos.memory = Memory(location=None)
yield
datasets._chaos.memory = Memory(os.path.join(_TEMPDIR, "datasets"), verbose=0)

"dataset_func",
[
    datasets.henon_map,
    datasets.logistic_map,
    datasets.lorenz,
    datasets.mackey_glass,
    datasets.multiscroll,
    datasets.doublescroll,
    datasets.rabinovich_fabrikant,
    datasets.narma,
    datasets.lorenz96,
    datasets.rossler,
    datasets.kuramoto_sivashinsky,
],

with no_cache():
    timesteps = 2000
    X = dataset_func(timesteps)

assert isinstance(X, np.ndarray)
assert len(X) == timesteps

"dataset_func,kwargs,expected",
[
    (datasets.logistic_map, {"r": -1}, ValueError),
    (datasets.logistic_map, {"x0": 1}, ValueError),
    (datasets.mackey_glass, {"seed": 1234}, None),
    (datasets.mackey_glass, {"seed": None}, None),
    (datasets.mackey_glass, {"tau": 0}, None),
    (datasets.narma, {"seed": 1234}, None),
    (datasets.lorenz96, {"N": 1}, ValueError),
    (datasets.lorenz96, {"x0": [0.1, 0.2, 0.3, 0.4, 0.5], "N": 4}, ValueError),
    (datasets.rossler, {"x0": [0.1, 0.2]}, ValueError),
    (
        datasets.kuramoto_sivashinsky,
        {"x0": np.random.normal(size=129), "N": 128},
        ValueError,
    ),
    (
        datasets.kuramoto_sivashinsky,
        {"x0": np.random.normal(size=128), "N": 128},
        None,
    ),
],

if expected is None:
    timesteps = 2000
    X = dataset_func(timesteps, **kwargs)

    assert isinstance(X, np.ndarray)
    assert len(X) == timesteps
else:
    with pytest.raises(expected):
        timesteps = 2000
        dataset_func(timesteps, **kwargs)

x1 = dataset_func(200)
x2 = dataset_func(200)

assert_allclose(x1, x2)

s = datasets.get_seed()
assert s == datasets._seed._DEFAULT_SEED

x1 = dataset_func(200)

datasets.set_seed(1234)
assert datasets._seed._DEFAULT_SEED == 1234
assert datasets.get_seed() == 1234

x2 = dataset_func(200)

assert (np.abs(x1 - x2) > 1e-3).sum() > 0

x = dataset_func(200)

x, y = to_forecasting(x, forecast=5)

assert x.shape[0] == 200 - 5
assert y.shape[0] == 200 - 5
assert x.shape[0] == y.shape[0]

x = dataset_func(200)

x, xt, y, yt = to_forecasting(x, forecast=5, test_size=10)

assert x.shape[0] == 200 - 5 - 10
assert y.shape[0] == 200 - 5 - 10
assert x.shape[0] == y.shape[0]
assert xt.shape[0] == yt.shape[0] == 10

X, Y, X_test, Y_test = datasets.japanese_vowels(reload=True)

assert len(X) == 270 == len(Y)
assert len(X_test) == 370 == len(Y_test)

assert Y[0].shape == (1, 9)

X, Y, X_test, Y_test = datasets.japanese_vowels(repeat_targets=True)

assert Y[0].shape == (X[0].shape[0], 9)

X, Y, X_test, Y_test = datasets.japanese_vowels(one_hot_encode=False)

```
# File: 3-General_Introduction_to_Reservoir_Computing.txt
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

from reservoirpy.datasets import mackey_glass

from reservoirpy.observables import nrmse, rsquare

timesteps = 2510

tau = 17

X = mackey_glass(timesteps, tau=tau)

# rescale between -1 and 1

X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

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

plot_mackey_glass(X, 500, tau)

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

from reservoirpy.datasets import to_forecasting

x, y = to_forecasting(X, forecast=10)

X_train1, y_train1 = x[:2000], y[:2000]

X_test1, y_test1 = x[2000:], y[2000:]

plot_train_test(X_train1, y_train1, X_test1, y_test1)

units = 100

leak_rate = 0.3

spectral_radius = 1.25

input_scaling = 1.0

connectivity = 0.1

input_connectivity = 0.2

regularization = 1e-8

seed = 1234

def reset_esn():

    from reservoirpy.nodes import Reservoir, Ridge

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,

                            lr=leak_rate, rc_connectivity=connectivity,

                            input_connectivity=input_connectivity, seed=seed)

    readout   = Ridge(1, ridge=regularization)

    return reservoir >> readout

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,

                        lr=leak_rate, rc_connectivity=connectivity,

                        input_connectivity=input_connectivity, seed=seed)

readout   = Ridge(1, ridge=regularization)

esn = reservoir >> readout

y = esn(X[0])  # initialisation

reservoir.Win is not None, reservoir.W is not None, readout.Wout is not None

np.all(readout.Wout == 0.0)

esn = esn.fit(X_train1, y_train1)

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

plot_readout(readout)

def plot_results(y_pred, y_test, sample=500):

    fig = plt.figure(figsize=(15, 7))

    plt.subplot(211)

    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")

    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")

    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")

    plt.legend()

    plt.show()

y_pred1 = esn.run(X_test1)

plot_results(y_pred1, y_test1)

rsquare(y_test1, y_pred1), nrmse(y_test1, y_pred1)

x, y = to_forecasting(X, forecast=100)

X_train2, y_train2 = x[:2000], y[:2000]

X_test2, y_test2 = x[2000:], y[2000:]

plot_train_test(X_train2, y_train2, X_test2, y_test2)

y_pred2 = esn.fit(X_train2, y_train2).run(X_test2)

plot_results(y_pred2, y_test2, sample=400)

rsquare(y_test2, y_pred2), nrmse(y_test2, y_pred2)

units = 500

leak_rate = 0.3

spectral_radius = 0.99

input_scaling = 1.0

connectivity = 0.1      # - density of reservoir internal matrix

input_connectivity = 0.2  # and of reservoir input matrix

regularization = 1e-4

seed = 1234             # for reproducibility

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

esn = reset_esn()

x, y = to_forecasting(X, forecast=1)

X_train3, y_train3 = x[:2000], y[:2000]

X_test3, y_test3 = x[2000:], y[2000:]

esn = esn.fit(X_train3, y_train3)

seed_timesteps = 100

warming_inputs = X_test3[:seed_timesteps]

warming_out = esn.run(warming_inputs, reset=True)  # warmup

nb_generations = 400

X_gen = np.zeros((nb_generations, 1))

y = warming_out[-1]

for t in range(nb_generations):  # generation

    y = esn(y)

    X_gen[t, :] = y

X_t = X_test3[seed_timesteps: nb_generations+seed_timesteps]

plot_generation(X_gen, X_t, nb_generations, warming_out=warming_out,

                warming_inputs=warming_inputs, seed_timesteps=seed_timesteps)

<img src="./static/online.png" width="700">

units = 100

leak_rate = 0.3

spectral_radius = 1.25

input_scaling = 1.0

connectivity = 0.1

input_connectivity = 0.2

seed = 1234

from reservoirpy.nodes import FORCE

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,

                        lr=leak_rate, rc_connectivity=connectivity,

                        input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)

esn_online = reservoir >> readout

outputs_pre = np.zeros(X_train1.shape)

for t, (x, y) in enumerate(zip(X_train1, y_train1)): # for each timestep of training data:

    outputs_pre[t, :] = esn_online.train(x, y)

plot_results(outputs_pre, y_train1, sample=100)

plot_results(outputs_pre, y_train1, sample=500)

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,

                        lr=leak_rate, rc_connectivity=connectivity,

                        input_connectivity=input_connectivity, seed=seed)

readout   = FORCE(1)

esn_online = reservoir >> readout

esn_online.train(X_train1, y_train1)

pred_online = esn_online.run(X_test1)  # Wout est maintenant figée

plot_results(pred_online, y_test1, sample=500)

rsquare(y_test1, pred_online), nrmse(y_test1, pred_online)

<img src="./static/sigmaban.gif" width="500">

import glob

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from joblib import delayed, Parallel

from tqdm import tqdm

features = ['com_x', 'com_y', 'com_z', 'trunk_pitch', 'trunk_roll', 'left_x', 'left_y',

            'right_x', 'right_y', 'left_ankle_pitch', 'left_ankle_roll', 'left_hip_pitch',

            'left_hip_roll', 'left_hip_yaw', 'left_knee', 'right_ankle_pitch',

            'right_ankle_roll', 'right_hip_pitch', 'right_hip_roll',

            'right_hip_yaw', 'right_knee']

prediction = ['fallen']

force = ['force_orientation', 'force_magnitude']

files = glob.glob("./r4-data/experiments/*")

dfs = []

with Parallel(n_jobs=-1) as parallel:

    dfs = parallel(delayed(pd.read_csv)(f, compression="gzip", header=0, sep=",") for f in tqdm(files))

X = []

Y = []

F = []

for i, df in enumerate(dfs):

    X.append(df[features].values)

    Y.append(df[prediction].values)

    F.append(df["force_magnitude"].values)

Y_train = []

for y in Y:

    y_shift = np.roll(y, -500)

    y_shift[-500:] = y[-500:]

    Y_train.append(y_shift)

def plot_robot(Y, Y_train, F):

    plt.figure(figsize=(10, 7))

    plt.plot(Y_train[1], label="Objective")

    plt.plot(Y[1], label="Fall indicator")

    plt.plot(F[1], label="Applied force")

    plt.legend()

    plt.show()

plot_robot(Y, Y_train, F)

X_train, X_test, y_train, y_test = train_test_split(X, Y_train, test_size=0.2, random_state=42)

from reservoirpy.nodes import ESN

reservoir = Reservoir(300, lr=0.5, sr=0.99, input_bias=False)

readout   = Ridge(1, ridge=1e-3)

esn = ESN(reservoir=reservoir, readout=readout, workers=-1)  # version distribuée

esn = esn.fit(X_train, y_train)

res = esn.run(X_test)

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

plot_robot_results(y_test, res)

print("Averaged RMSE :", f"{np.mean(scores):.4f}", "±", f"{np.std(scores):.5f}")

print("Averaged RMSE (with threshold) :", f"{np.mean(filt_scores):.4f}", "±", f"{np.std(filt_scores):.5f}")

<img src="./static/canary.png" width="500">

from IPython.display import Audio

audio = Audio(filename="./static/song.wav")

display(audio)

segment songs properly.

im = plt.imread("./static/canary_outputs.png")

plt.figure(figsize=(15, 15)); plt.imshow(im); plt.axis('off'); plt.show()

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

one_hot = OneHotEncoder(categories=[vocab], sparse_output=False)

Y = [one_hot.fit_transform(np.array(y)) for y in Y]

X_train, y_train = X[:-10], Y[:-10]

X_test, y_test = X[-10:], Y[-10:]

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

esn = esn.fit(X_train, y_train)

outputs = esn.run(X_test)

from sklearn.metrics import accuracy_score

scores = []

for y_t, y_p in zip(y_test, outputs):

    targets = np.vstack(one_hot.inverse_transform(y_t)).flatten()

    top_1 = np.argmax(y_p, axis=1)

    top_1 = np.array([vocab[t] for t in top_1])

    accuracy = accuracy_score(targets, top_1)

    scores.append(accuracy)

scores  # for each song in the testing set

```
# File: tests_xavier.txt
```python



mackey_glass, 

forecast=forecast, 

test_size=0.2





seed = 260_418

x_train, x_test, y_train, y_test = dataset

leak_rate = 0.3

spectral_radius = 1.25

input_scaling = 1.0

connectivity = 0.1

input_connectivity = 0.2

scores = []

for i in tqdm.tqdm(range(iters)):

    seed += 1

    reservoir = Reservoir(

        N,

        input_scaling=input_scaling, 

        sr=spectral_radius,

        lr=leak_rate, 

        rc_connectivity=connectivity,

        input_connectivity=input_connectivity, 

        seed=seed

    )

    # readout = Ridge(1, ridge=regularization)

    readout = ScikitLearnNode(linear_model.LinearRegression)

    model = reservoir >> readout

    

    model.fit(x_train, y_train)

    y_pred = model.run(x_test)

    score = reservoirpy.observables.nrmse(y_test, y_pred)

    scores.append(score)

return scores



scores = eval(N, 10, mg_dataset)

with open("./scores.txt", "a+") as file:

    print(scores, file=file)

all_scores.append(scores)











distr = scipy.stats.norm()

distr.random_state = rng

density = np.sqrt(1 - np.exp(np.log(1 - connectivity) / rank))

m = scipy.sparse.random(units, rank, density=density, random_state=rng, data_rvs=distr.rvs)

n = scipy.sparse.random(rank, units, density=density, random_state=rng, data_rvs=distr.rvs)

W = m @ n

sr = spec_rad(W, maxiter=300)

return W * spectral_radius / sr



seed = 260_418

x_train, x_test, y_train, y_test = dataset

leak_rate = 0.3

spectral_radius = 1.25

input_scaling = 1.0

connectivity = 0.1

input_connectivity = 0.2

scores = []

for i in tqdm.tqdm(range(iters)):

    seed += 1

    try:

        reservoir = Reservoir(

            W=sparse_with_rank(

                units=N,

                rank=rank,

                connectivity=connectivity,

                spectral_radius=spectral_radius,

                rng=seed+1000,

            ),

            input_scaling=input_scaling, 

            lr=leak_rate,

            input_connectivity=input_connectivity,

            seed=seed,

        )

    except scipy.sparse.linalg.ArpackNoConvergence:

        print("ArpackNoConvergence", rank, i)

        continue

        

    # readout = Ridge(1, ridge=regularization)

    readout = ScikitLearnNode(linear_model.LinearRegression)

    model = reservoir >> readout

    

    model.fit(x_train, y_train)

    y_pred = model.run(x_test)

    score = reservoirpy.observables.nrmse(y_test, y_pred)

    scores.append(score)

return scores





scores = eval_rank(N, rank, 10, mg_dataset)

with open("./scores_rank.txt", "a+") as file:

    print(scores, file=file)

all_scores_rank.append(scores)

plt.plot([rank]*len(scores), scores, '+', color='black')





scores = eval_rank(N, rank, 10, mg_dataset)

with open("./scores_rank_300.txt", "a+") as file:

    print(scores, file=file)

all_scores_rank300.append(scores)

plt.plot([rank]*len(scores), scores, '+', color='black')






```
# File: test_scikitlearnnode.txt
```python
ElasticNet,
Lars,
Lasso,
LassoCV,
LassoLars,
LinearRegression,
LogisticRegression,
MultiTaskLassoCV,
OrthogonalMatchingPursuitCV,
PassiveAggressiveClassifier,
Perceptron,
Ridge,
RidgeClassifier,
SGDClassifier,
SGDRegressor,

with pytest.raises(AttributeError):
    _ = ScikitLearnNode(PCA)
with pytest.raises(AttributeError):
    _ = ScikitLearnNode(DotProduct)

with pytest.raises(RuntimeError):
    _ = ScikitLearnNode(LinearRegression).initialize()

with pytest.raises(RuntimeError):
    _ = ScikitLearnNode(LinearRegression).initialize(np.ones((100, 2)))

_ = ScikitLearnNode(LinearRegression).initialize(
    np.ones((100, 2)), np.ones((100, 2))
)

linear_regressor = ScikitLearnNode(
    LinearRegression, output_dim=2, model_hypers={"positive": False}
).initialize(np.ones((100, 2)))

assert linear_regressor.model_hypers == {"positive": False}

"model, model_hypers",
[
    (LogisticRegression, {"random_state": 2341}),
    (PassiveAggressiveClassifier, {"random_state": 2341}),
    (Perceptron, {"random_state": 2341}),
    (RidgeClassifier, {"random_state": 2341}),
    (SGDClassifier, {"random_state": 2341}),
    (MLPClassifier, {"random_state": 2341}),
],

rng = np.random.default_rng(seed=2341)

X_train = rng.normal(0, 1, size=(10000, 2))
y_train = (X_train[:, 0:1] > 0.0).astype(np.float16)
X_test = rng.normal(0, 1, size=(100, 2))
y_test = (X_test[:, 0:1] > 0.0).astype(np.float16)

scikit_learn_node = ScikitLearnNode(model=model, model_hypers=model_hypers)

scikit_learn_node.fit(X_train, y_train)
y_pred = scikit_learn_node.run(X_test)
assert y_pred.shape == y_test.shape
assert np.all(y_pred == y_test)

"model, model_hypers",
[
    (LinearRegression, None),
    (Ridge, {"random_state": 2341}),
    (SGDRegressor, {"random_state": 2341}),
    (ElasticNet, {"alpha": 1e-4, "random_state": 2341}),
    (Lars, {"random_state": 2341}),
    (Lasso, {"alpha": 1e-4, "random_state": 2341}),
    (LassoLars, {"alpha": 1e-4, "random_state": 2341}),
    (OrthogonalMatchingPursuitCV, {}),
    (MLPRegressor, {"tol": 1e-6, "random_state": 2341}),
],

rng = np.random.default_rng(seed=2341)
X_train = list(rng.normal(0, 1, size=(30, 100, 2)))
y_train = [(x[:, 0:1] + x[:, 1:2]).astype(np.float16) for x in X_train]
X_test = rng.normal(0, 1, size=(100, 2))
y_test = (X_test[:, 0:1] + X_test[:, 1:2]).astype(np.float16)

scikit_learn_node = ScikitLearnNode(model=model, model_hypers=model_hypers)

scikit_learn_node.fit(X_train, y_train)
y_pred = scikit_learn_node.run(X_test)
assert y_pred.shape == y_test.shape
mse = np.mean(np.square(y_pred - y_test))
assert mse < 2e-4

rng = np.random.default_rng(seed=2341)
X_train = rng.normal(0, 1, size=(10000, 3))
y_train = X_train @ np.array([[0, 1, 0], [0, 1, 1], [-1, 0, 1]])
X_test = rng.normal(0, 1, size=(100, 3))

lasso = ScikitLearnNode(model=LassoCV, model_hypers={"random_state": 2341}).fit(
    X_train, y_train
)
lasso_pred = lasso.run(X_test)

mt_lasso = ScikitLearnNode(
    model=MultiTaskLassoCV, model_hypers={"random_state": 2341}
).fit(X_train, y_train)
mt_lasso_pred = mt_lasso.run(X_test)

assert type(lasso.params["instances"]) is list
assert type(mt_lasso.params["instances"]) is not list

coef_single = [
    lasso.params["instances"][0].coef_,
    lasso.params["instances"][1].coef_,
    lasso.params["instances"][2].coef_,
]
coef_multitask = mt_lasso.params["instances"].coef_
assert np.linalg.norm(coef_single[0] - coef_multitask[0]) < 1e-3
assert np.linalg.norm(coef_single[1] - coef_multitask[1]) < 1e-3
assert np.linalg.norm(coef_single[2] - coef_multitask[2]) < 1e-3

assert lasso_pred.shape == mt_lasso_pred.shape == (100, 3)
assert np.linalg.norm(mt_lasso_pred - lasso_pred) < 1e-2

rng = np.random.default_rng(seed=2341)
X_train = rng.normal(0, 1, size=(100, 3))
y_train = (X_train @ np.array([0.5, 1, 2])).reshape(-1, 1)
X_test = rng.normal(0, 1, size=(100, 3))

# Different scikit-learn random_states
reservoirpy.set_seed(0)
y_pred1 = (
    ScikitLearnNode(model=SGDRegressor, model_hypers={"random_state": 1})
    .fit(X_train, y_train)
    .run(X_test)
)

reservoirpy.set_seed(0)
y_pred2 = (
    ScikitLearnNode(model=SGDRegressor, model_hypers={"random_state": 2})
    .fit(X_train, y_train)
    .run(X_test)
)

assert not np.all(y_pred1 == y_pred2)

# Same scikit-learn random_states
reservoirpy.set_seed(0)
y_pred1 = (
    ScikitLearnNode(model=SGDRegressor, model_hypers={"random_state": 1})
    .fit(X_train, y_train)
    .run(X_test)
)

reservoirpy.set_seed(0)
y_pred2 = (
    ScikitLearnNode(model=SGDRegressor, model_hypers={"random_state": 1})
    .fit(X_train, y_train)
    .run(X_test)
)

assert np.all(y_pred1 == y_pred2)

rng = np.random.default_rng(seed=2341)
X_train = rng.normal(0, 1, size=(100, 3))
y_train = (X_train @ np.array([0.5, 1, 2])).reshape(-1, 1)
X_test = rng.normal(0, 1, size=(100, 3))

# Different ReservoirPy random generator
reservoirpy.set_seed(1)
y_pred1 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

reservoirpy.set_seed(2)
y_pred2 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

assert not np.all(y_pred1 == y_pred2)

# Same ReservoirPy random generator
reservoirpy.set_seed(0)
y_pred1 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

reservoirpy.set_seed(0)
y_pred2 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

```
# File: test_delay.txt
```python
delay1 = Delay(delay=10)
delay1.run(np.ones((10, 2)))
assert delay1.input_dim == 2
assert np.all(delay1.buffer[0] == 1.0)

delay2 = Delay(delay=10, input_dim=5)
delay2.initialize()
assert delay2.input_dim == 5
assert np.all(delay2.buffer[0] == 0.0)

delay3 = Delay(delay=10, initial_values=np.ones((10, 7)))
delay3.initialize()
assert delay3.input_dim == 7
assert np.all(delay3.buffer[0] == 1.0)

delay_node = Delay(delay=0)

x = np.array([0.2, 0.3])
y = delay_node(x)
assert np.all(x == y)

x = np.linspace(1, 12, num=12).reshape(-1, 2)
y = delay_node.run(x)
assert np.all(x == y)

delay_node = Delay(delay=1)

x1 = np.array([0.2, 0.3])
y = delay_node(x1)
assert np.all(y == 0.0)

x2 = np.linspace(1, 12, num=12).reshape(-1, 2)
y = delay_node.run(x2)
assert np.all(y[0] == x1)
assert np.all(y[1:] == x2[:-1])

# Note: this is quite slow... is deque the best format?
delay_node = Delay(delay=1_000)

x = np.array([0.2, 0.3])
y = delay_node(x)
assert np.all(y == 0.0)
assert np.all(delay_node.buffer[0] == x)
assert np.all(delay_node.buffer[-1] == 0.0)

delay_node.run(np.zeros((999, 2)))
y = delay_node(np.zeros((1, 2)))
assert np.all(y == x)

delay_node = Delay(delay=2)
readout = Ridge(ridge=1e-3)
model = delay_node >> readout

x = list(np.fromfunction(lambda i, j, k: i + j, (2, 4, 2)))
y = list(np.fromfunction(lambda i, j, k: i + j, (2, 4, 1)))

```
# File: test_mat_gen.txt
```python
assert_allclose,
assert_array_almost_equal,
assert_array_equal,
assert_raises,

bernoulli,
fast_spectral_initialization,
generate_input_weights,
generate_internal_weights,
normal,
ones,
random_sparse,
uniform,
zeros,

"shape,dist,connectivity,kwargs,expects",
[
    ((50, 50), "uniform", 0.1, {}, "sparse"),
    ((50, 50), "uniform", 0.1, {"loc": 5.0, "scale": 1.0}, "sparse"),
    ((50, 50), "uniform", 1.0, {}, "dense"),
    ((50, 50), "custom_bernoulli", 0.1, {}, "sparse"),
    ((50, 50, 50), "custom_bernoulli", 0.1, {"p": 0.9}, "dense"),
    ((50, 50), "custom_bernoulli", 1.0, {}, "dense"),
    ((50, 50), "foo", 0.1, {}, "raise"),
    ((50, 50), "uniform", 5.0, {}, "raise"),
    ((50, 50), "uniform", 0.1, {"p": 0.9}, "raise"),
    ((50, 5), "uniform", 0.1, {"degree": 23, "direction": "out"}, "sparse"),
    ((50, 5), "uniform", 0.1, {"degree": 3, "direction": "in"}, "sparse"),
    ((50, 5), "uniform", 0.1, {"degree": 6, "direction": "in"}, "raise"),
    ((50, 5), "uniform", 0.1, {"degree": -1000, "direction": "out"}, "raise"),
],

if expects == "sparse":
    init = random_sparse(dist=dist, connectivity=connectivity, seed=42, **kwargs)
    w0 = init(*shape)
    w1 = random_sparse(
        *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
    )

    w0 = w0.toarray()
    w1 = w1.toarray()

if expects == "dense":
    init = random_sparse(dist=dist, connectivity=connectivity, seed=42, **kwargs)
    w0 = init(*shape)
    w1 = random_sparse(
        *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
    )

if expects == "raise":
    with pytest.raises(Exception):
        init = random_sparse(
            dist=dist, connectivity=connectivity, seed=42, **kwargs
        )
        w0 = init(*shape)
    with pytest.raises(Exception):
        w1 = random_sparse(
            *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
        )
else:
    assert_array_equal(w1, w0)
    if kwargs.get("degree") is None:
        assert_allclose(np.count_nonzero(w0) / w0.size, connectivity, atol=1e-2)
    else:
        dim_length = {"in": shape[0], "out": shape[1]}
        assert (
            np.count_nonzero(w0)
            == kwargs["degree"] * dim_length[kwargs["direction"]]
        )

"shape,sr,input_scaling,kwargs,expects",
[
    ((50, 50), 2.0, None, {"connectivity": 0.1}, "sparse"),
    ((50, 50), None, -2.0, {"connectivity": 1.0}, "dense"),
    ((50, 50), 2.0, None, {"connectivity": 1.0}, "dense"),
    ((50, 50), None, -2.0, {"connectivity": 1.0}, "dense"),
    ((50, 50), None, np.ones((50,)) * 0.1, {"connectivity": 1.0}, "dense"),
    ((50, 50), None, np.ones((50,)) * 0.1, {"connectivity": 0.1}, "sparse"),
    ((50, 50), 2.0, None, {"connectivity": 0.0}, "sparse"),
    ((50, 50), 2.0, -2.0, {"connectivity": 0.1}, "raise"),
],

if expects == "sparse":
    init = random_sparse(
        dist="uniform", sr=sr, input_scaling=input_scaling, seed=42, **kwargs
    )
    w0 = init(*shape)
    w1 = random_sparse(
        *shape,
        dist="uniform",
        sr=sr,
        input_scaling=input_scaling,
        seed=42,
        **kwargs,
    )

    assert_allclose(w1.toarray(), w0.toarray(), atol=1e-12)

if expects == "dense":
    init = random_sparse(
        dist="uniform", sr=sr, input_scaling=input_scaling, seed=42, **kwargs
    )
    w0 = init(*shape)
    w1 = random_sparse(
        *shape,
        dist="uniform",
        sr=sr,
        input_scaling=input_scaling,
        seed=42,
        **kwargs,
    )
    assert_allclose(w1, w0, atol=1e-12)

if expects == "raise":
    with pytest.raises(Exception):
        init = random_sparse(
            dist="uniform", sr=sr, input_scaling=input_scaling, seed=42, **kwargs
        )
        w0 = init(*shape)
        w1 = random_sparse(
            *shape,
            dist="uniform",
            sr=sr,
            input_scaling=input_scaling,
            seed=42,
            **kwargs,
        )

"shape,dtype,sparsity_type,kwargs,expects",
[
    ((50, 50), np.float64, "csr", {"dist": "norm", "connectivity": 0.1}, "sparse"),
    ((50, 50), np.float32, "csc", {"dist": "norm", "connectivity": 0.1}, "sparse"),
    ((50, 50), np.int64, "coo", {"dist": "norm", "connectivity": 0.1}, "sparse"),
    ((50, 50), float, "dense", {"dist": "norm", "connectivity": 0.1}, "dense"),
],

all_sparse_types = {
    "csr": sparse.isspmatrix_csr,
    "coo": sparse.isspmatrix_coo,
    "csc": sparse.isspmatrix_csc,
}

if expects == "sparse":
    init = random_sparse(
        dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
    )
    w0 = init(*shape)
    w1 = random_sparse(
        *shape, dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
    )

    assert_allclose(w1.toarray(), w0.toarray(), atol=1e-12)
    assert w0.dtype == dtype
    assert all_sparse_types[sparsity_type](w0)

if expects == "dense":
    init = random_sparse(
        dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
    )
    w0 = init(*shape)
    w1 = random_sparse(
        *shape, dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
    )

    assert_allclose(w1, w0, atol=1e-12)
    assert w0.dtype == dtype

"initializer,shape,kwargs,expects",
[
    (uniform, (50, 50), {"connectivity": 0.1}, "sparse"),
    (uniform, (50, 50, 50), {"connectivity": 0.1}, "dense"),
    (uniform, (50, 50), {"connectivity": 0.1, "sparsity_type": "dense"}, "dense"),
    (uniform, (50, 50), {"connectivity": 0.1, "high": 5.0, "low": 2.0}, "sparse"),
    (uniform, (50, 50), {"connectivity": 0.1, "high": 5.0, "low": "a"}, "raise"),
    (normal, (50, 50), {"connectivity": 0.1}, "sparse"),
    (normal, (50, 50, 50), {"connectivity": 0.1}, "dense"),
    (normal, (50, 50), {"connectivity": 0.1, "sparsity_type": "dense"}, "dense"),
    (normal, (50, 50), {"connectivity": 0.1, "loc": 5.0, "scale": 2.0}, "sparse"),
    (normal, (50, 50), {"connectivity": 0.1, "loc": 5.0, "scale": "a"}, "raise"),
    (bernoulli, (50, 50), {"connectivity": 0.1}, "sparse"),
    (bernoulli, (50, 50, 50), {"connectivity": 0.1}, "dense"),
    (bernoulli, (50, 50), {"connectivity": 0.1, "sparsity_type": "dense"}, "dense"),
    (bernoulli, (50, 50), {"connectivity": 0.1, "p": 0.9}, "sparse"),
    (bernoulli, (50, 50), {"connectivity": 0.1, "p": 5.0}, "raise"),
],

if expects == "sparse":
    init = initializer(seed=42, **kwargs)
    w0 = init(*shape)
    w1 = initializer(*shape, seed=42, **kwargs)

    assert_allclose(w1.toarray(), w0.toarray(), atol=1e-12)

if expects == "dense":
    init = initializer(seed=42, **kwargs)
    w0 = init(*shape)
    w1 = initializer(*shape, seed=42, **kwargs)
    assert_allclose(w1, w0, atol=1e-12)

if expects == "raise":
    with pytest.raises(Exception):
        init = initializer(seed=42, **kwargs)
        w0 = init(*shape)
        w1 = initializer(*shape, seed=42, **kwargs)

w = ones(50, 50)
assert_allclose(w, 1.0)

w = ones(50, 50, dtype=np.float32)
assert_allclose(w, 1.0)
assert w.dtype == np.float32

w = zeros(50, 50)
assert_allclose(w, 0.0)

w = zeros(50, 50, dtype=np.float32)
assert_allclose(w, 0.0)
assert w.dtype == np.float32

with pytest.raises(ValueError):
    w = zeros(50, 50, sr=2.0)

"N,dim_input,input_bias,expected",
[
    (100, 20, False, (100, 20)),
    (100, 20, True, (100, 21)),
    (20, 100, True, (20, 101)),
],

with pytest.warns(DeprecationWarning):
    Win = generate_input_weights(N, dim_input, input_bias=input_bias)

assert Win.shape == expected

"N,dim_input,input_bias",
[
    (-1, 10, True),
    (100, -5, False),
],

with pytest.warns(DeprecationWarning):
    with pytest.raises(ValueError):
        generate_input_weights(N, dim_input, input_bias=input_bias)

with pytest.warns(DeprecationWarning):
    Win = generate_input_weights(100, 20, input_scaling=iss, proba=proba, seed=1234)

    with pytest.warns(DeprecationWarning):
        Win_noiss = generate_input_weights(
            100, 20, input_scaling=1.0, proba=proba, seed=1234
        )

        if sparse.issparse(Win):
            result_proba = np.count_nonzero(Win.toarray()) / Win.toarray().size
        else:
            result_proba = np.count_nonzero(Win) / Win.size

        assert_allclose(result_proba, proba, rtol=1e-2)

        if sparse.issparse(Win):
            assert_allclose(Win.toarray() / iss, Win_noiss.toarray(), rtol=1e-4)
        else:
            assert_allclose(Win / iss, Win_noiss, rtol=1e-4)

with pytest.warns(DeprecationWarning):
    with pytest.raises(Exception):
        generate_input_weights(100, 20, input_scaling=iss, proba=proba)

"N,expected", [(100, (100, 100)), (-1, Exception), ("foo", Exception)]

if expected is Exception:
    with pytest.raises(expected):
        with pytest.warns(DeprecationWarning):
            generate_internal_weights(N)
else:
    with pytest.warns(DeprecationWarning):
        W = generate_internal_weights(N)
    assert W.shape == expected

"sr,proba",
[
    (0.5, 0.1),
    (2.0, 1.0),
],

with pytest.warns(DeprecationWarning):
    W = generate_internal_weights(
        100, sr=sr, proba=proba, seed=1234, sparsity_type="dense"
    )

    assert_allclose(max(abs(linalg.eig(W)[0])), sr)
    assert_allclose(np.sum(W != 0.0) / W.size, proba)

with pytest.warns(DeprecationWarning):
    W = generate_internal_weights(
        100, sr=sr, proba=proba, sparsity_type="csr", seed=42
    )

    rho = max(
        abs(
            sparse.linalg.eigs(
                W,
                k=1,
                which="LM",
                maxiter=20 * W.shape[0],
                return_eigenvectors=False,
            )
        )
    )
    assert_allclose(rho, sr)

    if sparse.issparse(W):
        assert_allclose(np.sum(W.toarray() != 0.0) / W.toarray().size, proba)
    else:
        assert_allclose(np.sum(W != 0.0) / W.size, proba)

with pytest.warns(DeprecationWarning):
    with pytest.raises(Exception):
        generate_internal_weights(100, sr=sr, proba=proba)

"N,expected", [(100, (100, 100)), (-1, Exception), ("foo", Exception)]

if expected is Exception:
    with pytest.raises(expected):
        fast_spectral_initialization(N)
else:
    W = fast_spectral_initialization(N)
    assert W.shape == expected

W = fast_spectral_initialization(1000, sr=sr, connectivity=proba, seed=1234)

if sparse.issparse(W):
    rho = max(
        abs(
            sparse.linalg.eigs(
                W,
                k=1,
                which="LM",
                maxiter=20 * W.shape[0],
                return_eigenvectors=False,
            )
        )
    )
else:
    rho = max(abs(linalg.eig(W)[0]))

if proba == 0.0:
    assert_allclose(rho, 0.0)
else:
    assert_allclose(rho, sr, rtol=1e-1)

if 1.0 - proba < 1e-5:
    assert not sparse.issparse(W)
if sparse.issparse(W):
    assert_allclose(
        np.count_nonzero(W.toarray()) / W.toarray().size, proba, rtol=1e-1
    )
else:
    assert_allclose(np.count_nonzero(W) / W.size, proba, rtol=1e-1)

with pytest.raises(Exception):
    fast_spectral_initialization(100, sr=sr, connectivity=proba)

with pytest.raises(ValueError):
    fast_spectral_initialization(100, input_scaling=10.0, connectivity=proba)

seed0 = default_rng(78946312)
with pytest.warns(DeprecationWarning):
    W0 = generate_internal_weights(
        N=100, sr=1.2, proba=0.4, dist="uniform", loc=-1, scale=2, seed=seed0
    ).toarray()

seed1 = default_rng(78946312)
with pytest.warns(DeprecationWarning):
    W1 = generate_internal_weights(
        N=100, sr=1.2, proba=0.4, dist="uniform", loc=-1, scale=2, seed=seed1
    ).toarray()

seed2 = default_rng(6135435)
with pytest.warns(DeprecationWarning):
    W2 = generate_internal_weights(
        N=100, sr=1.2, proba=0.4, dist="uniform", loc=-1, scale=2, seed=seed2
    ).toarray()

assert_array_almost_equal(W0, W1)
assert_raises(AssertionError, assert_array_almost_equal, W0, W2)

seed0 = default_rng(78946312)
with pytest.warns(DeprecationWarning):
    W0 = generate_input_weights(100, 50, input_scaling=1.2, proba=0.4, seed=seed0)

seed1 = default_rng(78946312)
with pytest.warns(DeprecationWarning):
    W1 = generate_input_weights(100, 50, input_scaling=1.2, proba=0.4, seed=seed1)

seed2 = default_rng(6135435)
with pytest.warns(DeprecationWarning):
    W2 = generate_input_weights(100, 50, input_scaling=1.2, proba=0.4, seed=seed2)

assert_allclose(W0.toarray(), W1.toarray())

with pytest.raises(AssertionError):
    assert_allclose(W0.toarray(), W2.toarray())

seed0 = default_rng(78946312)
W0 = fast_spectral_initialization(
    100, sr=1.2, connectivity=0.4, seed=seed0
).toarray()

seed1 = default_rng(78946312)
W1 = fast_spectral_initialization(
    100, sr=1.2, connectivity=0.4, seed=seed1
).toarray()

seed2 = default_rng(6135435)
W2 = fast_spectral_initialization(
    100, sr=1.2, connectivity=0.4, seed=seed2
).toarray()

assert_array_almost_equal(W0, W1)
assert_raises(AssertionError, assert_array_almost_equal, W0, W2)

```
# File: esn_jax.txt
```python
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

```
# File: rls copy.txt
```python
_assemble_wout,
_compute_error,
_initialize_readout,
_prepare_inputs_for_learning,
_split_and_save_wout,
readout_forward,

"""Recursive Least Squares learning rule."""
k = np.dot(P, r)
rPr = np.dot(r.T, k).squeeze()
c = float(1.0 / (1.0 + rPr))
P = P - c * np.outer(k, k)

dw = -c * np.outer(e, k)

return dw, P

"""Train a readout using RLS learning rule."""
x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias, allow_reshape=True)

error, r = _compute_error(node, x, y)

P = node.P
dw, P = _rls(P, r, error)
wo = _assemble_wout(node.Wout, node.bias, node.input_bias)
wo = wo + dw.T

_split_and_save_wout(node, wo)

node.set_param("P", P)

readout: "RLS", x=None, y=None, init_func=None, bias_init=None, bias=None

_initialize_readout(readout, x, y, init_func, bias_init, bias)

if x is not None:
    input_dim, alpha = readout.input_dim, readout.alpha

    if readout.input_bias:
        input_dim += 1

    P = np.eye(input_dim) / alpha

    readout.set_param("P", P)

"""Single layer of neurons learning connections using Recursive Least Squares
algorithm.

The learning rules is well described in [1]_.

:py:attr:`RLS.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
``P``              Matrix :math:`\\mathbf{P}` of RLS rule.
================== =================================================================

:py:attr:`RLS.hypers` **list**

================== =================================================================
``alpha``          Diagonal value of matrix P (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by default).
``input_bias``     If True, learn a bias term (True by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
alpha : float or Python generator or iterable, default to 1e-6
    Diagonal value of matrix P.
Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

References
----------

.. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
        Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
        https://doi.org/10.1016/j.neuron.2009.07.018

Examples
--------
>>> x = np.random.normal(size=(100, 3))
>>> noise = np.random.normal(scale=0.1, size=(100, 1))
>>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.

>>> from reservoirpy.nodes import RLS
>>> rls_node = RLS(alpha=1e-1)

>>> rls_node.train(x[:5], y[:5])
>>> print(rls_node.Wout.T, rls_node.bias)
[[ 9.90731641 -0.06884784  6.87944632]] [[12.07802068]]
>>> rls_node.train(x[5:], y[5:])
>>> print(rls_node.Wout.T, rls_node.bias)
[[ 9.99223366 -0.20499636  6.98924066]] [[12.01128622]]
"""

```
# File: performances.txt
```python







print(units)

rpy_model = rpy.nodes.ESN(

    units=units,

    connectivity=CONNECTIVITY,

    input_connectivity=INPUT_CONNECTIVITY,

    ridge=1,

)

jax_model = esn_jax.ESN(

    units=units,

    connectivity=CONNECTIVITY,

    input_connectivity=INPUT_CONNECTIVITY,

    ridge=1,

)

start = time.time()

rpy_model.fit(x, y)

fit_time = time.time()

rpy_model.run(x)

stop = time.time()

print(stop - start)

fit_times[i, 0] = fit_time - start

run_times[i, 0] = stop - fit_time

start = time.time()

jax_model.fit(x, y)

fit_time = time.time()

jax_model.run(x)

stop = time.time()

print(stop - start)

fit_times[i, 1] = fit_time - start

run_times[i, 1] = stop - fit_time




f"timesteps={TIMESTEPS}, connectivity={CONNECTIVITY}, input_connectivity={INPUT_CONNECTIVITY}"



f"timesteps={TIMESTEPS}, connectivity={CONNECTIVITY}, input_connectivity={INPUT_CONNECTIVITY}"



print(units)

X = np.random.normal(size=(TIMESTEPS, units))

Y = np.random.normal(size=(TIMESTEPS, 1))



ridge_node = rpy.nodes.Ridge(ridge=1,input_bias=False)

start = time.time()

ridge_node.fit(X, Y)

stop = time.time()

print(stop - start)

times[i, 0] = stop - start

start = time.time()

esn_jax.ESN._ridge_regression(1, X, Y)

stop = time.time()

times[i, 1] = stop - start

print(stop - start)

start = time.time()

jax.jit(esn_jax.ESN._ridge_regression)(1, X, Y).block_until_ready()

stop = time.time()

times[i, 2] = stop - start

print(stop - start)





print(units)

rpy_model = rpy.nodes.ESN(

    units=units,

    connectivity=5/units,

    input_connectivity=5/units,

    ridge=1,

)

jax_model = esn_jax.ESN(

    units=units,

    connectivity=5/units,

    input_connectivity=5/units,

    ridge=1,

)

start = time.time()

rpy_model.fit(x, y)

fit_time = time.time()

rpy_model.run(x)

stop = time.time()

print(stop - start)

fit_times[i, 0] = fit_time - start

run_times[i, 0] = stop - fit_time

start = time.time()

jax_model.fit(x, y)

fit_time = time.time()

jax_model.run(x)

stop = time.time()

print(stop - start)

fit_times[i, 1] = fit_time - start

run_times[i, 1] = stop - fit_time



f"timesteps={TIMESTEPS}, connectivity={CONNECTIVITY}, degree={5}"





print(units)

jax_model = esn_jax.ESN(

    units=units,

    connectivity=5/units,

    input_connectivity=5/units,

    ridge=1,

)

start = time.time()

jax_model.fit(x, y)

stop = time.time()

print(stop - start)

times[i] = stop - start



raise ValueError(f"Invalid {n_batch=}, {n_dense=} for {shape=}")

tuple, split_list(shape, [n_batch, n_sparse])

raise ValueError(f"got {nse=}, expected to be between 0 and {sparse_size}")

nse = int(math.ceil(nse * sparse_size))

indices_dtype = dtypes.canonicalize_dtype(jnp.int_)

raise ValueError(

    f"{indices_dtype=} does not have enough range to generate "

    f"sparse indices of size {sparse_size}."

)

if not sparse_shape:

    return jnp.empty((nse, n_sparse), dtype=indices_dtype)

flat_ind = random.choice(

    key, sparse_size, shape=(nse,), replace=not unique_indices

).astype(indices_dtype)

return jnp.column_stack(jnp.unravel_index(flat_ind, sparse_shape))



if not sparse_shape:

    return jnp.empty((nse, n_sparse), dtype=jnp.float32)

print("XDDDDDDD")

flat_ind = random.choice(

    key, sparse_size, shape=(nse,), replace=not unique_indices

).astype(dtypes.canonicalize_dtype(jnp.int_))

print("AAAAAAAA")

unraveled = jnp.unravel_index(flat_ind, sparse_shape)

print("EEEEEEEE")

return jnp.column_stack(unraveled)





key=jax.random.PRNGKey(seed=1),

shape=(31622, 31622),

dtype=np.float32,

indices_dtype=int,

nse=100,

n_batch=0,

n_dense=0,

unique_indices=False,

generator=jax.random.normal,



W = a*sparse.random_bcoo(

    key=jax.random.PRNGKey(seed=1),

    shape=(31622, 31622),

    dtype=np.float32,

    indices_dtype=int,

    nse=0.001,

    generator=jax.random.normal,

)

Win = sparse.random_bcoo(

    key=jax.random.PRNGKey(seed=1),

    shape=(1, 31622),

    dtype=np.float32,

    indices_dtype=int,

    nse=0.001,

    generator=jax.random.normal,

)

return 1. * Win * W * Win.T




```
# File: add.txt
```python
if len(data) > 1:
    data = np.squeeze(data)
else:  # nothing to add
    data = data[0]
return np.sum(data, axis=0).reshape(1, -1)

if x is not None:
    # x is an array, add over axis 0
    if isinstance(x, np.ndarray):
        add.set_input_dim(x.shape)
        add.set_output_dim((1, x.shape[1]))

    elif is_sequence_set(x):
        shapes = [array.shape for array in x]

        if not all([s[0] == 1 for s in shapes]):
            raise ValueError(
                f"Each timestep of data must be represented "
                f"by a vector of shape (1, dimension) when "
                f"entering node {add.name}. Received inputs "
                f"of shape {shapes}."
            )

        add.set_input_dim((len(x), x[0].shape[1]))

        if len(set([s[1] for s in shapes])) > 1:
            raise ValueError(
                f"Impossible to sum inputs: inputs have "
                f"different dimensions  entering node "
                f"{add.name}. Received inputs of shape "
                f"{shapes}."
            )
        else:
            add.set_output_dim((1, x[0].shape[1]))

```
# File: base.txt
```python
readout, x=None, y=None, init_func=None, bias_init=None, bias=True

if x is not None:

    in_dim = x.shape[1]

    if readout.output_dim is not None:
        out_dim = readout.output_dim
    elif y is not None:
        out_dim = y.shape[1]
    else:
        raise RuntimeError(
            f"Impossible to initialize {readout.name}: "
            f"output dimension was not specified at "
            f"creation, and no teacher vector was given."
        )

    readout.set_input_dim(in_dim)
    readout.set_output_dim(out_dim)

    if callable(init_func):
        W = init_func(in_dim, out_dim, dtype=readout.dtype)
    elif isinstance(init_func, np.ndarray):
        W = (
            check_vector(init_func, caller=readout)
            .reshape(readout.input_dim, readout.output_dim)
            .astype(readout.dtype)
        )
    else:
        raise ValueError(
            f"Data type {type(init_func)} not "
            f"understood for matrix initializer "
            f"'Wout'. It should be an array or "
            f"a callable returning an array."
        )

    if bias:
        if callable(bias_init):
            bias = bias_init(1, out_dim, dtype=readout.dtype)
        elif isinstance(bias_init, np.ndarray):
            bias = (
                check_vector(bias_init)
                .reshape(1, readout.output_dim)
                .astype(readout.dtype)
            )
        else:
            raise ValueError(
                f"Data type {type(bias_init)} not "
                f"understood for matrix initializer "
                f"'bias'. It should be an array or "
                f"a callable returning an array."
            )
    else:
        bias = np.zeros((1, out_dim), dtype=readout.dtype)

    readout.set_param("Wout", W)
    readout.set_param("bias", bias)

if X is not None:

    if bias:
        X = add_bias(X)
    if not isinstance(X, np.ndarray):
        X = np.vstack(X)

    X = check_vector(X, allow_reshape=allow_reshape)

if Y is not None:

    if not isinstance(Y, np.ndarray):
        Y = np.vstack(Y)

    Y = check_vector(Y, allow_reshape=allow_reshape)

return X, Y

return (node.Wout.T @ x.reshape(-1, 1) + node.bias.T).T

wo = Wout
if has_bias:
    wo = np.r_[bias, wo]
return wo

if node.input_bias:
    Wout, bias = wo[1:, :], wo[0, :][np.newaxis, :]
    node.set_param("Wout", Wout)
    node.set_param("bias", bias)
else:
    node.set_param("Wout", wo)

```
# File: intrinsic_plasticity.txt
```python
from typing_extensions import Literal

from typing import Literal

"""KL loss gradients of neurons with tanh activation (~ Normal(mu, sigma))."""
sig2 = sigma**2
delta_b = -eta * (-(mu / sig2) + (y / sig2) * (2 * sig2 + 1 - y**2 + mu * y))
delta_a = (eta / a) + delta_b * x
return delta_a, delta_b

"""KL loss gradients of neurons with sigmoid activation
(~ Exponential(lambda=1/mu))."""
delta_b = eta * (1 - (2 + (1 / mu)) * y + (y**2) / mu)
delta_a = (eta / a) + delta_b * x
return delta_a, delta_b

"""Apply gradients on a and b parameters of intrinsic plasticity."""
a2 = a + delta_a
b2 = b + delta_b
return a2, b2

"""Perform one step of intrinsic plasticity.

Optimize a and b such that
post_state = f(a*pre_state+b) ~ Dist(params) where Dist can be normal or
exponential."""
a = reservoir.a
b = reservoir.b
mu = reservoir.mu
eta = reservoir.learning_rate

if reservoir.activation_type == "tanh":
    sigma = reservoir.sigma
    delta_a, delta_b = gaussian_gradients(
        x=pre_state.T, y=post_state.T, a=a, mu=mu, sigma=sigma, eta=eta
    )
else:  # sigmoid
    delta_a, delta_b = exp_gradients(
        x=pre_state.T, y=post_state.T, a=a, mu=mu, eta=eta
    )

return apply_gradients(a=a, b=b, delta_a=delta_a, delta_b=delta_b)

"""Activation of neurons f(a*x+b) where a and b are intrinsic plasticity
parameters."""
a, b = reservoir.a, reservoir.b
return f(a * state + b)

for e in range(reservoir.epochs):
    for seq in X:
        for u in seq:
            post_state = reservoir.call(u.reshape(1, -1))
            pre_state = reservoir.internal_state

            a, b = ip(reservoir, pre_state, post_state)

            reservoir.set_param("a", a)
            reservoir.set_param("b", b)

initialize_base(reservoir, *args, **kwargs)

a = np.ones((reservoir.output_dim, 1))
b = np.zeros((reservoir.output_dim, 1))

reservoir.set_param("a", a)
reservoir.set_param("b", b)

"""Pool of neurons with random recurrent connexions, tuned using Intrinsic
Plasticity.

Intrinsic Plasticity is applied as described in [1]_ and [2]_.

Reservoir neurons states, gathered in a vector :math:`\\mathbf{x}`, follow
the update rule below:

.. math::

    \\mathbf{r}[t+1] = (1 - \\mathrm{lr}) * \\mathbf{r}[t] + \\mathrm{lr}
    * (\\mathbf{W}_{in} \\cdot (\\mathbf{u}[t+1]+c_{in}*\\xi)
        + \\mathbf{W} \\cdot \\mathbf{x}[t]
    + \\mathbf{W}_{fb} \\cdot (g(\\mathbf{y}[t])+c_{fb}*\\xi) + \\mathbf{b}_{in})

.. math::

    \\mathbf{x}[t+1] = f(\\mathbf{a}*\\mathbf{r}[t+1]+\\mathbf{b}) + c * \\xi

Parameters :math:`\\mathbf{a}` and :math:`\\mathbf{b}` are updated following two
different rules:

- **1.** Neuron activation is tanh:

In that case, output distribution should be a Gaussian distribution of parameters
(:math:`\\mu`, :math:`\\sigma`). The learning rule to obtain this output
distribution is described in [2]_.

- **2.** Neuron activation is sigmoid:

In that case, output distribution should be an exponential distribution of
parameter :math:`\\mu = \\frac{1}{\\lambda}`.
The learning rule to obtain this output distribution is described in [1]_ and [2]_.

where:
    - :math:`\\mathbf{x}` is the output activation vector of the reservoir;
    - :math:`\\mathbf{r}` is the internal activation vector of the reservoir;
    - :math:`\\mathbf{u}` is the input timeseries;
    - :math:`\\mathbf{y}` is a feedback vector;
    - :math:`\\xi` is a random noise;
    - :math:`f` and :math:`g` are activation functions.

:py:attr:`IPReservoir.params` **list:**

================== =================================================================
``W``              Recurrent weights matrix (:math:`\\mathbf{W}`).
``Win``            Input weights matrix (:math:`\\mathbf{W}_{in}`).
``Wfb``            Feedback weights matrix (:math:`\\mathbf{W}_{fb}`).
``bias``           Input bias vector (:math:`\\mathbf{b}_{in}`).
``internal_state``  Internal state (:math:`\\mathbf{r}`).
``a``              Gain of reservoir activation (:math:`\\mathbf{a}`).
``b``              Bias of reservoir activation (:math:`\\mathbf{b}`).
================== =================================================================

:py:attr:`IPReservoir.hypers` **list:**

======================= ========================================================
``lr``                  Leaking rate (1.0 by default) (:math:`\\mathrm{lr}`).
``sr``                  Spectral radius of ``W`` (optional).
``mu``                  Mean of the target distribution (0.0 by default) (:math:`\\mu`).
``sigma``               Variance of the target distribution (1.0 by default) (:math:`\\sigma`).
``learning_rate``       Learning rate (5e-4 by default).
``epochs``              Number of epochs for training (1 by default).
``input_scaling``       Input scaling (float or array) (1.0 by default).
``fb_scaling``          Feedback scaling (float or array) (1.0 by default).
``rc_connectivity``     Connectivity (or density) of ``W`` (0.1 by default).
``input_connectivity``  Connectivity (or density) of ``Win`` (0.1 by default).
``fb_connectivity``     Connectivity (or density) of ``Wfb`` (0.1 by default).
``noise_in``            Input noise gain (0 by default) (:math:`c_{in} * \\xi`).
``noise_rc``            Reservoir state noise gain (0 by default) (:math:`c*\\xi`).
``noise_fb``            Feedback noise gain (0 by default) (:math:`c_{fb}*\\xi`).
``noise_type``          Distribution of noise (normal by default) (:math:`\\xi\\sim\\mathrm{Noise~type}`).
``activation``          Activation of the reservoir units (tanh by default) (:math:`f`).
``fb_activation``       Activation of the feedback units (identity by default) (:math:`g`).
``units``               Number of neuronal units in the reservoir.
``noise_generator``     A random state generator.
======================= ========================================================

Parameters
----------
units : int, optional
    Number of reservoir units. If None, the number of units will be inferred from
    the ``W`` matrix shape.
lr : float or array-like of shape (units,), default to 1.0
    Neurons leak rate. Must be in :math:`[0, 1]`.
sr : float, optional
    Spectral radius of recurrent weight matrix.
mu : float, default to 0.0
    Mean of the target distribution.
sigma : float, default to 1.0
    Variance of the target distribution.
learning_rate : float, default to 5e-4
    Learning rate.
epochs : int, default to 1
    Number of training iterations.
input_bias : bool, default to True
    If False, no bias is added to inputs.
noise_rc : float, default to 0.0
    Gain of noise applied to reservoir activations.
noise_in : float, default to 0.0
    Gain of noise applied to input inputs.
noise_fb : float, default to 0.0
    Gain of noise applied to feedback signal.
noise_type : str, default to "normal"
    Distribution of noise. Must be a Numpy random variable generator
    distribution (see :py:class:`numpy.random.Generator`).
noise_kwargs : dict, optional
    Keyword arguments to pass to the noise generator, such as `low` and `high`
    values of uniform distribution.
input_scaling : float or array-like of shape (features,), default to 1.0.
    Input gain. An array of the same dimension as the inputs can be used to
    set up different input scaling for each feature.
bias_scaling: float, default to 1.0
    Bias gain.
fb_scaling : float or array-like of shape (features,), default to 1.0
    Feedback gain. An array of the same dimension as the feedback can be used to
    set up different feedback scaling for each feature.
input_connectivity : float, default to 0.1
    Connectivity of input neurons, i.e. ratio of input neurons connected
    to reservoir neurons. Must be in :math:`]0, 1]`.
rc_connectivity : float, default to 0.1
    Connectivity of recurrent weight matrix, i.e. ratio of reservoir
    neurons connected to other reservoir neurons, including themselves.
    Must be in :math:`]0, 1]`.
fb_connectivity : float, default to 0.1
    Connectivity of feedback neurons, i.e. ratio of feedback neurons
    connected to reservoir neurons. Must be in :math:`]0, 1]`.
Win : callable or array-like of shape (units, features), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
    Input weights matrix or initializer. If a callable (like a function) is
    used,
    then this function should accept any keywords
    parameters and at least two parameters that will be used to define the
    shape of
    the returned weight matrix.
W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.uniform`
    Recurrent weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the
    shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the
    shape of
    the returned weight matrix.
Wfb : callable or array-like of shape (units, feedback), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
    Feedback weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the
    shape of
    the returned weight matrix.
fb_activation : str or callable, default to :py:func:`~reservoirpy.activationsfunc.identity`
    Feedback activation function.
    - If a str, should be a :py:mod:`~reservoirpy.activationsfunc`
    function name.
    - If a callable, should be an element-wise operator on arrays.
activation : {"tanh", "sigmoid"}, default to "tanh"
    Reservoir units activation function.
feedback_dim : int, optional
    Feedback dimension. Can be inferred at first call.
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
seed : int or :py:class:`numpy.random.Generator`, optional
    A random state seed, for noise generation.

References
----------
.. [1] Triesch, J. (2005). A Gradient Rule for the Plasticity of a
        Neuron’s Intrinsic Excitability. In W. Duch, J. Kacprzyk,
        E. Oja, & S. Zadrożny (Eds.), Artificial Neural Networks:
        Biological Inspirations – ICANN 2005 (pp. 65–70).
        Springer. https://doi.org/10.1007/11550822_11

.. [2] Schrauwen, B., Wardermann, M., Verstraeten, D., Steil, J. J.,
        & Stroobandt, D. (2008). Improving reservoirs using intrinsic
        plasticity. Neurocomputing, 71(7), 1159–1171.
        https://doi.org/10.1016/j.neucom.2007.12.020

Example
-------
>>> from reservoirpy.nodes import IPReservoir
>>> reservoir = IPReservoir(
...                 100, mu=0.0, sigma=0.1, sr=0.95, activation="tanh", epochs=10)

We can fit the intrinsic plasticity parameters to reach a normal distribution
of the reservoir activations.
Using the :py:func:`~reservoirpy.datasets.narma` timeseries:

>>> from reservoirpy.datasets import narma
>>> x = narma(1000)
>>> _ = reservoir.fit(x, warmup=100)
>>> states = reservoir.run(x)

.. plot:: ./api/intrinsic_plasticity_example.py

"""

def __init__(
    self,
    units: int = None,
    sr: Optional[float] = None,
    lr: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    learning_rate: float = 5e-4,
    epochs: int = 1,
    input_bias: bool = True,
    noise_rc: float = 0.0,
    noise_in: float = 0.0,
    noise_fb: float = 0.0,
    noise_type: str = "normal",
    noise_kwargs: Dict = None,
    input_scaling: Union[float, Sequence] = 1.0,
    bias_scaling: float = 1.0,
    fb_scaling: Union[float, Sequence] = 1.0,
    input_connectivity: Optional[float] = 0.1,
    rc_connectivity: Optional[float] = 0.1,
    fb_connectivity: Optional[float] = 0.1,
    Win: Union[Weights, Callable] = bernoulli,
    W: Union[Weights, Callable] = uniform,
    Wfb: Union[Weights, Callable] = bernoulli,
    bias: Union[Weights, Callable] = bernoulli,
    feedback_dim: int = None,
    fb_activation: Union[str, Callable] = identity,
    activation: Literal["tanh", "sigmoid"] = "tanh",
    name=None,
    seed=None,
    **kwargs,
):
    if units is None and not is_array(W):
        raise ValueError(
            "'units' parameter must not be None if 'W' parameter is not "
            "a matrix."
        )

    if activation not in ["tanh", "sigmoid"]:
        raise ValueError(
            f"Activation '{activation}' must be 'tanh' or 'sigmoid' when "
            "applying intrinsic plasticity."
        )

    rng = rand_generator(seed=seed)
    noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

    super(IPReservoir, self).__init__(
        fb_initializer=partial(
            initialize_feedback,
            Wfb_init=Wfb,
            fb_scaling=fb_scaling,
            fb_connectivity=fb_connectivity,
            seed=seed,
        ),
        params={
            "W": None,
            "Win": None,
            "Wfb": None,
            "bias": None,
            "a": None,
            "b": None,
            "internal_state": None,
        },
        hypers={
            "sr": sr,
            "lr": lr,
            "mu": mu,
            "sigma": sigma,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "input_bias": input_bias,
            "input_scaling": input_scaling,
            "fb_scaling": fb_scaling,
            "rc_connectivity": rc_connectivity,
            "input_connectivity": input_connectivity,
            "fb_connectivity": fb_connectivity,
            "noise_in": noise_in,
            "noise_rc": noise_rc,
            "noise_out": noise_fb,
            "noise_type": noise_type,
            "activation_type": activation,
            "activation": partial(
                ip_activation, reservoir=self, f=get_function(activation)
            ),
            "fb_activation": fb_activation,
            "units": units,
            "noise_generator": partial(noise, rng=rng, **noise_kwargs),
        },
        forward=forward_external,
        initializer=partial(
            initialize,
            input_bias=input_bias,
            bias_scaling=bias_scaling,
            sr=sr,
            input_scaling=input_scaling,
            input_connectivity=input_connectivity,
            rc_connectivity=rc_connectivity,
            W_init=W,
            Win_init=Win,
            bias_init=bias,
            seed=seed,
        ),
        backward=backward,
        output_dim=units,
        feedback_dim=feedback_dim,
        name=name,
        **kwargs,
    )

# TODO: handle unsupervised learners with a specific attribute
@property
def fitted(self):
    return True

def partial_fit(self, X_batch, Y_batch=None, warmup=0, **kwargs) -> "Node":
    """Partial offline fitting method of a Node.
    Can be used to perform batched fitting or to pre-compute some variables
    used by the fitting method.

    Parameters
    ----------
    X_batch : array-like of shape ([series], timesteps, features)
        A sequence or a batch of sequence of input data.
    Y_batch : array-like of shape ([series], timesteps, features), optional
        A sequence or a batch of sequence of teacher signals.
    warmup : int, default to 0
        Number of timesteps to consider as warmup and
        discard at the beginning of each timeseries before training.

    Returns
    -------
        Node
            Partially fitted Node.
    """
    X, _ = check_xy(self, X_batch, allow_n_inputs=False)

    X, _ = _init_with_sequences(self, X)

    self.initialize_buffers()

    for i in range(len(X)):
        X_seq = X[i]

        if X_seq.shape[0] <= warmup:
            raise ValueError(
                f"Warmup set to {warmup} timesteps, but one timeseries is only "
                f"{X_seq.shape[0]} long."
            )

        if warmup > 0:
            self.run(X_seq[:warmup])

        self._partial_backward(self, X_seq[warmup:])

```
# File: nvar.txt
```python
from scipy.special import comb

from math import comb

store = node.store
strides = node.strides
idxs = node._monomial_idx

# store the current input
new_store = np.roll(store, 1, axis=0)
new_store[0] = x
node.set_param("store", new_store)

output = np.zeros((node.output_dim, 1))

# select all previous inputs, including the current, with strides
linear_feats = np.ravel(new_store[::strides, :]).reshape(-1, 1)
linear_len = linear_feats.shape[0]

output[:linear_len, :] = linear_feats

# select monomial terms and compute them
output[linear_len:, :] = np.prod(linear_feats[idxs.astype(int)], axis=1)

return output.reshape(1, -1)

if x is not None:
    input_dim = x.shape[1]

    order = node.order
    delay = node.delay
    strides = node.strides

    linear_dim = delay * input_dim
    # number of non linear components is (d + n - 1)! / (d - 1)! n!
    # i.e. number of all unique monomials of order n made from the
    # linear components.
    nonlinear_dim = comb(linear_dim + order - 1, order)

    output_dim = int(linear_dim + nonlinear_dim)

    node.set_output_dim(output_dim)
    node.set_input_dim(input_dim)

    # for each monomial created in the non linear part, indices
    # of the n components involved, n being the order of the
    # monomials. Pre-compute them to improve efficiency.
    idx = np.array(
        list(it.combinations_with_replacement(np.arange(linear_dim), order))
    )

    node.set_param("_monomial_idx", idx)

    # to store the k*s last inputs, k being the delay and s the strides
    node.set_param("store", np.zeros((delay * strides, node.input_dim)))

"""Non-linear Vector AutoRegressive machine.

NVAR is implemented as described in [1]_.

The state :math:`\\mathbb{O}_{total}` of the NVAR first contains a series of linear
features :math:`\\mathbb{O}_{lin}` made of input data concatenated
with delayed inputs:

.. math::

    \\mathbb{O}_{lin}[t] = \\mathbf{X}[t] \\oplus \\mathbf{X}[t - s] \\oplus
    \\mathbf{X}[t - 2s] \\oplus \\dots \\oplus \\mathbf{X}[t - (k-1)s]

where :math:`\\mathbf{X}[t]` are the inputs at time :math:`t`, :math:`k` is the
delay and :math:`s` is the strides (only one input every :math:`s`
inputs within the delayed inputs is used).
The operator :math:`\\oplus` denotes the concatenation.

In addition to these linear features, nonlinear representations
:math:`\\mathbb{O}_{nonlin}^n` of the inputs are constructed using all unique
monomials of order :math:`n` of these inputs:

.. math::

    \\mathbb{O}_{nonlin}^n[t] = \\mathbb{O}_{lin}[t] \\otimes \\mathbb{O}_{lin}[t]
    \\overbrace{\\otimes \\dots \\otimes}^{n-1~\\mathrm{times}} \\mathbb{O}_{lin}[t]

where :math:`\\otimes` is the operator denoting an outer product followed by the
selection of all unique monomials generated by this outer product.

Note
----

    Under the hood,
    this product is computed by finding all unique combinations
    of input features and multiplying each combination of terms.

Finally, all representations are gathered to form the final feature
vector :math:`\\mathbb{O}_{total}`:

.. math::

    \\mathbb{O}_{total} = \\mathbb{O}_{lin}[t] \\oplus \\mathbb{O}_{nonlin}^n[t]

:py:attr:`NVAR.params` **list:**

================== ===================================================================
``store``          Time window over the inputs (of shape (delay * strides, features)).
================== ===================================================================

:py:attr:`NVAR.hypers` **list:**

================== =================================================================
``delay``          Maximum delay of inputs (:math:`k`).
``order``          Order of the non-linear monomials (:math:`n`).
``strides``        Strides between delayed inputs, by default 1 (:math:`s`).
================== =================================================================

Parameters
----------
delay : int
    Maximum delay of inputs.
order : int
    Order of the non-linear monomials.
strides : int, default to 1
    Strides between delayed inputs.
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.

References
----------
.. [1] Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
        Next generation reservoir computing. Nature Communications, 12(1), 5564.
        https://doi.org/10.1038/s41467-021-25801-2

Example
-------

>>> import numpy as np
>>> from reservoirpy.nodes import NVAR, Ridge
>>> nvar = NVAR(delay=2, order=2, strides=1)
>>> readout = Ridge(3, ridge=2.5e-6)
>>> model = nvar >> readout

Using the :py:func:`~reservoirpy.datasets.lorenz` timeseries and learning to
predict the next difference:

>>> from reservoirpy.datasets import lorenz
>>> X = lorenz(5400, x0=[17.677, 12.931, 43.914], h=0.025, method="RK23")
>>> Xi  = X[:600]
>>> dXi = X[1:601] - X[:600]  # difference u[t+1] - u[t]
>>> Y_test = X[600:]  # testing data
>>> _ = model.fit(Xi, dXi, warmup=200)

We can now predict the differences and integrate these predictions:

>>> u = X[600]
>>> res = np.zeros((5400-600, readout.output_dim))
>>> for i in range(5400-600):
...     u = u + model(u)
...     res[i, :] = u
...

.. plot:: ./api/nvar_example.py

"""

```
# File: delay.txt
```python
node.buffer.appendleft(x)
output = node.buffer.pop()

return output

if node.input_dim is not None:
    dim = node.input_dim
else:
    dim = x.shape[1]

node.set_input_dim(dim)
node.set_output_dim(dim)

if initial_values is None:
    initial_values = np.zeros((node.delay, node.input_dim), dtype=node.dtype)
node.set_param("buffer", deque(initial_values, maxlen=node.delay + 1))

"""Delays the data transmitted through this node without transformation.

:py:attr:`Delay.params` **list**

============= ======================================================================
``buffer``    (:py:class:`collections.deque`) Buffer of the values coming next.
============= ======================================================================

:py:attr:`Delay.hypers` **list**

============= ======================================================================
``delay``     (int) Number of timesteps before outputting the input.
============= ======================================================================

Parameters
----------
delay: int, defaults to 1.
    Number of timesteps before outputting the input.
initial_values: array of shape (delay, input_dim), defaults to
    `np.zeros((delay, input_dim))`.
    Initial outputs of the node.
input_dim : int, optional
    Input dimension. Can be inferred at first call.
dtype : Numpy dtype, defaults to np.float64
    Numerical type for node parameters.
name : str, optional
    Node name.

Examples
--------
>>> x = np.arange(10.).reshape(-1, 1)
>>>
>>> from reservoirpy.nodes import Delay
>>> delay_node = Delay(
>>>     delay=3,
>>>     initial_values=np.array([[-3.0], [-2.0], [-1.0]])
>>> )
>>>
>>> out = delay_node.run(x)
>>> print(out.T)
[[-1. -2. -3.  0.  1.  2.  3.  4.  5.  6.]]
>>> print(delay_node.buffer)
deque([array([[9.]]), array([[8.]]), array([[7.]])], maxlen=4)
"""

def __init__(
    self,
    delay=1,
    initial_values=None,
    input_dim=None,
    dtype=None,
    **kwargs,
):
    if input_dim is None and type(initial_values) is np.ndarray:
        input_dim = initial_values.shape[-1]

```
# File: lms copy.txt
```python
_assemble_wout,
_compute_error,
_initialize_readout,
_prepare_inputs_for_learning,
_split_and_save_wout,
readout_forward,

"""Least Mean Squares learning rule."""
# learning rate is a generator to allow scheduling
dw = -next(alpha) * np.outer(e, r)
return dw

"""Train a readout using LMS learning rule."""
x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias, allow_reshape=True)

error, r = _compute_error(node, x, y)

alpha = node._alpha_gen
dw = _lms(alpha, r, error)
wo = _assemble_wout(node.Wout, node.bias, node.input_bias)
wo = wo + dw.T

_split_and_save_wout(node, wo)

readout: "LMS", x=None, y=None, init_func=None, bias_init=None, bias=None

_initialize_readout(readout, x, y, init_func, bias_init, bias)

"""Single layer of neurons learning connections using Least Mean Squares
algorithm.

The learning rules is well described in [1]_.

:py:attr:`LMS.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
``P``              Matrix :math:`\\mathbf{P}` of RLS rule.
================== =================================================================

:py:attr:`LMS.hypers` **list**

================== =================================================================
``alpha``          Learning rate (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by default).
``input_bias``     If True, learn a bias term (True by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
alpha : float or Python generator or iterable, default to 1e-6
    Learning rate. If an iterable or a generator is provided, the learning rate can
    be changed at each timestep of training. A new learning rate will be drawn from
    the iterable or generator at each timestep.
Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

Examples
--------
>>> x = np.random.normal(size=(100, 3))
>>> noise = np.random.normal(scale=0.01, size=(100, 1))
>>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.

>>> from reservoirpy.nodes import LMS
>>> lms_node = LMS(alpha=1e-1)

>>> lms_node.train(x[:50], y[:50])
>>> print(lms_node.Wout.T, lms_node.bias)
[[ 9.156 -0.967   6.411]] [[11.564]]
>>> lms_node.train(x[50:], y[50:])
>>> print(lms_node.Wout.T, lms_node.bias)
[[ 9.998 -0.202  7.001]] [[12.005]]

References
----------

.. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
        Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
        https://doi.org/10.1016/j.neuron.2009.07.018
"""

def __init__(
    self,
    output_dim=None,
    alpha=1e-6,
    Wout=zeros,
    bias=zeros,
    input_bias=True,
    name=None,
):
    if isinstance(alpha, Number):

        def _alpha_gen():
            while True:
                yield alpha

        alpha_gen = _alpha_gen()
    elif isinstance(alpha, Iterable):
        alpha_gen = alpha
    else:
        raise TypeError(
            "'alpha' parameter should be a float or an iterable yielding floats."
        )

```
# File: _base.txt
```python
"""Get inputs for distant Nodes in a Model used as feedback or teacher.
These inputs should be already computed by other Nodes."""
input_data = {}
for p, c in model.edges:
    if p in model.input_nodes:
        input_data[c.name] = p.state_proxy()
return input_data

"""Remove inputs nodes from feedback Model and gather remaining nodes
into a new Model. Allow getting inputs for feedback model from its input
nodes states."""
from .model import Model

all_nodes = set(node.nodes)
input_nodes = set(node.input_nodes)
filtered_nodes = list(all_nodes - input_nodes)
filtered_edges = [edge for edge in node.edges if edge[0] not in input_nodes]

# return a single Node if Model - Inputs = Node
# else return Model - Inputs = Reduced Model
if len(filtered_nodes) == 1:
    return list(filtered_nodes)[0]
return Model(filtered_nodes, filtered_edges, name=str(uuid4()))

x: Union[np.ndarray, Sequence[np.ndarray]],
expected_dim=None,
caller=None,
allow_timespans=True,

caller_name = caller.name + "is " if caller is not None else ""

if expected_dim is not None and not hasattr(expected_dim, "__iter__"):
    expected_dim = (expected_dim,)

x_new = check_vector(
    x, allow_reshape=True, allow_timespans=allow_timespans, caller=caller
)
data_dim = x_new.shape[1:]

# Check x dimension
if expected_dim is not None:
    if len(expected_dim) != len(data_dim):
        raise ValueError(
            f"{caller_name} expecting {len(expected_dim)} inputs "
            f"but received {len(data_dim)}: {x_new}."
        )
    for dim in expected_dim:
        if all([dim != ddim for ddim in data_dim]):
            raise ValueError(
                f"{caller_name} expecting data of shape "
                f"{expected_dim} but received shape {data_dim}."
            )
return x_new

x,
expected_dim=None,
allow_n_sequences=True,
allow_n_inputs=True,
allow_timespans=True,
caller=None,

if expected_dim is not None:
    if not hasattr(expected_dim, "__iter__"):
        expected_dim = (expected_dim,)
    n_inputs = len(expected_dim)

    # I
    if n_inputs > 1:
        if isinstance(x, (list, tuple)):
            x_new = [x[i] for i in range(len(x))]
            timesteps = []
            for i in range(n_inputs):
                dim = (expected_dim[i],)
                x_new[i] = check_n_sequences(
                    x[i],
                    expected_dim=dim,
                    caller=caller,
                    allow_n_sequences=allow_n_sequences,
                    allow_timespans=allow_timespans,
                    allow_n_inputs=allow_n_inputs,
                )
                if isinstance(x_new[i], (list, tuple)):
                    timesteps.append(tuple([x_.shape[0] for x_ in x_new[i]]))
                else:
                    dim = dim[0]
                    if not hasattr(dim, "__len__"):
                        dim = (dim,)
                    if len(dim) + 2 > len(x_new[i].shape) >= len(dim) + 1:
                        timesteps.append((x_new[i].shape[0],))
                    else:
                        timesteps.append((x_new[i].shape[1],))

            if len(np.unique([len(t) for t in timesteps])) > 1 or any(
                [
                    len(np.unique([t[i] for t in timesteps])) > 1
                    for i in range(len(timesteps[0]))
                ]
            ):
                raise ValueError("Inputs with different timesteps")
        else:
            raise ValueError("Expecting several inputs.")
    else:  # L
        dim = expected_dim[0]
        if not hasattr(dim, "__len__"):
            dim = (dim,)

        if isinstance(x, (list, tuple)):
            if not allow_n_sequences:
                raise TypeError("No lists, only arrays.")
            x_new = [x[i] for i in range(len(x))]
            for i in range(len(x)):
                x_new[i] = check_one_sequence(
                    x[i],
                    allow_timespans=allow_timespans,
                    expected_dim=dim,
                    caller=caller,
                )
        else:
            if len(x.shape) <= len(dim) + 1:  # only one sequence
                x_new = check_one_sequence(
                    x,
                    expected_dim=dim,
                    allow_timespans=allow_timespans,
                    caller=caller,
                )
            elif len(x.shape) == len(dim) + 2:  # several sequences
                # if not allow_n_sequences:
                #     raise TypeError("No lists, only arrays.")
                x_new = x
                for i in range(len(x)):
                    x_new[i] = check_one_sequence(
                        x[i],
                        allow_timespans=allow_timespans,
                        expected_dim=dim,
                        caller=caller,
                    )
            else:  # pragma: no cover
                x_new = check_vector(
                    x,
                    allow_reshape=True,
                    allow_timespans=allow_timespans,
                    caller=caller,
                )
else:
    if isinstance(x, (list, tuple)):
        x_new = [x[i] for i in range(len(x))]
        for i in range(len(x)):
            if allow_n_inputs:
                x_new[i] = check_n_sequences(
                    x[i],
                    allow_n_sequences=allow_n_sequences,
                    allow_timespans=allow_timespans,
                    allow_n_inputs=False,
                    caller=caller,
                )
            elif allow_n_sequences:
                x_new[i] = check_n_sequences(
                    x[i],
                    allow_n_sequences=False,
                    allow_timespans=allow_timespans,
                    allow_n_inputs=False,
                    caller=caller,
                )
            else:
                raise ValueError("No lists, only arrays.")
    else:
        x_new = check_one_sequence(
            x, allow_timespans=allow_timespans, caller=caller
        )

return x_new

x,
receiver_nodes=None,
expected_dim=None,
caller=None,
io_type="input",
allow_n_sequences=True,
allow_n_inputs=True,
allow_timespans=True,

noteacher_msg = f"Nodes can not be used as {io_type}" + " for {}."
notonline_msg = "{} is not trained online."

x_new = None
# Caller is a Model
if receiver_nodes is not None:
    if not is_mapping(x):
        x_new = {n.name: x for n in receiver_nodes}
    else:
        x_new = x.copy()

    for node in receiver_nodes:
        if node.name not in x_new:
            # Maybe don't fit nodes a second time
            if io_type == "target" and node.fitted:
                continue
            else:
                raise ValueError(f"Missing {io_type} data for node {node.name}.")

        if (
            callable(x_new[node.name])
            and hasattr(x_new[node.name], "initialize")
            and hasattr(x_new[node.name], "is_initialized")
            and hasattr(x_new[node.name], "output_dim")
        ):
            if io_type == "target":
                if node.is_trained_online:
                    register_teacher(
                        node,
                        x_new.pop(node.name),
                        expected_dim=node.output_dim,
                    )
                else:
                    raise TypeError(
                        (noteacher_msg + notonline_msg).format(node.name, node.name)
                    )
            else:
                raise TypeError(noteacher_msg.format(node.name))
        else:
            if io_type == "target":
                dim = node.output_dim
            else:
                dim = node.input_dim

            x_new[node.name] = check_n_sequences(
                x_new[node.name],
                expected_dim=dim,
                caller=node,
                allow_n_sequences=allow_n_sequences,
                allow_n_inputs=allow_n_inputs,
                allow_timespans=allow_timespans,
            )
# Caller is a Node
else:
    if (
        callable(x)
        and hasattr(x, "initialize")
        and hasattr(x, "is_initialized")
        and hasattr(x, "output_dim")
    ):
        if io_type == "target":
            if caller.is_trained_online:
                register_teacher(
                    caller,
                    x,
                    expected_dim=expected_dim,
                )
            else:
                raise TypeError(
                    (noteacher_msg + notonline_msg).format(caller.name, caller.name)
                )
        else:
            raise TypeError(noteacher_msg.format(caller.name))
    else:
        x_new = check_n_sequences(
            x,
            expected_dim=expected_dim,
            caller=caller,
            allow_n_sequences=allow_n_sequences,
            allow_n_inputs=allow_n_inputs,
            allow_timespans=allow_timespans,
        )

# All x are teacher nodes, no data to return
if is_mapping(x_new) and io_type == "target" and len(x_new) == 0:
    return None

return x_new

target_dim = None
if teacher.is_initialized:
    target_dim = teacher.output_dim

if (
    expected_dim is not None
    and target_dim is not None
    and expected_dim != target_dim
):
    raise ValueError()

caller._teacher = DistantFeedback(
    sender=teacher, receiver=caller, callback_type="teacher"
)

caller,
x,
y=None,
input_dim=None,
output_dim=None,
allow_n_sequences=True,
allow_n_inputs=True,
allow_timespans=True,

"""Prepare one step of input and target data for a Node or a Model.

Preparation may include:
    - reshaping data to ([inputs], [sequences], timesteps, features);
    - converting non-array objects to array objects;
    - checking if n_features is equal to node input or output dimension.

This works on numerical data and teacher nodes.

Parameters
----------
caller: Node or Model
    Node or Model requesting inputs/targets preparation.
x : array-like of shape ([inputs], [sequences], timesteps, features)
    Input array or sequence of input arrays containing a single timestep of
    data.
y : array-like of shape ([sequences], timesteps, features) or Node, optional
    Target array containing a single timestep of data, or teacher Node or
    Model
    yielding target values.
input_dim, output_dim : int or tuple of ints, optional
    Expected input and target dimensions, if available.

Returns
-------
array-like of shape ([inputs], 1, n), array-like of shape (1, n) or Node
    Processed input and target vectors.
"""

if input_dim is None and hasattr(caller, "input_dim"):
    input_dim = caller.input_dim

# caller is a Model
if hasattr(caller, "input_nodes"):
    input_nodes = caller.input_nodes
# caller is a Node
else:
    input_nodes = None

x_new = _check_node_io(
    x,
    receiver_nodes=input_nodes,
    expected_dim=input_dim,
    caller=caller,
    io_type="input",
    allow_n_sequences=allow_n_sequences,
    allow_n_inputs=allow_n_inputs,
    allow_timespans=allow_timespans,
)

y_new = y
if y is not None:
    # caller is a Model
    if hasattr(caller, "trainable_nodes"):
        output_dim = None
        trainable_nodes = caller.trainable_nodes

    # caller is a Node
    else:
        trainable_nodes = None
        if output_dim is None and hasattr(caller, "output_dim"):
            output_dim = caller.output_dim

    y_new = _check_node_io(
        y,
        receiver_nodes=trainable_nodes,
        expected_dim=output_dim,
        caller=caller,
        io_type="target",
        allow_n_sequences=allow_n_sequences,
        allow_timespans=allow_timespans,
        allow_n_inputs=False,
    )

return x_new, y_new

def __init__(self, sender, receiver, callback_type="feedback"):
    self._sender = sender
    self._receiver = receiver
    self._callback_type = callback_type

    # used to store a reduced version of the feedback if needed
    # when feedback is a Model (inputs of the feedback Model are suppressed
    # in the reduced version, as we do not need then to re-run them
    # because we assume they have already run during the forward call)
    self._reduced_sender = None

    self._clamped = False
    self._clamped_value = None

def __call__(self):
    if not self.is_initialized:
        self.initialize()
    return self.call_distant_node()

@property
def is_initialized(self):
    return self._sender.is_initialized

@property
def output_dim(self):
    return self._sender.output_dim

@property
def name(self):
    return self._sender.name

def call_distant_node(self):
    """Call a distant Model for feedback or teaching
    (no need to run the input nodes again)"""
    if self._clamped:
        self._clamped = False
        return self._clamped_value

    if self._reduced_sender is not None:
        if len(np.unique([n._fb_flag for n in self._sender.nodes])) > 1:
            input_data = _distant_model_inputs(self._sender)

            if hasattr(self._reduced_sender, "nodes"):
                return self._reduced_sender.call(input_data)
            else:
                reduced_name = self._reduced_sender.name
                return self._reduced_sender.call(input_data[reduced_name])
        else:
            fb_outputs = [n.state() for n in self._sender.output_nodes]
            if len(fb_outputs) > 1:
                return fb_outputs
            else:
                return fb_outputs[0]
    else:
        return self._sender.state_proxy()

def initialize(self):
    """Initialize a distant Model or Node (used as feedback sender or teacher)."""
    msg = f"Impossible to get {self._callback_type} "
    msg += "from {} for {}: {} is not initialized or has no input/output_dim"

    reduced_model = None
    if hasattr(self._sender, "input_nodes"):
        for n in self._sender.input_nodes:
            if not n.is_initialized:
                try:
                    n.initialize()
                except RuntimeError:
                    raise RuntimeError(
                        msg.format(
                            self._sender.name,
                            self._receiver.name,
                            self._sender.name,
                        )
                    )

        input_data = _distant_model_inputs(self._sender)
        reduced_model = _remove_input_for_feedback(self._sender)

        if not reduced_model.is_initialized:
            if hasattr(reduced_model, "nodes"):
                reduced_model.initialize(x=input_data)
            else:
                reduced_name = reduced_model.name
                reduced_model.initialize(x=input_data[reduced_name])
            self._sender._is_initialized = True
    else:
        try:
            self._sender.initialize()
        except RuntimeError:  # raise more specific error
            raise RuntimeError(
                msg.format(
                    self._sender.name, self._receiver.name, self._sender.name
                )
            )

    self._reduced_sender = reduced_model

def zero_feedback(self):
    """A null feedback vector. Returns None if the Node receives
    no feedback."""
    if hasattr(self._sender, "output_nodes"):
        zeros = []
        for output in self._sender.output_nodes:
            zeros.append(output.zero_state())
        if len(zeros) == 1:
            return zeros[0]
        else:
            return zeros
    else:
        return self._sender.zero_state()

def clamp(self, value):
    self._clamped_value = check_n_sequences(
        value,
        expected_dim=self._sender.output_dim,
        caller=self._sender,
        allow_n_sequences=False,
    )
    self._clamped = True

"""One-step call, without input check."""
with node.with_state(from_state, stateful=stateful, reset=reset):
    state = node._forward(node, x)
    node._state = state.astype(node.dtype)
    node._flag_feedback()

return state

node,
X,
Y=None,
call_node=True,
force_teachers=True,
learn_every=1,
from_state=None,
stateful=True,
reset=False,

seq_len = X.shape[0]
seq = (
    progress(range(seq_len), f"Training {node.name}")
    if seq_len > 1
    else range(seq_len)
)

with node.with_state(from_state, stateful=stateful, reset=reset):
    states = np.zeros((seq_len, node.output_dim))
    for i in seq:
        x = np.atleast_2d(X[i, :])

        y = None
        if node._teacher is not None:
            y = node._teacher()
        elif Y is not None:
            y = np.atleast_2d(Y[i, :])

        if call_node:
            s = call(node, x)
        else:
            s = node.state()

        if force_teachers:
            node.set_state_proxy(y)

        if i % learn_every == 0 or seq_len == 1:
            node._train(node, x=x, y=y)

        states[i, :] = s

return states

"""Node base class for type checking and interface inheritance."""

_factory_id = -1
_registry = list()
_name: str

def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._factory_id = -1
    cls._registry = list()

def __repr__(self):
    klas = type(self).__name__
    hypers = [(str(k), str(v)) for k, v in self._hypers.items()]
    all_params = ["=".join((k, v)) for k, v in hypers]
    all_params += [f"in={self.input_dim}", f"out={self.output_dim}"]
    return f"'{self.name}': {klas}(" + ", ".join(all_params) + ")"

def __setstate__(self, state):
    curr_name = state.get("name")
    if curr_name in type(self)._registry:
        new_name = curr_name + "-(copy)"
        state["name"] = new_name
    self.__dict__ = state

def __del__(self):
    try:
        type(self)._registry.remove(self._name)
    except (ValueError, AttributeError):
        pass

def __getattr__(self, item):
    if item in ["_params", "_hypers"]:
        raise AttributeError()
    if item in self._params:
        return self._params.get(item)
    elif item in self._hypers:
        return self._hypers.get(item)
    else:
        raise AttributeError(f"'{str(item)}'")

def __call__(self, *args, **kwargs) -> np.ndarray:
    return self.call(*args, **kwargs)

def __rshift__(self, other: Union["_Node", Sequence["_Node"]]) -> "Model":
    from .ops import link

    return link(self, other)

def __rrshift__(self, other: Union["_Node", Sequence["_Node"]]) -> "Model":
    from .ops import link

    return link(other, self)

def __and__(self, other: Union["_Node", Sequence["_Node"]]) -> "Model":
    from .ops import merge

    return merge(self, other)

def _get_name(self, name=None):
    if name is None:
        type(self)._factory_id += 1
        _id = self._factory_id
        name = f"{type(self).__name__}-{_id}"

    if name in type(self)._registry:
        raise NameError(
            f"Name '{name}' is already taken "
            f"by another node. Node names should "
            f"be unique."
        )

    type(self)._registry.append(name)
    return name

@property
def name(self) -> str:
    """Name of the Node or Model."""
    return self._name

@name.setter
def name(self, value):
    type(self)._registry.remove(self.name)
    self._name = self._get_name(value)

@property
def params(self) -> Dict[str, Any]:
    """Parameters of the Node or Model."""
    return self._params

@property
def hypers(self) -> Dict[str, Any]:
    """Hyperparameters of the Node or Model."""
    return self._hypers

@property
def is_initialized(self) -> bool:
    return self._is_initialized

@property
@abstractmethod
def input_dim(self) -> Shape:
    raise NotImplementedError()

@property
@abstractmethod
def output_dim(self) -> Shape:
    raise NotImplementedError()

@property
@abstractmethod
def is_trained_offline(self) -> bool:
    raise NotImplementedError()

@property
@abstractmethod
def is_trained_online(self) -> bool:
    raise NotImplementedError()

@property
@abstractmethod
def is_trainable(self) -> bool:
    raise NotImplementedError()

@property
@abstractmethod
def fitted(self) -> bool:
    raise NotImplementedError()

@is_trainable.setter
@abstractmethod
def is_trainable(self, value: bool):
    raise NotImplementedError()

def get_param(self, name: str) -> Any:
    if name in self._params:
        return self._params.get(name)
    elif name in self._hypers:
        return self._hypers.get(name)
    else:
        raise NameError(f"No parameter named '{name}' found in node {self}")

@abstractmethod
def copy(
    self, name: str = None, copy_feedback: bool = False, shallow: bool = False
) -> "_Node":
    raise NotImplementedError()

@abstractmethod
def initialize(self, x: MappedData = None, y: MappedData = None):
    raise NotImplementedError()

@abstractmethod
def reset(self, to_state: np.ndarray = None) -> "_Node":
    raise NotImplementedError()

@contextmanager
@abstractmethod
def with_state(self, state=None, stateful=False, reset=False) -> Iterator["_Node"]:
    raise NotImplementedError()

```
# File: Intrinsic_Plasiticity_Schrauwen_et_al_2008.txt
```python
import reservoirpy as rpy

import matplotlib.pyplot as plt

import numpy as np

from reservoirpy.datasets import narma

from reservoirpy.mat_gen import uniform, bernoulli

from reservoirpy.nodes import IPReservoir

%matplotlib inline

from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""")

rpy.verbosity(0)

rpy.set_seed(123456789)

def heavyside(x):

    return 1.0 if x >= 0 else 0.0

def bounded(dist, x, mu, sigma, a, b):

    """Return the bounded version of a scipy.stats distribution.

    

    As described in the paper (section 2.1)"""

    num = dist.pdf(x, loc=mu, scale=sigma) * heavyside(x - a) * heavyside(b - x)

    den = dist.cdf(b, loc=mu, scale=sigma) - dist.cdf(a, loc=mu, scale=sigma)

    return num / den

steps = 1000

X = narma(steps)

X = (X - X.min()) / (X.ptp())

plt.plot(X[:200])

plt.ylabel("NARMA (order 30)")

plt.xlabel("Timestep")

plt.show()

# Parameters (from the paper)

activation = "sigmoid"

units = 100

connectivity = 0.1

sr = 0.95

input_scaling = 0.1

mu = 0.3

warmup = 100

learning_rate = 5e-4

epochs = 100

W_dist = uniform(high=1.0, low=-1.0)

Win_dist = bernoulli

reservoir = IPReservoir(

    units,

    sr=sr,

    mu=mu,

    learning_rate=learning_rate,

    input_scaling=input_scaling,

    W=W_dist,

    Win=Win_dist,

    rc_connectivity=connectivity,

    input_connectivity=connectivity,

    activation=activation,

    epochs=epochs

)

reservoir = reservoir.fit(X, warmup=warmup)

# Resetting and warming up

reservoir.run(X[:warmup], reset=True)

# Running

states = reservoir.run(X[warmup:])

from scipy.stats import expon

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))

ax1.set_xlim(0.0, 1.0)

ax1.set_ylim(0, 16)

for s in range(states.shape[1]):

    hist, edges = np.histogram(states[:, s], density=True, bins=200)

    points = [np.mean([edges[i], edges[i+1]]) for i in range(len(edges) - 1)]

    ax1.scatter(points, hist, s=0.2, color="gray", alpha=0.25)

ax1.hist(states.flatten(), density=True, bins=200, histtype="step", label="Global activation", lw=3.0)

x = np.linspace(0.0, 1.0, 200)

pdf = [bounded(expon, xi, 0.0, mu, 0.0, 1.0) for xi in x]

ax1.plot(x, pdf, label="Target distribution", linestyle="--", lw=3.0)

ax1.set_xlabel("Reservoir activations")

ax1.set_ylabel("Probability density")

plt.legend()

plt.show()

# Parameters (from the paper)

activation = "tanh"

units = 100

connectivity = 0.1

sr = 0.95

input_scaling = 0.1

mu = 0.0

sigma = 0.1

warmup = 100

learning_rate = 5e-4

epochs = 100

W_dist = uniform(high=1.0, low=-1.0)

Win_dist = bernoulli

reservoir = IPReservoir(

    units,

    sr=sr,

    mu=mu,

    sigma=sigma,

    learning_rate=learning_rate,

    input_scaling=input_scaling,

    W=W_dist,

    Win=Win_dist,

    rc_connectivity=connectivity,

    input_connectivity=connectivity,

    activation=activation,

    epochs=epochs

)

reservoir = reservoir.fit(X, warmup=warmup)

# Resetting and warming up

reservoir.run(X[:warmup], reset=True)

# Running

states = reservoir.run(X[warmup:])

from scipy.stats import norm

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))

ax1.set_xlim(-1.0, 1.0)

ax1.set_ylim(0, 16)

for s in range(states.shape[1]):

    hist, edges = np.histogram(states[:, s], density=True, bins=200)

    points = [np.mean([edges[i], edges[i+1]]) for i in range(len(edges) - 1)]

    ax1.scatter(points, hist, s=0.2, color="gray", alpha=0.25)

ax1.hist(states.flatten(), density=True, bins=200, histtype="step", label="Global activation", lw=3.0)

x = np.linspace(-1.0, 1.0, 200)

pdf = [bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in x]

ax1.plot(x, pdf, label="Target distribution", linestyle="--", lw=3.0)

ax1.set_xlabel("Reservoir activations")

ax1.set_ylabel("Probability density")

plt.legend()

```
# File: batchforce.txt
```python
_initialize_readout,
_prepare_inputs_for_learning,
readout_forward,

step[:] = np.zeros_like(step)
rTPs[:] = np.zeros_like(rTPs)
factors[:] = np.zeros_like(factors)

if x is not None:

    x, y = _prepare_inputs_for_learning(
        x, y, bias=readout.has_bias, allow_reshape=True
    )

    W = readout.Wout
    if readout.has_bias:
        bias = readout.bias
        W = np.c_[bias, W]

    P = readout.P
    r = x.T
    output = readout.state()

    factors = readout.get_buffer("factors")
    rTPs = readout.get_buffer("rTPs")
    steps = readout.get_buffer("step")
    step = int(steps[0])

    error = output.T - y.T

    rt = r.T
    rTP = (rt @ P) - (rt @ (factors * rTPs)) @ rTPs.T
    factor = float(1.0 / (1.0 + rTP @ r))

    factors[step] = factor
    rTPs[:, step] = rTP

    new_rTP = rTP * (1 - factor * (rTP @ r).item())

    W -= error @ new_rTP

    if readout.has_bias:
        readout.set_param("Wout", W[:, 1:])
        readout.set_param("bias", W[:, :1])
    else:
        readout.set_param("Wout", W)

    step += 1

    if step == readout.batch_size:
        P -= (factors * rTPs) @ rTPs.T
        _reset_buffers(steps, rTPs, factors)

_initialize_readout(readout, x, y, init_func, bias)

if x is not None:
    input_dim, alpha = readout.input_dim, readout.alpha

    if readout.has_bias:
        input_dim += 1

    P = np.asmatrix(np.eye(input_dim)) / alpha

    readout.set_param("P", P)

bias_dim = 0
if readout.has_bias:
    bias_dim = 1

readout.create_buffer("rTPs", (readout.input_dim + bias_dim, readout.batch_size))
readout.create_buffer("factors", (readout.batch_size,))
readout.create_buffer("step", (1,))

# A special thanks to Lionel Eyraud-Dubois and
# Olivier Beaumont for their improvement of this method.

```
# File: low_rank_reservoir.txt
```python



mackey_glass, 

forecast=forecast, 

test_size=0.2



x_train, x_test, y_train, y_test = dataset

# repris du tuto 3

units = 100

leak_rate = 0.3

spectral_radius = 1.25

input_scaling = 1.0

connectivity = 0.1

input_connectivity = 0.2

regularization = 1e-8

seed = 1234

from reservoirpy.nodes import Reservoir, Ridge

if W is None:

    reservoir = Reservoir(

        units, 

        input_scaling=input_scaling, 

        sr=spectral_radius,

        lr=leak_rate, 

        rc_connectivity=connectivity,

        input_connectivity=input_connectivity, 

        seed=seed

    )

else:

    reservoir = Reservoir(

        W=W,

        input_scaling=input_scaling, 

        sr=spectral_radius,

        lr=leak_rate, 

        input_connectivity=input_connectivity, 

        seed=seed

    )

readout = Ridge(1, ridge=regularization)

model = reservoir >> readout



model(x_train[0])

model.fit(x_train, y_train)

y_pred = model.run(x_test)

return reservoirpy.observables.rsquare(y_test, y_pred), reservoirpy.observables.nrmse(y_test, y_pred)




print("shape =", W.shape)

if type(W) != numpy.ndarray:

    print("Rank =", numpy.linalg.matrix_rank(W.toarray()))

else:

    print("Rank =", numpy.linalg.matrix_rank(W))

print("Non-zeros count =", W.nonzero()[0].shape[0])

print("Non-zeros ratio =", W.nonzero()[0].shape[0] / (W.shape[0] * W.shape[1]))


reservoir = reservoirpy.nodes.Reservoir(units=units, rc_connectivity=rc_connectivity)

reservoir.initialize(numpy.eye(1))

ranks.append(numpy.linalg.matrix_rank(reservoir.W.toarray()))

nrmse.append(eval_reservoir(reservoir.W)[1])



















nrmses_in = []

r2s_in = []

for i in range(iters):

    density = np.sqrt(1 - np.exp(np.log(1 - connectivity) / rank))

    m = scipy.sparse.random(units, rank, density=density, random_state=rng, data_rvs=scipy.stats.norm().rvs)

    n = scipy.sparse.random(rank, units, density=density, random_state=rng, data_rvs=scipy.stats.norm().rvs)

    W = m @ n

    r2, nrmse = eval_reservoir(W)

    nrmses_in.append(nrmse)

    r2s_in.append(r2)



nrmses.append(np.median(nrmses_in))

r2s.append(np.median(r2s_in))



density = np.sqrt(1 - np.exp(np.log(1 - connectivity) / rank))

m = scipy.sparse.random(units, rank, density=density, random_state=rng, data_rvs=scipy.stats.norm().rvs)

n = scipy.sparse.random(rank, units, density=density, random_state=rng, data_rvs=scipy.stats.norm().rvs)

W = m @ n

matrix_stat(W)

r2, nrmse = eval_reservoir(W)

nrmses_in.append(nrmse)

r2s_in.append(r2)


```
# File: _chaos.txt
```python
"""
Mackey-Glass time delay differential equation, at values x(t) and x(t-tau).
"""
return -b * xt + a * xtau / (1 + xtau**n)

"""
Runge-Kuta method (RK4) for Mackey-Glass timeseries discretization.
"""
k1 = h * _mg_eq(xt, xtau, a, b, n)
k2 = h * _mg_eq(xt + 0.5 * k1, xtau, a, b, n)
k3 = h * _mg_eq(xt + 0.5 * k2, xtau, a, b, n)
k4 = h * _mg_eq(xt + k3, xtau, a, b, n)

return xt + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

n_timesteps: int,
a: float = 1.4,
b: float = 0.3,
x0: Union[list, np.ndarray] = [0.0, 0.0],
**kwargs,

"""Hénon map discrete timeseries [2]_ [3]_.

.. math::

    x(n+1) &= 1 - ax(n)^2 + y(n)\\\\
    y(n+1) &= bx(n)

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
a : float, default to 1.4
    :math:`a` parameter of the system.
b : float, default to 0.3
    :math:`b` parameter of the system.
x0 : array-like of shape (2,), default to [0.0, 0.0]
    Initial conditions of the system.

Returns
-------
array of shape (n_timesteps, 2)
    Hénon map discrete timeseries.

References
----------
.. [2] M. Hénon, ‘A two-dimensional mapping with a strange
        attractor’, Comm. Math. Phys., vol. 50, no. 1, pp. 69–77, 1976.

.. [3] `Hénon map <https://en.wikipedia.org/wiki/H%C3%A9non_map>`_
        on Wikipedia

"""
states = np.zeros((n_timesteps, 2))
states[0] = np.asarray(x0)

for i in range(1, n_timesteps):
    states[i][0] = 1 - a * states[i - 1][0] ** 2 + states[i - 1][1]
    states[i][1] = b * states[i - 1][0]

return states

n_timesteps: int, r: float = 3.9, x0: float = 0.5, **kwargs

"""Logistic map discrete timeseries [4]_ [5]_.

.. math::

    x(n+1) = rx(n)(1-x(n))

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
r : float, default to 3.9
    :math:`r` parameter of the system.
x0 : float, default to 0.5
    Initial condition of the system.

Returns
-------
array of shape (n_timesteps, 1)
    Logistic map discrete timeseries.

References
----------
.. [4] R. M. May, ‘Simple mathematical models with very
        complicated dynamics’, Nature, vol. 261, no. 5560,
        Art. no. 5560, Jun. 1976, doi: 10.1038/261459a0.

.. [5] `Logistic map <https://en.wikipedia.org/wiki/Logistic_map>`_
        on Wikipedia
"""
if r > 0 and 0 < x0 < 1:
    X = np.zeros(n_timesteps)
    X[0] = x0

    for i in range(1, n_timesteps):
        X[i] = r * X[i - 1] * (1 - X[i - 1])

    return X.reshape(-1, 1)
elif r <= 0:
    raise ValueError("r should be positive.")
else:
    raise ValueError("Initial condition x0 should be in ]0;1[.")

n_timesteps: int,
rho: float = 28.0,
sigma: float = 10.0,
beta: float = 8.0 / 3.0,
x0: Union[list, np.ndarray] = [1.0, 1.0, 1.0],
h: float = 0.03,
**kwargs,

"""Lorenz attractor timeseries as defined by Lorenz in 1963 [6]_ [7]_.

.. math::

    \\frac{\\mathrm{d}x}{\\mathrm{d}t} &= \\sigma (y-x) \\\\
    \\frac{\\mathrm{d}y}{\\mathrm{d}t} &= x(\\rho - z) - y \\\\
    \\frac{\\mathrm{d}z}{\\mathrm{d}t} &= xy - \\beta z

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
rho : float, default to 28.0
    :math:`\\rho` parameter of the system.
sigma : float, default to 10.0
    :math:`\\sigma` parameter of the system.
beta : float, default to 8/3
    :math:`\\beta` parameter of the system.
x0 : array-like of shape (3,), default to [1.0, 1.0, 1.0]
    Initial conditions of the system.
h : float, default to 0.03
    Time delta between two discrete timesteps.
**kwargs:
    Other parameters to pass to the `scipy.integrate.solve_ivp`
    solver.

Returns
-------
array of shape (n_timesteps, 3)
    Lorenz attractor timeseries.

References
----------
.. [6] E. N. Lorenz, ‘Deterministic Nonperiodic Flow’,
        Journal of the Atmospheric Sciences, vol. 20, no. 2,
        pp. 130–141, Mar. 1963,
        doi: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.

.. [7] `Lorenz system <https://en.wikipedia.org/wiki/Lorenz_system>`_
        on Wikipedia.
"""

def lorenz_diff(t, state):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

t_max = n_timesteps * h

t_eval = np.linspace(0.0, t_max, n_timesteps)

sol = solve_ivp(lorenz_diff, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs)

return sol.y.T

n_timesteps: int,
tau: int = 17,
a: float = 0.2,
b: float = 0.1,
n: int = 10,
x0: float = 1.2,
h: float = 1.0,
seed: Union[int, RandomState, Generator] = None,
**kwargs,

"""Mackey-Glass timeseries [8]_ [9]_, computed from the Mackey-Glass
delayed differential equation.

.. math::

    \\frac{x}{t} = \\frac{ax(t-\\tau)}{1+x(t-\\tau)^n} - bx(t)

Parameters
----------
n_timesteps : int
    Number of timesteps to compute.
tau : int, default to 17
    Time delay :math:`\\tau` of Mackey-Glass equation.
    By defaults, equals to 17. Other values can
    change the chaotic behaviour of the timeseries.
a : float, default to 0.2
    :math:`a` parameter of the equation.
b : float, default to 0.1
    :math:`b` parameter of the equation.
n : int, default to 10
    :math:`n` parameter of the equation.
x0 : float, optional, default to 1.2
    Initial condition of the timeseries.
h : float, default to 1.0
    Time delta between two discrete timesteps.
seed : int or :py:class:`numpy.random.Generator`, optional
    Random state seed for reproducibility.

Returns
-------
array of shape (n_timesteps, 1)
    Mackey-Glass timeseries.

Note
----
    As Mackey-Glass is defined by delayed time differential equations,
    the first timesteps of the timeseries can't be initialized at 0
    (otherwise, the first steps of computation involving these
    not-computed-yet-timesteps would yield inconsistent results).
    A random number generator is therefore used to produce random
    initial timesteps based on the value of the initial condition
    passed as parameter. A default seed is hard-coded to ensure
    reproducibility in any case. It can be changed with the
    :py:func:`set_seed` function.

References
----------
.. [8] M. C. Mackey and L. Glass, ‘Oscillation and chaos in
        physiological
        control systems’, Science, vol. 197, no. 4300, pp. 287–289,
        Jul. 1977,
        doi: 10.1126/science.267326.

.. [9] `Mackey-Glass equations
        <https://en.wikipedia.org/wiki/Mackey-Glass_equations>`_
        on Wikipedia.

"""
# a random state is needed as the method used to discretize
# the timeseries needs to use randomly generated initial steps
# based on the initial condition passed as parameter.
if seed is None:
    seed = get_seed()

rs = rand_generator(seed)

# generate random first step based on the value
# of the initial condition
history_length = int(np.floor(tau / h))
history = collections.deque(
    x0 * np.ones(history_length) + 0.2 * (rs.random(history_length) - 0.5)
)
xt = x0

X = np.zeros(n_timesteps)

for i in range(0, n_timesteps):
    X[i] = xt

    if tau == 0:
        xtau = 0.0
    else:
        xtau = history.popleft()
        history.append(xt)

    xth = _mg_rk4(xt, xtau, a=a, b=b, n=n, h=h)

    xt = xth

return X.reshape(-1, 1)

n_timesteps: int,
a: float = 40.0,
b: float = 3.0,
c: float = 28.0,
x0: Union[list, np.ndarray] = [-0.1, 0.5, -0.6],
h: float = 0.01,
**kwargs,

"""Double scroll attractor timeseries [10]_ [11]_,
a particular case of multiscroll attractor timeseries.

.. math::

    \\frac{\\mathrm{d}x}{\\mathrm{d}t} &= a(y - x) \\\\
    \\frac{\\mathrm{d}y}{\\mathrm{d}t} &= (c - a)x - xz + cy \\\\
    \\frac{\\mathrm{d}z}{\\mathrm{d}t} &= xy - bz

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
a : float, default to 40.0
    :math:`a` parameter of the system.
b : float, default to 3.0
    :math:`b` parameter of the system.
c : float, default to 28.0
    :math:`c` parameter of the system.
x0 : array-like of shape (3,), default to [-0.1, 0.5, -0.6]
    Initial conditions of the system.
h : float, default to 0.01
    Time delta between two discrete timesteps.

Returns
-------
array of shape (n_timesteps, 3)
    Multiscroll attractor timeseries.

References
----------
.. [10] G. Chen and T. Ueta, ‘Yet another chaotic attractor’,
        Int. J. Bifurcation Chaos, vol. 09, no. 07, pp. 1465–1466,
        Jul. 1999, doi: 10.1142/S0218127499001024.

.. [11] `Chen double scroll attractor
        <https://en.wikipedia.org/wiki/Multiscroll_attractor
        #Chen_attractor>`_
        on Wikipedia.

"""

def multiscroll_diff(t, state):
    x, y, z = state
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return dx, dy, dz

t_max = n_timesteps * h

t_eval = np.linspace(0.0, t_max, n_timesteps)

sol = solve_ivp(
    multiscroll_diff, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs
)

return sol.y.T

n_timesteps: int,
r1: float = 1.2,
r2: float = 3.44,
r4: float = 0.193,
ir: float = 2 * 2.25e-5,
beta: float = 11.6,
x0: Union[list, np.ndarray] = [0.37926545, 0.058339, -0.08167691],
h: float = 0.25,
**kwargs,

"""Double scroll attractor timeseries [10]_ [11]_,
a particular case of multiscroll attractor timeseries.

.. math::

    \\frac{\\mathrm{d}V_1}{\\mathrm{d}t} &= \\frac{V_1}{R_1} - \\frac{\\Delta V}{R_2} -
    2I_r \\sinh(\\beta\\Delta V) \\\\
    \\frac{\\mathrm{d}V_2}{\\mathrm{d}t} &= \\frac{\\Delta V}{R_2} +2I_r \\sinh(\\beta\\Delta V) - I\\\\
    \\frac{\\mathrm{d}I}{\\mathrm{d}t} &= V_2 - R_4 I

where :math:`\\Delta V = V_1 - V_2`.

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
r1 : float, default to 1.2
    :math:`R_1` parameter of the system.
r2 : float, default to 3.44
    :math:`R_2` parameter of the system.
r4 : float, default to 0.193
    :math:`R_4` parameter of the system.
ir : float, default to 2*2e.25e-5
    :math:`I_r` parameter of the system.
beta : float, default to 11.6
    :math:`\\beta` parameter of the system.
x0 : array-like of shape (3,), default to [0.37926545, 0.058339, -0.08167691]
    Initial conditions of the system.
h : float, default to 0.01
    Time delta between two discrete timesteps.

Returns
-------
array of shape (n_timesteps, 3)
    Multiscroll attractor timeseries.

References
----------
.. [10] G. Chen and T. Ueta, ‘Yet another chaotic attractor’,
        Int. J. Bifurcation Chaos, vol. 09, no. 07, pp. 1465–1466,
        Jul. 1999, doi: 10.1142/S0218127499001024.

.. [11] `Chen double scroll attractor
        <https://en.wikipedia.org/wiki/Multiscroll_attractor
        #Chen_attractor>`_
        on Wikipedia.
"""

def doublescroll_diff(t, state):
    V1, V2, i = state

    dV = V1 - V2
    factor = (dV / r2) + ir * np.sinh(beta * dV)
    dV1 = (V1 / r1) - factor
    dV2 = factor - i
    dI = V2 - r4 * i

    return dV1, dV2, dI

t_max = n_timesteps * h

t_eval = np.linspace(0.0, t_max, n_timesteps)

sol = solve_ivp(
    doublescroll_diff, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs
)

return sol.y.T

n_timesteps: int,
alpha: float = 1.1,
gamma: float = 0.89,
x0: Union[list, np.ndarray] = [-1, 0, 0.5],
h: float = 0.05,
**kwargs,

"""Rabinovitch-Fabrikant system [12]_ [13]_ timeseries.

.. math::

    \\frac{\\mathrm{d}x}{\\mathrm{d}t} &= y(z - 1 + x^2) + \\gamma x \\\\
    \\frac{\\mathrm{d}y}{\\mathrm{d}t} &= x(3z + 1 - x^2) + \\gamma y \\\\
    \\frac{\\mathrm{d}z}{\\mathrm{d}t} &= -2z(\\alpha + xy)

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
alpha : float, default to 1.1
    :math:`\\alpha` parameter of the system.
gamma : float, default to 0.89
    :math:`\\gamma` parameter of the system.
x0 : array-like of shape (3,), default to [-1, 0, 0.5]
    Initial conditions of the system.
h : float, default to 0.05
    Time delta between two discrete timesteps.
**kwargs:
    Other parameters to pass to the `scipy.integrate.solve_ivp`
    solver.

Returns
-------
array of shape (n_timesteps, 3)
    Rabinovitch-Fabrikant system timeseries.

References
----------
.. [12] M. I. Rabinovich and A. L. Fabrikant,
        ‘Stochastic self-modulation of waves in
        nonequilibrium media’, p. 8, 1979.

.. [13] `Rabinovich-Fabrikant equations
        <https://en.wikipedia.org/wiki/Rabinovich%E2%80
        %93Fabrikant_equations>`_
        on Wikipedia.

"""

def rabinovich_fabrikant_diff(t, state):
    x, y, z = state
    dx = y * (z - 1 + x**2) + gamma * x
    dy = x * (3 * z + 1 - x**2) + gamma * y
    dz = -2 * z * (alpha + x * y)
    return dx, dy, dz

t_max = n_timesteps * h

t_eval = np.linspace(0.0, t_max, n_timesteps)

sol = solve_ivp(
    rabinovich_fabrikant_diff, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs
)

return sol.y.T

n_timesteps: int,
order: int = 30,
a1: float = 0.2,
a2: float = 0.04,
b: float = 1.5,
c: float = 0.001,
x0: Union[list, np.ndarray] = [0.0],
seed: Union[int, RandomState] = None,
u: np.ndarray = None,

"""Non-linear Autoregressive Moving Average (NARMA) timeseries,
as first defined in [14]_, and as used in [15]_.

NARMA n-th order dynamical system is defined by the recurrent relation:

.. math::

    y[t+1] = a_1 y[t] + a_2 y[t] (\\sum_{i=0}^{n-1} y[t-i]) + b u[t-(
    n-1)]u[t] + c

where :math:`u[t]` are sampled following a uniform distribution in
:math:`[0, 0.5]`.

Note
----
In most reservoir computing benchmarks, $u$ is given as an input. If you want
access to its value, you should create the `u` array and pass it as an argument
to the function as shown below.
This choice is made to avoid breaking changes. In future ReservoirPy versions, `u`
will be returned with `y`.
See `related discussion <https://github.com/reservoirpy/reservoirpy/issues/142>`_.

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
order: int, default to 30
    Order of the system.
a1 : float, default to 0.2
    :math:`a_1` parameter of the system.
a2 : float, default to 0.04
    :math:`a_2` parameter of the system.
b : float, default to 1.5
    :math:`b` parameter of the system.
c : float, default to 0.001
    :math:`c` parameter of the system.
x0 : array-like of shape (init_steps,), default to [0.0]
    Initial conditions of the system.
seed : int or :py:class:`numpy.random.Generator`, optional
    Random state seed for reproducibility.
u : array of shape (`n_timesteps` + `order`, 1), default to None.
    Input timeseries (usually uniformly distributed). See above note.

Returns
-------
array of shape (n_timesteps, 1)
    NARMA timeseries.

Example
-------
>>> import numpy as np
>>> from reservoirpy.nodes import Reservoir, Ridge
>>> from reservoirpy.datasets import narma
>>> model = Reservoir(100) >> Ridge()
>>> n_timesteps, order = 2000, 30
>>> rng = np.random.default_rng(seed=2341)
>>> u = rng.uniform(0, 0.5, size=(n_timesteps + order, 1))
>>> y = narma(n_timesteps=n_timesteps, order=order, u=u)
>>> model = model.fit(u[order:], y)

References
----------
.. [14] A. F. Atiya and A. G. Parlos, ‘New results on recurrent
        network training: unifying the algorithms and accelerating
        convergence,‘ in IEEE Transactions on Neural Networks,
        vol. 11, no. 3, pp. 697-709, May 2000,
        doi: 10.1109/72.846741.

.. [15] B.Schrauwen, M. Wardermann, D. Verstraeten, J. Steil,
        D. Stroobandt, ‘Improving reservoirs using intrinsic
        plasticity‘,
        Neurocomputing, 71. 1159-1171, 2008,
        doi: 10.1016/j.neucom.2007.12.020.
"""
if seed is None:
    seed = get_seed()
rs = rand_generator(seed)

y = np.zeros((n_timesteps + order, 1))

x0 = check_vector(np.atleast_2d(np.asarray(x0)))
y[: x0.shape[0], :] = x0

if u is None:
    u = rs.uniform(0, 0.5, size=(n_timesteps + order, 1))

for t in range(order, n_timesteps + order - 1):
    y[t + 1] = (
        a1 * y[t]
        + a2 * y[t] * np.sum(y[t - order : t])
        + b * u[t - order] * u[t]
        + c
    )
return y[order:, :]

n_timesteps: int,
warmup: int = 0,
N: int = 36,
F: float = 8.0,
dF: float = 0.01,
h: float = 0.01,
x0: Union[list, np.ndarray] = None,
**kwargs,

"""Lorenz96 attractor timeseries as defined by Lorenz in 1996 [17]_.

.. math::

    \\frac{\\mathrm{d}x_i}{\\mathrm{d} t} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F

where :math:`i = 1, \\dots, N` and :math:`x_{-1} = x_{N-1}`
and :math:`x_{N+1} = x_1` and :math:`N \\geq 4`.

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
warmup : int, default to 0
    Number of timesteps to discard at the beginning of the signal, to remove
    transient states.
N: int, default to 36
    Dimension of the system.
F : float, default to F
    :math:`F` parameter of the system.
dF : float, default to 0.01
    Perturbation applied to initial condition if x0 is None.
h : float, default to 0.01
    Time delta between two discrete timesteps.
x0 : array-like of shape (N,), default to None
    Initial conditions of the system. If None, the array is initialized to
    an array of shape (N, ) with value F, except for the first value of the
    array that takes the value F + dF.
**kwargs:
    Other parameters to pass to the `scipy.integrate.solve_ivp`
    solver.

Returns
-------
array of shape (n_timesteps - warmup, N)
    Lorenz96 timeseries.

References
----------
.. [17] Lorenz, E. N. (1996, September).
        Predictability: A problem partly solved. In Proc.
        Seminar on predictability (Vol. 1, No. 1).
"""
if N < 4:
    raise ValueError("N must be >= 4.")

if x0 is None:
    x0 = F * np.ones(N)
    x0[0] = F + dF

if len(x0) != N:
    raise ValueError(
        f"x0 should have shape ({N},), but have shape {np.asarray(x0).shape}"
    )

def lorenz96_diff(t, state):
    ds = np.zeros(N)
    for i in range(N):
        ds[i] = (state[(i + 1) % N] - state[i - 2]) * state[i - 1] - state[i] + F
    return ds

t_max = (warmup + n_timesteps) * h

t_eval = np.linspace(0.0, t_max * h, n_timesteps)

sol = solve_ivp(
    lorenz96_diff,
    y0=x0,
    t_span=(0.0, t_max * h),
    t_eval=t_eval,
    **kwargs,
)

return sol.y.T[warmup:]

n_timesteps: int,
a: float = 0.2,
b: float = 0.2,
c: float = 5.7,
x0: Union[list, np.ndarray] = [-0.1, 0.0, 0.02],
h: float = 0.1,
**kwargs,

"""Rössler attractor timeseries [18]_.

.. math::

    \\frac{\\mathrm{d}x}{\\mathrm{d}t} &= -y - z \\\\
    \\frac{\\mathrm{d}y}{\\mathrm{d}t} &= x + a y \\\\
    \\frac{\\mathrm{d}z}{\\mathrm{d}t} &= b + z (x - c)

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
a : float, default to 0.2
    :math:`a` parameter of the system.
b : float, default to 0.2
    :math:`b` parameter of the system.
c : float, default to 5.7
    :math:`c` parameter of the system.
x0 : array-like of shape (3,), default to [-0.1, 0.0, 0.02]
    Initial conditions of the system.
h : float, default to 0.1
    Time delta between two discrete timesteps.
**kwargs:
    Other parameters to pass to the `scipy.integrate.solve_ivp`
    solver.

Returns
-------
array of shape (n_timesteps, 3)
    Rössler attractor timeseries.

References
----------

.. [18] O.E. Rössler, "An equation for continuous chaos", Physics Letters A,
        vol 57, Issue 5, Pages 397-398, ISSN 0375-9601, 1976,
        https://doi.org/10.1016/0375-9601(76)90101-8.
"""
if len(x0) != 3:
    raise ValueError(
        f"x0 should have shape (3,), but have shape {np.asarray(x0).shape}"
    )

def rossler_diff(t, state):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return dx, dy, dz

t_max = n_timesteps * h

t_eval = np.linspace(0.0, t_max, n_timesteps)

sol = solve_ivp(rossler_diff, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs)

return sol.y.T

"""A single step of EDTRK4 to solve Kuramoto-Sivashinsky equation.

Kassam, A. K., & Trefethen, L. N. (2005). Fourth-order time-stepping for stiff PDEs.
SIAM Journal on Scientific Computing, 26(4), 1214-1233.
"""

Nv = g * fft(np.real(ifft(v)) ** 2)
a = E2 * v + Q * Nv
Na = g * fft(np.real(ifft(a)) ** 2)
b = E2 * v + Q * Na
Nb = g * fft(np.real(ifft(b)) ** 2)
c = E2 * a + Q * (2 * Nb - Nv)
Nc = g * fft(np.real(ifft(c)) ** 2)
v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

return v

# initial conditions
v0 = fft(x0)

# ETDRK4 scalars
k = np.conj(np.r_[np.arange(0, N / 2), [0], np.arange(-N / 2 + 1, 0)]) / M

L = k**2 - k**4

E = np.exp(h * L)
E2 = np.exp(h * L / 2)

r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)

Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))

f1 = (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3
f1 = h * np.real(np.mean(f1, axis=1))

f2 = (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3
f2 = h * np.real(np.mean(f2, axis=1))

f3 = (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3
f3 = h * np.real(np.mean(f3, axis=1))

g = -0.5j * k

# integration using ETDRK4 method
v = np.zeros((n_timesteps, N), dtype=complex)
v[0] = v0
for n in range(1, n_timesteps):
    v[n] = _kuramoto_sivashinsky_etdrk4(
        v[n - 1], g=g, E=E, E2=E2, Q=Q, f1=f1, f2=f2, f3=f3
    )

return np.real(ifft(v[warmup:]))

n_timesteps: int,
warmup: int = 0,
N: int = 128,
M: float = 16,
x0: Union[list, np.ndarray] = None,
h: float = 0.25,

"""Kuramoto-Sivashinsky oscillators [19]_ [20]_ [21]_.

.. math::

    y_t = -yy_x - y_{xx} - y_{xxxx}, ~~ x \\in [0, 32\\pi]

This 1D partial differential equation is solved using ETDRK4
(Exponential Time-Differencing 4th order Runge-Kutta) method, as described in [22]_.

Parameters
----------
n_timesteps : int
    Number of timesteps to generate.
warmup : int, default to 0
    Number of timesteps to discard at the beginning of the signal, to remove
    transient states.
N : int, default to 128
    Dimension of the system.
M : float, default to 0.2
    Number of points for complex means. Modify behaviour of the resulting
    multivariate timeseries.
x0 : array-like of shape (N,), default to None.
    Initial conditions of the system. If None, x0 is equal to
    :math:`\\cos (\\frac{y}{M}) * (1 + \\sin(\\frac{y}{M}))`
    with :math:`y = 2M\\pi x / N, ~~ x \\in [1, N]`.
h : float, default to 0.25
    Time delta between two discrete timesteps.

Returns
-------
array of shape (n_timesteps - warmup, N)
    Kuramoto-Sivashinsky equation solution.

References
----------

.. [19] Kuramoto, Y. (1978). Diffusion-Induced Chaos in Reaction Systems.
        Progress of Theoretical Physics Supplement, 64, 346–367.
        https://doi.org/10.1143/PTPS.64.346

.. [20] Sivashinsky, G. I. (1977). Nonlinear analysis of hydrodynamic instability
        in laminar flames—I. Derivation of basic equations.
        Acta Astronautica, 4(11), 1177–1206.
        https://doi.org/10.1016/0094-5765(77)90096-0

.. [21] Sivashinsky, G. I. (1980). On Flame Propagation Under Conditions
        of Stoichiometry. SIAM Journal on Applied Mathematics, 39(1), 67–82.
        https://doi.org/10.1137/0139007

.. [22] Kassam, A. K., & Trefethen, L. N. (2005).
        Fourth-order time-stepping for stiff PDEs.
        SIAM Journal on Scientific Computing, 26(4), 1214-1233.
"""
if x0 is None:
    x = 2 * M * np.pi * np.arange(1, N + 1) / N
    x0 = np.cos(x / M) * (1 + np.sin(x / M))
else:
    if not np.asarray(x0).shape[0] == N:
        raise ValueError(
            f"Initial condition x0 should be of shape {N} (= N) but "
            f"has shape {np.asarray(x0).shape}"
        )
    else:
        x0 = np.asarray(x0)

```
# File: activations.txt
```python
return node.f(x, **kwargs)

if x is not None:
    node.set_input_dim(x.shape[1])
    node.set_output_dim(x.shape[1])

"""Softmax activation function.

.. math::

    y_k = \\frac{e^{\\beta x_k}}{\\sum_{i=1}^{n} e^{\\beta x_i}}

:py:attr:`Softmax.hypers` **list**

============= ======================================================================
``f``         Activation function (:py:func:`reservoir.activationsfunc.softmax`).
``beta``      Softmax :math:`\\beta` parameter (1.0 by default).
============= ======================================================================

Parameters
----------
beta: float, default to 1.0
    Beta parameter of softmax.
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
"""

def __init__(self, beta=1.0, **kwargs):
    super(Softmax, self).__init__(
        hypers={"f": get_function("softmax"), "beta": beta},
        forward=partial(forward, beta=beta),
        initializer=initialize,
        **kwargs,
    )

"""Softplus activation function.

.. math::

    f(x) = \\mathrm{ln}(1 + e^{x})

:py:attr:`Softplus.hypers` **list**

============= ======================================================================
``f``         Activation function (:py:func:`reservoir.activationsfunc.softplus`).
============= ======================================================================

Parameters
----------
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
"""

def __init__(self, **kwargs):
    super(Softplus, self).__init__(
        hypers={"f": get_function("softplus")},
        forward=forward,
        initializer=initialize,
        **kwargs,
    )

"""Sigmoid activation function.

.. math::

    f(x) = \\frac{1}{1 + e^{-x}}

:py:attr:`Sigmoid.hypers` **list**

============= ======================================================================
``f``         Activation function (:py:func:`reservoir.activationsfunc.sigmoid`).
============= ======================================================================

Parameters
----------
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
"""

def __init__(self, **kwargs):
    super(Sigmoid, self).__init__(
        hypers={"f": get_function("sigmoid")},
        forward=forward,
        initializer=initialize,
        **kwargs,
    )

"""Hyperbolic tangent activation function.

.. math::

    f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

:py:attr:`Tanh.hypers` **list**

============= ======================================================================
``f``         Activation function (:py:func:`reservoir.activationsfunc.tanh`).
============= ======================================================================

Parameters
----------
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
"""

def __init__(self, **kwargs):
    super(Tanh, self).__init__(
        hypers={"f": get_function("tanh")},
        forward=forward,
        initializer=initialize,
        **kwargs,
    )

"""Identity function.

.. math::

    f(x) = x

Provided for convenience.

:py:attr:`Identity.hypers` **list**

============= ======================================================================
``f``         Activation function (:py:func:`reservoir.activationsfunc.identity`).
============= ======================================================================

Parameters
----------
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
"""

def __init__(self, **kwargs):
    super(Identity, self).__init__(
        hypers={"f": get_function("identity")},
        forward=forward,
        initializer=initialize,
        **kwargs,
    )

"""ReLU activation function.

.. math::

    f(x) = x ~~ \\mathrm{if} ~~ x > 0 ~~ \\mathrm{else} ~~ 0

:py:attr:`ReLU.hypers` **list**

============= ======================================================================
``f``         Activation function (:py:func:`reservoir.activationsfunc.relu`).
============= ======================================================================

Parameters
----------
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
"""

```
# File: 4-Understand_and_optimize_hyperparameters.txt
```python
UNITS = 100               # - number of neurons

LEAK_RATE = 0.3           # - leaking rate

SPECTRAL_RADIUS = 1.25    # - spectral radius of W

INPUT_SCALING = 1.0       # - input scaling

RC_CONNECTIVITY = 0.1     # - density of reservoir internal matrix

INPUT_CONNECTIVITY = 0.2  # and of reservoir input matrix

REGULARIZATION = 1e-8     # - regularization coefficient for ridge regression

SEED = 1234               # for reproductibility

import numpy as np

import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge

from reservoirpy.datasets import mackey_glass

import reservoirpy as rpy

rpy.verbosity(0)

X = mackey_glass(2000)

# rescale between -1 and 1

X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

plt.figure()

plt.xlabel("$t$")

plt.title("Mackey-Glass timeseries")

plt.plot(X[:500])

plt.show()

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

UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))

for i, s in enumerate(states):

    plt.subplot(len(spectral_radii), 1, i+1)

    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)

    plt.ylabel(f"$sr={spectral_radii[i]}$")

plt.xlabel(f"Activations ({UNITS_SHOWN} neurons)")

plt.show()

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

UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))

for i, s in enumerate(states):

    plt.subplot(len(input_scalings), 1, i+1)

    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)

    plt.ylabel(f"$iss={input_scalings[i]}$")

plt.xlabel(f"Activations ({UNITS_SHOWN} neurons)")

plt.show()

def correlation(states, inputs):

    correlations = [np.corrcoef(states[:, i].flatten(), inputs.flatten())[0, 1] for i in range(states.shape[1])]

    return np.mean(np.abs(correlations))

print("input_scaling    correlation")

for i, s in enumerate(states):

    corr = correlation(states[i], X[:500])

    print(f"{input_scalings[i]: <13}    {corr}")

saturation)

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

UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))

for i, s in enumerate(states):

    plt.subplot(len(leaking_rates), 1, i+1)

    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)

    plt.ylabel(f"$lr={leaking_rates[i]}$")

plt.xlabel(f"States ({UNITS_SHOWN} neurons)")

plt.show()

from reservoirpy.datasets import doublescroll

timesteps = 2000

x0 = [0.37926545, 0.058339, -0.08167691]

X = doublescroll(timesteps, x0=x0, method="RK23")

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

from reservoirpy.observables import nrmse, rsquare

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

train_len = 1000

forecast = 2

X_train = X[:train_len]

Y_train = X[forecast : train_len + forecast]

X_test = X[train_len : -forecast]

Y_test = X[train_len + forecast:]

dataset = (X_train, Y_train, X_test, Y_test)

from reservoirpy.datasets import to_forecasting

X_train, X_test, Y_train, Y_test = to_forecasting(X, forecast=forecast, test_size=train_len-forecast)

from reservoirpy.hyper import research

best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

from reservoirpy.hyper import plot_hyperopt_report

fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr", "ridge"), metric="r2")

```
# File: _utils.txt
```python
if folder_path is None:
    folder_path = DATA_FOLDER
else:
    folder_path = Path(folder_path)

if not folder_path.exists():
    folder_path.mkdir(parents=True)

```
# File: test_observables.txt
```python
"obs,ytest,ypred,kwargs,expects",
[
    (mse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
    (rmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
    (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
    (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm": "var"}, None),
    (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm": "q1q3"}, None),
    (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm": "foo"}, "raise"),
    (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm_value": 3.0}, None),
    (rsquare, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
    (mse, [1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5], {}, "raise"),
    (rmse, [[1.0, 2.0, 3.0]], [1.5, 2.5, 3.5], {}, "raise"),
    (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5, 4.2], {}, "raise"),
    (rsquare, [1.0, 2.0, 3.0, 0.0], [1.5, 2.5, 3.5], {}, "raise"),
],

if expects == "raise":
    with pytest.raises(ValueError):
        obs(ytest, ypred, **kwargs)
else:
    m = obs(ytest, ypred, **kwargs)
    assert isinstance(m, float)

rng = np.random.default_rng(1234)

w = rng.uniform(size=(100, 100))

rho = spectral_radius(w)

assert isinstance(rho, float)

idxs = rng.random(size=(100, 100))
w[idxs < 0.5] = 0
w = csr_matrix(w)

rho = spectral_radius(w)

assert isinstance(rho, float)

rho = spectral_radius(w, maxiter=500)

assert isinstance(rho, float)

```
# File: node.txt
```python
    ~Node.call
    ~Node.clean_buffers
    ~Node.copy
    ~Node.create_buffer
    ~Node.feedback
    ~Node.fit
    ~Node.get_buffer
    ~Node.get_param
    ~Node.initialize
    ~Node.initialize_buffers
    ~Node.initialize_feedback
    ~Node.link_feedback
    ~Node.partial_fit
    ~Node.reset
    ~Node.run
    ~Node.set_buffer
    ~Node.set_feedback_dim
    ~Node.set_input_dim
    ~Node.set_output_dim
    ~Node.set_param
    ~Node.set_state_proxy
    ~Node.state
    ~Node.state_proxy
    ~Node.train
    ~Node.with_feedback
    ~Node.with_state
    ~Node.zero_feedback
    ~Node.zero_state

    ~Node.feedback_dim
    ~Node.fitted
    ~Node.has_feedback
    ~Node.hypers
    ~Node.input_dim
    ~Node.is_fb_initialized
    ~Node.is_initialized
    ~Node.is_trainable
    ~Node.is_trained_offline
    ~Node.is_trained_online
    ~Node.name
    ~Node.output_dim
    ~Node.params

ReservoirPy Node API was heavily inspired by Explosion.ai *Thinc*
functional deep learning library [1]_, and *Nengo* core API [2]_.
It also follows some *scikit-learn* schemes and guidelines [3]_.

.. [1] `Thinc <https://thinc.ai/>`_ website
.. [2] `Nengo <https://www.nengo.ai/>`_ website
.. [3] `scikit-learn <https://scikit-learn.org/stable/>`_ website

BackwardFn,
Data,
EmptyInitFn,
ForwardFn,
ForwardInitFn,
PartialBackFn,
Shape,
global_dtype,

"""Initialize a Node with a sequence of inputs/targets."""
X = to_ragged_seq_set(X)

if Y is not None:
    Y = to_ragged_seq_set(Y)
else:
    Y = [None for _ in range(len(X))]

if not node.is_initialized:
    node.initialize(X[0], Y[0])

return X, Y

msg = f"Impossible to initialize node {node.name}: "
in_msg = (
    msg + "input_dim is unknown and no input data x was given "
    "to call/run the node."
)

x_init = y_init = None
if isinstance(x, np.ndarray):
    x_init = np.atleast_2d(check_vector(x, caller=node))
elif isinstance(x, list):
    x_init = list()
    for i in range(len(x)):
        x_init.append(np.atleast_2d(check_vector(x[i], caller=node)))
elif x is None:
    if node.input_dim is not None:
        if hasattr(node.input_dim, "__iter__"):
            x_init = [np.empty((1, d)) for d in node.input_dim]
        else:
            x_init = np.empty((1, node.input_dim))
    else:
        raise RuntimeError(in_msg)

if y is not None:
    y_init = np.atleast_2d(check_vector(y, caller=node))
elif node.output_dim is not None:
    y_init = np.empty((1, node.output_dim))
else:
    # check if output dimension can be inferred from a teacher node
    if node._teacher is not None and node._teacher.output_dim is not None:
        y_init = np.empty((1, node._teacher.output_dim))

return x_init, y_init

"""By default, for offline learners, partial_fit simply stores inputs and
targets, waiting for fit to be called."""

node._X.append(X_batch)

if Y_batch is not None:
    node._Y.append(Y_batch)

return

"""Void feedback initializer. Works in any case."""
fb_dim = None
if isinstance(fb, list):
    fb_dim = tuple([fb.shape[1] for fb in fb])
elif isinstance(fb, np.ndarray):
    fb_dim = fb.shape[1]

node.set_feedback_dim(fb_dim)

"""Node base class.

Parameters
----------
params : dict, optional
    Parameters of the Node. Parameters are mutable, and can be modified
    through learning or by the effect of hyperparameters.
hypers : dict, optional
    Hyperparameters of the Node. Hyperparameters are immutable, and define
    the architecture and properties of the Node.
forward : callable, optional
    A function defining the computation performed by the Node on some data
    point :math:`x_t`, and that would update the Node internal state from
    :math:`s_t` to :math:`s_{t+1}`.
backward : callable, optional
    A function defining an offline learning rule, applied on a whole
    dataset, or on pre-computed values stored in buffers.
partial_backward : callable, optional
    A function defining an offline learning rule, applied on a single batch
    of data.
train : callable, optional
    A function defining an online learning, applied on a single step of
    a sequence or of a timeseries.
initializer : callable, optional
    A function called at first run of the Node, defining the dimensions and
    values of its parameters based on the dimension of input data and its
    hyperparameters.
fb_initializer : callable, optional
    A function called at first run of the Node, defining the dimensions and
    values of its parameters based on the dimension of data received as
    a feedback from another Node.
buffers_initializer : callable, optional
    A function called at the beginning of an offline training session to
    create buffers used to store intermediate results, for batch or
    multi-sequence offline learning.
input_dim : int
    Input dimension of the Node.
output_dim : int
    Output dimension of the Node. Dimension of its state.
feedback_dim :
    Dimension of the feedback signal received by the Node.
name : str
    Name of the Node. It must be a unique identifier.

See also
--------
    Model
        Object used to compose node operations and create computational
        graphs.
"""

_name: str

_state: Optional[np.ndarray]
_state_proxy: Optional[np.ndarray]
_feedback: Optional[DistantFeedback]
_teacher: Optional[DistantFeedback]

_params: Dict[str, Any]
_hypers: Dict[str, Any]
_buffers: Dict[str, Any]

_input_dim: int
_output_dim: int
_feedback_dim: int

_forward: ForwardFn
_backward: BackwardFn
_partial_backward: PartialBackFn
_train: PartialBackFn

_initializer: ForwardInitFn
_buffers_initializer: EmptyInitFn
_feedback_initializer: ForwardInitFn

_trainable: bool
_fitted: bool

_X: List  # For partial_fit default behavior (store first, then fit)
_Y: List

def __init__(
    self,
    params: Dict[str, Any] = None,
    hypers: Dict[str, Any] = None,
    forward: ForwardFn = None,
    backward: BackwardFn = None,
    partial_backward: PartialBackFn = _partial_backward_default,
    train: PartialBackFn = None,
    initializer: ForwardInitFn = None,
    fb_initializer: ForwardInitFn = _initialize_feedback_default,
    buffers_initializer: EmptyInitFn = None,
    input_dim: int = None,
    output_dim: int = None,
    feedback_dim: int = None,
    name: str = None,
    dtype: np.dtype = global_dtype,
):

    self._params = dict() if params is None else params
    self._hypers = dict() if hypers is None else hypers
    # buffers are all node state components that should not live
    # outside the node training loop, like partial computations for
    # linear regressions. They can also be shared across multiple processes
    # when needed.
    self._buffers = dict()

    self._forward = forward
    self._backward = backward
    self._partial_backward = partial_backward
    self._train = train

    self._initializer = initializer
    self._feedback_initializer = fb_initializer
    self._buffers_initializer = buffers_initializer

    self._input_dim = input_dim
    self._output_dim = output_dim
    self._feedback_dim = feedback_dim

    self._name = self._get_name(name)
    self._dtype = dtype

    self._is_initialized = False
    self._is_fb_initialized = False
    self._state_proxy = None
    self._feedback = None
    self._teacher = None
    self._fb_flag = True  # flag is used to trigger distant feedback model update

    self._trainable = self._backward is not None or self._train is not None

    self._fitted = False if self.is_trainable and self.is_trained_offline else True

    self._X, self._Y = [], []

def __lshift__(self, other) -> "_Node":
    return self.link_feedback(other)

def __ilshift__(self, other) -> "_Node":
    return self.link_feedback(other, inplace=True)

def __iand__(self, other):
    raise TypeError(
        f"Impossible to merge nodes in-place: {self} is not a Model instance."
    )

def _flag_feedback(self):
    self._fb_flag = not self._fb_flag

def _unregister_teacher(self):
    self._teacher = None

@property
def input_dim(self):
    """Node input dimension."""
    return self._input_dim

@property
def output_dim(self):
    """Node output and internal state dimension."""
    return self._output_dim

@property
def feedback_dim(self):
    """Node feedback signal dimension."""
    return self._feedback_dim

@property
def is_initialized(self):
    """Returns if the Node is initialized or not."""
    return self._is_initialized

@property
def has_feedback(self):
    """Returns if the Node receives feedback or not."""
    return self._feedback is not None

@property
def is_trained_offline(self):
    """Returns if the Node can be fitted offline or not."""
    return self.is_trainable and self._backward is not None

@property
def is_trained_online(self):
    """Returns if the Node can be trained online or not."""
    return self.is_trainable and self._train is not None

@property
def is_trainable(self):
    """Returns if the Node can be trained."""
    return self._trainable

@is_trainable.setter
def is_trainable(self, value: bool):
    """Freeze or unfreeze the Node. If set to False,
    learning is stopped."""
    if self.is_trained_offline or self.is_trained_online:
        if type(value) is bool:
            self._trainable = value
        else:
            raise TypeError("'is_trainable' must be a boolean.")

@property
def fitted(self):
    """Returns if the Node parameters have fitted already, using an
    offline learning rule. If the node is trained online, returns True."""
    return self._fitted

@property
def is_fb_initialized(self):
    """Returns if the Node feedback initializer has been called already."""
    return self._is_fb_initialized

@property
def dtype(self):
    """Numpy numerical type of node parameters."""
    return self._dtype

@property
def unsupervised(self):
    return False

def state(self) -> Optional[np.ndarray]:
    """Node current internal state.

    Returns
    -------
    array of shape (1, output_dim), optional
        Internal state of the Node.
    """
    if not self.is_initialized:
        return None
    return self._state

def state_proxy(self) -> Optional[np.ndarray]:
    """Returns the internal state frozen to be sent to other Nodes,
    connected through a feedback connection. This prevents any change
    occurring on the Node before feedback have reached the other Node to
    propagate to the other Node to early.

    Returns
    -------
    array of shape (1, output_dim), optional
        Internal state of the Node.
    """
    if self._state_proxy is None:
        return self._state
    return self._state_proxy

def feedback(self) -> np.ndarray:
    """State of the Nodes connected to this Node through feedback
    connections.

    Returns
    -------
    array-like of shape ([n_feedbacks], 1, feedback_dim), optional
        State of the feedback Nodes, i.e. the feedback signal.
    """
    if self.has_feedback:
        return self._feedback()
    else:
        raise RuntimeError(
            f"Node {self} is not connected to any feedback Node or Model."
        )

def set_state_proxy(self, value: np.ndarray = None):
    """Change the frozen state of the Node. Used internally to send
    the current state to feedback receiver Nodes during the next call.

    Parameters
    ----------
    value : array of shape (1, output_dim)
        State to freeze, waiting to be sent to feedback receivers.
    """
    if value is not None:
        if self.is_initialized:
            value = check_one_sequence(
                value, self.output_dim, allow_timespans=False, caller=self
            ).astype(self.dtype)
            self._state_proxy = value
        else:
            raise RuntimeError(f"{self.name} is not initialized yet.")

def set_input_dim(self, value: int):
    """Set the input dimension of the Node. Can only be called once,
    during Node initialization."""
    if not self._is_initialized:
        if self._input_dim is not None and value != self._input_dim:
            raise ValueError(
                f"Impossible to use {self.name} with input "
                f"data of dimension {value}. Node has input "
                f"dimension {self._input_dim}."
            )
        self._input_dim = value
    else:
        raise TypeError(
            f"Input dimension of {self.name} is immutable after initialization."
        )

def set_output_dim(self, value: int):
    """Set the output dimension of the Node. Can only be called once,
    during Node initialization."""
    if not self._is_initialized:
        if self._output_dim is not None and value != self._output_dim:
            raise ValueError(
                f"Impossible to use {self.name} with target "
                f"data of dimension {value}. Node has output "
                f"dimension {self._output_dim}."
            )
        self._output_dim = value
    else:
        raise TypeError(
            f"Output dimension of {self.name} is immutable after initialization."
        )

def set_feedback_dim(self, value: int):
    """Set the feedback dimension of the Node. Can only be called once,
    during Node initialization."""
    if not self.is_fb_initialized:
        self._feedback_dim = value
    else:
        raise TypeError(
            f"Output dimension of {self.name} is immutable after initialization."
        )

def get_param(self, name: str):
    """Get one of the parameters or hyperparameters given its name."""
    if name in self._params:
        return self._params.get(name)
    elif name in self._hypers:
        return self._hypers.get(name)
    else:
        raise AttributeError(f"No attribute named '{name}' found in node {self}")

def set_param(self, name: str, value: Any):
    """Set the value of a parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    value : array-like
        Parameter new value.
    """
    if name in self._params:
        if hasattr(value, "dtype"):
            if issparse(value):
                value.data = value.data.astype(self.dtype)
            else:
                value = value.astype(self.dtype)
        self._params[name] = value
    elif name in self._hypers:
        self._hypers[name] = value
    else:
        raise KeyError(
            f"No param named '{name}' "
            f"in {self.name}. Available params are: "
            f"{list(self._params.keys())}."
        )

def create_buffer(
    self, name: str, shape: Shape = None, data: np.ndarray = None, as_memmap=True
):
    """Create a buffer array on disk, using numpy.memmap. This can be
    used to store transient variables on disk. Typically, called inside
    a `buffers_initializer` function.

    Parameters
    ----------
    name : str
        Name of the buffer array.
    shape : tuple of int, optional
        Shape of the buffer array.
    data : array-like
        Data to store in the buffer array.
    """
    if as_memmap:
        self._buffers[name] = memmap_buffer(self, data=data, shape=shape, name=name)
    else:
        if data is not None:
            self._buffers[name] = data
        else:
            self._buffers[name] = np.empty(shape)

def set_buffer(self, name: str, value: np.ndarray):
    """Dump data in the buffer array.

    Parameters
    ----------
    name : str
        Name of the buffer array.
    value : array-like
        Data to store in the buffer array.
    """
    self._buffers[name][:] = value.astype(self.dtype)

def get_buffer(self, name) -> np.memmap:
    """Get data from a buffer array.

    Parameters
    ----------
    name : str
        Name of the buffer array.

    Returns
    -------
        numpy.memmap
            Data as Numpy memory map.
    """
    if self._buffers.get(name) is None:
        raise AttributeError(f"No buffer named '{name}' in {self}.")
    return self._buffers[name]

def initialize(self, x: Data = None, y: Data = None) -> "Node":
    """Call the Node initializers on some data points.
    Initializers are functions called at first run of the Node,
    defining the dimensions and values of its parameters based on the
    dimension of some input data and its hyperparameters.

    Data point `x` is used to infer the input dimension of the Node.
    Data point `y` is used to infer the output dimension of the Node.

    Parameters
    ----------
    x : array-like of shape ([n_inputs], 1, input_dim)
        Input data.
    y : array-like of shape (1, output_dim)
        Ground truth data. Used to infer output dimension
        of trainable nodes.

    Returns
    -------
    Node
        Initialized Node.
    """
    if not self.is_initialized:
        x_init, y_init = _init_vectors_placeholders(self, x, y)
        self._initializer(self, x=x_init, y=y_init)
        self.reset()
        self._is_initialized = True
    return self

def initialize_feedback(self) -> "Node":
    """Call the Node feedback initializer. The feedback initializer will
    determine feedback dimension given some feedback signal, and initialize
    all parameters related to the feedback connection.

    Feedback sender Node must be initialized, as the feedback initializer
    will probably call the :py:meth:`Node.feedback` method to get
    a sample of feedback signal.

    Returns
    -------
    Node
        Initialized Node.
    """
    if self.has_feedback:
        if not self.is_fb_initialized:
            self._feedback.initialize()
            self._feedback_initializer(self, self.zero_feedback())
            self._is_fb_initialized = True
    return self

def initialize_buffers(self) -> "Node":
    """Call the Node buffer initializer. The buffer initializer will create
    buffer array on demand to store transient values of the parameters,
    typically during training.

    Returns
    -------
    Node
        Initialized Node.
    """
    if self._buffers_initializer is not None:
        if len(self._buffers) == 0:
            self._buffers_initializer(self)

    return self

def clean_buffers(self):
    """Clean Node's buffer arrays."""
    if len(self._buffers) > 0:
        self._buffers = dict()
        clean_tempfile(self)

    # Empty possibly stored inputs and targets in default buffer.
    self._X = self._Y = []

def reset(self, to_state: np.ndarray = None) -> "Node":
    """Reset the last state saved to zero or to
    another state value `to_state`.

    Parameters
    ----------
    to_state : array of shape (1, output_dim), optional
        New state value.

    Returns
    -------
    Node
        Reset Node.
    """
    if to_state is None:
        self._state = self.zero_state()
    else:
        self._state = check_one_sequence(
            to_state, self.output_dim, allow_timespans=False, caller=self
        ).astype(self.dtype)
    return self

@contextmanager
def with_state(
    self, state: np.ndarray = None, stateful: bool = False, reset: bool = False
) -> "Node":
    """Modify the state of the Node using a context manager.
    The modification will have effect only within the context defined,
    before the state returns back to its previous value.

    Parameters
    ----------
    state : array of shape (1, output_dim), optional
        New state value.
    stateful : bool, default to False
        If set to True, then all modifications made in the context manager
        will remain after leaving the context.
    reset : bool, default to False
        If True, the Node will be reset using its :py:meth:`Node.reset`
        method.

    Returns
    -------
    Node
        Modified Node.
    """
    if not self._is_initialized:
        raise RuntimeError(
            f"Impossible to set state of node {self.name}: node"
            f"is not initialized yet."
        )

    current_state = self._state

    if state is None:
        if reset:
            state = self.zero_state()
        else:
            state = current_state

    self.reset(to_state=state)
    yield self

    if not stateful:
        self._state = current_state

@contextmanager
def with_feedback(
    self, feedback: np.ndarray = None, stateful=False, reset=False
) -> "Node":
    """Modify the feedback received or sent by the Node using
    a context manager.
    The modification will have effect only within the context defined,
    before the feedback returns to its previous state.

    If the Node is receiving feedback, then this function will alter the
    state of the Node connected to it through feedback connections.

    If the Node is sending feedback, then this function will alter the
    state (or state proxy, see :py:meth:`Node.state_proxy`) of the Node.

    Parameters
    ----------
    feedback : array of shape (1, feedback_dim), optional
        New feedback signal.
    stateful : bool, default to False
        If set to True, then all modifications made in the context manager
        will remain after leaving the context.
    reset : bool, default to False
        If True, the feedback  will be reset to zero.

    Returns
    -------
    Node
        Modified Node.
    """
    if self.has_feedback:
        if reset:
            feedback = self.zero_feedback()
        if feedback is not None:
            self._feedback.clamp(feedback)

        yield self

    else:  # maybe a feedback sender then?
        current_state_proxy = self._state_proxy

        if feedback is None:
            if reset:
                feedback = self.zero_state()
            else:
                feedback = current_state_proxy

        self.set_state_proxy(feedback)

        yield self

        if not stateful:
            self._state_proxy = current_state_proxy

def zero_state(self) -> np.ndarray:
    """A null state vector."""
    if self.output_dim is not None:
        return np.zeros((1, self.output_dim), dtype=self.dtype)

def zero_feedback(self) -> Optional[Union[List[np.ndarray], np.ndarray]]:
    """A null feedback vector. Returns None if the Node receives
    no feedback."""
    if self._feedback is not None:
        return self._feedback.zero_feedback()
    return None

def link_feedback(
    self, node: _Node, inplace: bool = False, name: str = None
) -> "_Node":
    """Create a feedback connection between the Node and another Node or
    Model.

    Parameters
    ----------
    node : Node or Model
        Feedback sender Node or Model.
    inplace : bool, default to False
        If False, then this function returns a copy of the current Node
        with feedback enabled. If True, feedback is directly added to the
        current Node.
    name : str, optional
        Name of the node copy, if `inplace` is False.

    Returns
    -------
    Node
        A Node with a feedback connection.
    """
    from .ops import link_feedback

    return link_feedback(self, node, inplace=inplace, name=name)

def call(
    self,
    x: Data,
    from_state: np.ndarray = None,
    stateful: bool = True,
    reset: bool = False,
) -> np.ndarray:
    """Call the Node forward function on a single step of data.
    Can update the state of the
    Node.

    Parameters
    ----------
    x : array of shape ([n_inputs], 1, input_dim)
        One single step of input data.
    from_state : array of shape (1, output_dim), optional
        Node state value to use at beginning of computation.
    stateful : bool, default to True
        If True, Node state will be updated by this operation.
    reset : bool, default to False
        If True, Node state will be reset to zero before this operation.

    Returns
    -------
    array of shape (1, output_dim)
        An output vector.
    """
    x, _ = check_xy(
        self,
        x,
        allow_timespans=False,
        allow_n_sequences=False,
    )

    if not self._is_initialized:
        self.initialize(x)

    return call(self, x, from_state=from_state, stateful=stateful, reset=reset)

def run(self, X: np.array, from_state=None, stateful=True, reset=False):
    """Run the Node forward function on a sequence of data.
    Can update the state of the
    Node several times.

    Parameters
    ----------
    X : array-like of shape ([n_inputs], timesteps, input_dim)
        A sequence of data of shape (timesteps, features).
    from_state : array of shape (1, output_dim), optional
        Node state value to use at beginning of computation.
    stateful : bool, default to True
        If True, Node state will be updated by this operation.
    reset : bool, default to False
        If True, Node state will be reset to zero before this operation.

    Returns
    -------
    array of shape (timesteps, output_dim)
        A sequence of output vectors.
    """
    X_, _ = check_xy(
        self,
        X,
        allow_n_sequences=False,
    )

    if isinstance(X_, np.ndarray):
        if not self._is_initialized:
            self.initialize(np.atleast_2d(X_[0]))
        seq_len = X_.shape[0]
    else:  # multiple inputs?
        if not self._is_initialized:
            self.initialize([np.atleast_2d(x[0]) for x in X_])
        seq_len = X_[0].shape[0]

    with self.with_state(from_state, stateful=stateful, reset=reset):
        states = np.zeros((seq_len, self.output_dim))
        for i in progress(range(seq_len), f"Running {self.name}: "):
            if isinstance(X_, (list, tuple)):
                x = [np.atleast_2d(Xi[i]) for Xi in X_]
            else:
                x = np.atleast_2d(X_[i])

            s = call(self, x)
            states[i, :] = s

    return states

def train(
    self,
    X: np.ndarray,
    Y: Union[_Node, np.ndarray] = None,
    force_teachers: bool = True,
    call: bool = True,
    learn_every: int = 1,
    from_state: np.ndarray = None,
    stateful: bool = True,
    reset: bool = False,
) -> np.ndarray:
    """Train the Node parameters using an online learning rule, if
    available.

    Parameters
    ----------
    X : array-like of shape ([n_inputs], timesteps, input_dim)
        Input sequence of data.
    Y : array-like of shape (timesteps, output_dim), optional.
        Target sequence of data. If None, the Node will search a feedback
        signal, or train in an unsupervised way, if possible.
    force_teachers : bool, default to True
        If True, this Node will broadcast the available ground truth signal
        to all Nodes using this Node as a feedback sender. Otherwise,
        the real state of this Node will be sent to the feedback receivers.
    call : bool, default to True
        It True, call the Node and update its state before applying the
        learning rule. Otherwise, use the train method
        on the current state.
    learn_every : int, default to 1
        Time interval at which training must occur, when dealing with a
        sequence of input data. By default, the training method is called
        every time the Node receive an input.
    from_state : array of shape (1, output_dim), optional
        Node state value to use at beginning of computation.
    stateful : bool, default to True
        If True, Node state will be updated by this operation.
    reset : bool, default to False
        If True, Node state will be reset to zero before this operation.

    Returns
    -------
    array of shape (timesteps, output_dim)
        All outputs computed during the training. If `call` is False,
        outputs will be the result of :py:meth:`Node.zero_state`.
    """
    if not self.is_trained_online:
        raise TypeError(f"Node {self} has no online learning rule implemented.")

    X_, Y_ = check_xy(
        self,
        X,
        Y,
        allow_n_sequences=False,
        allow_n_inputs=False,
    )

    if not self._is_initialized:
        x_init = np.atleast_2d(X_[0])
        y_init = None
        if hasattr(Y, "__iter__"):
            y_init = np.atleast_2d(Y_[0])

        self.initialize(x=x_init, y=y_init)
        self.initialize_buffers()

    states = train(
        self,
        X_,
        Y_,
        call_node=call,
        force_teachers=force_teachers,
        learn_every=learn_every,
        from_state=from_state,
        stateful=stateful,
        reset=reset,
    )

    self._unregister_teacher()

    return states

def partial_fit(
    self,
    X_batch: Data,
    Y_batch: Data = None,
    warmup=0,
    **kwargs,
) -> "Node":
    """Partial offline fitting method of a Node.
    Can be used to perform batched fitting or to pre-compute some variables
    used by the fitting method.

    Parameters
    ----------
    X_batch : array-like of shape ([n_inputs], [series], timesteps, input_dim)
        A sequence or a batch of sequence of input data.
    Y_batch : array-like of shape ([series], timesteps, output_dim), optional
        A sequence or a batch of sequence of teacher signals.
    warmup : int, default to 0
        Number of timesteps to consider as warmup and
        discard at the beginning of each timeseries before training.

    Returns
    -------
    Node
        Partially fitted Node.
    """
    if not self.is_trained_offline:
        raise TypeError(f"Node {self} has no offline learning rule implemented.")

    X, Y = check_xy(self, X_batch, Y_batch, allow_n_inputs=False)

    X, Y = _init_with_sequences(self, X, Y)

    self.initialize_buffers()

    for i in range(len(X)):
        X_seq = X[i]
        Y_seq = None
        if Y is not None:
            Y_seq = Y[i]

        if X_seq.shape[0] <= warmup:
            raise ValueError(
                f"Warmup set to {warmup} timesteps, but one timeseries is only "
                f"{X_seq.shape[0]} long."
            )

        if Y_seq is not None:
            self._partial_backward(self, X_seq[warmup:], Y_seq[warmup:], **kwargs)
        else:
            self._partial_backward(self, X_seq[warmup:], **kwargs)

    return self

def fit(self, X: Data = None, Y: Data = None, warmup=0) -> "Node":
    """Offline fitting method of a Node.

    Parameters
    ----------
    X : array-like of shape ([n_inputs], [series], timesteps, input_dim), optional
        Input sequences dataset. If None, the method will try to fit
        the parameters of the Node using the precomputed values returned
        by previous call of :py:meth:`partial_fit`.
    Y : array-like of shape ([series], timesteps, output_dim), optional
        Teacher signals dataset. If None, the method will try to fit
        the parameters of the Node using the precomputed values returned
        by previous call of :py:meth:`partial_fit`, or to fit the Node in
        an unsupervised way, if possible.
    warmup : int, default to 0
        Number of timesteps to consider as warmup and
        discard at the beginning of each timeseries before training.

    Returns
    -------
    Node
        Node trained offline.
    """

    if not self.is_trained_offline:
        raise TypeError(f"Node {self} has no offline learning rule implemented.")

    self._fitted = False

    # Call the partial backward function on the dataset if it is
    # provided all at once.
    if X is not None:
        if self._partial_backward is not None:
            self.partial_fit(X, Y, warmup=warmup)

    elif not self._is_initialized:
        raise RuntimeError(
            f"Impossible to fit node {self.name}: node"
            f"is not initialized, and fit was called "
            f"without input and teacher data."
        )

    self._backward(self, self._X, self._Y)

    self._fitted = True
    self.clean_buffers()

    return self

def copy(
    self, name: str = None, copy_feedback: bool = False, shallow: bool = False
):
    """Returns a copy of the Node.

    Parameters
    ----------
    name : str
        Name of the Node copy.
    copy_feedback : bool, default to False
        If True, also copy the Node feedback senders.
    shallow : bool, default to False
        If False, performs a deep copy of the Node.

    Returns
    -------
    Node
        A copy of the Node.
    """
    if shallow:
        new_obj = copy(self)
    else:
        if self.has_feedback:
            # store feedback node
            fb = self._feedback
            # temporarily remove it
            self._feedback = None

            # copy and restore feedback, deep copy of feedback depends
            # on the copy_feedback parameter only
            new_obj = deepcopy(self)
            new_obj._feedback = fb
            self._feedback = fb

        else:
            new_obj = deepcopy(self)

    if copy_feedback:
        if self.has_feedback:
            fb_copy = deepcopy(self._feedback)
            new_obj._feedback = fb_copy

    n = self._get_name(name)
    new_obj._name = n

    return new_obj

```
# File: _japanese_vowels.txt
```python
"DESCR": "JapaneseVowels.data.html",
"train": "ae.train",
"test": "ae.test",
"train_sizes": "size_ae.train",
"test_sizes": "size_ae.test",

"""Load and parse data from downloaded files."""

X = []
Y = []

data = data.split("\n\n")[:-1]

block_cursor = 0
speaker_cursor = 0
for block in data:

    if block_cursor >= block_numbers[speaker_cursor]:
        block_cursor = 0
        speaker_cursor += 1

    X.append(np.loadtxt(io.StringIO(block)))

    if one_hot_encode:
        Y.append(ONE_HOT_SPEAKERS[speaker_cursor].reshape(1, -1))
    else:
        Y.append(np.array([SPEAKERS[speaker_cursor]]).reshape(1, 1))

    block_cursor += 1

return X, Y

"""Download data from source into the reservoirpy data local directory."""

logger.info(f"Downloading {SOURCE_URL}.")

with urlopen(SOURCE_URL) as zipresp:
    with zipfile.ZipFile(io.BytesIO(zipresp.read())) as zfile:
        zfile.extractall(data_folder)

"""Repeat target label/vector along block's time axis."""

repeated_targets = []
for block, target in zip(blocks, targets):
    timesteps = block.shape[0]
    target_series = np.repeat(target, timesteps, axis=0)
    repeated_targets.append(target_series)

return repeated_targets

one_hot_encode=True, repeat_targets=False, data_folder=None, reload=False

"""Load the Japanese vowels [16]_ dataset.

This is a classic audio discrimination task. Nine male Japanese speakers
pronounced the ` \\ae\\ ` vowel. The task consists in inferring the speaker
identity from the audio recording.

Audio recordings are series of 12 LPC cepstrum coefficient. Series contains
between 7 and 29 timesteps. Each series (or "block") is one utterance of ` \\ae\\ `
vowel from one speaker.

============================   ===============================
Classes                                                      9
Samples per class (training)       30 series of 7-29 timesteps
Samples per class (testing)     29-50 series of 7-29 timesteps
Samples total                                              640
Dimensionality                                              12
Features                                                  real
============================   ===============================

Data is downloaded from:
https://doi.org/10.24432/C5NS47

Parameters
----------
one_hot_encode : bool, default to True
    If True, returns class label as a one-hot encoded vector.
repeat_targets : bool, default to False
    If True, repeat the target label or vector along the time axis of the
    corresponding sample.
data_folder : str or Path-like object, optional
    Local destination of the downloaded data.
reload : bool, default to False
    If True, re-download data from remote repository. Else, if a cached version
    of the dataset exists, use the cached dataset.

Returns
-------
X_train, Y_train, X_test, Y_test
    Lists of arrays of shape (timesteps, features) or (timesteps, target)
    or (target,).

References
----------
.. [16] M. Kudo, J. Toyama and M. Shimbo. (1999).
        "Multidimensional Curve Classification Using Passing-Through Regions".
        Pattern Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.

"""

data_folder = _get_data_folder(data_folder)

complete = True
for file_role, file_name in REMOTE_FILES.items():
    if not (data_folder / file_name).exists():
        complete = False
        break

if reload or not complete:
    _download(data_folder)

data_files = {}
for file_role, file_name in REMOTE_FILES.items():

    with open(data_folder / file_name, "r") as fp:

        if file_role in ["train_sizes", "test_sizes"]:
            data = fp.read().split(" ")
            # remove empty characters and spaces
            data = [
                int(s) for s in filter(lambda s: s not in ["", "\n", " "], data)
            ]

        else:
            data = fp.read()

    data_files[file_role] = data

X_train, Y_train = _format_data(
    data_files["train"], data_files["train_sizes"], one_hot_encode
)

X_test, Y_test = _format_data(
    data_files["test"], data_files["test_sizes"], one_hot_encode
)

if repeat_targets:
    Y_train = _repeat_target(X_train, Y_train)
    Y_test = _repeat_target(X_test, Y_test)

```
# File: model.txt
```python
    ~Model.call
    ~Model.fit
    ~Model.get_node
    ~Model.initialize
    ~Model.initialize_buffers
    ~Model.reset
    ~Model.run
    ~Model.train
    ~Model.update_graph
    ~Model.with_feedback
    ~Model.with_state

    ~Model.data_dispatcher
    ~Model.edges
    ~Model.feedback_nodes
    ~Model.fitted
    ~Model.hypers
    ~Model.input_dim
    ~Model.input_nodes
    ~Model.is_empty
    ~Model.is_initialized
    ~Model.is_trainable
    ~Model.is_trained_offline
    ~Model.is_trained_online
    ~Model.name
    ~Model.node_names
    ~Model.nodes
    ~Model.output_dim
    ~Model.output_nodes
    ~Model.params
    ~Model.trainable_nodes

DataDispatcher,
dispatch,
find_entries_and_exits,
get_offline_subgraphs,
topological_sort,

allocate_returned_states,
build_forward_sumodels,
dist_states_to_next_subgraph,
fold_mapping,
to_data_mapping,

submodel,
complete_model,
offlines,
relations,
x_seq,
y_seq,
warmup,
stateful=True,
reset=False,
return_states=None,
force_teachers=True,

"""Run a submodel and call its partial fit function."""

if not submodel.is_empty:
    x_seq = {n: x for n, x in x_seq.items() if n in submodel.node_names}

    if force_teachers and y_seq is not None:
        y_seq = {
            n: y
            for n, y in y_seq.items()
            if n in [o.name for o in complete_model.nodes if o.is_trained_offline]
        }
    else:
        y_seq = None

    submodel._is_initialized = True
    states = run_submodel(
        complete_model,
        submodel,
        x_seq,
        y_seq,
        return_states=return_states,
        stateful=stateful,
        reset=reset,
    )

    dist_states = dist_states_to_next_subgraph(states, relations)
else:
    dist_states = x_seq

if y_seq is not None:
    for node in offlines:
        node.partial_fit(
            dist_states.get(node.name), y_seq.get(node.name), warmup=warmup
        )
else:
    for node in offlines:
        node.partial_fit(dist_states.get(node.name), warmup=warmup)

return dist_states

model,
submodel,
X: MappedData,
forced_feedbacks: Dict[str, np.ndarray] = None,
from_state: Dict[str, np.ndarray] = None,
stateful=True,
reset=False,
shift_fb=True,
return_states: Sequence[str] = None,

X_, forced_feedbacks_ = to_data_mapping(submodel, X, forced_feedbacks)

submodel._initialize_on_sequence(X_[0], forced_feedbacks_[0])

states = []
for X_seq, fb_seq in zip(X_, forced_feedbacks_):
    with model.with_state(reset=reset, stateful=stateful):
        states_seq = model._run(
            X_seq,
            fb_seq,
            from_state=from_state,
            stateful=stateful,
            shift_fb=shift_fb,
            return_states=return_states,
            submodel=submodel,
        )

        states.append(states_seq)

return fold_mapping(submodel, states, return_states)

"""Function applied by a :py:class:`Model` instance.

This function if basically a composition of the forward functions of all
nodes involved in the model architecture. For instance, let :math:`f`
be the forward function of a first node, and let :math:`g` be the forward
function of a second node. If first node is connected to second node in a
model, then the model forward function :math:`h` will compute, at each
timestep :math:`t` of a timeseries :math:`X`:

.. math::

    h(X_t) = g(f(X_t)) = (g \\circ f)(X_t)

Parameters
----------
model : Model
    A :py:class:`Model` instance.
x : dict or array-like of shape ([n_inputs], 1, input_dim)
    A vector corresponding to a timestep of data, or
    a dictionary mapping node names to input vectors.

Returns
-------
array-like of shape (n_outputs, 1, output_dim)
    New states of all terminal nodes of the model, i.e. its output.
"""
data = model.data_dispatcher.load(x)

for node in model.nodes:
    _base.call(node, data[node].x)

return [out_node.state() for out_node in model.output_nodes]

"""Training function for a Model. Run all train functions of all online
nodes within the Model. Nodes have already been called. Only training
is performed."""

data = model.data_dispatcher.load(X=x, Y=y)

for node in model.nodes:
    if node.is_trained_online:
        _base.train(
            node,
            data[node].x,
            data[node].y,
            force_teachers=force_teachers,
            call_node=False,
        )

"""Initializes a :py:class:`Model` instance at runtime, using samples of
data to infer all :py:class:`Node` dimensions.

Parameters
----------
model : Model
    A :py:class:`Model` instance.
x : numpy.ndarray or dict of numpy.ndarray
    A vector of shape `(1, ndim)` corresponding to a timestep of data, or
    a dictionary mapping node names to vector of shapes
    `(1, ndim of corresponding node)`.
y : numpy.ndarray or dict of numpy.ndarray
    A vector of shape `(1, ndim)` corresponding to a timestep of target
    data or feedback, or a dictionary mapping node names to vector of
    shapes `(1, ndim of corresponding node)`.
"""
data = model.data_dispatcher.load(x, y)

# first, probe network to init forward flow
# (no real call, only zero states)
for node in model.nodes:
    node.initialize(x=data[node].x, y=data[node].y)

# second, probe feedback demanding nodes to
# init feedback flow
for fb_node in model.feedback_nodes:
    fb_node.initialize_feedback()

"""Model base class.

Parameters
----------
nodes : list of Node, optional
    Nodes to include in the Model.
edges : list of (Node, Node), optional
    Edges between Nodes in the graph. An edge between a
    Node A and a Node B is created as a tuple (A, B).
name : str, optional
    Name of the Model.
"""

_node_registry: Dict[str, _Node]
_nodes: List[_Node]
_edges: List[Tuple[_Node, _Node]]
_inputs: List[_Node]
_outputs: List[_Node]
_dispatcher: "DataDispatcher"

def __init__(
    self,
    nodes: Sequence[_Node] = None,
    edges: Sequence[Tuple[_Node, _Node]] = None,
    name: str = None,
):

    if nodes is None:
        nodes = tuple()
    if edges is None:
        edges = tuple()

    self._forward = forward
    self._train = train
    self._initializer = initializer

    self._name = self._get_name(name)

    nodes, edges = self._concat_multi_inputs(nodes, edges)

    self._edges = edges

    # always maintain nodes in topological order
    if len(nodes) > 0:
        self._inputs, self._outputs = find_entries_and_exits(nodes, edges)
        self._nodes = topological_sort(nodes, edges, self._inputs)
    else:
        self._inputs = list()
        self._outputs = list()
        self._nodes = nodes

    self._is_initialized = False
    self._trainable = any([n.is_trainable for n in nodes])
    self._fitted = all([n.fitted for n in nodes])

    self._params = {n.name: n.params for n in nodes}
    self._hypers = {n.name: n.hypers for n in nodes}

    self._node_registry = {n.name: n for n in self.nodes}

    self._dispatcher = DataDispatcher(self)

def __repr__(self):
    klas = self.__class__.__name__
    nodes = [n.name for n in self._nodes]
    return f"'{self.name}': {klas}('" + "', '".join(nodes) + "')"

def __getitem__(self, item):
    return self.get_node(item)

def __iand__(self, other) -> "Model":
    from .ops import merge

    return merge(self, other, inplace=True)

@staticmethod
def _concat_multi_inputs(nodes, edges):
    from .ops import concat_multi_inputs

    return concat_multi_inputs(nodes, edges)

def _check_if_only_online(self):
    if any([n.is_trained_offline and not n.fitted for n in self.nodes]):
        raise RuntimeError(
            f"Impossible to train model {self.name} using "
            f"online method: model contains untrained "
            f"offline nodes."
        )

def _load_proxys(self, keep=False):
    """Save states of all nodes into their state_proxy"""
    for node in self._nodes:
        if keep and node._state_proxy is not None:
            continue
        node._state_proxy = node.state()

def _clean_proxys(self):
    """Destroy state_proxy of all nodes"""
    for node in self._nodes:
        node._state_proxy = None

def _initialize_on_sequence(self, X=None, Y=None):
    if not self._is_initialized:
        x_init = None
        if X is not None:
            if is_mapping(X):
                x_init = {name: np.atleast_2d(x[0]) for name, x in X.items()}
            else:
                x_init = np.atleast_2d(X[0])

        y_init = None
        if Y is not None:
            if is_mapping(Y):
                y_init = {name: np.atleast_2d(y[0]) for name, y in Y.items()}
            else:
                y_init = np.atleast_2d(Y[0])

        self.initialize(x_init, y_init)

def _call(self, x=None, return_states=None, submodel=None, *args, **kwargs):

    if submodel is None:
        submodel = self

    self._forward(submodel, x)

    state = {}
    if return_states == "all":
        for node in submodel.nodes:
            state[node.name] = node.state()

    elif hasattr(return_states, "__iter__"):
        for name in return_states:
            state[name] = submodel[name].state()

    else:
        if len(submodel.output_nodes) > 1:
            for out_node in submodel.output_nodes:
                state[out_node.name] = out_node.state()
        else:
            state = submodel.output_nodes[0].state()

    return state

def _run(
    self,
    X,
    feedback,
    from_state=None,
    stateful=True,
    shift_fb=True,
    return_states=None,
    submodel=None,
):
    if submodel is None:
        submodel = self

    states = allocate_returned_states(submodel, X, return_states)

    seq = progress(
        dispatch(X, feedback, shift_fb=shift_fb),
        f"Running {self.name}",
        total=len(X),
    )

    with self.with_state(from_state, stateful=stateful):
        # load proxys after state update to make it have an
        # impact on feedback
        self._load_proxys(keep=True)
        for i, (x, forced_fb, _) in enumerate(seq):

            with self.with_feedback(forced_fb):
                state = submodel._call(x, return_states=return_states)

            if is_mapping(state):
                for name, value in state.items():
                    states[name][i, :] = value
            else:
                states[submodel.output_nodes[0].name][i, :] = state

            self._load_proxys()

    self._clean_proxys()

    return states

def _unregister_teachers(self):
    """Remove teacher nodes refs from student nodes."""
    for node in self.trainable_nodes:
        node._teacher = None

def update_graph(
    self,
    new_nodes: Sequence[_Node],
    new_edges: Sequence[Tuple[_Node, _Node]],
) -> "Model":
    """Update current Model's with new nodes and edges, inplace (a copy
    is not performed).

    Parameters
    ----------
    new_nodes : list of Node
        New nodes.
    new_edges : list of (Node, Node)
        New edges between nodes.

    Returns
    -------
    Model
        The updated Model.
    """
    nodes = list(set(new_nodes) | set(self.nodes))
    edges = list(set(new_edges) | set(self.edges))

    nodes, edges = self._concat_multi_inputs(nodes, edges)

    self._nodes = nodes
    self._edges = edges

    self._params = {n.name: n.params for n in self._nodes}
    self._hypers = {n.name: n.hypers for n in self._nodes}

    self._inputs, self._outputs = find_entries_and_exits(self._nodes, self._edges)
    self._nodes = topological_sort(self._nodes, self._edges, self._inputs)
    self._node_registry = {n.name: n for n in self.nodes}

    self._dispatcher = DataDispatcher(self)

    self._fitted = all([n.fitted for n in self.nodes])
    self._is_initialized = False

    return self

def get_node(self, name: str) -> _Node:
    """Get Node in Model, by name.

    Parameters
    ----------
    name : str
        Node name.

    Returns
    -------
    Node
        The requested Node.

    Raises
    ------
    KeyError
        Node not found.
    """
    if self._node_registry.get(name) is not None:
        return self._node_registry[name]
    else:
        raise KeyError(f"No node named '{name}' found in model {self.name}.")

@property
def nodes(self) -> List[_Node]:
    """Nodes in the Model, in topological order."""
    return self._nodes

@property
def node_names(self):
    """Names of all the Nodes in the Model."""
    return list(self._node_registry.keys())

@property
def edges(self):
    """All edges between Nodes, in the form (sender, receiver)."""
    return self._edges

@property
def input_dim(self):
    """Input dimension of the Model;
    input dimensions of all input Nodes."""
    if self.is_initialized:
        dims = [n.input_dim for n in self.input_nodes]
        if len(dims) == 0:
            return 0
        elif len(dims) < 2:
            return dims[0]
        return dims
    else:
        return None

@property
def output_dim(self):
    """Output dimension of the Model;
    output dimensions of all output Nodes."""
    if self.is_initialized:
        dims = [n.output_dim for n in self.output_nodes]
        if len(dims) == 0:
            return 0
        elif len(dims) < 2:
            return dims[0]
        return dims
    else:
        return None

@property
def input_nodes(self):
    """First Nodes in the graph held by the Model."""
    return self._inputs

@property
def output_nodes(self):
    """Last Nodes in the graph held by the Model."""
    return self._outputs

@property
def trainable_nodes(self):
    """Returns all offline and online
    trainable Nodes in the Model."""
    return [n for n in self.nodes if n.is_trainable]

@property
def feedback_nodes(self):
    """Returns all Nodes equipped with a feedback connection
    in the Model."""
    return [n for n in self.nodes if n.has_feedback]

@property
def data_dispatcher(self):
    """DataDispatcher object of the Model. Used to
    distribute data to Nodes when
    calling/running/fitting the Model."""
    return self._dispatcher

@property
def is_empty(self):
    """Returns True if the Model contains no Nodes."""
    return len(self.nodes) == 0

@property
def is_trainable(self) -> bool:
    """Returns True if at least one Node in the Model is trainable."""
    return any([n.is_trainable for n in self.nodes])

@is_trainable.setter
def is_trainable(self, value):
    """Freeze or unfreeze trainable Nodes in the Model."""
    trainables = [
        n for n in self.nodes if n.is_trained_offline or n.is_trained_online
    ]
    for node in trainables:
        node.is_trainable = value

@property
def is_trained_online(self) -> bool:
    """Returns True if all nodes are online learners."""
    return all([n.is_trained_online or n.fitted for n in self.nodes])

@property
def is_trained_offline(self) -> bool:
    """Returns True if all nodes are offline learners."""
    return all([n.is_trained_offline or n.fitted for n in self.nodes])

@property
def fitted(self) -> bool:
    """Returns True if all nodes are fitted."""
    return all([n.fitted for n in self.nodes])

@contextmanager
def with_state(
    self, state: Dict[str, np.ndarray] = None, stateful=False, reset=False
) -> "Model":
    """Modify the state of one or several Nodes in the Model
    using a context manager.
    The modification will have effect only within the context defined,
    before all states return to their previous value.

    Parameters
    ----------
    state : dict of arrays of shape (1, output_dim), optional
        Pairs of keys and values, where keys are Model nodes names and
        value are new ndarray state vectors.
    stateful : bool, default to False
        If set to True, then all modifications made in the context manager
        will remain after leaving the context.
    reset : bool, default to False
        If True, all Nodes will be reset using their :py:meth:`Node.reset`
        method.

    Returns
    -------
    Model
        Modified Model.
    """
    if state is None and not reset:
        current_state = None
        if not stateful:
            current_state = {n.name: n.state() for n in self.nodes}
        yield self
        if not stateful:
            self.reset(to_state=current_state)
    elif isinstance(state, np.ndarray):
        raise TypeError(
            f"Impossible to set state of {self.name} with "
            f"an array. State should be a dict mapping state "
            f"to a Node name within the model."
        )
    else:
        if state is None:
            state = {}

        with ExitStack() as stack:
            for node in self.nodes:
                value = state.get(node.name)
                stack.enter_context(
                    node.with_state(value, stateful=stateful, reset=reset)
                )
            yield self

@contextmanager
def with_feedback(
    self, feedback: Dict[str, np.ndarray] = None, stateful=False, reset=False
) -> "Model":
    """Modify the feedback received or sent by Nodes in the Model using
    a context manager.
    The modification will have effect only within the context defined,
    before the feedbacks return to their previous states.

    If the Nodes are receiving feedback, then this function will alter the
    states of the Nodes connected to it through feedback connections.

    If the Nodes are sending feedback, then this function will alter the
    states (or state proxies, see :py:meth:`Node.state_proxy`) of the
    Nodes.

    Parameters
    ----------
    feedback : dict of arrays of shape (1, output_dim), optional
        Pairs of keys and values, where keys are Model nodes names and
        value are new ndarray feedback vectors.
    stateful : bool, default to False
        If set to True, then all modifications made in the context manager
        will remain after leaving the context.
    reset : bool, default to False
        If True, all feedbacks  will be reset to zero.

    Returns
    -------
    Model
        Modified Model.
    """

    if feedback is None and not reset:
        yield self
        return

    elif feedback is not None:
        with ExitStack() as stack:
            for node in self.nodes:
                value = feedback.get(node.name)
                # maybe node does not send feedback but receives it?
                if value is None and node.has_feedback:
                    value = feedback.get(node._feedback.name)
                stack.enter_context(
                    node.with_feedback(value, stateful=stateful, reset=reset)
                )

            yield self
    else:
        yield self
        return

def reset(self, to_state: Dict[str, np.ndarray] = None):
    """Reset the last state saved to zero for all Nodes in
    the Model or to other state values.

    Parameters
    ----------
    to_state : dict of arrays of shape (1, output_dim), optional
        Pairs of keys and values, where keys are Model nodes names and
        value are new ndarray state vectors.
    """
    if to_state is None:
        for node in self.nodes:
            node.reset()
    else:
        for node_name, current_state in to_state.items():
            self[node_name].reset(to_state=current_state)
    return self

def initialize(self, x=None, y=None) -> "Model":
    """Call the Model initializers on some data points.
    Model will be virtually run to infer shapes of all nodes given
    inputs and targets vectors.

    Parameters
    ----------
    x : dict or array-like of shape ([n_inputs], 1, input_dim)
        Input data.
    y : dict or array-like of shape ([n_outputs], 1, output_dim)
        Ground truth data. Used to infer output dimension
        of trainable nodes.

    Returns
    -------
    Model
        Initialized Model.
    """
    self._is_initialized = False
    self._initializer(self, x=x, y=y)
    self.reset()
    self._is_initialized = True
    return self

def initialize_buffers(self) -> "Model":
    """Call all Nodes buffer initializers. Buffer initializers will create
    buffer arrays on demand to store transient values of the parameters,
    typically during training.

    Returns
    -------
    Model
        Initialized Model.
    """
    for node in self.nodes:
        if node._buffers_initializer is not None:
            node.initialize_buffers()

def call(
    self,
    x: MappedData,
    forced_feedback: MappedData = None,
    from_state: Dict[str, np.ndarray] = None,
    stateful=True,
    reset=False,
    return_states: Sequence[str] = None,
) -> MappedData:
    """Call the Model forward function on a single step of data.
    Model forward function is a composition of all its Nodes forward
    functions.

    Can update the state its Nodes.

    Parameters
    ----------
    x : dict or array-like of shape ([n_inputs], 1, input_dim)
        One single step of input data. If dict, then
        pairs of keys and values, where keys are Model input
        nodes names and values are single steps of input data.
    forced_feedback: dict of arrays of shape ([n_feedbacks], 1, feedback_dim), optional
        Pairs of keys and values, where keys are Model nodes names and
        value are feedback vectors to force into the nodes.
    from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim), optional
        Pairs of keys and values, where keys are Model nodes names and
        value are new ndarray state vectors.
    stateful : bool, default to True
        If True, Node state will be updated by this operation.
    reset : bool, default to False
        If True, Nodes states will be reset to zero before this operation.
    return_states: list of str, optional
        Names of Nodes from which to return states as output.

    Returns
    -------
    dict or array-like of shape ([n_outputs], 1, output_dim)
        An output vector or pairs of keys and values
        where keys are output nodes names and values
        are corresponding output vectors.
    """

    x, _ = check_xy(
        self,
        x,
        allow_timespans=False,
        allow_n_sequences=False,
    )

    if not self._is_initialized:
        self.initialize(x)

    with self.with_state(from_state, stateful=stateful, reset=reset):
        # load current states in proxys interfaces accessible
        # through feedback. These proxys are not updated during the
        # graph call.
        self._load_proxys(keep=True)

        # maybe load forced feedbacks in proxys?
        with self.with_feedback(forced_feedback, stateful=stateful, reset=reset):
            state = self._call(x, return_states)

    # wash states proxys
    self._clean_proxys()

    return state

def run(
    self,
    X: MappedData,
    forced_feedbacks: Dict[str, np.ndarray] = None,
    from_state: Dict[str, np.ndarray] = None,
    stateful=True,
    reset=False,
    shift_fb=True,
    return_states: Sequence[str] = None,
) -> MappedData:
    """Run the Model forward function on a sequence of data.
    Model forward function is a composition of all its Nodes forward
    functions.
    Can update the state of the
    Nodes several times.

    Parameters
    ----------
    X : dict or array-like of shape ([n_inputs], timesteps, input_dim)
        A sequence of input data.
        If dict, then pairs of keys and values, where keys are Model input
        nodes names and values are sequence of input data.
    forced_feedbacks: dict of array-like of shape ([n_feedbacks], timesteps, feedback_dim)
        Pairs of keys and values, where keys are Model nodes names and
        value are sequences of feedback vectors to force into the nodes.
    from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim)
        Pairs of keys and values, where keys are Model nodes names and
        value are new ndarray state vectors.
    stateful : bool, default to True
        If True, Node state will be updated by this operation.
    reset : bool, default to False
        If True, Nodes states will be reset to zero before this operation.
    shift_fb: bool, defaults to True
        If True, then forced feedbacks are fed to nodes with a
        one timestep delay. If forced feedbacks are a sequence
        of target vectors matching the sequence of input
        vectors, then this parameter should be set to True.
    return_states: list of str, optional
        Names of Nodes from which to return states as output.

    Returns
    -------
    dict or array-like of shape ([n_outputs], timesteps, output_dim)
        A sequence of output vectors or pairs of keys and values
        where keys are output nodes names and values
        are corresponding sequences of output vectors.
    """
    X_, forced_feedbacks_ = to_data_mapping(self, X, forced_feedbacks)

    self._initialize_on_sequence(X_[0], forced_feedbacks_[0])

    states = []
    for X_seq, fb_seq in zip(X_, forced_feedbacks_):
        with self.with_state(reset=reset, stateful=stateful):
            states_seq = self._run(
                X_seq,
                fb_seq,
                from_state=from_state,
                stateful=stateful,
                shift_fb=shift_fb,
                return_states=return_states,
            )

            states.append(states_seq)

    return fold_mapping(self, states, return_states)

def train(
    self,
    X,
    Y=None,
    force_teachers=True,
    learn_every=1,
    from_state=None,
    stateful=True,
    reset=False,
    return_states=None,
) -> MappedData:
    """Train all online Nodes in the Model
    using their online learning rule.

    Parameters
    ----------
    X : dict or array-like of shape ([n_inputs], timesteps, input_dim)
        Input sequence of data. If dict, then pairs
        of keys and values, where keys are Model input
        nodes names and values are sequence of input data.
    Y : dict or array-like of shape ([onlines], timesteps, onlines.output_dim), optional.
        Target sequence of data.
        If dict, then pairs of keys and values, where keys are Model
        online trainable nodes names values are sequences
        of target data. If None, the Nodes will search a feedback
        signal, or train in an unsupervised way, if possible.
    force_teachers : bool, default to True
        If True, this Model will broadcast the available ground truth
        signal
        to all online nodes sending feedback to other nodes. Otherwise,
        the real state of these nodes will be sent to the feedback
        receivers
        during training.
    learn_every : int, default to 1
        Time interval at which training must occur, when dealing with a
        sequence of input data. By default, the training method is called
        every time the Model receive an input.
    from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim)
        Pairs of keys and values, where keys are Model nodes names and
        value are new ndarray state vectors.
    stateful : bool, default to True
        If True, Node state will be updated by this operation.
    reset : bool, default to False
        If True, Nodes states will be reset to zero before this operation.
    return_states: list of str, optional
        Names of Nodes from which to return states as output.

    Returns
    -------
    dict or array-like of shape ([n_outputs], timesteps, output_dim)
        All outputs computed during the training
        or pairs of keys and values
        where keys are output nodes names and values
        are corresponding outputs computed.
        If `call` is False,
        outputs will be null vectors.
    """

    self._check_if_only_online()

    X_, Y_ = check_xy(self, X, Y, allow_n_sequences=False)

    self._initialize_on_sequence(X_, Y_)

    states = allocate_returned_states(self, X_, return_states)

    dispatched_data = dispatch(
        X_, Y_, return_targets=True, force_teachers=force_teachers
    )

    with self.with_state(from_state, stateful=stateful, reset=reset):
        # load proxys after state update to make it have an
        # impact on feedback
        self._load_proxys(keep=True)
        for i, (x, forced_feedback, y) in enumerate(dispatched_data):

            if not force_teachers:
                forced_feedback = None

            with self.with_feedback(forced_feedback):
                state = self._call(x, return_states=return_states)

            # reload feedbacks from training. Some nodes may need
            # updated feedback signals to train.
            self._load_proxys()

            if i % learn_every == 0 or len(X) == 1:
                self._train(self, x=x, y=y, force_teachers=force_teachers)

            if is_mapping(state):
                for name, value in state.items():
                    states[name][i, :] = value
            else:
                states[self.output_nodes[0].name][i, :] = state

    self._clean_proxys()
    self._unregister_teachers()

    # dicts are only relevant if model has several outputs (len > 1) or
    # if we want to return states from hidden nodes
    if len(states) == 1 and return_states is None:
        return states[self.output_nodes[0].name]

    return states

def fit(
    self,
    X: MappedData,
    Y: MappedData,
    warmup=0,
    force_teachers=True,
    from_state=None,
    stateful=True,
    reset=False,
) -> "Model":
    """Fit all offline Nodes in the Model
    using their offline learning rule.

    Parameters
    ----------
    X : dict or array-like of shape ([n_inputs], [series], timesteps, input_dim)
        Input sequence of data. If dict, then pairs
        of keys and values, where keys are Model input
        nodes names and values are sequence of input data.
    Y : dict or array-like of shape ([offlines], [series], timesteps, offlines.output_dim)
        Target sequence of data. If dict, then pairs
        of keys and values, where keys are Model input
        nodes names and values are sequence of target data.
    warmup : int, default to 0
        Number of timesteps to consider as warmup and
        discard at the beginning of each timeseries before training.
    force_teachers : bool, default to True
        If True, this Model will broadcast the available ground truth
        signal
        to all online nodes sending feedback to other nodes. Otherwise,
        the real state of these nodes will be sent to the feedback
        receivers
        during training.
    from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim)
        Pairs of keys and values, where keys are Model nodes names and
        value are new ndarray state vectors.
    stateful : bool, default to True
        If True, Node state will be updated by this operation.
    reset : bool, default to False
        If True, Nodes states will be reset to zero before this operation.

    Returns
    -------
    Model
        Model trained offline.
    """

    if not any([n for n in self.trainable_nodes if n.is_trained_offline]):
        raise TypeError(
            f"Impossible to fit model {self} offline: "
            "no offline nodes found in model."
        )

    X, Y = to_data_mapping(self, X, Y)

    self._initialize_on_sequence(X[0], Y[0])
    self.initialize_buffers()

    subgraphs = get_offline_subgraphs(self.nodes, self.edges)

    trained = set()
    next_X = None

    with self.with_state(from_state, reset=reset, stateful=stateful):
        for i, ((nodes, edges), relations) in enumerate(subgraphs):
            submodel, offlines = build_forward_sumodels(nodes, edges, trained)

            if next_X is not None:
                for j in range(len(X)):
                    X[j].update(next_X[j])

            return_states = None
            if len(relations) > 0:
                return_states = list(relations.keys())

            # next inputs for next submodel
            next_X = []
            seq = progress(X, f"Running {self.name}")

            _partial_fit_fn = partial(
                run_and_partial_fit,
                force_teachers=force_teachers,
                complete_model=self,
                submodel=submodel,
                warmup=warmup,
                reset=reset,
                offlines=offlines,
                relations=relations,
                stateful=stateful,
                return_states=return_states,
            )

            # for seq/batch in dataset
            for x_seq, y_seq in zip(seq, Y):
                next_X += [_partial_fit_fn(x_seq=x_seq, y_seq=y_seq)]

            for node in offlines:
                if verbosity():
                    print(f"Fitting node {node.name}...")
                node.fit()

            trained |= set(offlines)

    return self

def copy(self, *args, **kwargs):
    raise NotImplementedError()

"""A FrozenModel is a Model that can not be
linked to other nodes or models.
"""

def __init__(self, *args, **kwargs):
    super(FrozenModel, self).__init__(*args, **kwargs)

```
# File: test_base.txt
```python
if isinstance(val, np.ndarray):
    return str(val.shape)
if isinstance(val, list):
    return f"list[{len(val)}]"
if isinstance(val, dict):
    return str(val)
else:
    return val

"x,kwargs,expects",
[
    (np.ones((1, 5)), {}, np.ones((1, 5))),
    (np.ones((5,)), {}, np.ones((1, 5))),
    (np.ones((1, 5)), {"expected_dim": 6}, ValueError),
    ("foo", {}, TypeError),
    (1, {}, np.ones((1, 1))),
    ([np.ones((1, 5)), np.ones((1, 6))], {}, TypeError),
    (np.ones((5, 5)), {}, np.ones((5, 5))),
    (np.ones((2, 5)), {}, np.ones((2, 5))),
    (np.ones((2, 5)), {"expected_dim": 5}, np.ones((2, 5))),
    (np.ones((2, 5)), {"expected_dim": 2}, ValueError),
    (np.ones((2, 5, 2)), {"expected_dim": 2}, ValueError),
    (np.ones((2, 5, 2)), {"expected_dim": (5, 2)}, np.ones((2, 5, 2))),
    (
        [np.ones((2, 5)), np.ones((2, 6))],
        {},
        TypeError,
    ),
],
ids=idfn,

if isinstance(expects, (np.ndarray, list)):
    x = check_one_sequence(x, **kwargs)
    assert_equal(x, expects)
else:
    with pytest.raises(expects):
        x = check_one_sequence(x, **kwargs)

"x,kwargs,expects",
[
    (np.ones((1, 5)), {}, np.ones((1, 5))),
    (np.ones((5,)), {}, np.ones((1, 5))),
    (np.ones((1, 5)), {"expected_dim": (6,)}, ValueError),
    ("foo", {}, TypeError),
    (1, {}, np.ones((1, 1))),
    (
        [np.ones((1, 5)), np.ones((1, 6))],
        {},
        [np.ones((1, 5)), np.ones((1, 6))],
    ),
    (
        [np.ones((5,)), np.ones((6,))],
        {},
        [np.ones((1, 5)), np.ones((1, 6))],
    ),
    (
        [np.ones((1, 5)), np.ones((1, 6))],
        {"expected_dim": (5, 6)},
        [np.ones((1, 5)), np.ones((1, 6))],
    ),
    (
        [np.ones((1, 5)), np.ones((1, 6))],
        {"expected_dim": (6, 5)},
        ValueError,
    ),
    ([np.ones((1, 5)), np.ones((1, 6))], {"expected_dim": (5,)}, ValueError),
    ([np.ones((1, 5)), np.ones((1, 6))], {"expected_dim": (5, 8)}, ValueError),
    (["foo", np.ones((1, 6))], {}, TypeError),
    ([1, np.ones((1, 6))], {}, [np.ones((1, 1)), np.ones((1, 6))]),
    (np.ones((5, 5)), {}, np.ones((5, 5))),
    (np.ones((2, 5)), {}, np.ones((2, 5))),
    (np.ones((2, 5)), {"expected_dim": (5,)}, np.ones((2, 5))),
    (np.ones((2, 5)), {"expected_dim": (2,)}, ValueError),
    (
        [np.ones((2, 5)), np.ones((2, 6))],
        {},
        [np.ones((2, 5)), np.ones((2, 6))],
    ),
    (
        [np.ones((5, 5)), np.ones((5, 6))],
        {"expected_dim": (5, 6)},
        [np.ones((5, 5)), np.ones((5, 6))],
    ),
    ([np.ones((5, 5)), np.ones((5, 6))], {"expected_dim": (5,)}, ValueError),
    ([np.ones((8, 5)), np.ones((8, 6))], {"expected_dim": (5, 8)}, ValueError),
    ([np.ones((7, 5)), np.ones((8, 6))], {"expected_dim": (5, 6)}, ValueError),
    (
        [np.ones((2, 7, 5)), np.ones((2, 8))],
        {"expected_dim": ((7, 5), 8)},
        [np.ones((2, 7, 5)), np.ones((2, 8))],
    ),
    ([np.ones((2, 5)) for _ in range(5)], {}, [np.ones((2, 5)) for _ in range(5)]),
    ([np.ones((5,)) for _ in range(5)], {}, [np.ones((1, 5)) for _ in range(5)]),
    (
        [np.ones((2, 5, 7)) for _ in range(5)],
        {"expected_dim": ((5, 7),)},
        [np.ones((2, 5, 7)) for _ in range(5)],
    ),
    (
        [np.ones((5, 2, 7)), np.ones((5, 2, 6))],
        {"expected_dim": (7, 6)},
        [np.ones((5, 2, 7)), np.ones((5, 2, 6))],
    ),
    (
        [
            [np.ones((i, 7)) for i in range(10)],
            [np.ones((i, 6)) for i in range(10)],
        ],
        {},
        [
            [np.ones((i, 7)) for i in range(10)],
            [np.ones((i, 6)) for i in range(10)],
        ],
    ),
    (
        [
            [np.ones((i, 7)) for i in range(10)],
            [np.ones((i, 6)) for i in range(10)],
        ],
        {"expected_dim": (7, 6)},
        [
            [np.ones((i, 7)) for i in range(10)],
            [np.ones((i, 6)) for i in range(10)],
        ],
    ),
    (np.ones((5, 7, 6)), {"expected_dim": (7, 6)}, ValueError),
],
ids=idfn,

if isinstance(expects, (np.ndarray, list)):
    x = check_n_sequences(x, **kwargs)
    assert_equal(x, expects)
else:
    with pytest.raises(expects):
        x = check_n_sequences(x, **kwargs)

"caller,x,y,kwargs,expects",
[
    (PlusNode(), np.ones((1, 5)), None, {}, (np.ones((1, 5)), None)),
    (PlusNode(), np.ones((5,)), None, {}, (np.ones((1, 5)), None)),
    (PlusNode(), np.ones((1, 5)), None, {"input_dim": (6,)}, ValueError),
    (PlusNode(input_dim=6), np.ones((1, 5)), None, {}, ValueError),
    (
        PlusNode(),
        np.ones((1, 5)),
        np.ones((1, 5)),
        {},
        (np.ones((1, 5)), np.ones((1, 5))),
    ),
    (
        PlusNode(),
        np.ones((5,)),
        np.ones((6,)),
        {},
        (np.ones((1, 5)), np.ones((1, 6))),
    ),
    (
        Offline(),
        np.ones((1, 5)),
        np.ones((1, 6)),
        {"output_dim": (7,)},
        ValueError,
    ),
    (PlusNode(output_dim=7), np.ones((1, 5)), np.ones((1, 6)), {}, ValueError),
    (
        PlusNode(name="plus0") & MinusNode(name="minus0"),
        {"plus0": np.ones((1, 5)), "minus0": np.ones((1, 6))},
        None,
        {},
        ({"plus0": np.ones((1, 5)), "minus0": np.ones((1, 6))}, None),
    ),
    (
        PlusNode(name="plus1") & MinusNode(name="minus1"),
        np.ones((1, 5)),
        None,
        {},
        ({"plus1": np.ones((1, 5)), "minus1": np.ones((1, 5))}, None),
    ),
    (
        PlusNode(name="plus2") & MinusNode(name="minus2"),
        {"plus2": np.ones((1, 5))},
        None,
        {},
        ValueError,
    ),
    (OnlineNode(output_dim=6), np.ones((1, 5)), np.ones((1, 5)), {}, ValueError),
    (
        OnlineNode(),
        np.ones((1, 5)),
        PlusNode(input_dim=5, output_dim=5).initialize(),
        {},
        (np.ones((1, 5)), None),
    ),
    (OnlineNode(), PlusNode(), np.ones((1, 5)), {}, TypeError),
    (Offline(), np.ones((1, 5)), PlusNode(), {}, TypeError),
    (MinusNode(), PlusNode(), np.ones((1, 5)), {}, TypeError),
    (
        OnlineNode(output_dim=6),
        np.ones((1, 5)),
        PlusNode(input_dim=5, output_dim=5).initialize(),
        {},
        ValueError,
    ),
    (
        PlusNode(name="plus3") & MinusNode(name="minus3"),
        {"plus3": np.ones((1, 5)), "minus3": np.ones((1, 6))},
        np.ones((1, 5)),
        {},
        ({"plus3": np.ones((1, 5)), "minus3": np.ones((1, 6))}, None),
    ),
    (
        PlusNode(name="plus4") >> OnlineNode(name="online4"),
        np.ones((1, 5)),
        PlusNode(),
        {},
        ({"plus4": np.ones((1, 5))}, None),
    ),
    (
        PlusNode(name="plus5") >> OnlineNode(name="online51")
        & MinusNode(name="minus5") >> OnlineNode(name="online52"),
        {"plus5": np.ones((1, 5)), "minus5": np.ones((1, 5))},
        {"online51": PlusNode(), "online52": np.ones((1, 5))},
        {},
        (
            {"plus5": np.ones((1, 5)), "minus5": np.ones((1, 5))},
            {"online52": np.ones((1, 5))},
        ),
    ),
    (
        PlusNode(name="plus6") >> OnlineNode(name="online61")
        & MinusNode(name="minus6") >> Offline(name="offline62"),
        {"plus6": np.ones((1, 5)), "minus6": np.ones((1, 5))},
        {"offline62": PlusNode(), "online61": np.ones((1, 5))},
        {},
        TypeError,
    ),
    (
        PlusNode(name="plus8") >> OnlineNode(name="online81")
        & MinusNode(name="minus8") >> Offline(name="offline82"),
        {"plus8": PlusNode(), "minus8": np.ones((1, 5))},
        {"offline82": PlusNode(), "online81": np.ones((1, 5))},
        {},
        TypeError,
    ),
    (
        PlusNode(name="plus7")
        >> Offline().fit(np.ones((10, 5)), np.ones((10, 5)))
        >> Offline(name="off7"),
        np.ones((1, 5)),
        {"off7": np.ones((1, 5))},
        {},
        ({"plus7": np.ones((1, 5))}, {"off7": np.ones((1, 5))}),
    ),
],
ids=idfn,

if isinstance(expects, (np.ndarray, list, dict, tuple)):
    x, y = check_xy(caller, x, y, **kwargs)
    assert_equal(x, expects[0])
    if y is not None:
        assert_equal(y, expects[1])
else:
    with pytest.raises(expects):
        x = check_xy(caller, x, y, **kwargs)

sender = PlusNode(input_dim=5, output_dim=5)
fb = DistantFeedback(sender, feedback_node)

fb.initialize()

assert sender.is_initialized
assert_equal(sender.state_proxy(), fb())

fb = DistantFeedback(plus_node, feedback_node)

with pytest.raises(RuntimeError):
    fb.initialize()

plus = PlusNode(input_dim=5, output_dim=5) >> Inverter()
minus = MinusNode(output_dim=5)
sender = plus >> minus
fb = DistantFeedback(sender, feedback_node)

fb.initialize()

assert sender.is_initialized
assert_equal(minus.state_proxy(), fb())

fb = DistantFeedback(plus_node, feedback_node)

with pytest.raises(RuntimeError):
    fb.initialize()

plus = PlusNode()
minus = MinusNode()
sender = plus >> minus
fb = DistantFeedback(sender, feedback_node)

```
# File: base reservoir.txt
```python
"""Reservoir base forward function.

Computes: s[t+1] = W.r[t] + Win.(u[t] + noise) + Wfb.(y[t] + noise) + bias
"""
W = reservoir.W
Win = reservoir.Win
bias = reservoir.bias

g_in = reservoir.noise_in
dist = reservoir.noise_type
noise_gen = reservoir.noise_generator

pre_s = W @ r + Win @ (u + noise_gen(dist=dist, shape=u.shape, gain=g_in)) + bias

if reservoir.has_feedback:
    Wfb = reservoir.Wfb
    g_fb = reservoir.noise_out
    h = reservoir.fb_activation

    y = reservoir.feedback().reshape(-1, 1)
    y = h(y) + noise_gen(dist=dist, shape=y.shape, gain=g_fb)

    pre_s += Wfb @ y

return np.array(pre_s)

"""Reservoir with internal activation function:

.. math::

    r[n+1] = (1 - \\alpha) \\cdot r[t] + \\alpha
        \\cdot f (W_{in} \\cdot u[n] + W \\cdot r[t])

where :math:`r[n]` is the state and the output of the reservoir."""
lr = reservoir.lr
f = reservoir.activation
dist = reservoir.noise_type
g_rc = reservoir.noise_rc
noise_gen = reservoir.noise_generator

u = x.reshape(-1, 1)
r = reservoir.state().T

s_next = (
    np.multiply((1 - lr), r.T).T
    + np.multiply(lr, f(reservoir_kernel(reservoir, u, r)).T).T
    + noise_gen(dist=dist, shape=r.shape, gain=g_rc)
)

return s_next.T

"""Reservoir with external activation function:

.. math::

    x[n+1] = (1 - \\alpha) \\cdot x[t] + \\alpha
        \\cdot f (W_{in} \\cdot u[n] + W \\cdot r[t])

    r[n+1] = f(x[n+1])

where :math:`x[n]` is the internal state of the reservoir and :math:`r[n]`
is the response of the reservoir."""
lr = reservoir.lr
f = reservoir.activation
dist = reservoir.noise_type
g_rc = reservoir.noise_rc
noise_gen = reservoir.noise_generator

u = x.reshape(-1, 1)
r = reservoir.state().T
s = reservoir.internal_state.T

s_next = (
    np.multiply((1 - lr), s.T).T
    + np.multiply(lr, reservoir_kernel(reservoir, u, r).T).T
    + noise_gen(dist=dist, shape=r.shape, gain=g_rc)
)

reservoir.set_param("internal_state", s_next.T)

return f(s_next).T

reservoir,
x=None,
y=None,
sr=None,
input_scaling=None,
bias_scaling=None,
input_connectivity=None,
rc_connectivity=None,
W_init=None,
Win_init=None,
bias_init=None,
input_bias=None,
seed=None,

if x is not None:
    reservoir.set_input_dim(x.shape[1])

    dtype = reservoir.dtype
    dtype_msg = (
        "Data type {} not understood in {}. {} should be an array or a "
        "callable returning an array."
    )

    if is_array(W_init):
        W = W_init
        if W.shape[0] != W.shape[1]:
            raise ValueError(
                "Dimension mismatch inside W: "
                f"W is {W.shape} but should be "
                f"a square matrix."
            )

        if W.shape[0] != reservoir.output_dim:
            reservoir._output_dim = W.shape[0]
            reservoir.hypers["units"] = W.shape[0]

    elif callable(W_init):
        W = W_init(
            reservoir.output_dim,
            reservoir.output_dim,
            sr=sr,
            connectivity=rc_connectivity,
            dtype=dtype,
            seed=seed,
        )
    else:
        raise ValueError(dtype_msg.format(str(type(W_init)), reservoir.name, "W"))

    reservoir.set_param("units", W.shape[0])
    reservoir.set_param("W", W.astype(dtype))

    out_dim = reservoir.output_dim

    Win_has_bias = False
    if is_array(Win_init):
        Win = Win_init

        msg = (
            f"Dimension mismatch in {reservoir.name}: Win input dimension is "
            f"{Win.shape[1]} but input dimension is {x.shape[1]}."
        )

        # is bias vector inside Win ?
        if Win.shape[1] == x.shape[1] + 1:
            if input_bias:
                Win_has_bias = True
            else:
                bias_msg = (
                    " It seems Win has a bias column, but 'input_bias' is False."
                )
                raise ValueError(msg + bias_msg)
        elif Win.shape[1] != x.shape[1]:
            raise ValueError(msg)

        if Win.shape[0] != out_dim:
            raise ValueError(
                f"Dimension mismatch in {reservoir.name}: Win internal dimension "
                f"is {Win.shape[0]} but reservoir dimension is {out_dim}"
            )

    elif callable(Win_init):
        Win = Win_init(
            reservoir.output_dim,
            x.shape[1],
            input_scaling=input_scaling,
            connectivity=input_connectivity,
            dtype=dtype,
            seed=seed,
        )
    else:
        raise ValueError(
            dtype_msg.format(str(type(Win_init)), reservoir.name, "Win")
        )

    if input_bias:
        if not Win_has_bias:
            if callable(bias_init):
                bias = bias_init(
                    reservoir.output_dim,
                    1,
                    input_scaling=bias_scaling,
                    connectivity=input_connectivity,
                    dtype=dtype,
                    seed=seed,
                )
            elif is_array(bias_init):
                bias = bias_init
                if bias.shape[0] != reservoir.output_dim or (
                    bias.ndim > 1 and bias.shape[1] != 1
                ):
                    raise ValueError(
                        f"Dimension mismatch in {reservoir.name}: bias shape is "
                        f"{bias.shape} but should be {(reservoir.output_dim, 1)}"
                    )
            else:
                raise ValueError(
                    dtype_msg.format(str(type(bias_init)), reservoir.name, "bias")
                )
        else:
            bias = Win[:, :1]
            Win = Win[:, 1:]
    else:
        bias = zeros(reservoir.output_dim, 1, dtype=dtype)

    reservoir.set_param("Win", Win.astype(dtype))
    reservoir.set_param("bias", bias.astype(dtype))
    reservoir.set_param("internal_state", reservoir.zero_state())

reservoir,
feedback=None,
Wfb_init=None,
fb_scaling=None,
fb_connectivity=None,
seed=None,

if reservoir.has_feedback:
    fb_dim = feedback.shape[1]
    reservoir.set_feedback_dim(fb_dim)

    if is_array(Wfb_init):
        Wfb = Wfb_init
        if not fb_dim == Wfb.shape[1]:
            raise ValueError(
                "Dimension mismatch between Wfb and feedback "
                f"vector in {reservoir.name}: Wfb is "
                f"{Wfb.shape} "
                f"and feedback is {(1, fb_dim)} "
                f"({fb_dim} != {Wfb.shape[0]})"
            )

        if not Wfb.shape[0] == reservoir.output_dim:
            raise ValueError(
                f"Dimension mismatch between Wfb and W in "
                f"{reservoir.name}: Wfb is {Wfb.shape} and "
                f"W is "
                f"{reservoir.W.shape} ({Wfb.shape[1]} "
                f"!= {reservoir.output_dim})"
            )

    elif callable(Wfb_init):
        Wfb = Wfb_init(
            reservoir.output_dim,
            fb_dim,
            input_scaling=fb_scaling,
            connectivity=fb_connectivity,
            seed=seed,
            dtype=reservoir.dtype,
        )
    else:
        raise ValueError(
            f"Data type {type(Wfb_init)} not understood "
            f"for matrix initializer 'Wfb_init' in "
            f"{reservoir.name}. Wfb should be an array "
            f"or a callable returning an array."
        )

```
# File: sklearn_node.txt
```python
instances = readout.params.get("instances")
if type(instances) is not list:
    return instances.predict(X)
else:
    return np.concatenate([instance.predict(X) for instance in instances], axis=-1)

# Concatenate all the batches as one np.ndarray
# of shape (timeseries*timesteps, features)
X_ = np.concatenate(X, axis=0)
Y_ = np.concatenate(Y, axis=0)

instances = readout.params.get("instances")
if type(instances) is not list:
    if readout.output_dim > 1:
        # Multi-output node and multi-output sklearn model
        instances.fit(X_, Y_)
    else:
        # Y_ should have 1 feature so we reshape to
        # (timeseries, ) to avoid scikit-learn's DataConversionWarning
        instances.fit(X_, Y_[..., 0])
else:
    for i, instance in enumerate(instances):
        instance.fit(X_, Y_[..., i])

if x is not None:
    in_dim = x.shape[1]
    if readout.output_dim is not None:
        out_dim = readout.output_dim
    elif y is not None:
        out_dim = y.shape[1]
    else:
        raise RuntimeError(
            f"Impossible to initialize {readout.name}: "
            f"output dimension was not specified at "
            f"creation, and no teacher vector was given."
        )

    readout.set_input_dim(in_dim)
    readout.set_output_dim(out_dim)

    first_instance = readout.model(**deepcopy(model_hypers))
    # If there are multiple output but the specified model doesn't support
    # multiple outputs, we create an instance of the model for each output.
    if out_dim > 1 and not first_instance._get_tags().get("multioutput"):
        instances = [
            readout.model(**deepcopy(model_hypers)) for i in range(out_dim)
        ]
        readout.set_param("instances", instances)
    else:
        readout.set_param("instances", first_instance)

    return

"""
A node interfacing a scikit-learn linear model that can be used as an offline
readout node.

The ScikitLearnNode takes a scikit-learn model as parameter and creates a
node with the specified model.

We currently support classifiers (like
:py:class:`sklearn.linear_model.LogisticRegression` or
:py:class:`sklearn.linear_model.RidgeClassifier`) and regressors (like
:py:class:`sklearn.linear_model.Lasso` or
:py:class:`sklearn.linear_model.ElasticNet`).

For more information on the above-mentioned estimators,
please visit scikit-learn linear model API reference
<https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model>`_

:py:attr:`ScikitLearnNode.params` **list**

================== =================================================================
``instances``      Instance(s) of the model class used to fit and predict. If
                    :py:attr:`ScikitLearnNode.output_dim` > 1 and the model doesn't
                    support multi-outputs, `instances` is a list of instances, one
                    for each output feature.
================== =================================================================

:py:attr:`ScikitLearnNode.hypers` **list**

================== =================================================================
``model``          (class) Underlying scikit-learn model.
``model_hypers``   (dict) Keyword arguments for the scikit-learn model.
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
model : str, optional
    Node name.
model_hypers
    (dict) Additional keyword arguments for the scikit-learn model.

Example
-------
>>> from reservoirpy.nodes import Reservoir, ScikitLearnNode
>>> from sklearn.linear_model import Lasso
>>> reservoir = Reservoir(units=100)
>>> readout = ScikitLearnNode(model=Lasso, model_hypers={"alpha":1e-5})
>>> model = reservoir >> readout
"""

def __init__(self, model, model_hypers=None, output_dim=None, **kwargs):
    if model_hypers is None:
        model_hypers = {}

    if not hasattr(model, "fit"):
        model_name = model.__name__
        raise AttributeError(
            f"Specified model {model_name} has no method called 'fit'."
        )
    if not hasattr(model, "predict"):
        model_name = model.__name__
        raise AttributeError(
            f"Specified model {model_name} has no method called 'predict'."
        )

    # Ensure reproducibility
    # scikit-learn currently only supports RandomState
    if (
        model_hypers.get("random_state") is None
        and "random_state" in model.__init__.__kwdefaults__
    ):

        generator = rand_generator()
        model_hypers.update(
            {
                "random_state": np.random.RandomState(
                    seed=generator.integers(1 << 32)
                )
            }
        )

```
# File: activationsfunc.txt
```python
get_function
identity
sigmoid
tanh
relu
softmax
softplus

"""Vectorize a function to apply it
on arrays.
"""
vect = np.vectorize(func)

@wraps(func)
def vect_wrapper(*args, **kwargs):
    u = np.asanyarray(args)
    v = vect(u)
    return v[0]

return vect_wrapper

"""Return an activation function from name.

Parameters
----------
name : str
    Name of the activation function.
    Can be one of {'softmax', 'softplus',
    'sigmoid', 'tanh', 'identity', 'relu'} or
    their respective short names {'smax', 'sp',
    'sig', 'id', 're'}.

Returns
-------
callable
    An activation function.
"""
index = {
    "softmax": softmax,
    "softplus": softplus,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "identity": identity,
    "relu": relu,
    "smax": softmax,
    "sp": softplus,
    "sig": sigmoid,
    "id": identity,
    "re": relu,
}

if index.get(name) is None:
    raise ValueError(f"Function name must be one of {[k for k in index.keys()]}")
else:
    return index[name]

"""Softmax activation function.

.. math::

    y_k = \\frac{e^{\\beta x_k}}{\\sum_{i=1}^{n} e^{\\beta x_i}}

Parameters
----------
x : array
    Input array.
beta: float, default to 1.0
    Beta parameter of softmax.
Returns
-------
array
    Activated vector.
"""
_x = np.asarray(x)
return np.exp(beta * _x) / np.exp(beta * _x).sum()

"""Softplus activation function.

.. math::

    f(x) = \\mathrm{ln}(1 + e^{x})

Can be used as a smooth version of ReLU.

Parameters
----------
x : array
    Input array.
Returns
-------
array
    Activated vector.
"""
return np.log(1 + np.exp(x))

"""Sigmoid activation function.

.. math::

    f(x) = \\frac{1}{1 + e^{-x}}

Parameters
----------
x : array
    Input array.
Returns
-------
array
    Activated vector.
"""
if x < 0:
    u = np.exp(x)
    return u / (u + 1)
return 1 / (1 + np.exp(-x))

"""Hyperbolic tangent activation function.

.. math::

    f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

Parameters
----------
x : array
    Input array.
Returns
-------
array
    Activated vector.
"""
return np.tanh(x)

"""Identity function.

.. math::

    f(x) = x

Provided for convenience.

Parameters
----------
x : array
    Input array.
Returns
-------
array
    Activated vector.
"""
return x

"""ReLU activation function.

.. math::

    f(x) = x ~~ \\mathrm{if} ~~ x > 0 ~~ \\mathrm{else} ~~ 0

```
# File: test_node.txt
```python
assert plus_node.name == "PlusNode-0"
assert plus_node.params["c"] is None
assert plus_node.hypers["h"] == 1
assert plus_node.input_dim is None
assert plus_node.output_dim is None
assert not plus_node.is_initialized
assert hasattr(plus_node, "c")
assert hasattr(plus_node, "h")
assert plus_node.state() is None

assert plus_node.get_param("c") is None
assert plus_node.get_param("h") == 1

plus_node.set_param("c", 1)

assert plus_node.get_param("c") == 1

with pytest.raises(AttributeError):
    plus_node.get_param("foo")

with pytest.raises(KeyError):
    plus_node.set_param("foo", 1)

plus_node.params["a"] = 2

assert plus_node.get_param("a") == 2
plus_node.set_param("a", 3)
assert plus_node.get_param("a") == 3
assert plus_node.a == 3
assert plus_node.c == 1
assert plus_node.h == 1

plus_node.h = 5
assert plus_node.h == 5

data = np.zeros((1, 5))

res = plus_node(data)

assert_array_equal(res, data + 2)
assert plus_node.is_initialized
assert plus_node.input_dim == 5
assert plus_node.output_dim == 5
assert plus_node.c == 1

data = np.zeros((1, 8))

with pytest.raises(ValueError):
    plus_node(data)

with pytest.raises(TypeError):
    plus_node.set_input_dim(9)
with pytest.raises(TypeError):
    plus_node.set_output_dim(45)

plus_noinit = PlusNode(input_dim=5)
plus_noinit.initialize()

assert plus_noinit.input_dim == 5
assert plus_noinit.c == 1
assert_array_equal(plus_noinit.state(), np.zeros((1, 5)))

multiinput = MultiInput(input_dim=(5, 2))
multiinput.initialize()

assert multiinput.input_dim == (5, 2)

with pytest.raises(RuntimeError):
    plus_noinit = PlusNode()
    plus_noinit.initialize()

data = np.zeros((1, 5))

plus_node.set_input_dim(5)

plus_node.initialize()

assert plus_node.is_initialized
assert plus_node.input_dim == 5
assert plus_node.output_dim == 5
assert plus_node.c == 1

data = np.zeros((1, 8))

with pytest.raises(ValueError):
    plus_node(data)

with pytest.raises(TypeError):
    plus_node.set_input_dim(9)
with pytest.raises(TypeError):
    plus_node.set_output_dim(45)

data = np.zeros((1, 5))
res = plus_node(data)

assert_array_equal(res, data + 2)
assert plus_node.state() is not None
assert_array_equal(data + 2, plus_node.state())

res2 = plus_node(data)
assert_array_equal(res2, data + 4)
assert_array_equal(plus_node.state(), data + 4)

res3 = plus_node(data, stateful=False)
assert_array_equal(res3, data + 6)
assert_array_equal(plus_node.state(), data + 4)

res4 = plus_node(data, reset=True)
assert_array_equal(res4, res)
assert_array_equal(plus_node.state(), data + 2)

data = np.zeros((1, 5))
res = plus_node(data)

# input size mismatch
with pytest.raises(ValueError):
    data = np.zeros((1, 6))
    plus_node(data)

# input size mismatch in run,
# no matter how many timesteps are given
with pytest.raises(ValueError):
    data = np.zeros((5, 6))
    plus_node.run(data)

with pytest.raises(ValueError):
    data = np.zeros((1, 6))
    plus_node.run(data)

# no timespans in call, only single timesteps
with pytest.raises(ValueError):
    data = np.zeros((2, 5))
    plus_node(data)

data = np.zeros((1, 5))

with pytest.raises(RuntimeError):
    with plus_node.with_state(np.ones((1, 5))):
        plus_node(data)

plus_node(data)
assert_array_equal(plus_node.state(), data + 2)

with plus_node.with_state(np.ones((1, 5))):
    res_w = plus_node(data)
    assert_array_equal(res_w, data + 3)
assert_array_equal(plus_node.state(), data + 2)

with plus_node.with_state(np.ones((1, 5)), stateful=True):
    res_w = plus_node(data)
    assert_array_equal(res_w, data + 3)
assert_array_equal(plus_node.state(), data + 3)

with plus_node.with_state(reset=True):
    res_w = plus_node(data)
    assert_array_equal(res_w, data + 2)
assert_array_equal(plus_node.state(), data + 3)

with pytest.raises(ValueError):
    with plus_node.with_state(np.ones((1, 8))):
        plus_node(data)

data = np.zeros((3, 5))
res = plus_node.run(data)
expected = np.array([[2] * 5, [4] * 5, [6] * 5])

assert_array_equal(res, expected)
assert_array_equal(res[-1][np.newaxis, :], plus_node.state())

res2 = plus_node.run(data, stateful=False)
expected2 = np.array([[8] * 5, [10] * 5, [12] * 5])

assert_array_equal(res2, expected2)
assert_array_equal(res[-1][np.newaxis, :], plus_node.state())

res3 = plus_node.run(data, reset=True)

assert_array_equal(res3, expected)
assert_array_equal(res[-1][np.newaxis, :], plus_node.state())

X = np.ones((10, 5)) * 0.5
Y = np.ones((10, 5))

assert offline_node.b == 0

offline_node.partial_fit(X, Y)

assert_array_equal(offline_node.get_buffer("b"), np.array([0.5]))

offline_node.fit()

assert_array_equal(offline_node.b, np.array([0.5]))

X = np.ones((10, 5)) * 2.0
Y = np.ones((10, 5))

offline_node.fit(X, Y)

assert_array_equal(offline_node.b, np.array([1.0]))

X = [np.ones((10, 5)) * 2.0] * 3
Y = [np.ones((10, 5))] * 3

offline_node.fit(X, Y)

assert_array_equal(offline_node.b, np.array([3.0]))

offline_node.partial_fit(X, Y)

assert_array_equal(offline_node.get_buffer("b"), np.array([3.0]))

X = np.ones((10, 5))

assert unsupervised_node.b == 0

unsupervised_node.partial_fit(X)

assert_array_equal(unsupervised_node.get_buffer("b"), np.array([1.0]))

unsupervised_node.fit()

assert_array_equal(unsupervised_node.b, np.array([1.0]))

X = np.ones((10, 5)) * 2.0

unsupervised_node.fit(X)

assert_array_equal(unsupervised_node.b, np.array([2.0]))

X = [np.ones((10, 5)) * 2.0] * 3

unsupervised_node.fit(X)

assert_array_equal(unsupervised_node.b, np.array([6.0]))

unsupervised_node.partial_fit(X)

assert_array_equal(unsupervised_node.get_buffer("b"), np.array([6.0]))

X = np.ones((10, 5))

assert online_node.b == 0

online_node.train(X)

assert_array_equal(online_node.b, np.array([10.0]))

X = np.ones((10, 5)) * 2.0

online_node.train(X)

assert_array_equal(online_node.b, np.array([30.0]))

X = [np.ones((10, 5)) * 2.0] * 3

with pytest.raises(TypeError):
    online_node.train(X)

X = np.ones((10, 5))
Y = np.ones((10, 5))

assert online_node.b == 0

online_node.train(X, Y)

assert_array_equal(online_node.b, np.array([20.0]))

X = np.ones((10, 5)) * 2.0

online_node.train(X, Y)

assert_array_equal(online_node.b, np.array([50.0]))

X = [np.ones((10, 5)) * 2.0] * 3

with pytest.raises(TypeError):
    online_node.train(X, Y)

X = np.ones((10, 5))
Y = np.ones((10, 5))

assert online_node.b == 0

online_node.train(X, Y, learn_every=2)

assert_array_equal(online_node.b, np.array([10.0]))

X = np.ones((10, 5)) * 2.0

online_node.train(X, Y, learn_every=2)

assert_array_equal(online_node.b, np.array([25.0]))

X = np.ones((1, 5))

# using not initialized node
with pytest.raises(RuntimeError):
    online_node.train(X, plus_node)

plus_node(np.ones((1, 5)))

online_node.train(X, plus_node)

assert_array_equal(online_node.b, np.array([4.0]))

X = np.ones((10, 5))
Y = np.ones((10, 5))

with pytest.raises(TypeError):
    online_node.fit(X, Y)

with pytest.raises(TypeError):
    plus_node.fit(X, Y)

with pytest.raises(TypeError):
    online_node.partial_fit(X, Y)

with pytest.raises(TypeError):
    offline_node.train(X, Y)

with pytest.raises(TypeError):
    plus_node.train(X, Y)

X = np.ones((10, 5))
Y = np.ones((10, 5))

with pytest.raises(ValueError):
    offline_node.fit(X, Y, warmup=10)

X = np.ones((10, 5))
Y = np.ones((10, 5))

basic_offline_node.partial_fit(X, Y, warmup=2)
assert_array_equal(basic_offline_node._X[0], X[2:])

multi_noinit = MultiInput(input_dim=(5, 2))
multi_noinit.initialize()

with pytest.raises(RuntimeError):
    multi_noinit = MultiInput()
    multi_noinit.initialize()

x = [np.ones((1, 5)), np.ones((1, 2))]

r = multiinput(x)

assert r.shape == (1, 7)
assert multiinput.input_dim == (5, 2)

multi_noinit = MultiInput()
x = [np.ones((2, 5)), np.ones((2, 2))]
r = multi_noinit.run(x)

assert multi_noinit.input_dim == (5, 2)

with pytest.raises(RuntimeError):
    feedback_node.feedback()

inv_notinit = Inverter(input_dim=5, output_dim=5)

feedback_node <<= inv_notinit

data = np.ones((1, 5))

feedback_node(data)

assert_array_equal(feedback_node.feedback(), inv_notinit.state())

inv_notinit = Inverter(input_dim=5, output_dim=5)
feedback_node <<= inv_notinit

data = np.ones((1, 5))

feedback_node.initialize_feedback()
res = feedback_node(data)

fb = feedback_node.feedback()
inv_state = inv_notinit.state()

assert_array_equal(inv_state, fb)

inv_notinit = Inverter(input_dim=5, output_dim=5)
plus_noinit = PlusNode(input_dim=5, output_dim=5)

# default feedback initializer (plus_node is not supposed to handle feedback)
plus_noinit <<= inv_notinit
plus_noinit.initialize_feedback()
res = plus_noinit(data)

fb = plus_noinit.feedback()
inv_state = inv_notinit.state()

assert_array_equal(inv_state, fb)

feedback_node <<= plus_node >> inverter_node

with pytest.raises(RuntimeError):
    feedback_node.initialize_feedback()

with pytest.raises(RuntimeError):
    data = np.ones((1, 5))
    feedback_node(data)

plus_node.initialize(data)

feedback_node.initialize_feedback()

fb = feedback_node.feedback()
inv_state = inverter_node.state()

assert_array_equal(inv_state, fb)

feedback_node, plus_node, minus_node, inverter_node

feedback_node <<= plus_node >> minus_node >> inverter_node

with pytest.raises(RuntimeError):
    feedback_node.initialize_feedback()

with pytest.raises(RuntimeError):
    data = np.ones((1, 5))
    feedback_node(data)

plus_node.initialize(data)
feedback_node.initialize_feedback()

fb = feedback_node.feedback()
inv_state = inverter_node.state()

```
# File: test_ridge.txt
```python
node = Ridge(10, ridge=1e-8)

data = np.ones((1, 100))
res = node(data)

assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)
assert node.ridge == 1e-8

data = np.ones((10000, 100))
res = node.run(data)

assert res.shape == (10000, 10)

node = Ridge(10, ridge=1e-8)

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))
res = node.fit(X, Y)

assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

node = Ridge(10, ridge=1e-8)

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = node.partial_fit(x, y)

node.fit()

assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

data = np.ones((100, 100))
res = node.run(data)

assert res.shape == (100, 10)

readout = Ridge(10, ridge=1e-8)
reservoir = Reservoir(100)

esn = reservoir >> readout

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))
res = esn.fit(X, Y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)

data = np.ones((100, 100))
res = esn.run(data)

assert res.shape == (100, 10)

readout = Ridge(10, ridge=1e-8)
reservoir = Reservoir(100)

esn = reservoir >> readout

reservoir <<= readout

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))
res = esn.fit(X, Y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)
assert reservoir.Wfb.shape == (100, 10)

data = np.ones((100, 100))
res = esn.run(data)

assert res.shape == (100, 10)

reservoir1 = Reservoir(100, input_dim=5, name="h1")
readout1 = Ridge(ridge=1e-8, name="r1")

reservoir2 = Reservoir(100, name="h2")
readout2 = Ridge(ridge=1e-8, name="r2")

esn = reservoir1 >> readout1 >> reservoir2 >> readout2

X, Y = np.ones((1, 200, 5)), {
    "r1": np.ones((1, 200, 10)),
    "r2": np.ones((1, 200, 3)),
}

res = esn.fit(X, Y)

assert readout1.Wout.shape == (100, 10)
assert readout1.bias.shape == (1, 10)

assert readout2.Wout.shape == (100, 3)
assert readout2.bias.shape == (1, 3)

assert reservoir1.Win.shape == (100, 5)
assert reservoir2.Win.shape == (100, 10)

data = np.ones((100, 5))
res = esn.run(data)

assert res.shape == (100, 3)

if sys.platform in ["win32", "cygwin"] and sys.version_info < (3, 8):
    # joblib>=1.3.0 & Windows & Python<3.8 & loky combined are incompatible
    # see https://github.com/joblib/loky/issues/411
    return

process_count = 4 * os.cpu_count()

rng = np.random.default_rng(seed=42)
x = rng.random((40000, 10))
y = x[:, 2::-1] + rng.random((40000, 3)) / 10
x_run = rng.random((20, 10))

def run_ridge(i):
    readout = Ridge(ridge=1e-8)
    return readout.fit(x, y).run(x_run)

parallel = Parallel(n_jobs=process_count, return_as="generator")
results = list(parallel(delayed(run_ridge)(i) for i in range(process_count)))

```
# File: model_utils.txt
```python
"""Separate unfitted offline nodes from fitted nodes and gather all fitted
nodes in submodels."""
from ..model import Model

offline_nodes = [
    n for n in nodes if n.is_trained_offline and n not in already_trained
]

forward_nodes = list(set(nodes) - set(offline_nodes))
forward_edges = [edge for edge in edges if edge[1] not in offline_nodes]

submodel = Model(forward_nodes, forward_edges, name=f"SubModel-{uuid4()}")

submodel.already_trained = already_trained

return submodel, offline_nodes

"""Map submodel output state vectors to input nodes of next submodel.

Edges between first and second submodel are stored in 'relations'.
"""
dist_states = {}
for curr_node, next_nodes in relations.items():
    if len(next_nodes) > 1:
        for next_node in next_nodes:
            if dist_states.get(next_node) is None:
                dist_states[next_node] = list()
            dist_states[next_node].append(states[curr_node])
    else:
        dist_states[next_nodes[0]] = states[curr_node]

return dist_states

"""Allocate output states matrices."""
seq_len = inputs[list(inputs.keys())[0]].shape[0]

# pre-allocate states
if return_states == "all":
    states = {n.name: np.zeros((seq_len, n.output_dim)) for n in model.nodes}
elif isinstance(return_states, Iterable):
    states = {
        n.name: np.zeros((seq_len, n.output_dim))
        for n in [model[name] for name in return_states]
    }
else:
    states = {n.name: np.zeros((seq_len, n.output_dim)) for n in model.output_nodes}

return states

"""Convert dataset from mapping/array of sequences
to lists of mappings of sequences."""
# data is a dict
if is_mapping(data):
    new_data = {}
    for name, datum in data.items():
        if not is_sequence_set(datum):
            # all sequences must at least be 2D (seq length, num features)
            # 1D sequences are converted to (1, num features) by default.
            new_datum = [np.atleast_2d(datum)]
        else:
            new_datum = datum
        new_data[name] = new_datum
    return new_data
# data is an array or a list
else:
    if not is_sequence_set(data):
        if data.ndim < 3:
            return [np.atleast_2d(data)]
        else:
            return data
    else:
        return data

"""Map input/target data to input/trainable nodes in the model."""
data = to_ragged_seq_set(data)
if not is_mapping(data):
    if io_type == "input":
        data_map = {n.name: data for n in nodes}
    elif io_type == "target":
        # Remove unsupervised or already fitted nodes from the mapping
        data_map = {n.name: data for n in nodes if not n.unsupervised}
    else:
        raise ValueError(
            f"Unknown io_type: '{io_type}'. "
            f"Accepted io_types are 'input' and 'target'."
        )
else:
    data_map = data.copy()

return data_map

"""Convert a mapping of sequence lists into a list of sequence to nodes mappings."""
seq_numbers = [len(data_map[n]) for n in data_map.keys()]
if len(np.unique(seq_numbers)) > 1:
    seq_numbers = {n: len(data_map[n]) for n in data_map.keys()}
    raise ValueError(
        f"Found dataset with inconsistent number of sequences for each node. "
        f"Current number of sequences per node: {seq_numbers}"
    )

# select an input dataset and check
n_sequences = len(data_map[list(data_map.keys())[0]])

mapped_sequences = []
for i in range(n_sequences):
    sequence = {n: data_map[n][i] for n in data_map.keys()}
    mapped_sequences.append(sequence)

return mapped_sequences

"""Convert a list of sequence to nodes mappings into a mapping of lists or a
simple array if possible."""
n_sequences = len(states)
if n_sequences == 1:
    states_map = states[0]
else:
    states_map = defaultdict(list)
    for i in range(n_sequences):
        for node_name, seq in states[i].items():
            states_map[node_name] += [seq]

if len(states_map) == 1 and return_states is None:
    return states_map[model.output_nodes[0].name]

return states_map

"""Map dataset to input/target nodes in the model."""
X_map = build_mapping(model.input_nodes, X, io_type="input")

Y_map = None
if Y is not None:
    Y_map = build_mapping(model.trainable_nodes, Y, io_type="target")

X_map, Y_map = check_xy(model, x=X_map, y=Y_map)

X_sequences = unfold_mapping(X_map)

if Y_map is None:
    n_sequences = len(X_sequences)
    Y_sequences = [None] * n_sequences
else:
    Y_sequences = unfold_mapping(Y_map)

```
# File: 2-Advanced_Features.txt
```python
import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here

rpy.set_seed(42)  # make everything reproducible!

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

from reservoirpy.nodes import Reservoir, Ridge, Input

data = Input()

reservoir = Reservoir(100, lr=0.5, sr=0.9)

readout = Ridge(ridge=1e-7)

esn_model = data >> reservoir >> readout & data >> readout

connection_1 = data >> reservoir >> readout

connection_2 = data >> readout

esn_model = connection_1 & connection_2

esn_model = [data, data >> reservoir] >> readout

esn_model.node_names

from reservoirpy.nodes import Reservoir, Ridge, Input, Concat

data = Input()

reservoir = Reservoir(100, lr=0.5, sr=0.9)

readout = Ridge(ridge=1e-7)

concatenate = Concat()

esn_model = [data, data >> reservoir] >> concatenate >> readout

esn_model.node_names

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)

readout = Ridge(ridge=1e-7)

reservoir <<= readout

esn_model = reservoir >> readout

esn_model = esn_model.fit(X_train, Y_train)

esn_model(X[0].reshape(1, -1))

print("Feedback received (reservoir):", reservoir.feedback())

print("State sent: (readout):", readout.state())

esn_model = esn_model.fit(X_train, Y_train, force_teachers=True)  # by default

pred = esn_model.run(X_train, forced_feedbacks=Y_train, shift_fb=True)

random_feedback = np.random.normal(0, 1, size=(1, readout.output_dim))

with reservoir.with_feedback(random_feedback):

    reservoir(X[0].reshape(1, -1))

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)

ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=10)

warmup_y = esn_model.run(X_train[-10:], reset=True)

Y_pred = np.empty((100, 1))

x = warmup_y[-1].reshape(1, -1)

for i in range(100):

    x = esn_model(x)

    Y_pred[i] = x

plt.figure(figsize=(10, 3))

plt.title("100 timesteps of a sine wave.")

plt.xlabel("$t$")

plt.plot(Y_pred, label="Generated sin(t)")

plt.legend()

plt.show()

from reservoirpy.nodes import Reservoir

def normal_w(n, m, **kwargs):

    return np.random.normal(0, 1, size=(n, m))

reservoir = Reservoir(100, W=normal_w)

reservoir(X[0].reshape(1, -1))

plt.figure(figsize=(5, 5))

plt.title("Weights distribution in $W$")

plt.hist(reservoir.W.ravel(), bins=50)

plt.show()

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

from reservoirpy.nodes import Reservoir

W_matrix = np.random.normal(0, 1, size=(100, 100))

bias_vector = np.ones((100, 1))

reservoir = Reservoir(W=W_matrix, bias=bias_vector)

states = reservoir(X[0].reshape(1, -1))

X = np.array([[np.sin(np.linspace(0, 12*np.pi, 1000)) 

                for j in range(50)] 

                for i in range(500)]).reshape(-1, 1000, 50)

Y = np.array([[np.sin(np.linspace(0, 12*np.pi, 1000))

                for j in range(40)] 

                for i in range(500)]).reshape(-1, 1000, 40)

print(X.shape, Y.shape)

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

flows from node to node in the chain.

pathways. Merging two chains of nodes will create a new model
containing all the nodes in the two chains along with all the
connections between them.

from reservoirpy.nodes import Reservoir, Ridge, Input

reservoir1 = Reservoir(100, name="res1-1")

reservoir2 = Reservoir(100, name="res2-1")

readout1 = Ridge(ridge=1e-5, name="readout1-1")

readout2 = Ridge(ridge=1e-5, name="readout2-1")

model = reservoir1 >> readout1 >> reservoir2 >> readout2

model = model.fit(X_train, {"readout1-1": Y_train, "readout2-1": Y_train})

from reservoirpy.nodes import Reservoir, Ridge, Input

reservoir1 = Reservoir(100, name="res1-2")

reservoir2 = Reservoir(100, name="res2-2")

reservoir3 = Reservoir(100, name="res3-2")

readout = Ridge(name="readout-2")

model = reservoir1 >> reservoir2 >> reservoir3 & \

        data >> [reservoir1, reservoir2, reservoir3] >> readout

from reservoirpy.nodes import Reservoir, Ridge, Input

reservoir1 = Reservoir(100, name="res1-3")

reservoir2 = Reservoir(100, name="res2-3")

reservoir3 = Reservoir(100, name="res3-3")

readout1 = Ridge(name="readout2")

readout2 = Ridge(name="readout1")

model = [reservoir1, reservoir2] >> readout1 & \

```
# File: test_model.txt
```python
clean_registry(Model)

model1 = plus_node >> minus_node
model2 = minus_node >> plus_node

assert model1.name == "Model-0"
assert model1.params["PlusNode-0"]["c"] is None
assert model1.hypers["PlusNode-0"]["h"] == 1
assert model1["PlusNode-0"].input_dim is None

assert model2.name == "Model-1"
assert model2.params["PlusNode-0"]["c"] is None
assert model2.hypers["PlusNode-0"]["h"] == 1
assert model2["PlusNode-0"].input_dim is None

assert model1.edges == [(plus_node, minus_node)]
assert model2.edges == [(minus_node, plus_node)]
assert set(model1.nodes) == set(model2.nodes)

with pytest.raises(RuntimeError):
    model1 & model2

with pytest.raises(RuntimeError):
    plus_node >> minus_node >> plus_node

with pytest.raises(RuntimeError):
    plus_node >> plus_node

model = Model()
assert model.is_empty

model = plus_node >> minus_node

data = np.zeros((1, 5))
res = model(data)

assert_array_equal(res, data)

input_node = Input()
branch1 = input_node >> plus_node
branch2 = input_node >> minus_node

model = branch1 & branch2

res = model(data)

for name, arr in res.items():
    assert name in [out.name for out in model.output_nodes]
    if name == "PlusNode-0":
        assert_array_equal(arr, data + 2)
    else:
        assert_array_equal(arr, data - 2)

res = model(data)

for name, arr in res.items():
    assert name in [out.name for out in model.output_nodes]
    if name == "PlusNode-0":
        assert_array_equal(arr, data + 4)
    else:
        assert_array_equal(arr, data)

res = model(data, reset=True)

for name, arr in res.items():
    assert name in [out.name for out in model.output_nodes]
    if name == "PlusNode-0":
        assert_array_equal(arr, data + 2)
    else:
        assert_array_equal(arr, data - 2)

res = model(data, stateful=False)

for name, arr in res.items():
    assert name in [out.name for out in model.output_nodes]
    if name == "PlusNode-0":
        assert_array_equal(arr, data + 4)
    else:
        assert_array_equal(arr, data)

for node in model.output_nodes:
    if node.name == "PlusNode-0":
        assert_array_equal(node.state(), data + 2)
    else:
        assert_array_equal(node.state(), data - 2)

model = plus_node >> minus_node

data = np.zeros((1, 5))
res = model(data)

assert_array_equal(res, data)

input_node = Input()
branch1 = input_node >> plus_node
branch2 = input_node >> minus_node

model = branch1 & branch2

res = model(data)

with model.with_state(state={plus_node.name: np.zeros_like(plus_node.state())}):
    assert_array_equal(plus_node.state(), np.zeros_like(plus_node.state()))

with pytest.raises(TypeError):
    with model.with_state(state=np.zeros_like(plus_node.state())):
        pass

input_node = Input()
branch1 = input_node >> plus_node
branch2 = input_node >> minus_node

model = merge(branch1, branch2)

data = np.zeros((3, 5))
res = model.run(data)

expected_plus = np.array([[2] * 5, [4] * 5, [6] * 5])
expected_minus = np.array([[-2] * 5, [0] * 5, [-2] * 5])

for name, arr in res.items():
    assert name in [out.name for out in model.output_nodes]
    if name == "PlusNode-0":
        assert_array_equal(arr, expected_plus)
        assert_array_equal(arr[-1][np.newaxis, :], plus_node.state())
    else:
        assert_array_equal(arr, expected_minus)
        assert_array_equal(arr[-1][np.newaxis, :], minus_node.state())

res = model.run(data, reset=True)

expected_plus = np.array([[2] * 5, [4] * 5, [6] * 5])
expected_minus = np.array([[-2] * 5, [0] * 5, [-2] * 5])

for name, arr in res.items():
    assert name in [out.name for out in model.output_nodes]
    if name == "PlusNode-0":
        assert_array_equal(arr, expected_plus)
        assert_array_equal(arr[-1][np.newaxis, :], plus_node.state())
    else:
        assert_array_equal(arr, expected_minus)
        assert_array_equal(arr[-1][np.newaxis, :], minus_node.state())

res = model.run(data, stateful=False)

expected_plus2 = np.array([[8] * 5, [10] * 5, [12] * 5])
expected_minus2 = np.array([[0] * 5, [-2] * 5, [0] * 5])

for name, arr in res.items():
    assert name in [out.name for out in model.output_nodes]
    if name == "PlusNode-0":
        assert_array_equal(arr, expected_plus2)
        assert_array_equal(expected_plus[-1][np.newaxis, :], plus_node.state())
    else:
        assert_array_equal(arr, expected_minus2)
        assert_array_equal(expected_minus[-1][np.newaxis, :], minus_node.state())

input_node = Input()
branch1 = input_node >> plus_node
branch2 = input_node >> minus_node

model = branch1 & branch2

data = np.zeros((5, 3, 5))
res = model.run(data)

assert set(res.keys()) == {plus_node.name, minus_node.name}
assert len(res[plus_node.name]) == 5
assert len(res[minus_node.name]) == 5
assert res[plus_node.name][0].shape == (3, 5)

input_node = Input()
branch1 = input_node >> plus_node
branch2 = input_node >> minus_node

model = branch1 & branch2

data = [np.zeros((3, 5)), np.zeros((8, 5))]
res = model.run(data)

assert set(res.keys()) == {plus_node.name, minus_node.name}
assert len(res[plus_node.name]) == 2
assert len(res[minus_node.name]) == 2
assert res[plus_node.name][0].shape == (3, 5)
assert res[plus_node.name][1].shape == (8, 5)

model = plus_node >> feedback_node >> minus_node
feedback_node <<= minus_node

data = np.zeros((1, 5))
res = model(data)

assert_array_equal(res, data + 1)
assert_array_equal(feedback_node.state(), data + 3)

res = model(data)
assert_array_equal(res, data + 3)
assert_array_equal(feedback_node.state(), data + 6)

model = plus_node >> feedback_node >> minus_node
feedback_node <<= minus_node

data = np.zeros((3, 5))
res = model.run(data)

expected = np.array([[1] * 5, [3] * 5, [5] * 5])

assert_array_equal(res, expected)
assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 10)

model = plus_node >> feedback_node >> minus_node
feedback_node <<= minus_node

data = np.zeros((3, 5))
res = model.run(data, forced_feedbacks={"MinusNode-0": data + 1}, shift_fb=False)
expected = np.array([[2] * 5, [2] * 5, [4] * 5])

assert_array_equal(res, expected)
assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 8)

model = plus_node >> feedback_node >> minus_node
feedback_node <<= minus_node

data = np.zeros((3, 5))
res = model.run(data, forced_feedbacks={"FBNode-0": data + 1}, shift_fb=False)
expected = np.array([[2] * 5, [2] * 5, [4] * 5])

assert_array_equal(res, expected)
assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 8)

model = plus_node >> feedback_node >> minus_node
feedback_node <<= plus_node  # feedback in time, not in space anymore

data = np.zeros((3, 5))
res = model.run(data)

expected = np.array([[1] * 5, [4] * 5, [5] * 5])

assert_array_equal(res, expected)
assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 11)

model = plus_node >> feedback_node
feedback_node <<= plus_node >> inverter_node

data = np.zeros((1, 5))
res = model(data)

assert_array_equal(res, data + 3)
assert_array_equal(plus_node.state(), data + 2)
assert_array_equal(inverter_node.state(), data)

res = model(data)
assert_array_equal(res, data + 3)
assert_array_equal(plus_node.state(), data + 4)
assert_array_equal(inverter_node.state(), data - 2)

plus_node, feedback_node, inverter_node, minus_node

model = plus_node >> feedback_node
fb_model = plus_node >> inverter_node >> minus_node
feedback_node <<= fb_model

data = np.zeros((1, 5))
res = model(data)

assert_array_equal(res, data + 1)
assert_array_equal(plus_node.state(), data + 2)
assert_array_equal(minus_node.state(), data - 2)

res = model(data)

assert_array_equal(res, data + 3)
assert_array_equal(plus_node.state(), data + 4)
assert_array_equal(minus_node.state(), data - 2)

model = plus_node >> offline_node

X = np.ones((5, 5)) * 0.5
Y = np.ones((5, 5))

model.fit(X, Y)

assert_array_equal(offline_node.b, np.array([6.5]))

X = np.ones((3, 5, 5)) * 0.5
Y = np.ones((3, 5, 5))

model.fit(X, Y)

assert_array_equal(offline_node.b, np.array([94.5]))

model.fit(X, Y, reset=True)

assert_array_equal(offline_node.b, np.array([19.5]))

res = model.run(X[0], reset=True)

exp = np.tile(np.array([22.0, 24.5, 27.0, 29.5, 32.0]), 5).reshape(5, 5).T

assert_array_equal(exp, res)

basic_offline_node, offline_node2, plus_node, minus_node, feedback_node

model = plus_node >> feedback_node >> basic_offline_node
feedback_node <<= basic_offline_node

X = np.ones((5, 5)) * 0.5
Y = np.ones((5, 5))

model.fit(X, Y)

assert_array_equal(basic_offline_node.b, np.array([9.3]))

model = plus_node >> feedback_node >> basic_offline_node
feedback_node <<= basic_offline_node

X = np.ones((3, 5, 5)) * 0.5
Y = np.ones((3, 5, 5))

model.fit(X, Y)

assert_array_equal(basic_offline_node.b, np.array([11.4]))

model.fit(X, Y, reset=True)

assert_array_equal(basic_offline_node.b, np.array([5.15]))

res = model.run(X[0], reset=True)

exp = np.tile(np.array([8.65, 19.8, 33.45, 49.6, 68.25]), 5).reshape(5, 5).T

assert_array_equal(exp, res)

model = plus_node >> feedback_node >> basic_offline_node
feedback_node <<= basic_offline_node

X = np.ones((3, 5, 5)) * 0.5
Y = np.ones((3, 5, 5))

model.fit(X, Y, force_teachers=False)

basic_offline_node, offline_node2, plus_node, minus_node, feedback_node

model = [plus_node >> basic_offline_node, plus_node] >> minus_node >> offline_node2

X = np.ones((5, 5, 5)) * 0.5
Y_1 = np.ones((5, 5, 5))
Y_2 = np.ones((5, 5, 10))  # after concat

model.fit(X, Y={"BasicOffline-0": Y_1, "Offline2-0": Y_2})

res = model.run(X[0])

assert res.shape == (5, 10)

model = plus_node >> online_node

X = np.ones((5, 5)) * 0.5
Y = np.ones((5, 5))

model.train(X, Y)

assert_array_equal(online_node.b, np.array([42.5]))

model.train(X, Y, reset=True)

assert_array_equal(online_node.b, np.array([85]))

model = plus_node >> feedback_node >> online_node

feedback_node <<= online_node

X = np.ones((5, 5)) * 0.5
Y = np.ones((5, 5))

model.train(X, Y)

assert_array_equal(online_node.b, np.array([51.5]))

model.train(X, Y, reset=True)

assert_array_equal(online_node.b, np.array([103.0]))

model = plus_node >> feedback_node >> online_node

feedback_node <<= online_node

X = np.ones((5, 5)) * 0.5
Y = np.ones((5, 5))

model.train(X, Y, force_teachers=False)

assert_array_equal(online_node.b, np.array([189.5]))

model.train(X, Y, reset=True, force_teachers=False)

assert_array_equal(online_node.b, np.array([3221.5]))

X = np.ones((5, 5)) * 0.5
model = plus_node >> online_node

with pytest.raises(RuntimeError):
    model.train(X, minus_node)  # Impossible to init node nor infer shape

model = plus_node >> [minus_node, online_node]

minus_node.set_output_dim(5)

model.train(X, minus_node)

assert_array_equal(online_node.b, np.array([54.0]))

model.train(X, minus_node, reset=True)

assert_array_equal(online_node.b, np.array([108.0]))

off = Offline(name="offline")
plus = PlusNode(name="plus")
minus = MinusNode(name="minus")
inverter = Inverter(name="inv")

model = plus >> [minus, off >> inverter]

X = np.ones((5, 5)) * 0.5
Y = np.ones((5, 5))

model.fit(X, Y)

res = model.run(X)

assert set(res.keys()) == {"minus", "inv"}

res = model.run(X, return_states="all")

assert set(res.keys()) == {"minus", "inv", "offline", "plus"}

res = model.run(X, return_states=["offline"])

assert set(res.keys()) == {"offline"}

import numpy as np

from reservoirpy.nodes import Input, Reservoir

```
# File: force copy.txt
```python
"""Single layer of neurons learning connections through online learning rules.

Warning
-------

This class is deprecated since v0.3.4 and will be removed in future versions.
Please use :py:class:`~reservoirpy.LMS` or :py:class:`~reservoirpy.RLS` instead.

The learning rules involved are similar to Recursive Least Squares (``rls`` rule)
as described in [1]_ or Least Mean Squares (``lms`` rule, similar to Hebbian
learning) as described in [2]_.

"FORCE" name refers to the training paradigm described in [1]_.

:py:attr:`FORCE.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
``P``              Matrix :math:`\\mathbf{P}` of RLS rule (optional).
================== =================================================================

:py:attr:`FORCE.hypers` **list**

================== =================================================================
``alpha``          Learning rate (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by
                    default).
``input_bias``     If True, learn a bias term (True by default).
``rule``           One of RLS or LMS rule ("rls" by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
alpha : float or Python generator or iterable, default to 1e-6
    Learning rate. If an iterable or a generator is provided and the learning
    rule is "lms", then the learning rate can be changed at each timestep of
    training. A new learning rate will be drawn from the iterable or generator
    at each timestep.
rule : {"rls", "lms"}, default to "rls"
    Learning rule applied for online training.
Wout : callable or array-like of shape (units, targets), default to
    :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to
    :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

References
----------

.. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
        Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
        https://doi.org/10.1016/j.neuron.2009.07.018

.. [2] Hoerzer, G. M., Legenstein, R., & Maass, W. (2014). Emergence of Complex
        Computational Structures From Chaotic Neural Networks Through
        Reward-Modulated Hebbian Learning. Cerebral Cortex, 24(3), 677–690.
        https://doi.org/10.1093/cercor/bhs348
"""

def __init__(
    self,
    output_dim=None,
    alpha=1e-6,
    rule="rls",
    Wout=zeros,
    bias=zeros,
    input_bias=True,
    name=None,
):

    warnings.warn(
        "'FORCE' is deprecated since v0.3.4 and will be removed "
        "in "
        "future versions. Consider using 'RLS' or 'LMS'.",
        DeprecationWarning,
    )

    params = {"Wout": None, "bias": None}

    if rule not in RULES:
        raise ValueError(
            f"Unknown rule for FORCE learning. "
            f"Available rules are {self._rules}."
        )
    else:
        if rule == "lms":
            train = lms_like_train
            initialize = initialize_lms
        else:
            train = rls_like_train
            initialize = initialize_rls
            params["P"] = None

    if isinstance(alpha, Number):

        def _alpha_gen():
            while True:
                yield alpha

        alpha_gen = _alpha_gen()
    elif isinstance(alpha, Iterable):
        alpha_gen = alpha
    else:
        raise TypeError(
            "'alpha' parameter should be a float or an iterable yielding floats."
        )

```
# File: base readouts.txt
```python
readout, x=None, y=None, init_func=None, bias_init=None, bias=True

if x is not None:

    in_dim = x.shape[1]

    if readout.output_dim is not None:
        out_dim = readout.output_dim
    elif y is not None:
        out_dim = y.shape[1]
    else:
        raise RuntimeError(
            f"Impossible to initialize {readout.name}: "
            f"output dimension was not specified at "
            f"creation, and no teacher vector was given."
        )

    readout.set_input_dim(in_dim)
    readout.set_output_dim(out_dim)

    if callable(init_func):
        W = init_func(in_dim, out_dim, dtype=readout.dtype)
    elif isinstance(init_func, np.ndarray):
        W = (
            check_vector(init_func, caller=readout)
            .reshape(readout.input_dim, readout.output_dim)
            .astype(readout.dtype)
        )
    else:
        raise ValueError(
            f"Data type {type(init_func)} not "
            f"understood for matrix initializer "
            f"'Wout'. It should be an array or "
            f"a callable returning an array."
        )

    if bias:
        if callable(bias_init):
            bias = bias_init(1, out_dim, dtype=readout.dtype)
        elif isinstance(bias_init, np.ndarray):
            bias = (
                check_vector(bias_init)
                .reshape(1, readout.output_dim)
                .astype(readout.dtype)
            )
        else:
            raise ValueError(
                f"Data type {type(bias_init)} not "
                f"understood for matrix initializer "
                f"'bias'. It should be an array or "
                f"a callable returning an array."
            )
    else:
        bias = np.zeros((1, out_dim), dtype=readout.dtype)

    readout.set_param("Wout", W)
    readout.set_param("bias", bias)

if X is not None:

    if bias:
        X = add_bias(X)
    if not isinstance(X, np.ndarray):
        X = np.vstack(X)

    X = check_vector(X, allow_reshape=allow_reshape)

if Y is not None:

    if not isinstance(Y, np.ndarray):
        Y = np.vstack(Y)

    Y = check_vector(Y, allow_reshape=allow_reshape)

return X, Y

return (node.Wout.T @ x.reshape(-1, 1) + node.bias.T).T

wo = Wout
if has_bias:
    wo = np.r_[bias, wo]
return wo

if node.input_bias:
    Wout, bias = wo[1:, :], wo[0, :][np.newaxis, :]
    node.set_param("Wout", Wout)
    node.set_param("bias", bias)
else:
    node.set_param("Wout", wo)

```
# File: sklearn_node copy.txt
```python
instances = readout.params.get("instances")
if type(instances) is not list:
    return instances.predict(X)
else:
    return np.concatenate([instance.predict(X) for instance in instances], axis=-1)

# Concatenate all the batches as one np.ndarray
# of shape (timeseries*timesteps, features)
X_ = np.concatenate(X, axis=0)
Y_ = np.concatenate(Y, axis=0)

instances = readout.params.get("instances")
if type(instances) is not list:
    if readout.output_dim > 1:
        # Multi-output node and multi-output sklearn model
        instances.fit(X_, Y_)
    else:
        # Y_ should have 1 feature so we reshape to
        # (timeseries, ) to avoid scikit-learn's DataConversionWarning
        instances.fit(X_, Y_[..., 0])
else:
    for i, instance in enumerate(instances):
        instance.fit(X_, Y_[..., i])

if x is not None:
    in_dim = x.shape[1]
    if readout.output_dim is not None:
        out_dim = readout.output_dim
    elif y is not None:
        out_dim = y.shape[1]
    else:
        raise RuntimeError(
            f"Impossible to initialize {readout.name}: "
            f"output dimension was not specified at "
            f"creation, and no teacher vector was given."
        )

    readout.set_input_dim(in_dim)
    readout.set_output_dim(out_dim)

    first_instance = readout.model(**deepcopy(model_hypers))
    # If there are multiple output but the specified model doesn't support
    # multiple outputs, we create an instance of the model for each output.
    if out_dim > 1 and not first_instance._get_tags().get("multioutput"):
        instances = [
            readout.model(**deepcopy(model_hypers)) for i in range(out_dim)
        ]
        readout.set_param("instances", instances)
    else:
        readout.set_param("instances", first_instance)

    return

"""
A node interfacing a scikit-learn linear model that can be used as an offline
readout node.

The ScikitLearnNode takes a scikit-learn model as parameter and creates a
node with the specified model.

We currently support classifiers (like
:py:class:`sklearn.linear_model.LogisticRegression` or
:py:class:`sklearn.linear_model.RidgeClassifier`) and regressors (like
:py:class:`sklearn.linear_model.Lasso` or
:py:class:`sklearn.linear_model.ElasticNet`).

For more information on the above-mentioned estimators,
please visit scikit-learn linear model API reference
<https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model>`_

:py:attr:`ScikitLearnNode.params` **list**

================== =================================================================
``instances``      Instance(s) of the model class used to fit and predict. If
                    :py:attr:`ScikitLearnNode.output_dim` > 1 and the model doesn't
                    support multi-outputs, `instances` is a list of instances, one
                    for each output feature.
================== =================================================================

:py:attr:`ScikitLearnNode.hypers` **list**

================== =================================================================
``model``          (class) Underlying scikit-learn model.
``model_hypers``   (dict) Keyword arguments for the scikit-learn model.
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
model : str, optional
    Node name.
model_hypers
    (dict) Additional keyword arguments for the scikit-learn model.

Example
-------
>>> from reservoirpy.nodes import Reservoir, ScikitLearnNode
>>> from sklearn.linear_model import Lasso
>>> reservoir = Reservoir(units=100)
>>> readout = ScikitLearnNode(model=Lasso, model_hypers={"alpha":1e-5})
>>> model = reservoir >> readout
"""

def __init__(self, model, model_hypers=None, output_dim=None, **kwargs):
    if model_hypers is None:
        model_hypers = {}

    if not hasattr(model, "fit"):
        model_name = model.__name__
        raise AttributeError(
            f"Specified model {model_name} has no method called 'fit'."
        )
    if not hasattr(model, "predict"):
        model_name = model.__name__
        raise AttributeError(
            f"Specified model {model_name} has no method called 'predict'."
        )

    # Ensure reproducibility
    # scikit-learn currently only supports RandomState
    if (
        model_hypers.get("random_state") is None
        and "random_state" in model.__init__.__kwdefaults__
    ):

        generator = rand_generator()
        model_hypers.update(
            {
                "random_state": np.random.RandomState(
                    seed=generator.integers(1 << 32)
                )
            }
        )

```
# File: graphflow.txt
```python
"""Returns two dicts linking nodes to their parents and children in the graph."""
parents = defaultdict(list)
children = defaultdict(list)

# Kludge to always have the parents and children in the same order at every run.
# TODO: refactor the graphflow part.
edges = sorted(list(edges), key=lambda x: x[0].name + x[1].name)

for edge in edges:
    parent, child = edge
    parents[child] += [parent]
    children[parent] += [child]

return parents, children

"""Topological sort of nodes in a Model, to determine execution order."""
if inputs is None:
    inputs, _ = find_entries_and_exits(nodes, edges)

parents, children = find_parents_and_children(edges)

# using Kahn's algorithm
ordered_nodes = []
edges = set(edges)
inputs = deque(inputs)
while len(inputs) > 0:
    n = inputs.pop()
    ordered_nodes.append(n)
    for m in children.get(n, ()):
        edges.remove((n, m))
        parents[m].remove(n)
        if parents.get(m) is None or len(parents[m]) < 1:
            inputs.append(m)
if len(edges) > 0:
    raise RuntimeError(
        "Model has a cycle: impossible "
        "to automatically determine operations "
        "order in the model."
    )
else:
    return ordered_nodes

"""Cut a graph into several subgraphs where output nodes are untrained offline
learner nodes."""
inputs, outputs = find_entries_and_exits(nodes, edges)
parents, children = find_parents_and_children(edges)

offlines = set(
    [n for n in nodes if n.is_trained_offline and not n.is_trained_online]
)
included, trained = set(), set()
subgraphs, required = [], []
_nodes = nodes.copy()
while trained != offlines:
    subnodes, subedges = [], []
    for node in _nodes:
        if node in inputs or all([p in included for p in parents.get(node)]):

            if node.is_trained_offline and node not in trained:
                trained.add(node)
                subnodes.append(node)
            else:
                if node not in outputs:
                    subnodes.append(node)
                included.add(node)

    subedges = [
        edge for edge in edges if edge[0] in subnodes and edge[1] in subnodes
    ]
    subgraphs.append((subnodes, subedges))
    _nodes = [n for n in nodes if n not in included]

required = _get_required_nodes(subgraphs, children)

return list(zip(subgraphs, required))

"""Get nodes whose outputs are required to run/fit children nodes."""
req = []
fitted = set()
for i in range(1, len(subgraphs)):
    currs = set(subgraphs[i - 1][0])
    nexts = set(subgraphs[i][0])

    req.append(_get_links(currs, nexts, children))

    fitted |= set([node for node in currs if node.is_trained_offline])

nexts = set(
    [n for n in subgraphs[-1][0] if n.is_trained_offline and n not in fitted]
)
currs = set(
    [n for n in subgraphs[-1][0] if not n.is_trained_offline or n in fitted]
)

req.append(_get_links(currs, nexts, children))

return req

"""Returns graphs edges between two subgraphs."""
links = {}
for n in previous:
    next_children = []
    if n not in nexts:
        next_children = [c.name for c in children.get(n, []) if c in nexts]

    if len(next_children) > 0:
        links[n.name] = next_children

return links

"""Find outputs and inputs nodes of a directed acyclic graph."""
nodes = set(nodes)
senders = set([n for n, _ in edges])
receivers = set([n for _, n in edges])

lonely = nodes - senders - receivers

entrypoints = senders - receivers | lonely
endpoints = receivers - senders | lonely

return list(entrypoints), list(endpoints)

X,
Y=None,
shift_fb=True,
return_targets=False,
force_teachers=True,

"""Transform data from a dict of arrays
([node], timesteps, dimension) to an iterator yielding
a node: data mapping for each timestep."""
X_map, Y_map = X, Y
current_node = list(X_map.keys())[0]
sequence_length = len(X_map[current_node])

for i in range(sequence_length):
    x = {node: X_map[node][np.newaxis, i] for node in X_map.keys()}
    if Y_map is not None:
        y = None
        if return_targets:
            y = {node: Y_map[node][np.newaxis, i] for node in Y_map.keys()}
        # if feedbacks vectors are meant to be fed
        # with a delay in time of one timestep w.r.t. 'X_map'
        if shift_fb:
            if i == 0:
                if force_teachers:
                    fb = {
                        node: np.zeros_like(Y_map[node][np.newaxis, i])
                        for node in Y_map.keys()
                    }
                else:
                    fb = {node: None for node in Y_map.keys()}
            else:
                fb = {node: Y_map[node][np.newaxis, i - 1] for node in Y_map.keys()}
        # else assume that all feedback vectors must be instantaneously
        # fed to the network. This means that 'Y_map' already contains
        # data that is delayed by one timestep w.r.t. 'X_map'.
        else:
            fb = {node: Y_map[node][np.newaxis, i] for node in Y_map.keys()}
    else:
        fb = y = None

    yield x, fb, y

"""A utility used to feed data to nodes in a Model."""

_inputs: List
_parents: Dict

def __init__(self, model):
    self._nodes = model.nodes
    self._trainables = model.trainable_nodes
    self._inputs = model.input_nodes
    self.__parents, _ = find_parents_and_children(model.edges)

    self._parents = safe_defaultdict_copy(self.__parents)
    self._teachers = dict()

def __getitem__(self, item):
    return self.get(item)

def _check_inputs(self, input_mapping):
    if is_mapping(input_mapping):
        for node in self._inputs:
            if input_mapping.get(node.name) is None:
                raise KeyError(
                    f"Node {node.name} not found "
                    f"in data mapping. This node requires "
                    f"data to run."
                )

def _check_targets(self, target_mapping):
    if is_mapping(target_mapping):
        for node in self._nodes:
            if (
                node in self._trainables
                and not node.fitted
                and target_mapping.get(node.name) is None
            ):
                raise KeyError(
                    f"Trainable node {node.name} not found "
                    f"in target/feedback data mapping. This "
                    f"node requires "
                    f"target values."
                )

def get(self, item):
    parents = self._parents.get(item, ())
    teacher = self._teachers.get(item, None)

    x = []
    for parent in parents:
        if isinstance(parent, _Node):
            x.append(parent.state())
        else:
            x.append(parent)

    # in theory, only operators can support several incoming signal
    # i.e. several operands, so unpack data if the list is unecessary
    if len(x) == 1:
        x = x[0]

    return DataPoint(x=x, y=teacher)

def load(self, X=None, Y=None):
    """Load input and target data for dispatch."""
    self._parents = safe_defaultdict_copy(self.__parents)
    self._teachers = dict()

    if X is not None:
        self._check_inputs(X)
        if is_mapping(X):
            for node in self._nodes:
                if X.get(node.name) is not None:
                    self._parents[node] += [X[node.name]]

        else:
            for inp_node in self._inputs:
                self._parents[inp_node] += [X]

```
# File: ridge.txt
```python
"""Solve Tikhonov regression."""
return linalg.solve(XXT + ridge, YXT.T, assume_a="sym")

"""Aggregate Xi.Xi^T and Yi.Xi^T matrices from a state sequence i."""
XXT = readout.get_buffer("XXT")
YXT = readout.get_buffer("YXT")
XXT += xxt
YXT += yxt

"""Pre-compute XXt and YXt before final fit."""
X, Y = _prepare_inputs_for_learning(
    X_batch,
    Y_batch,
    bias=readout.input_bias,
    allow_reshape=True,
)

xxt = X.T.dot(X)
yxt = Y.T.dot(X)

if lock is not None:
    # This is not thread-safe using Numpy memmap as buffers
    # ok for parallelization then with a lock (see ESN object)
    with lock:
        _accumulate(readout, xxt, yxt)
else:
    _accumulate(readout, xxt, yxt)

ridge = readout.ridge
XXT = readout.get_buffer("XXT")
YXT = readout.get_buffer("YXT")

input_dim = readout.input_dim
if readout.input_bias:
    input_dim += 1

ridgeid = ridge * np.eye(input_dim, dtype=global_dtype)

Wout_raw = _solve_ridge(XXT, YXT, ridgeid)

if readout.input_bias:
    Wout, bias = Wout_raw[1:, :], Wout_raw[0, :][np.newaxis, :]
    readout.set_param("Wout", Wout)
    readout.set_param("bias", bias)
else:
    readout.set_param("Wout", Wout_raw)

_initialize_readout(
    readout, x, y, bias=readout.input_bias, init_func=Wout_init, bias_init=bias_init
)

"""create memmaped buffers for matrices X.X^T and Y.X^T pre-computed
in parallel for ridge regression
! only memmap can be used ! Impossible to share Numpy arrays with
different processes in r/w mode otherwise (with proper locking)
"""
input_dim = readout.input_dim
output_dim = readout.output_dim

if readout.input_bias:
    input_dim += 1

readout.create_buffer("XXT", (input_dim, input_dim))
readout.create_buffer("YXT", (output_dim, input_dim))

"""A single layer of neurons learning with Tikhonov linear regression.

Output weights of the layer are computed following:

.. math::

    \\hat{\\mathbf{W}}_{out} = \\mathbf{YX}^\\top ~ (\\mathbf{XX}^\\top +
    \\lambda\\mathbf{Id})^{-1}

Outputs :math:`\\mathbf{y}` of the node are the result of:

.. math::

    \\mathbf{y} = \\mathbf{W}_{out}^\\top \\mathbf{x} + \\mathbf{b}

where:
    - :math:`\\mathbf{X}` is the accumulation of all inputs during training;
    - :math:`\\mathbf{Y}` is the accumulation of all targets during training;
    - :math:`\\mathbf{b}` is the first row of :math:`\\hat{\\mathbf{W}}_{out}`;
    - :math:`\\mathbf{W}_{out}` is the rest of :math:`\\hat{\\mathbf{W}}_{out}`.

If ``input_bias`` is True, then :math:`\\mathbf{b}` is non-zero, and a constant
term is added to :math:`\\mathbf{X}` to compute it.

:py:attr:`Ridge.params` **list**

================== =================================================================
``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
``bias``           Learned bias (:math:`\\mathbf{b}`).
================== =================================================================

:py:attr:`Ridge.hypers` **list**

================== =================================================================
``ridge``          Regularization parameter (:math:`\\lambda`) (0.0 by default).
``input_bias``     If True, learn a bias term (True by default).
================== =================================================================

Parameters
----------
output_dim : int, optional
    Number of units in the readout, can be inferred at first call.
ridge: float, default to 0.0
    L2 regularization parameter.
Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Output weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
input_bias : bool, default to True
    If True, then a bias parameter will be learned along with output weights.
name : str, optional
    Node name.

Example
-------

>>> x = np.random.normal(size=(100, 3))
>>> noise = np.random.normal(scale=0.1, size=(100, 1))
>>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.
>>>
>>> from reservoirpy.nodes import Ridge
>>> ridge_regressor = Ridge(ridge=0.001)
>>>
>>> ridge_regressor.fit(x, y)
>>> ridge_regressor.Wout, ridge_regressor.bias
array([[ 9.992, -0.205,  6.989]]).T, array([[12.011]])
"""

```
# File: 1-Getting_Started.txt
```python
import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here

rpy.set_seed(42)  # make everything reproducible!

neurons;

reservoir. It controls the chaoticity of the reservoir dynamics.

from reservoirpy.nodes import Reservoir

reservoir = Reservoir(100, lr=0.5, sr=0.9)

import numpy as np

import matplotlib.pyplot as plt

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

plt.figure(figsize=(10, 3))

plt.title("A sine wave.")

plt.ylabel("$sin(t)$")

plt.xlabel("$t$")

plt.plot(X)

plt.show()

s = reservoir(X[0].reshape(1, -1))

print("New state vector shape: ", s.shape)

s = reservoir.state()

states = np.empty((len(X), reservoir.output_dim))

for i in range(len(X)):

    states[i] = reservoir(X[i].reshape(1, -1))

plt.figure(figsize=(10, 3))

plt.title("Activation of 20 reservoir neurons.")

plt.ylabel("$reservoir(sin(t))$")

plt.xlabel("$t$")

plt.plot(states[:, :20])

plt.show()

states = reservoir.run(X)

reservoir = reservoir.reset()

states_from_null = reservoir.run(X, reset=True)

a_state_vector = np.random.uniform(-1, 1, size=(1, reservoir.output_dim))

states_from_a_starting_state = reservoir.run(X, from_state=a_state_vector)

previous_states = reservoir.run(X)

with reservoir.with_state(reset=True):

    states_from_null = reservoir.run(X)

    

# as if the with_state never happened!

states_from_previous = reservoir.run(X) 

from reservoirpy.nodes import Ridge

readout = Ridge(ridge=1e-7)

X_train = X[:50]

Y_train = X[1:51]

plt.figure(figsize=(10, 3))

plt.title("A sine wave and its future.")

plt.xlabel("$t$")

plt.plot(X_train, label="sin(t)", color="blue")

plt.plot(Y_train, label="sin(t+1)", color="red")

plt.legend()

plt.show()

train_states = reservoir.run(X_train, reset=True)

readout = readout.fit(train_states, Y_train, warmup=10)

test_states = reservoir.run(X[50:])

Y_pred = readout.run(test_states)

plt.figure(figsize=(10, 3))

plt.title("A sine wave and its future.")

plt.xlabel("$t$")

plt.plot(Y_pred, label="Predicted sin(t)", color="blue")

plt.plot(X[51:], label="Real sin(t+1)", color="red")

plt.legend()

plt.show()

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)

ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=10)

print(reservoir.is_initialized, readout.is_initialized, readout.fitted)

Y_pred = esn_model.run(X[50:])

plt.figure(figsize=(10, 3))

plt.title("A sine wave and its future.")

plt.xlabel("$t$")

plt.plot(Y_pred, label="Predicted sin(t+1)", color="blue")

plt.plot(X[51:], label="Real sin(t+1)", color="red")

plt.legend()

```
# File: test_io.txt
```python
inp = Input()
x = np.ones((1, 10))
out = inp(x)
assert_equal(out, x)
x = np.ones((10, 10))
out = inp.run(x)
assert_equal(out, x)

with pytest.raises(ValueError):
    inp = Input(input_dim=9)
    inp.run(x)

```
# File: 5-Classification-with-RC.txt
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

X_train, Y_train, X_test, Y_test = japanese_vowels()

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

# repeat_target ensure that we obtain one label per timestep, and not one label per utterance.

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

from reservoirpy.nodes import Reservoir, Ridge, Input

source = Input()

reservoir = Reservoir(500, sr=0.9, lr=0.1)

readout = Ridge(ridge=1e-6)

model = [source >> reservoir, source] >> readout

Y_pred = model.fit(X_train, Y_train, stateful=False, warmup=2).run(X_test, stateful=False)

Y_pred_class = [np.argmax(y_p, axis=1) for y_p in Y_pred]

Y_test_class = [np.argmax(y_t, axis=1) for y_t in Y_test]

score = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(Y_pred_class, axis=0))

print("Accuracy: ", f"{score * 100:.3f} %")

X_train, Y_train, X_test, Y_test = japanese_vowels()

from reservoirpy.nodes import Reservoir, Ridge, Input

source = Input()

reservoir = Reservoir(500, sr=0.9, lr=0.1)

readout = Ridge(ridge=1e-6)

model = source >> reservoir >> readout

the reservoir.run method.

sequence.

states_train = []

for x in X_train:

    states = reservoir.run(x, reset=True)

    states_train.append(states[-1, np.newaxis])

readout.fit(states_train, Y_train)

Y_pred = []

for x in X_test:

    states = reservoir.run(x, reset=True)

    y = readout.run(states[-1, np.newaxis])

    Y_pred.append(y)

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]

Y_test_class = [np.argmax(y_t) for y_t in Y_test]

score = accuracy_score(Y_test_class, Y_pred_class)

```
# File: test_force.txt
```python
with pytest.warns(DeprecationWarning):
    node = FORCE(10)

data = np.ones((1, 100))
res = node(data)

assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)
assert node.alpha == 1e-6

data = np.ones((10000, 100))
res = node.run(data)

assert res.shape == (10000, 10)

with pytest.warns(DeprecationWarning):
    node = FORCE(10)

x = np.ones((5, 2))
y = np.ones((5, 10))

for x, y in zip(x, y):
    res = node.train(x, y)

assert node.Wout.shape == (2, 10)
assert node.bias.shape == (1, 10)
assert node.alpha == 1e-6

data = np.ones((10000, 2))
res = node.run(data)

assert res.shape == (10000, 10)

with pytest.warns(DeprecationWarning):
    node = FORCE(10)

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

with pytest.warns(DeprecationWarning):
    node = FORCE(10)

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

with pytest.warns(DeprecationWarning):
    node = FORCE(10)

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = node.train(x, y)

assert node.Wout.shape == (100, 10)
assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
assert node.bias.shape == (1, 10)
assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

data = np.ones((10000, 100))
res = node.run(data)

assert res.shape == (10000, 10)

with pytest.warns(DeprecationWarning):
    node = FORCE(10, rule="lms")

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)

def alpha_gen():
    while True:
        yield 1e-6

with pytest.warns(DeprecationWarning):
    node = FORCE(10, rule="lms", alpha=alpha_gen())

X, Y = np.ones((200, 100)), np.ones((200, 10))

res = node.train(X, Y)

assert res.shape == (200, 10)
assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)

with pytest.warns(DeprecationWarning):
    node = FORCE(10, rule="lms", alpha=alpha_gen())

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = node.train(x, y, learn_every=5)

assert node.Wout.shape == (100, 10)
assert node.bias.shape == (1, 10)

data = np.ones((10000, 100))
res = node.run(data)

assert res.shape == (10000, 10)

with pytest.warns(DeprecationWarning):
    readout = FORCE(10)
reservoir = Reservoir(100)

esn = reservoir >> readout

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = esn.train(x, y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)

data = np.ones((10000, 100))
res = esn.run(data)

assert res.shape == (10000, 10)


with pytest.warns(DeprecationWarning):
    readout = FORCE(10, rule="lms")
reservoir = Reservoir(100)

esn = reservoir >> readout

X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

for x, y in zip(X, Y):
    res = esn.train(x, y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)

data = np.ones((10000, 100))
res = esn.run(data)

assert res.shape == (10000, 10)

with pytest.warns(DeprecationWarning):
    readout = FORCE(10)
reservoir = Reservoir(100)

esn = reservoir >> readout

reservoir <<= readout

X, Y = np.ones((5, 200, 8)), np.ones((5, 200, 10))
for x, y in zip(X, Y):
    res = esn.train(x, y)

assert readout.Wout.shape == (100, 10)
assert readout.bias.shape == (1, 10)
assert reservoir.Wfb.shape == (100, 10)

data = np.ones((10000, 8))
res = esn.run(data)

assert res.shape == (10000, 10)

with pytest.warns(DeprecationWarning):
    readout1 = FORCE(10, name="r1")
reservoir1 = Reservoir(100)
with pytest.warns(DeprecationWarning):
    readout2 = FORCE(3, name="r2")
reservoir2 = Reservoir(100)

esn = reservoir1 >> readout1 >> reservoir2 >> readout2

X, Y = np.ones((200, 5)), {"r1": np.ones((200, 10)), "r2": np.ones((200, 3))}
res = esn.train(X, Y)

assert readout1.Wout.shape == (100, 10)
assert readout1.bias.shape == (1, 10)

assert readout2.Wout.shape == (100, 3)
assert readout2.bias.shape == (1, 3)

assert reservoir1.Win.shape == (100, 5)
assert reservoir2.Win.shape == (100, 10)

data = np.ones((10000, 5))
res = esn.run(data)

assert res.shape == (10000, 3)

with pytest.warns(DeprecationWarning):
    readout1 = FORCE(1, name="r1")
reservoir1 = Reservoir(100)
with pytest.warns(DeprecationWarning):
    readout2 = FORCE(1, name="r2")
reservoir2 = Reservoir(100)

reservoir1 <<= readout1
reservoir2 <<= readout2

branch1 = reservoir1 >> readout1
branch2 = reservoir2 >> readout2

model = branch1 & branch2

X = np.ones((200, 5))

res = model.train(X, Y={"r1": readout2, "r2": readout1}, force_teachers=True)

assert readout1.Wout.shape == (100, 1)
assert readout1.bias.shape == (1, 1)

assert readout2.Wout.shape == (100, 1)
assert readout2.bias.shape == (1, 1)

```
# File: test_ops.txt
```python
model1 = plus_node >> minus_node
model2 = minus_node >> plus_node

assert model1.edges == [(plus_node, minus_node)]
assert model2.edges == [(minus_node, plus_node)]
assert set(model1.nodes) == set(model2.nodes)

model3 = plus_node >> offline_node
model4 = minus_node >> offline_node2

model = model3 >> model4

assert set(model.edges) == {
    (plus_node, offline_node),
    (offline_node, minus_node),
    (minus_node, offline_node2),
}
assert set(model.nodes) == set(model3.nodes) | set(model4.nodes)

# cycles in the model!
with pytest.raises(RuntimeError):
    model1 & model2

with pytest.raises(RuntimeError):
    plus_node >> minus_node >> plus_node

with pytest.raises(RuntimeError):
    plus_node >> plus_node

x = np.ones((1, 5))
x2 = np.ones((1, 6))
plus_node(x)
minus_node(x2)

# bad dimensions
with pytest.raises(ValueError):
    plus_node >> minus_node

with pytest.raises(ValueError):
    model1(x)

# merge inplace on a node
with pytest.raises(TypeError):
    plus_node &= minus_node

model = [plus_node, minus_node] >> offline_node

assert len(model.nodes) == 4
assert len(model.edges) == 3

model = plus_node >> [offline_node, minus_node]

assert set(model.nodes) == {plus_node, minus_node, offline_node}
assert set(model.edges) == {(plus_node, offline_node), (plus_node, minus_node)}

fb_plus_node = plus_node << minus_node

assert id(fb_plus_node._feedback._sender) == id(minus_node)
assert plus_node._feedback is None

plus_node <<= minus_node
assert id(plus_node._feedback._sender) == id(minus_node)

branch1 = plus_node >> minus_node
branch2 = plus_node >> basic_offline_node

model = branch1 & branch2

assert set(model.nodes) == {plus_node, minus_node, basic_offline_node}
assert set(model.edges) == {
    (plus_node, minus_node),
    (plus_node, basic_offline_node),
}

branch1 &= branch2

```
# File: reservoir.txt
```python
from typing_extensions import Literal

from typing import Literal

"""Pool of leaky-integrator neurons with random recurrent connexions.

Reservoir neurons states, gathered in a vector :math:`\\mathbf{x}`, may follow
one of the two update rules below:

- **1.** Activation function is part of the neuron internal state
    (equation called ``internal``):

.. math::

    \\mathbf{x}[t+1] = (1 - \\mathrm{lr}) * \\mathbf{x}[t] + \\mathrm{lr}
        * f(\\mathbf{W}_{in} \\cdot (\\mathbf{u}[t+1]+c_{in}*\\xi)
        + \\mathbf{W} \\cdot \\mathbf{x}[t]
    + \\mathbf{W}_{fb} \\cdot (g(\\mathbf{y}[t])+c_{fb}*\\xi) + \\mathbf{b})
    + c * \\xi

- **2.** Activation function is applied on emitted internal states
    (equation called ``external``):

.. math::

    \\mathbf{r}[t+1] = (1 - \\mathrm{lr}) * \\mathbf{r}[t] + \\mathrm{lr}
    * (\\mathbf{W}_{in} \\cdot (\\mathbf{u}[t+1]+c_{in}*\\xi)
        + \\mathbf{W} \\cdot \\mathbf{x}[t]
    + \\mathbf{W}_{fb} \\cdot (g(\\mathbf{y}[t])+c_{fb}*\\xi) + \\mathbf{b})

.. math::

    \\mathbf{x}[t+1] = f(\\mathbf{r}[t+1]) + c * \\xi

where:
    - :math:`\\mathbf{x}` is the output activation vector of the reservoir;
    - :math:`\\mathbf{r}` is the (optional) internal activation vector of the reservoir;
    - :math:`\\mathbf{u}` is the input timeseries;
    - :math:`\\mathbf{y}` is a feedback vector;
    - :math:`\\xi` is a random noise;
    - :math:`f` and :math:`g` are activation functions.

:py:attr:`Reservoir.params` **list:**

================== ===================================================================
``W``              Recurrent weights matrix (:math:`\\mathbf{W}`).
``Win``            Input weights matrix (:math:`\\mathbf{W}_{in}`).
``Wfb``            Feedback weights matrix (:math:`\\mathbf{W}_{fb}`).
``bias``           Input bias vector (:math:`\\mathbf{b}`).
``internal_state``  Internal state used with equation="external" (:math:`\\mathbf{r}`).
================== ===================================================================

:py:attr:`Reservoir.hypers` **list:**

======================= ========================================================
``lr``                  Leaking rate (1.0 by default) (:math:`\\mathrm{lr}`).
``sr``                  Spectral radius of ``W`` (optional).
``input_scaling``       Input scaling (float or array) (1.0 by default).
``fb_scaling``          Feedback scaling (float or array) (1.0 by default).
``rc_connectivity``     Connectivity (or density) of ``W`` (0.1 by default).
``input_connectivity``  Connectivity (or density) of ``Win`` (0.1 by default).
``fb_connectivity``     Connectivity (or density) of ``Wfb`` (0.1 by default).
``noise_in``            Input noise gain (0 by default) (:math:`c_{in} * \\xi`).
``noise_rc``            Reservoir state noise gain (0 by default) (:math:`c * \\xi`).
``noise_fb``            Feedback noise gain (0 by default) (:math:`c_{fb} * \\xi`).
``noise_type``          Distribution of noise (normal by default) (:math:`\\xi \\sim \\mathrm{Noise~type}`).
``activation``          Activation of the reservoir units (tanh by default) (:math:`f`).
``fb_activation``       Activation of the feedback units (identity by default) (:math:`g`).
``units``               Number of neuronal units in the reservoir.
``noise_generator``     A random state generator.
======================= ========================================================

Parameters
----------
units : int, optional
    Number of reservoir units. If None, the number of units will be inferred from
    the ``W`` matrix shape.
lr : float or array-like of shape (units,), default to 1.0
    Neurons leak rate. Must be in :math:`[0, 1]`.
sr : float, optional
    Spectral radius of recurrent weight matrix.
input_bias : bool, default to True
    If False, no bias is added to inputs.
noise_rc : float, default to 0.0
    Gain of noise applied to reservoir activations.
noise_in : float, default to 0.0
    Gain of noise applied to input inputs.
noise_fb : float, default to 0.0
    Gain of noise applied to feedback signal.
noise_type : str, default to "normal"
    Distribution of noise. Must be a Numpy random variable generator
    distribution (see :py:class:`numpy.random.Generator`).
noise_kwargs : dict, optional
    Keyword arguments to pass to the noise generator, such as `low` and `high`
    values of uniform distribution.
input_scaling : float or array-like of shape (features,), default to 1.0.
    Input gain. An array of the same dimension as the inputs can be used to
    set up different input scaling for each feature.
bias_scaling: float, default to 1.0
    Bias gain.
fb_scaling : float or array-like of shape (features,), default to 1.0
    Feedback gain. An array of the same dimension as the feedback can be used to
    set up different feedback scaling for each feature.
input_connectivity : float, default to 0.1
    Connectivity of input neurons, i.e. ratio of input neurons connected
    to reservoir neurons. Must be in :math:`]0, 1]`.
rc_connectivity : float, default to 0.1
    Connectivity of recurrent weight matrix, i.e. ratio of reservoir
    neurons connected to other reservoir neurons, including themselves.
    Must be in :math:`]0, 1]`.
fb_connectivity : float, default to 0.1
    Connectivity of feedback neurons, i.e. ratio of feedback neurons
    connected to reservoir neurons. Must be in :math:`]0, 1]`.
Win : callable or array-like of shape (units, features), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
    Input weights matrix or initializer. If a callable (like a function) is used,
    then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.normal`
    Recurrent weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
    Bias weights vector or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
Wfb : callable or array-like of shape (units, feedback), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
    Feedback weights matrix or initializer. If a callable (like a function) is
    used, then this function should accept any keywords
    parameters and at least two parameters that will be used to define the shape of
    the returned weight matrix.
fb_activation : str or callable, default to :py:func:`~reservoirpy.activationsfunc.identity`
    Feedback activation function.
    - If a str, should be a :py:mod:`~reservoirpy.activationsfunc`
    function name.
    - If a callable, should be an element-wise operator on arrays.
activation : str or callable, default to :py:func:`~reservoirpy.activationsfunc.tanh`
    Reservoir units activation function.
    - If a str, should be a :py:mod:`~reservoirpy.activationsfunc`
    function name.
    - If a callable, should be an element-wise operator on arrays.
equation : {"internal", "external"}, default to "internal"
    If "internal", will use equation defined in equation 1 to update the state of
    reservoir units. If "external", will use the equation defined in equation 2
    (see above).
feedback_dim : int, optional
    Feedback dimension. Can be inferred at first call.
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
seed : int or :py:class:`numpy.random.Generator`, optional
    A random state seed, for noise generation.

Note
----

If W, Win, bias or Wfb are initialized with an array-like matrix, then all
initializers parameters such as spectral radius (``sr``) or input scaling
(``input_scaling``) are ignored.
See :py:mod:`~reservoirpy.mat_gen` for more information.

Example
-------

>>> from reservoirpy.nodes import Reservoir
>>> reservoir = Reservoir(100, lr=0.2, sr=0.8) # a 100 neurons reservoir

Using the :py:func:`~reservoirpy.datasets.mackey_glass` timeseries:

>>> from reservoirpy.datasets import mackey_glass
>>> x = mackey_glass(200)
>>> states = reservoir.run(x)

.. plot::

    from reservoirpy.nodes import Reservoir
    reservoir = Reservoir(100, lr=0.2, sr=0.8)
    from reservoirpy.datasets import mackey_glass
    x = mackey_glass(200)
    states = reservoir.run(x)
    fig, ax = plt.subplots(6, 1, figsize=(7, 10), sharex=True)
    ax[0].plot(x)
    ax[0].grid()
    ax[0].set_title("Neuron states (on Mackey-Glass)")
    for i in range(1, 6):
        ax[i].plot(states[:, i], label=f"Neuron {i}")
        ax[i].legend()
        ax[i].grid()
    ax[-1].set_xlabel("Timesteps")
"""

def __init__(
    self,
    units: int = None,
    lr: float = 1.0,
    sr: Optional[float] = None,
    input_bias: bool = True,
    noise_rc: float = 0.0,
    noise_in: float = 0.0,
    noise_fb: float = 0.0,
    noise_type: str = "normal",
    noise_kwargs: Dict = None,
    input_scaling: Union[float, Sequence] = 1.0,
    bias_scaling: float = 1.0,
    fb_scaling: Union[float, Sequence] = 1.0,
    input_connectivity: float = 0.1,
    rc_connectivity: float = 0.1,
    fb_connectivity: float = 0.1,
    Win: Union[Weights, Callable] = bernoulli,
    W: Union[Weights, Callable] = normal,
    Wfb: Union[Weights, Callable] = bernoulli,
    bias: Union[Weights, Callable] = bernoulli,
    fb_activation: Union[str, Callable] = identity,
    activation: Union[str, Callable] = tanh,
    equation: Literal["internal", "external"] = "internal",
    input_dim: Optional[int] = None,
    feedback_dim: Optional[int] = None,
    seed=None,
    **kwargs,
):
    if units is None and not is_array(W):
        raise ValueError(
            "'units' parameter must not be None if 'W' parameter is not "
            "a matrix."
        )

    if equation == "internal":
        forward = forward_internal
    elif equation == "external":
        forward = forward_external
    else:
        raise ValueError(
            "'equation' parameter must be either 'internal' or 'external'."
        )

    if type(activation) is str:
        activation = get_function(activation)
    if type(fb_activation) is str:
        fb_activation = get_function(fb_activation)

    rng = rand_generator(seed)

    noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

```
# File: 6-Interfacing_with_scikit-learn.txt
```python
import numpy as np

import matplotlib.pyplot as plt

import reservoirpy

from reservoirpy.observables import nrmse, rsquare

reservoirpy.set_seed(42)

reservoirpy.verbosity(0)

from sklearn import linear_model

from reservoirpy.nodes import ScikitLearnNode

import reservoirpy

reservoirpy.verbosity(0)

reservoirpy.set_seed(42)

readout = ScikitLearnNode(linear_model.Lasso)

readout = ScikitLearnNode(

    model = linear_model.Lasso, 

    model_hypers = {"alpha": 1e-3},

    name = "Lasso"

)

# create the model

reservoir = reservoirpy.nodes.Reservoir(

    units = 500,

    lr = 0.3,

    sr = 0.9,

)

model = reservoir >> readout

# create the dataset to train our model on

from reservoirpy.datasets import mackey_glass, to_forecasting

mg = mackey_glass(n_timesteps=10_000, tau=17)

# rescale between -1 and 1

mg = 2 * (mg - mg.min()) / mg.ptp() - 1

X_train, X_test, y_train, y_test = to_forecasting(mg, forecast=10, test_size=0.2)

model.fit(X_train, y_train, warmup=100)

def plot_results(y_pred, y_test, sample=500):

    fig = plt.figure(figsize=(15, 7))

    plt.subplot(211)

    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")

    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")

    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")

    plt.legend()

    plt.show()

y_pred = model.run(X_test)

plot_results(y_pred, y_test)

rsquare(y_test, y_pred), nrmse(y_test, y_pred)

node = ScikitLearnNode(linear_model.PassiveAggressiveRegressor)

node.initialize(x=np.ones((10, 3)), y=np.ones((10, 1)))

str(node.instances)

node = ScikitLearnNode(linear_model.PassiveAggressiveRegressor)

# we now have 2 output features !

node.initialize(x=np.ones((10, 3)), y=np.ones((10, 2)))

node.instances

import numpy as np

from reservoirpy.datasets import japanese_vowels

# repeat_target ensure that we obtain one label per timestep, and not one label per utterance.

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

# Y_train and Y_test are one-hot encoded, but we want qualitative values here.

Y_train = [np.argmax(sample, 1, keepdims=True) for sample in Y_train]

Y_test = [np.argmax(sample, 1, keepdims=True) for sample in Y_test]

X_train[0].shape, Y_train[0].shape

from reservoirpy.nodes import Reservoir, ScikitLearnNode

from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron

reservoir = Reservoir(500, sr=0.9, lr=0.1)

sk_ridge = ScikitLearnNode(RidgeClassifier, name="RidgeClassifier")

sk_logistic = ScikitLearnNode(LogisticRegression, name="LogisticRegression")

sk_perceptron = ScikitLearnNode(Perceptron, name="Perceptron")

# One reservoir for 3 readout. That's the magic of reservoir computing!

model = reservoir >> [sk_ridge, sk_logistic, sk_perceptron]

model.fit(X_train, Y_train, stateful=False, warmup=2)

Y_pred = model.run(X_test, stateful=False)

from sklearn.metrics import accuracy_score

speaker = np.concatenate(Y_test, dtype=np.float64)

for model, pred in Y_pred.items():

    model_pred = np.concatenate(pred)

    score = accuracy_score(speaker, model_pred)

```
# File: NG-RC_Gauthier_et_al_2021.txt
```python
import matplotlib.pyplot as plt

import numpy as np

from reservoirpy.datasets import lorenz, doublescroll

from reservoirpy.observables import nrmse

from reservoirpy.nodes import Ridge, NVAR

%matplotlib inline

from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""")

# time step duration (in time unit)

dt = 0.025

# training time (in time unit)

train_time  = 10.

# testing time (idem)

test_time   = 120.

# warmup time (idem): should always be > k * s

warm_time   = 5.

# discretization

train_steps = round(train_time / dt)

test_steps  = round(test_time  / dt)

warm_steps  = round(warm_time  / dt)

x0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]

n_timesteps = train_steps + test_steps + warm_steps

X = lorenz(n_timesteps, x0=x0, h=dt, method="RK23")

N = train_steps + warm_steps + test_steps

fig = plt.figure(figsize=(10, 10))

ax  = fig.add_subplot(111, projection='3d')

ax.set_title("Lorenz attractor (1963)")

ax.set_xlabel("$x$")

ax.set_ylabel("$y$")

ax.set_zlabel("$z$")

ax.grid(False)

for i in range(N-1):

    ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color=plt.cm.magma(255*i//N), lw=1.0)

plt.show()

nvar = NVAR(delay=2, order=2, strides=1)

readout = Ridge(3, ridge=2.5e-6)

model = nvar >> readout

Xi  = X[:train_steps+warm_steps-1]

dXi = X[1:train_steps+warm_steps] - X[:train_steps+warm_steps-1]

model = model.fit(Xi, dXi, warmup=warm_steps)

lin = ["$x_t$", "$y_t$", "$z_t$", "$x_{t-1}$", "$y_{t-1}$", "$z_{t-1}$"]

nonlin = []

for idx in nvar._monomial_idx:

    idx = idx.astype(int)

    if idx[0] == idx[1]:

        c = lin[idx[0]][:-1] + "^2$"

    else:

        c = " ".join((lin[idx[0]][:-1], lin[idx[1]][1:]))

    nonlin.append(c)

coefs = ["$c$"] + lin + nonlin

fig = plt.figure(figsize=(10, 10))

Wout = np.r_[readout.bias, readout.Wout]

x_Wout, y_Wout, z_Wout = Wout[:, 0], Wout[:, 1], Wout[:, 2]

ax = fig.add_subplot(131)

ax.set_xlim(-0.2, 0.2)

ax.grid(axis="y")

ax.set_xlabel("$[W_{out}]_x$")

ax.set_ylabel("Features")

ax.set_yticks(np.arange(len(coefs)))

ax.set_yticklabels(coefs[::-1])

ax.barh(np.arange(x_Wout.size), x_Wout.ravel()[::-1])

ax1 = fig.add_subplot(132)

ax1.set_xlim(-0.2, 0.2)

ax1.grid(axis="y")

ax1.set_yticks(np.arange(len(coefs)))

ax1.set_xlabel("$[W_{out}]_y$")

ax1.barh(np.arange(y_Wout.size), y_Wout.ravel()[::-1])

ax2 = fig.add_subplot(133)

ax2.set_xlim(-0.2, 0.2)

ax2.grid(axis="y")

ax2.set_yticks(np.arange(len(coefs)))

ax2.set_xlabel("$[W_{out}]_z$")

ax2.barh(np.arange(z_Wout.size), z_Wout.ravel()[::-1])

plt.show()

nvar.run(X[warm_steps+train_steps-2:warm_steps+train_steps])

u = X[warm_steps+train_steps]

res = np.zeros((test_steps, readout.output_dim))

for i in range(test_steps):

    u = u + model(u)

    res[i, :] = u

N = test_steps

Y = X[warm_steps+train_steps:]

fig = plt.figure(figsize=(15, 10))

ax  = fig.add_subplot(121, projection='3d')

ax.set_title("Generated attractor")

ax.set_xlabel("$x$")

ax.set_ylabel("$y$")

ax.set_zlabel("$z$")

ax.grid(False)

for i in range(N-1):

    ax.plot(res[i:i+2, 0], res[i:i+2, 1], res[i:i+2, 2], color=plt.cm.magma(255*i//N), lw=1.0)

ax2 = fig.add_subplot(122, projection='3d')

ax2.set_title("Real attractor")

ax2.grid(False)

for i in range(N-1):

    ax2.plot(Y[i:i+2, 0], Y[i:i+2, 1], Y[i:i+2, 2], color=plt.cm.magma(255*i//N), lw=1.0)

dt = 0.25

train_time  = 100.

test_time   = 800.

warm_time   = 1.

train_steps = round(train_time / dt)

test_steps  = round(test_time  / dt)

warm_steps  = round(warm_time  / dt)

x0 = [0.37926545, 0.058339, -0.08167691]

n_timesteps = train_steps + test_steps + warm_steps

X = doublescroll(n_timesteps, x0=x0, h=dt, method="RK23")

N = train_steps + warm_steps + test_steps

fig = plt.figure(figsize=(10, 10))

ax  = fig.add_subplot(111, projection='3d')

ax.set_title("Double scroll attractor (1998)")

ax.set_xlabel("x")

ax.set_ylabel("y")

ax.set_zlabel("z")

ax.grid(False)

for i in range(N-1):

    ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color=plt.cm.cividis(255*i//N), lw=1.0)

plt.show()

nvar = NVAR(delay=2, order=3, strides=1)

# The attractor is centered around (0, 0, 0), no bias is required

readout = Ridge(3, ridge=2.5e-6, input_bias=False)

model = nvar >> readout

Xi  = X[:train_steps+warm_steps-1]

dXi = X[1:train_steps+warm_steps] - X[:train_steps+warm_steps-1]

model = model.fit(Xi, dXi, warmup=warm_steps)

nvar.run(X[warm_steps+train_steps-2:warm_steps+train_steps])

u = X[warm_steps+train_steps]

res = np.zeros((test_steps, readout.output_dim))

for i in range(test_steps):

    u = u + model(u)

    res[i, :] = u

N = test_steps

Y = X[warm_steps+train_steps:]

fig = plt.figure(figsize=(15, 10))

ax  = fig.add_subplot(121, projection='3d')

ax.set_title("Generated attractor")

ax.set_xlabel("$x$")

ax.set_ylabel("$y$")

ax.set_zlabel("$z$")

ax.grid(False)

for i in range(N-1):

    ax.plot(res[i:i+2, 0], res[i:i+2, 1], res[i:i+2, 2], color=plt.cm.cividis(255*i//N), lw=1.0)

ax2 = fig.add_subplot(122, projection='3d')

ax2.set_title("Real attractor")

ax2.grid(False)

for i in range(N-1):

    ax2.plot(Y[i:i+2, 0], Y[i:i+2, 1], Y[i:i+2, 2], color=plt.cm.cividis(255*i//N), lw=1.0)

# time step duration (in time unit)

dt = 0.05

# training time (in time unit)

train_time  = 20.

# testing time (idem)

test_time   = 45.

# warmup time (idem): should always be > k * s

warm_time   = 5.

# discretization

train_steps = round(train_time / dt)

test_steps  = round(test_time  / dt)

warm_steps  = round(warm_time  / dt)

x0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]

n_timesteps = train_steps + test_steps + warm_steps

X = lorenz(n_timesteps, x0=x0, h=dt, method="RK23")

nvar = NVAR(delay=4, order=2, strides=5)

readout = Ridge(1, ridge=0.05)

model = nvar >> readout

xy  = X[:train_steps+warm_steps-1, :2]

z   = X[1:train_steps+warm_steps, 2][:, np.newaxis]

model = model.fit(xy, z, warmup=warm_steps)

_ = nvar.run(X[train_steps:warm_steps+train_steps, :2])

xy_test = X[warm_steps+train_steps:-1, :2]

res = model.run(xy_test)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111)

ax.plot(res, label="Inferred")

ax.plot(X[warm_steps+train_steps+1:, 2], label="Truth", linestyle="--")

ax.plot(abs(res[:, 0] - X[warm_steps+train_steps+1:, 2]), label="Absolute deviation")

ax.set_ylabel("$z$")

ax.set_xlabel("time")

ax.set_title("Lorenz attractor $z$ component inferred value")

ax.set_xticks(np.linspace(0, 900, 5))

ax.set_xticklabels(np.linspace(0, 900, 5) * dt + train_time + warm_time)

plt.legend()

```
# File: io.txt
```python
if x is not None:
    if io_node.input_dim is None:
        io_node.set_input_dim(x.shape[1])
        io_node.set_output_dim(x.shape[1])

return x

"""Node feeding input data to other nodes in the models.

Allow creating an input source and connecting it to several nodes at once.

This node has no parameters and no hyperparameters.

Parameters
----------
input_dim : int
    Input dimension. Can be inferred at first call.
name : str
    Node name.

Example
-------

An input source feeding three different nodes in parallel.

>>> from reservoirpy.nodes import Reservoir, Input
>>> source = Input()
>>> res1, res2, res3 = Reservoir(100), Reservoir(100), Reservoir(100)
>>> model = source >> [res1, res2, res3]

A model with different input sources. Use names to identify each source at runtime.

>>> import numpy as np
>>> from reservoirpy.nodes import Reservoir, Input
>>> source1, source2 = Input(name="s1"), Input(name="s2")
>>> res1, res2 = Reservoir(100), Reservoir(100)
>>> model = source1 >> [res1, res2] & source2 >> [res1, res2]
>>> outputs = model.run({"s1": np.ones((10, 5)), "s2": np.ones((10, 3))})
"""

def __init__(self, input_dim=None, name=None, **kwargs):
    super(Input, self).__init__(
        forward=_input_forward,
        initializer=_io_initialize,
        input_dim=input_dim,
        output_dim=input_dim,
        name=name,
        **kwargs,
    )

"""Convenience node which can be used to add an output to a model.

For instance, this node can be connected to a reservoir within a model to inspect
its states.

Parameters
----------
name : str
    Node name.

Example
-------

We can use the :py:class:`Output` node to probe the hidden states of Reservoir
in an Echo State Network:

>>> import numpy as np
>>> from reservoirpy.nodes import Reservoir, Ridge, Output
>>> reservoir = Reservoir(100)
>>> readout = Ridge()
>>> probe = Output(name="reservoir-states")
>>> esn = reservoir >> readout & reservoir >> probe
>>> _ = esn.initialize(np.ones((1,1)), np.ones((1,1)))

When running the model, states can then be retrieved as an output:

>>> data = np.ones((10, 1))
>>> outputs = esn.run(data)
>>> states = outputs["reservoir-states"]
"""

```
# File: sklearn.txt
```python
import sklearn

sklearn = None

return np.atleast_2d(estimator.estimator.predict(x))

return np.atleast_2d(estimator.estimator.transform(x))

estimator.hypers["estimator"] = estimator.estimator.fit(X, Y)

...

def __init__(self, estimator):
    if hasattr(estimator, "predict"):
        forward = forward_predict
    elif hasattr(estimator, "transform"):
        forward = forward_transform
    else:
        raise TypeError(
            f"Estimator {estimator} has no 'predict' or 'transform' attribute."
        )
```