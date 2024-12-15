**Beginner**

- What is a reservoir in the context of machine learning?

- What is reservoir computing?

- What is readout in the context of reservoir computing?

- Why the name ‘reservoir’ in the context of reservoir computing?

- Why the name ‘readout’ in the context of reservoir computing?

- On which tasks is reservoir computing good? 

- On which tasks is reservoir computing bad?

- Approximately how many neurons should be used (10, 100, 1000, 1 million) in the context of reservoir computing?

- What is the purpose of the ridge in readout in the context of reservoir computing?

- How are the weights set in the reservoir in the context of reservoir computing?

- Do the weights in the input learn from the neurons in the reservoir in the context of reservoir computing?

- Create a dataset on the normalised Mackey-Glass time series, with a prediction at 20 time steps (import of Mackey-Glass, normalisation, X/Y separation, train/test, etc) using the reservoirPy library.

- Create a simple reservoir/ESN, and train it on a dataset containing several time series (with the ESN or Reservoir+Ridge node) using the reservoirPy library.

- Creates an echo state network with parallelization using the reservoirPy library.

**Intermediate**

- What is the difference between ‘echo state network’ and ‘reservoir computing’ in the context of reservoir computing?

- Are there other forms of reservoir computing?

- Why is it called ‘computing at the edge of chaos’ in the context of reservoir computing?

- What is the ‘echo state property’ in the context of reservoir computing?

- Which paper introduces reservoir computing?

- Which paper introduces echo state network?

- What are all the hyper-parameters in the context of reservoir computing?

- How do you choose the hyper-parameters in the context of reservoir computing?

- Write a code to display the evolution of the reservoir neurons on the Lorenz series using the reservoirPy library.

- Create an NVAR model with online learning using the reservoirPy library.

- Create a reservoir in which all the neurons are connected online, and the input is connected to the first neuron using the reservoirPy library.

- Creates a DeepESN model using the reservoirPy library.

- Creates a model with 10 parallel reservoirs connected to the same readout using the reservoirPy library.

**Advanced**

- What is a liquid state machine in the context of reservoir computing?

- How explainable are reservoir computing models?

- To what extent do the results vary between two differently initialised reservoirs in the context of reservoir computing?

- What influence does the sparsity of the weight matrix have on performance in the context of reservoir computing?

- Create a ReservoirPy node that adds Gaussian noise to the input it receives using the reservoirPy library.

- Write a hyper-parameter search using the TPE sampler, on 300 instances, and evaluating the NRMSE, the R² and the maximum error using the reservoirPy library.