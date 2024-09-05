**Beginner**

- What is a reservoir?

- What is reservoir computing?

- What is readout?

- Why the name ‘reservoir’?

- Why the name ‘readout’?

- On which tasks is reservoir computing good? 

- On which tasks is reservoir computing bad?

- Approximately how many neurons should be used (10, 100, 1000, 1 million)?

- What is the purpose of the ridge in readout?

- How are the weights set in the reservoir?

- Do the weights in the input learn from the neurons in the reservoir?

- Create a dataset on the normalised Mackey-Glass time series, with a prediction at 20 time steps (import of Mackey-Glass, normalisation, X/Y separation, train/test, etc).

- Create a simple reservoir/ESN, and train it on a dataset containing several time series (with the ESN or Reservoir+Ridge node)

- Creates an echo state network with parallelization

**Intermediate**

- What is the difference between ‘echo state network’ and ‘reservoir computing’?

- Are there other forms of reservoir computing?

- Why is it called ‘computing at the edge of chaos’?

- What is the ‘echo state property’?

- Which paper introduces reservoir computing?

- Which paper introduces echo state network?

- What are all the hyper-parameters?

- How do you choose the hyper-parameters?

- Write a code to display the evolution of the reservoir neurons on the Lorenz series.

- Create an NVAR model with online learning

- Create a reservoir in which all the neurons are connected online, and the input is connected to the first neuron

- Creates a DeepESN model

- Creates a model with 10 parallel reservoirs connected to the same readout

**Advanced**

- What is a liquid state machine?

- How explainable are reservoir computing models?

- To what extent do the results vary between two differently initialised reservoirs?

- What influence does the sparsity of the weight matrix have on performance?

- Create a ReservoirPy node that adds Gaussian noise to the input it receives.

- Write a hyper-parameter search using the TPE sampler, on 300 instances, and evaluating the NRMSE, the R² and the maximum error.