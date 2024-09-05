# Contents

# Echo state network

An echo state network (ESN)[1][2] is a type of reservoir computer that uses a recurrent neural network with a sparsely connected hidden layer (with typically 1% connectivity). The connectivity and weights of hidden neurons are fixed and randomly assigned. The weights of output neurons can be learned so that the network can produce or reproduce specific temporal patterns. The main interest of this network is that although its behavior is non-linear, the only weights that are modified during training are for the synapses that connect the hidden neurons to output neurons. Thus, the error function is quadratic with respect to the parameter vector and can be differentiated easily to a linear system.

Alternatively, one may consider a nonparametric Bayesian formulation of the output layer, under which: (i) a prior distribution is imposed over the output weights; and (ii) the output weights are marginalized out in the context of prediction generation, given the training data. This idea has been demonstrated in[3] by using Gaussian priors, whereby a Gaussian process model with ESN-driven kernel function is obtained. Such a solution was shown to outperform ESNs with trainable (finite) sets of weights in several benchmarks.

Some publicly available implementations of ESNs are: (i) aureservoir: an efficient C++ library for various kinds of echo state networks with python/numpy bindings; (ii) Matlab code: an efficient matlab for an echo state network; (iii) ReservoirComputing.jl: an efficient Julia-based implementation of various types of echo state networks; and (iv) pyESN: simple echo state networks in Python.

# Background[edit]

The Echo State Network (ESN)[4] belongs to the Recurrent Neural Network (RNN) family and provide their architecture and supervised learning principle. Unlike Feedforward Neural Networks, Recurrent Neural Networks are dynamic systems and not functions. Recurrent Neural Networks are typically used for:

For the training of RNNs a number of learning algorithms are available: backpropagation through time, real-time recurrent learning. Convergence is not guaranteed due to instability and bifurcation phenomena.[4]

The main approach of the ESN is firstly to operate a random, large, fixed, recurring neural network with the input signal, which induces a nonlinear response signal in each neuron within this "reservoir" network, and secondly connect a desired output signal by a trainable linear combination of all these response signals.[2]

Another feature of the ESN is the autonomous operation in prediction: if the Echo State Network is trained with an input that is a backshifted version of the output, then it can be used for signal generation/prediction by using the previous output as input.[4][5]

The main idea of ESNs is tied to Liquid State Machines (LSM), which were independently and simultaneously developed with ESNs by Wolfgang Maass.[6] LSMs, ESNs and the newly researched Backpropagation Decorrelation learning rule for RNNs[7] are more and more summarized under the name Reservoir Computing.

Schiller and Steil[7] also demonstrated that in conventional training approaches for RNNs, in which all weights (not only output weights) are adapted, the dominant changes are in output weights. In cognitive neuroscience, Peter F. Dominey analysed a related process related to the modelling of sequence processing in the mammalian brain, in particular speech recognition in the human brain.[8] The basic idea also included a model of temporal input discrimination in biological neuronal networks.[9] An early clear formulation of the reservoir computing idea is due to K. Kirby, who disclosed this concept in a largely forgotten conference contribution.[10] The first formulation of the reservoir computing idea known today stems from L. Schomaker,[11] who described how a desired target output could be obtained from an RNN by learning to combine signals from a randomly configured ensemble of spiking neural oscillators.[2]

# Variants[edit]

Echo state networks can be built in different ways. They can be set up with or without directly trainable input-to-output connections, with or without output reservation feedback, with different neurotypes, different reservoir internal connectivity patterns etc. The output weight can be calculated for linear regression with all algorithms whether they are online or offline. In addition to the solutions for errors with smallest squares, margin maximization criteria, so-called training support vector machines, are used to determine the output values.[12] Other variants of echo state networks seek to change the formulation to better match common models of physical systems, such as those typically those defined by differential equations. Work in this direction includes echo state networks which partially include physical models,[13] hybrid echo state networks,[14] and continuous-time echo state networks.[15]

The fixed RNN acts as a random, nonlinear medium whose dynamic response, the "echo", is used as a signal base.  The linear combination of this base can be trained to reconstruct the desired output by minimizing some error criteria.[2]

# Significance[edit]

RNNs were rarely used in practice before the introduction of the ESN, because of the complexity involved in adjusting their connections (e.g., lack of autodifferentiation, susceptibility to vanishing/exploding gradients, etc.). RNN training algorithms were slow and often vulnerable to issues, such as branching errors.[16]  Convergence could therefore not be guaranteed. On the other hand, ESN training does not have a problem with branching and is easy to implement. In early studies, ESNs were shown to perform well on time series prediction tasks from synthetic datasets.[1][17]

However, today, many of the problems that made RNNs slow and error-prone have been addressed with the advent of autodifferentiation (deep learning) libraries, as well as more stable architectures such as LSTM and GRU—thus, the unique selling point of ESNs has been lost. In addition, RNNs have proven themselves in several practical areas, such as language processing. To cope with tasks of similar complexity using reservoir calculation methods requires memory of excessive size.

However, ESNs are used in some areas, such as many signal processing applications. In particular, they have been widely used as a computing principle that mixes well with non-digital computer substrates. Since ESNs do not need to modify the parameters of the RNN, they make it possible to use many different objects as their nonlinear "reservoir″. For example, optical microchips, mechanical nanooscillators, polymer mixtures, or even artificial soft limbs.[2]

# See also[edit]

# References[edit]

## References

