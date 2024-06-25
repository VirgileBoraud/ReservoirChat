# Contents

# Recurrent neural network

A recurrent neural network (RNN) is one of the two broad types of artificial neural network, characterized by direction of the flow of information between its layers. In contrast to the uni-directional feedforward neural network, it is a bi-directional artificial neural network, meaning that it allows the output from some nodes to affect subsequent input to the same nodes. Their ability to use internal state (memory) to process arbitrary sequences of inputs[1][2][3] makes them applicable to tasks such as unsegmented, connected handwriting recognition[4] or speech recognition.[5][6] The term "recurrent neural network" is used to refer to the class of networks with an infinite impulse response, whereas "convolutional neural network" refers to the class of finite impulse response. Both classes of networks exhibit temporal dynamic behavior.[7] A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that cannot be unrolled.

Additional stored states and the storage under direct control by the network can be added to both infinite-impulse and finite-impulse networks. Another network or graph can also replace the storage if that incorporates time delays or has feedback loops. Such controlled states are referred to as gated states or gated memory and are part of long short-term memory networks (LSTMs) and gated recurrent units. This is also called Feedback Neural Network (FNN). Recurrent neural networks are theoretically Turing complete and can run arbitrary programs to process arbitrary sequences of inputs.[8]

# History[edit]

The Ising model (1925) by Wilhelm Lenz[9] and Ernst Ising[10][11]
was the first RNN architecture that did not learn. Shun'ichi Amari made it adaptive in 1972.[12][13] This was also called the Hopfield network (1982). See also David Rumelhart's work in 1986.[14]  In 1993, a neural history compressor system solved a "Very Deep Learning" task that required more than 1000 subsequent layers in an RNN unfolded in time.[15]

# LSTM[edit]

Long short-term memory (LSTM) networks were invented by Hochreiter and Schmidhuber in 1997 and set accuracy records in multiple applications domains.[16]

Around 2007, LSTM started to revolutionize speech recognition, outperforming traditional models in certain speech applications.[17] In 2009, a Connectionist Temporal Classification (CTC)-trained LSTM network was the first RNN to win pattern recognition contests when it won several competitions in connected handwriting recognition.[18][19] In 2014, the Chinese company Baidu used CTC-trained RNNs to break the 2S09 Switchboard Hub5'00 speech recognition dataset[20] benchmark without using any traditional speech processing methods.[21]

LSTM also improved large-vocabulary speech recognition[5][6] and text-to-speech synthesis[22] and was used in Google Android.[18][23] In 2015, Google's speech recognition reportedly experienced a dramatic performance jump of 49%[citation needed] through CTC-trained LSTM.[24]

LSTM broke records for improved machine translation,[25] Language Modeling[26] and Multilingual Language Processing.[27] LSTM combined with convolutional neural networks (CNNs) improved automatic image captioning.[28]

# Architectures[edit]

RNNs come in many variants.

# Fully recurrent[edit]

Fully recurrent neural networks (FRNN) connect the outputs of all neurons to the inputs of all neurons.  This is the most general neural network topology because all other topologies can be represented by setting some connection weights to zero to simulate the lack of connections between those neurons. The illustration to the right may be misleading to many because practical neural network topologies are frequently organized in "layers" and the drawing gives that appearance. However, what appears to be layers are, in fact, different steps in time of the same fully recurrent neural network. The left-most item in the illustration shows the recurrent connections as the arc labeled 'v'.  It is "unfolded" in time to produce the appearance of layers.

# Elman networks and Jordan networks[edit]

An Elman network is a three-layer network (arranged horizontally as x, y, and z in the illustration) with the addition of a set of context units (u in the illustration). The middle (hidden) layer is connected to these context units fixed with a weight of one.[29] At each time step, the input is fed forward and a learning rule is applied. The fixed back-connections save a copy of the previous values of the hidden units in the context units (since they propagate over the connections before the learning rule is applied). Thus the network can maintain a sort of state, allowing it to perform tasks such as sequence-prediction that are beyond the power of a standard multilayer perceptron.

Jordan networks are similar to Elman networks. The context units are fed from the output layer instead of the hidden layer. The context units in a Jordan network are also called the state layer. They have a recurrent connection to themselves.[29]

Elman and Jordan networks are also known as "Simple recurrent networks" (SRN).

$h

t





=

σ

h


(

W

h



x

t


+

U

h



h

t
−
1


+

b

h


)





y

t





=

σ

y


(

W

y



h

t


+

b

y


)






{\displaystyle {\begin{aligned}h_{t}&=\sigma _{h}(W_{h}x_{t}+U_{h}h_{t-1}+b_{h})\\y_{t}&=\sigma _{y}(W_{y}h_{t}+b_{y})\end{aligned}}}$

$h

t





=

σ

h


(

W

h



x

t


+

U

h



y

t
−
1


+

b

h


)





y

t





=

σ

y


(

W

y



h

t


+

b

y


)






{\displaystyle {\begin{aligned}h_{t}&=\sigma _{h}(W_{h}x_{t}+U_{h}y_{t-1}+b_{h})\\y_{t}&=\sigma _{y}(W_{y}h_{t}+b_{y})\end{aligned}}}$

Variables and functions

$x

t




{\displaystyle x_{t}}$

$h

t




{\displaystyle h_{t}}$

$y

t




{\displaystyle y_{t}}$

$W


{\displaystyle W}$

$U


{\displaystyle U}$

$b


{\displaystyle b}$

$σ

h




{\displaystyle \sigma _{h}}$

$σ

y




{\displaystyle \sigma _{y}}$

# Hopfield[edit]

The Hopfield network is an RNN in which all connections across layers are equally sized. It requires stationary inputs and is thus not a general RNN, as it does not process sequences of patterns. However, it guarantees that it will converge. If the connections are trained using Hebbian learning, then the Hopfield network can perform as robust content-addressable memory, resistant to connection alteration.

# Bidirectional associative memory[edit]

Introduced by Bart Kosko,[32] a bidirectional associative memory (BAM) network is a variant of a Hopfield network that stores associative data as a vector. The bi-directionality comes from passing information through a matrix and its transpose. Typically, bipolar encoding is preferred to binary encoding of the associative pairs. Recently, stochastic BAM models using Markov stepping were optimized for increased network stability and relevance to real-world applications.[33]

A BAM network has two layers, either of which can be driven as an input to recall an association and produce an output on the other layer.[34]

# Echo state[edit]

Echo state networks (ESN) have a sparsely connected random hidden layer. The weights of output neurons are the only part of the network that can change (be trained). ESNs are good at reproducing certain time series.[35] A variant for spiking neurons is known as a liquid state machine.[36]

# Independently RNN (IndRNN)[edit]

The independently recurrent neural network (IndRNN)[37] addresses the gradient vanishing and exploding problems in the traditional fully connected RNN. Each neuron in one layer only receives its own past state as context information (instead of full connectivity to all other neurons in this layer) and thus neurons are independent of each other's history. The gradient backpropagation can be regulated to avoid gradient vanishing and exploding in order to keep long or short-term memory. The cross-neuron information is explored in the next layers. IndRNN can be robustly trained with non-saturated nonlinear functions such as ReLU. Deep networks can be trained using skip connections.

# Recursive[edit]

A recursive neural network[38] is created by applying the same set of weights recursively over a differentiable graph-like structure by traversing the structure in topological order. Such networks are typically also trained by the reverse mode of automatic differentiation.[39][40] They can process distributed representations of structure, such as logical terms. A special case of recursive neural networks is the RNN whose structure corresponds to a linear chain. Recursive neural networks have been applied to natural language processing.[41] The Recursive Neural Tensor Network uses a tensor-based composition function for all nodes in the tree.[42]

# Neural history compressor[edit]

The neural history compressor is an unsupervised stack of RNNs.[43] At the input level, it learns to predict its next input from the previous inputs. Only unpredictable inputs of some RNN in the hierarchy become inputs to the next higher level RNN, which therefore recomputes its internal state only rarely. Each higher level RNN thus studies a compressed representation of the information in the RNN below. This is done such that the input sequence can be precisely reconstructed from the representation at the highest level.

The system effectively minimizes the description length or the negative logarithm of the probability of the data.[44] Given a lot of learnable predictability in the incoming data sequence, the highest level RNN can use supervised learning to easily classify even deep sequences with long intervals between important events.

It is possible to distill the RNN hierarchy into two RNNs: the "conscious" chunker (higher level) and the "subconscious" automatizer (lower level).[43] Once the chunker has learned to predict and compress inputs that are unpredictable by the automatizer, then the automatizer can be forced in the next learning phase to predict or imitate through additional units the hidden units of the more slowly changing chunker. This makes it easy for the automatizer to learn appropriate, rarely changing memories across long intervals. In turn, this helps the automatizer to make many of its once unpredictable inputs predictable, such that the chunker can focus on the remaining unpredictable events.[43]

A generative model partially overcame the vanishing gradient problem[45] of automatic differentiation or backpropagation in neural networks in 1992. In 1993, such a system solved a "Very Deep Learning" task that required more than 1000 subsequent layers in an RNN unfolded in time.[15]

# Second order RNNs[edit]

Second-order RNNs use higher order weights 



w




i
j
k




{\displaystyle w{}_{ijk}}

 instead of the standard 



w




i
j




{\displaystyle w{}_{ij}}

 weights, and states can be a product. This allows a direct mapping to a finite-state machine both in training, stability, and representation.[46][47] Long short-term memory is an example of this but has no such formal mappings or proof of stability.

$w




i
j
k




{\displaystyle w{}_{ijk}}$

$w




i
j




{\displaystyle w{}_{ij}}$

# Long short-term memory[edit]

Long short-term memory (LSTM) is a deep learning system that avoids the vanishing gradient problem. LSTM is normally augmented by recurrent gates called "forget gates".[48] LSTM prevents backpropagated errors from vanishing or exploding.[45] Instead, errors can flow backward through unlimited numbers of virtual layers unfolded in space. That is, LSTM can learn tasks[18] that require memories of events that happened thousands or even millions of discrete time steps earlier. Problem-specific LSTM-like topologies can be evolved.[49] LSTM works even given long delays between significant events and can handle signals that mix low and high-frequency components.

Many applications use stacks of LSTM RNNs[50] and train them by connectionist temporal classification (CTC)[51] to find an RNN weight matrix that maximizes the probability of the label sequences in a training set, given the corresponding input sequences. CTC achieves both alignment and recognition.

LSTM can learn to recognize context-sensitive languages unlike previous models based on hidden Markov models (HMM) and similar concepts.[52]

# Gated recurrent unit[edit]

Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks introduced in 2014. They are used in the full form and several simplified variants.[53][54] Their performance on polyphonic music modeling and speech signal modeling was found to be similar to that of long short-term memory.[55] They have fewer parameters than LSTM, as they lack an output gate.[56]

# Bi-directional[edit]

Bi-directional RNNs use a finite sequence to predict or label each element of the sequence based on the element's past and future contexts. This is done by concatenating the outputs of two RNNs, one processing the sequence from left to right, and the other one from right to left. The combined outputs are the predictions of the teacher-given target signals. This technique has been proven to be especially useful when combined with LSTM RNNs.[57][58]

# Continuous-time[edit]

A continuous-time recurrent neural network (CTRNN) uses a system of ordinary differential equations to model the effects on a neuron of the incoming inputs.

For a neuron 



i


{\displaystyle i}

 in the network with activation 




y

i




{\displaystyle y_{i}}

, the rate of change of activation is given by:

$i


{\displaystyle i}$

$y

i




{\displaystyle y_{i}}$

$τ

i






y
˙




i


=
−

y

i


+

∑

j
=
1


n



w

j
i


σ
(

y

j


−

Θ

j


)
+

I

i


(
t
)


{\displaystyle \tau _{i}{\dot {y}}_{i}=-y_{i}+\sum _{j=1}^{n}w_{ji}\sigma (y_{j}-\Theta _{j})+I_{i}(t)}$

Where:

$τ

i




{\displaystyle \tau _{i}}$

$y

i




{\displaystyle y_{i}}$

$y
˙




i




{\displaystyle {\dot {y}}_{i}}$

$w




j
i




{\displaystyle w{}_{ji}}$

$σ
(
x
)


{\displaystyle \sigma (x)}$

$σ
(
x
)
=
1

/

(
1
+

e

−
x


)


{\displaystyle \sigma (x)=1/(1+e^{-x})}$

$y

j




{\displaystyle y_{j}}$

$Θ

j




{\displaystyle \Theta _{j}}$

$I

i


(
t
)


{\displaystyle I_{i}(t)}$

CTRNNs have been applied to evolutionary robotics where they have been used to address vision,[59] co-operation,[60] and minimal cognitive behaviour.[61]

Note that, by the Shannon sampling theorem, discrete-time recurrent neural networks can be viewed as continuous-time recurrent neural networks where the differential equations have transformed into equivalent difference equations.[62] This transformation can be thought of as occurring after the post-synaptic node activation functions 




y

i


(
t
)


{\displaystyle y_{i}(t)}

 have been low-pass filtered but prior to sampling.

$y

i


(
t
)


{\displaystyle y_{i}(t)}$

# Hierarchical recurrent neural network[edit]

Hierarchical recurrent neural networks (HRNN) connect their neurons in various ways to decompose hierarchical behavior into useful subprograms.[43][63] Such hierarchical structures of cognition are present in theories of memory presented by philosopher Henri Bergson, whose philosophical views have inspired hierarchical models.[64]

Hierarchical recurrent neural networks are useful in forecasting, helping to predict disaggregated inflation components of the consumer price index (CPI). The HRNN model leverages information from higher levels in the CPI hierarchy to enhance lower-level predictions. Evaluation of a substantial dataset from the US CPI-U index demonstrates the superior performance of the HRNN model compared to various established inflation prediction methods.[65]

# Recurrent multilayer perceptron network[edit]

Generally, a recurrent multilayer perceptron network (RMLP network) consists of cascaded subnetworks, each containing multiple layers of nodes. Each subnetwork is feed-forward except for the last layer, which can have feedback connections. Each of these subnets is connected only by feed-forward connections.[66]

# Multiple timescales model[edit]

A multiple timescales recurrent neural network (MTRNN) is a neural-based computational model that can simulate the functional hierarchy of the brain through self-organization depending on the spatial connection between neurons and on distinct types of neuron activities, each with distinct time properties.[67][68] With such varied neuronal activities, continuous sequences of any set of behaviors are segmented into reusable primitives, which in turn are flexibly integrated into diverse sequential behaviors. The biological approval of such a type of hierarchy was discussed in the memory-prediction theory of brain function by Hawkins in his book On Intelligence.[citation needed] Such a hierarchy also agrees with theories of memory posited by philosopher Henri Bergson, which have been incorporated into an MTRNN model.[64][69]

# Neural Turing machines[edit]

Neural Turing machines (NTMs) are a method of extending recurrent neural networks by coupling them to external memory resources which they can interact with by attentional processes. The combined system is analogous to a Turing machine or Von Neumann architecture but is differentiable end-to-end, allowing it to be efficiently trained with gradient descent.[70]

# Differentiable neural computer[edit]

Differentiable neural computers (DNCs) are an extension of Neural Turing machines, allowing for the usage of fuzzy amounts of each memory address and a record of chronology.

# Neural network pushdown automata[edit]

Neural network pushdown automata (NNPDA) are similar to NTMs, but tapes are replaced by analog stacks that are differentiable and trained. In this way, they are similar in complexity to recognizers of context free grammars (CFGs).[71]

# Memristive networks[edit]

Greg Snider of HP Labs describes a system of cortical computing with memristive nanodevices.[72] The memristors (memory resistors) are implemented by thin film materials in which the resistance is electrically tuned via the transport of ions or oxygen vacancies within the film. DARPA's SyNAPSE project has funded IBM Research and HP Labs, in collaboration with the Boston University Department of Cognitive and Neural Systems (CNS), to develop neuromorphic architectures that may be based on memristive systems.
Memristive networks are a particular type of physical neural network that have very similar properties to (Little-)Hopfield networks, as they have continuous dynamics, a limited memory capacity and natural relaxation via the minimization of a function which is asymptotic to the Ising model. In this sense, the dynamics of a memristive circuit have the advantage compared to a Resistor-Capacitor network to have a more interesting non-linear behavior. From this point of view, engineering analog memristive networks account for a peculiar type of neuromorphic engineering in which the device behavior depends on the circuit wiring or topology.
The evolution of these networks can be studied analytically using variations of the Caravelli–Traversa–Di Ventra equation.[73]

# Pseudocode[edit]

Given a time series x of length sequence_length.
In the recurrent neural network, there is a loop that processes all entries of the time series x through the layers neural_network one after another. These have as return value in each time step i both the prediction y_pred[i] and an updated hidden state hidden, which has the length hidden_size. As a result, after the loop, the collection of all predictions y_pred is returned.
The following pseudocode (based on the programming language Python) illustrates the functionality of a recurrent neural network.[74]

Modern libraries provide runtime-optimized implementations of the above functionality or allow to speed up the slow loop by just-in-time compilation.

# Training[edit]

# Gradient descent[edit]

Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. In neural networks, it can be used to minimize the error term by changing each weight in proportion to the derivative of the error with respect to that weight, provided the non-linear activation functions are differentiable. Various methods for doing so were developed in the 1980s and early 1990s by Werbos, Williams, Robinson, Schmidhuber, Hochreiter, Pearlmutter and others.

The standard method is called "backpropagation through time" or BPTT, and is a generalization of back-propagation for feed-forward networks.[75][76] Like that method, it is an instance of automatic differentiation in the reverse accumulation mode of Pontryagin's minimum principle. A more computationally expensive online variant is called "Real-Time Recurrent Learning" or RTRL,[77][78] which is an instance of automatic differentiation in the forward accumulation mode with stacked tangent vectors. Unlike BPTT, this algorithm is local in time but not local in space.

In this context, local in space means that a unit's weight vector can be updated using only information stored in the connected units and the unit itself such that update complexity of a single unit is linear in the dimensionality of the weight vector. Local in time means that the updates take place continually (on-line) and depend only on the most recent time step rather than on multiple time steps within a given time horizon as in BPTT. Biological neural networks appear to be local with respect to both time and space.[79][80]

For recursively computing the partial derivatives, RTRL has a time-complexity of O(number of hidden x number of weights) per time step for computing the Jacobian matrices, while BPTT only takes O(number of weights) per time step, at the cost of storing all forward activations within the given time horizon.[81] An online hybrid between BPTT and RTRL with intermediate complexity exists,[82][83] along with variants for continuous time.[84]

A major problem with gradient descent for standard RNN architectures is that error gradients vanish exponentially quickly with the size of the time lag between important events.[45][85] LSTM combined with a BPTT/RTRL hybrid learning method attempts to overcome these problems.[16] This problem is also solved in the independently recurrent neural network (IndRNN)[37] by reducing the context of a neuron to its own past state and the cross-neuron information can then be explored in the following layers. Memories of different ranges including long-term memory can be learned without the gradient vanishing and exploding problem.

The on-line algorithm called causal recursive backpropagation (CRBP), implements and combines BPTT and RTRL paradigms for locally recurrent networks.[86] It works with the most general locally recurrent networks. The CRBP algorithm can minimize the global error term. This fact improves the stability of the algorithm, providing a unifying view of gradient calculation techniques for recurrent networks with local feedback.

One approach to gradient information computation in RNNs with arbitrary architectures is based on signal-flow graphs diagrammatic derivation.[87] It uses the BPTT batch algorithm, based on Lee's theorem for network sensitivity calculations.[88] It was proposed by Wan and Beaufays, while its fast online version was proposed by Campolucci, Uncini and Piazza.[88]

# Global optimization methods[edit]

Training the weights in a neural network can be modeled as a non-linear global optimization problem. A target function can be formed to evaluate the fitness or error of a particular weight vector as follows: First, the weights in the network are set according to the weight vector. Next, the network is evaluated against the training sequence. Typically, the sum-squared difference between the predictions and the target values specified in the training sequence is used to represent the error of the current weight vector. Arbitrary global optimization techniques may then be used to minimize this target function.

The most common global optimization method for training RNNs is genetic algorithms, especially in unstructured networks.[89][90][91]

Initially, the genetic algorithm is encoded with the neural network weights in a predefined manner where one gene in the chromosome represents one weight link. The whole network is represented as a single chromosome. The fitness function is evaluated as follows:

Many chromosomes make up the population; therefore, many different neural networks are evolved until a stopping criterion is satisfied. A common stopping scheme is:

The fitness function evaluates the stopping criterion as it receives the mean-squared error reciprocal from each network during training. Therefore, the goal of the genetic algorithm is to maximize the fitness function, reducing the mean-squared error.

Other global (and/or evolutionary) optimization techniques may be used to seek a good set of weights, such as simulated annealing or particle swarm optimization.

# Related fields and models[edit]

RNNs may behave chaotically. In such cases, dynamical systems theory may be used for analysis.

They are in fact recursive neural networks with a particular structure: that of a linear chain. Whereas recursive neural networks operate on any hierarchical structure, combining child representations into parent representations, recurrent neural networks operate on the linear progression of time, combining the previous time step and a hidden representation into the representation for the current time step.

In particular, RNNs can appear as nonlinear versions of finite impulse response and infinite impulse response filters and also as a nonlinear autoregressive exogenous model (NARX).[92]

The effect of memory-based learning for the recognition of sequences can also be implemented by a more biological-based model which uses the silencing mechanism exhibited in neurons with a relatively high frequency spiking activity.[93]

# Libraries[edit]

# Applications[edit]

Applications of recurrent neural networks include:

# References[edit]

# Further reading[edit]

# External links[edit]

## Libraries

## References

