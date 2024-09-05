# Echo state network

Herbert Jaeger, Jacobs University Bremen, Bremen, Germany

Echo state networks (ESN) provide an architecture and supervised learning principle for recurrent neural networks (RNNs). The main idea is (i) to drive a random, large, fixed recurrent neural network with the input signal, thereby inducing in each neuron within this "reservoir" network a nonlinear response signal, and (ii) combine a desired output signal by a trainable linear combination of all of these response signals.

The basic idea of ESNs is shared with Liquid State Machines (LSM), which were developed independently from and simultaneously with ESNs by Wolfgang Maass (Maass W., Natschlaeger T., Markram H. 2002). Increasingly often, LSMs, ESNs and the more recently explored Backpropagation Decorrelation learning rule for RNNs (Schiller and Steil 2005) are subsumed under the name of Reservoir Computing. Schiller and Steil (2005) also showed that in traditional training methods for RNNs, where all weights (not only the output weights) are adapted, the dominant changes are in the output weights. In cognitive neuroscience, a related mechanism has been investigated by Peter F. Dominey in the context of modelling sequence processing in mammalian brains, especially speech recognition in the human brain (e.g., Dominey 1995, Dominey, Hoen and Inui 2006).  The basic idea also informed a model of temporal input discrimination in biological neural networks (Buonomano and Merzenich 1995).  An early clear formulation of the reservoir computing idea is due to K. Kirby who exposed this concept in a largely forgotten (1 Google cite, as of 2017) conference contribution (Kirby 1991). The earliest currently known formulation of the reservoir computing idea was given by L. Schomaker (1990 [= chapter 7 in Schomaker 1991], 1992), who described how a desired target output from an RNN can be obtained by learning to combine signals from a randomly configured ensemble of spiking neural oscillators.

For an illustration, consider the task of training an RNN to behave
as a tunable frequency generator (download the MATLAB code of this example). The
input signal \(u(n)\) is a slowly varying frequency setting, the desired
output \(y(n)\) is a sinewave of a frequency indicated by the current
input.  Assume that a training input-output sequence \(D = (u(1),y(1)),
\ldots, (u(n_{max}),y(n_{max}))\)
is given (see the input and output signals in  ; here
the input is a slow random step function indicating frequencies
ranging from 1/16 to 1/4 Hz). The task is to train a RNN from these
training data such that on slow test input signals, the output is
again a sinewave of the input-determined frequency.

In the ESN approach, this task is solved by the following steps.

# Contents

# Variants

Echo state networks can be set up with or without direct trainable input-to-output
connections, with or without output-to-reservoir feedback, with
different neuron types, different reservoir-internal connectivity
patterns, etc. Furthermore, the output weights can be computed with
any of the available offline or online algorithms for linear regression. Besides least-mean-square error solutions (i.e., linear regression weights),
margin-maximization criteria known from training support vector
machines have been used to determine output weights (Schmidhuber et al. 2007)

The unifying theme throughout all these variations is to use a
fixed RNN as a random nonlinear excitable medium, whose
high-dimensional dynamical "echo" response to a driving input
(and/or output feedback) is used as a non-orthogonal signal basis to
reconstruct the desired output by a linear combination, minimizing
some error criteria.

# Formalism and theory

System equations. The basic
discrete-time, sigmoid-unit echo state network with \(N\) reservoir units,
\(K\) inputs and \(L\) outputs is governed by the
state update equation

(1)  \(\mathbf{x}(n+1) = f(\mathbf{W} \mathbf{x}(n) + \mathbf{W}^{in}
\mathbf{u}(n+1) + \mathbf{W}^{fb} \mathbf{y}(n))\ ,\)

where \(\mathbf{x}(n)\) is the \(N\)-dimensional
reservoir state, \(f\) is a sigmoid function (usually the
logistic sigmoid or the tanh function), \(\mathbf{W}\) is the
\(N \times N\) reservoir weight matrix,
\(\mathbf{W}^{in}\) is the \(N \times K\) input
weight matrix, \(\mathbf{u}(n)\) is the
\(K\)dimensional input signal, \(\mathbf{W}^{fb}\)
is the \(N \times L\) output feedback matrix, and
\(\mathbf{y}(n)\) is the \(L\)-dimensional output
signal. In tasks where no output feedback is required,
\(\mathbf{W}^{fb}\) is nulled. The extended system state
\(\mathbf{z}(n) = [\mathbf{x}(n); \mathbf{u}(n)]\) at
time \(n\) is the concatenation of the reservoir and input
states. The output is obtained from the extended system state by

(2)  \(\mathbf{y}(n) = g(\mathbf{W}^{out} \mathbf{z}(n))\ ,\)

where \(g\) is an output activation function (typically the
identity or a sigmoid) and \(\mathbf{W}^{out}\) is a
\(L \times (K+N)\)-dimensional matrix of output
weights.

Learning equations. In the state harvesting stage of the
training, the ESN is driven by an input sequence \(\mathbf{u}(1),
\ldots, \mathbf{u}(n_{max})\ ,\) which yields a sequence
\(\mathbf{z}(1), \ldots, \mathbf{z}(n_{max})\) of extended
system states. The system equations (1), (2) are used here. If the
model includes output feedback (i.e., nonzero
\(\mathbf{W}^{fb}\)), then during the generation of the
system states, the correct outputs \(\mathbf{d}(n)\) (part of
the training data) are written into the output units ("teacher
forcing"). The obtained extended system states are filed row-wise into
a state collection matrix \(\mathbf{S}\) of size
\(n_{max} \times (N + K)\ .\) Usually some initial portion of
the states thus collected are discarded to accommodate for a washout of
the arbitrary (random or zero) initial reservoir state needed at time
1. Likewise, the desired outputs \(\mathbf{d}(n)\) are sorted
row-wise into a teacher output collection matrix
\(\mathbf{D}\) of size \(n_{max} \times L\ .\)

The desired output weights \(\mathbf{W}^{out}\) are the
linear regression weights of the desired outputs
\(\mathbf{d}(n)\) on the harvested extended states
\(\mathbf{z}(n)\ .\) A mathematically straightforward way to compute \(\mathbf{W}^{out}\) is to invoke the pseudoinverse (denoted by \(\cdot^{\dagger}\)) of \(\mathbf{S}\ :\)

(3)  \(\mathbf{W}^{out} =
     (\mathbf{S}^{\dagger}\mathbf{D})'\ \),

which is an  offline algorithm (the prime denotes matrix transpose). Online adaptive methods known from linear signal processing can also be used to compute output weights (Jaeger 2003).

Echo state property. In order for the ESN principle to work, the
reservoir must have the echo state property (ESP), which relates
asymptotic properties of the excited reservoir dynamics to the driving
signal. Intuitively, the ESP states that the reservoir will
asymptotically wash out any information from initial conditions. The
ESP is guaranteed for additive-sigmoid neuron reservoirs, if the
reservoir weight matrix (and the leaking rates) satisfy certain
algebraic conditions in terms of singular values. For such reservoirs
with a tanh sigmoid, the ESP is violated for zero input if the
spectral radius of the reservoir weight matrix is larger than
unity. Conversely, it is empirically observed that the ESP is granted
for any input if this spectral radius is smaller than unity. This has
led in the literature to a far-spread but erroneous identification of
the ESP with a spectral radius below 1. Specifically, the larger the
input amplitude, the further above unity the spectral radius may be
while still obtaining the ESP. An abstract characterization of the ESP
for arbitrary reservoir types, and algebraic conditions for
additive-sigmoid neuron reservoirs are given in Jaeger (2001a); for an
important subclass of reservoirs, tighter algebraic conditions are
given in Buehner and Young (2006) and Yildiz et al. (2012); for leaky integrator neurons,
algebraic conditions are spelled out in Jaeger et al. (2007). The relationship between input signal characteristics and the ESP are explored in Manjunath and Jaeger (2013), where a fundamental 0-1-law is shown: if the input comes from a stationary source, the ESP holds with probability 1 or 0.

Memory capacity. Due to the auto-feedback nature of RNNs, the
reservoir states \(\mathbf{x}(n)\) reflect traces of the past input
history. This can be seen as a dynamical short-term memory. For a
single-input ESN, this short-term memory's capacity \(C\) can be
quantified by \(C = \sum_{i = 1, 2, \ldots} r^2(u(n-i), y_i(n))\ ,\) where
\(r^2(u(n-i), y_i(n))\) is the squared correlation coefficient between
the input signal delayed by \(i\) and a trained output signal \(y_i(n)\)
which was trained on the task to retrodict (memorize) \(u(n-i)\) on the
input signal \(u(n)\ .\) It turns out that for i.i.d. input, the memory
capacity \(C\) of an echo state network of size \(N\) is bounded by \(N\ ;\) in the absence
of numerical errors and with a linear reservoir the bound is attained
(Jaeger 2002a; White and Sompolinsky 2004; Hermans & Schrauwen 2009). These findings imply that
it is impossible to train ESNs on tasks which require unbounded-time
memory, like for instance context-free
grammar parsing tasks (Schmidhuber et al. 2007). However, if output
units with feedback to the reservoir are trained as attractor memory
units, unbounded memory spans can be realized with ESNs, too (cf. the
multistable switch example in Jaeger 2002a; beginnings of a theory of feedback-induced memory-hold attractors in Maass, Joshi & Sontag 2007; an ESN based model of working memory with stable attractor states in Pascanu & Jaeger 2010).

Universal computation and approximation properties. ESNs can
realize every nonlinear filter with bounded memory arbitrarily
well. This line of theoretical research has been started and advanced
in the field of Liquid State Machines (Maass, Natschlaeger & Markram 2002; Maass, Joshi & Sontag 2007), and the reader is
referred to the LSM article for detail.

# Practical issues: tuning global controls and regularization

When using ESNs in practical nonlinear modeling tasks, the ultimate
objective is to minimize the test error. A standard method in machine
learning to get an
estimate of the test error is to use only a part of the available
training data for model estimation, and monitor the model's
performance on the withheld portion of the original training
data (the validation set).  The question is, how can the ESN models be optimized in order
to reduce the error on the validation set? In the terminology of
machine learning, this boils down to the question how one can equip
the ESN models with a task-appropriate bias. With ESNs, there are
three sorts of bias (in a wide sense) which one  should adjust.

The first sort of bias is to employ regularization. This
essentially means that the models are smoothed. Two standard ways to
achieve some kind of smoothing are the following:

where  \(\mathbf{R} = 1/n_{max} \; \mathbf{S}'\mathbf{S}\) is the correlation matrix of the extended reservoir states, \(\mathbf{P} = 1/n_{max} \; \mathbf{S}'\mathbf{D}\) is the cross-correlation matrix of the states vs. the desired outputs, \(\alpha^2\) is some nonnegative number (the larger, the stronger the smoothing effect), and \(\mathbf{I}\) is the identity matrix.

Both methods lead to smaller output weights. Adding state noise is computationally more expensive, but appears to have the additional benefit of stabilizing solutions in models with output feedback (Jaeger 2002a; Jaeger, Lukosevicius, Popovici & Siewert 2007).

The second sort of bias is effected by making the echo state network, as one could say, "dynamically similar" to the system that one wants to model. For instance, if the original system is evolving on a slow timescale, the ESN should do the same; or if the original system has long memory spans, so should the ESN. This shaping of major dynamical characteristics is realized by adjusting a small number of global control parameters:

Finally, a third sort of bias (here the terminology is stretched a bit) is simply the reservoir size \(N\ .\) In the sense of statistical learning theory, increasing the reservoir size is the most direct way of increasing the model capacity.

All these kinds of bias have to be optimized jointly. The current standard practice to do this is manual experimentation. Practical "tricks of the trade" are collected in Lukoševičius (2012).

# Significance

A number of algorithms for the supervised training of RNNs have been
known since the early 1990s, most notably real-time recurrent learning (Williams and Zipser 1989), backpropagation through time
(Werbos 1990), extended Kalman filtering based methods (Puskorius
and Feldkamp 2004), and the Atiya-Parlos algorithm (Atiya and Parlos
2000). All of these algorithms adapt all connections (input,
recurrent, output) by some version of gradient descent. This renders
these algorithms slow, and what is maybe even more cumbersome, makes
the learning process prone to become disrupted by bifurcations (Doya 1992); convergence cannot be guaranteed. As a consequence,
RNNs were rarely fielded in practical engineering
applications at the time when ESNs were introduced. ESN training, by contrast, is fast, does not suffer from
bifurcations, and is easy to implement. On a number of benchmark
tasks, ESNs have starkly outperformed all other methods of nonlinear
dynamical modelling (Jaeger and Haas 2004, Jaeger et al. 2007).

Today (as of 2017), with the advent of Deep Learning, the problems faced by gradient descent based training of recurrent neural networks can be considered solved. The original unique selling point of ESNs, stable and simple training algorithms, has dissolved. Moreover, deep learning methods for RNNs have proven effective for highly complex modeling tasks especially in language and speech processing. Reaching similar levels of complexity would demand reservoirs of inordinate size. Methods of reservoir computing are nonetheless an alternative worth considering when the modeled system is not too complex, and when cheap, fast and adaptive training is desired. This holds true for many applications in signal processing, as for example in biosignal processing (Kudithipudi et al. 2015), remote sensing (Antonelo 2017)  or robot motor control (Polydoros et al. 2015).

Starting around 2010, echo state networks have become relevant and quite popular as a computational principle that blends well with non-digital computational substrates, for instance optical microchips (Vandoorne et al. 2014), mechanical nano-oscillators (Coulombe, York and Sylvestre 2017), memristor-based neuromorphic microchips (Bürger et al. 2015), carbon-nanotube / polymer mixtures (Dale et al. 2016) or even artificial soft limbs (Nakajima, Hauser and Pfeifer 2015). Such nonstandard computational materials and the microdevices made from them often lack numerical precision, exhibit significant device mismatch, and ways to emulate classical logical switching circuits are unknown. But, often nonlinear dynamics can be elicited from suitably interconnected ensembles of such elements -- that is, physical reservoirs can be built, opening the door for training such systems with ESN methods.

# See Also

Liquid State Machine, Recurrent Neural Networks, Supervised Learning