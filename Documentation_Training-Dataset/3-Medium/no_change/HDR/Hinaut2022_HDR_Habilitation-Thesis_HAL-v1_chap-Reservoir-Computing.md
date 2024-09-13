Chapter 3
Diving into Reservoirs
Contents
3.1
Context
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18
3.2
Intuitions in (almost) one page . . . . . . . . . . . . . . . . .
20
3.3
Some equations . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
Before going more in depth in my research, I will brieﬂy introduce the Reservoir
Computing (RC) paradigm. It is central in my work since the beginning of my PhD
thesis, during which I worked within the FP7 European Project Organic which
gathered the European founders of Reservoir Computing. That’s why I want to
make a short overview to enable readers to have a little idea of what is RC before
what will follow.
Random weights
Learned weights
Activation through time
Inputs
Outputs
Figure 3.1: The Reservoir Computing (RC) paradigm to train Recurrent Neural
Networks (RNNs). Input and recurrent weights are ﬁxed and random while output
weights are trained. Time series provided as input generate a non-linear combination
of dynamics inside the reservoir – the recurrent part in the middle. The output layer
linearly reads out some of these dynamical combinations – it makes a weighted sum
of reservoir states. Image from [Juven & Hinaut 2020].
Reservoir insight. To start, let’s dive in reservoir computing with a quick example
in Figure 3.1. Inputs are fed to a recurrent layer of neurons, called the reservoir.
18
Chapter 3.
Diving into Reservoirs
Reservoir states combine these incoming inputs together with its previous states
thanks to the recurrent connections. The reservoir states are also sent to an output
layer, called the read-out. Input connections and recurrent connections are often
ﬁxed and random.
Usually, only the output layer connections are trained in a
supervised way with a variant of linear regression. We will later see in more details
how it works exactly. First, let’s jump to the context in which it appeared.
3.1
Context
Reservoir Computing emerged several times.
It is often stated that RC
has emerged twice in 1995 [Buonomano & Merzenich 1995, Dominey 1995] from
the computational neuroscience side, although it can be argued that similar forms
have appeared previously several times (see the references collected by Herbert
Jaeger on Scholarpedia1 [Jaeger 2007]).
Thus, it appeared only some years af-
ter the famous Simple Recurrent Network (SRN) from Elmann in 1990, which
was itself featured a few years after the invention of Back-Propagation Through
Time (BPTT) [Werbos 1988, Werbos 1990]. Thus, Reservoir Computing can be seen
as a possible “end of the road” of simpliﬁcation of Recurrent Neural Network (RNN)
training: ﬁrst RNNs were fully trained with BPTT, then only one step back in time
of BPTT is performed with SRNs, and ﬁnally inputs and recurrent weights are not
learnt anymore with the RC paradigm.
RC has again emerged in early 2000’s with the Echo State Network (ESN) of
Jaeger [Jaeger 2001] and with the Liquid State Machines (LSM) of Wolfgang Maass
and colleagues [Maass et al. 2002]. A RC community started to take shape: machine
learning community was more focused on ESNs and computational neuroscience
more on LSMs 2. This movement was probably enhanced because of the nice perfor-
mances obtained by Jaeger on chaotic time series prediction [Jaeger & Haas 2004].
Some authors did go further in trying to “simplify” the reservoir by removing
as much randomness as possible [Rodan & Tino 2010]. In my opinion, randomness
seems one of the simplest and most eﬃcient way one can get from a biological point
of view, at least to obtain generic computational properties (see Biology paragraph).
Creating random neuronal networks seems simple: it does not require to have speciﬁc
gene expression or other regulatory process for controlling precisely the connections.
On the other side of the spectrum, Long Short-Term Memory network (LSTM)
coined in 1997 [Hochreiter & Schmidhuber 1997] were another answer to the “Hard
Problem” that we will describe now.
Hard problem.
Indeed, training connections of a RNN with classical back-
propagation through time is known to be a hard problem [Bengio et al. 1994,
Pascanu et al. 2013]. Because, the error gradient tends to vanish or explode when
going further back in time in order to capture longer time dependencies. Intuitively,
1http://www.scholarpedia.org/article/Echo_state_network
2Even if ESNs or equivalent (rate-coded RNNs) were also used in computational neuroscience,
e.g.
3.1.
Context
19
changing one connection of one neuron can have an impact on all neurons a few
timesteps later. That’s why Back-Propagation have to be applied “Through Time”
(BPTT), in order to send the error gradient “back in time”, like a time machine that
will change the past in order to change the “present” error. This is done by taking
care of the unrolling of events in between3. It is hard, because this time machine
can “loose track” of the changes needed while going back in time: the error gradient
either decreases so much that no connections are modiﬁed anymore, or increases so
much that the modiﬁcations become exponentially huge. In both cases this means
that no learning can occur anymore far enough in time.
The LSTM network [Hochreiter & Schmidhuber 1997] was created in order to
solve this problem of vanishing or exploding gradient. LSTMs have internal recur-
rent units that were engineered to enable the BPTT algorithm to be more eﬀective
by enabling the error gradient to be kept constant. Inside each unit, they have
three (for the 1997 original version [Hochreiter & Schmidhuber 1997]) or four (with
an additional forget gate [Gers et al. 2000]) parameters. Input gate, forget gate,
output gate and the “cell” state.
This cell state is the special one that enables
to keep the gradient constant if needed: this is the solution provided by LSTMs
in order to prevent the gradient from vanishing or exploding. Even though it is
an elegant solution, it makes the LSTMs more demanding to train in computa-
tional resources because it has more parameters. That’s why LSTMs became very
popular only in the 2010’s with the revolution of deep learning due to new mathe-
matical and implementation tricks [Martens et al. 2010, Martens & Sutskever 2011,
Sutskever et al. 2011, Pascanu et al. 2013] along with the popularization of Graph-
ical Processing Units (GPUs) enabling to train these networks faster.
Biology. As we said earlier, Reservoir Computing (RC) emerged at start from the
computational neuroscience side [Buonomano & Merzenich 1995, Dominey 1995,
Dominey et al. 1995, Maass et al. 2002], before emerging also in the machine learn-
ing side [Jaeger 2001, Jaeger 2002, Jaeger & Haas 2004].
Indeed, a reservoir can
be seen as “a canonical computation unit” [Haeusler & Maass 2007]; it could model
“a cortical column”: what computational neuroscientists often consider as a generic
unit of computation. Since 1995 [Dominey 1995], my PhD supervisor Peter Dominey
have used it to model the cortico-basal network: the reservoir playing the role of
the (prefrontal) cortex and the output layer playing the role of the striatum (input
of the basal ganglia from the cortex). Dominey [Dominey et al. 1995] showed that
even with random networks (that were not called reservoirs yet) it was possible to
observe similar neuronal activation patterns then in studies on sequence processing
in monkey prefrontal cortex [Barone & Joseph 1989]. RC developed much faster
in the machine learning community since the 2000’s, but in the 2010’s it became
more popular from the experimental neuroscientists side. Neuroscientists started
using this idea of high-dimensional non-linear representations that can be decoded
by a linear classiﬁer.
It was a new way to interpret electrophysiological record-
ings from monkeys [Machens et al. 2010, Rigotti et al. 2013, Enel et al. 2016]: the
3Which is not usually the case for time machines.
20
Chapter 3.
Diving into Reservoirs
idea was no longer to ﬁnd particular sequential pattern in neural activity (like in
[Barone & Joseph 1989]), but rather to just decode linearly if some information were
present.
3.2
Intuitions in (almost) one page
Short deﬁnition: Reservoir Computing is a paradigm to train Recurrent Neural
Networks (RNN) without training all connections.
Intuition. The names “reservoir” for the recurrent layer, and “read-out” for the
output layer, come from the fact that a lot of input combinations are made inside
the recurrent layer (thanks to random projections).
The “reservoir” is literally
a reservoir of calculations (= “reservoir computing”) that are non-linear.
From
this “reservoir” one linearly decodes (= ”reads-out”) the combinations that will be
useful for the task to be solved. Reservoirs can be implemented on various kinds of
physical substrates [Tanaka et al. 2019] (e.g. electronic, photonic, mechanical RC).
Figure 3.2: Projection of inputs in a higher dimensional space.
The kernel trick. An intuitive way to understand how reservoir computing works
is to think it as a temporal Support Vector Machine (SVM) [Verstraeten 2009]. Like
in Figure 3.2, suppose you want to separate blue dots from red dots, but in your
initial 2D space you cannot separate them with a line. With a SVM [Vapnik 1999]
you project theses inputs (i.e.
the dots) into a higher dimensional space.
In
this high dimensional space you can ﬁnd a hyperplane (an equivalent of a line in
higher dimensions) that separates your blue dots from your red dots. Finding this
hyperplane is equivalent to perform a linear regression.
You can have diﬀerent
types of kernel with an SVM; in reservoirs this kernel is random.
3.3.
Some equations
21
Multi-task hub. Once a reservoir is trained for a task, it can still be used for
another task. Since the computations inside the reservoir are independent of the
read-out layer (if there is no feedback connections), new read-out units can be
connected to perform a new task. Thus, a reservoir can be seen as a “multi-hub
task”.
This property is interesting to understand how brain areas could share
computations:
some areas compute and represent information in a way that
could be used by several other areas.
This “hub” area do not have to compute
anything speciﬁc or represent information useful for one particular “task”: it just
have to make “some kind of non-linear computation”.
The “useful information”
is only computed when reading-out and projecting to another area.
As we dis-
cussed in the Introduction Chapter 1, Broca area (LIFG) seems to be involved in
representing hierarchical-like structures for language, sequence of actions, music, etc.
Less training data? If we come back to the idea that reservoir computing is like
having a temporal SVM, we can imagine that we do not necessarily need much
data points to be able to draw a hyperplane to separate our data. Indeed, a SVM
can only keep track of points that are close to the the decision boundary4 – the
support vectors.
In practice, we have shown that reservoirs needed less data to
generalize on an audio classiﬁcation task [Trouvain & Hinaut 2021] and a on a
language task [Variengien & Hinaut 2020, Oota et al. 2022].
Extended deﬁnition:
Reservoir Computing is a paradigm that can use any
physical substrate to obtain a suitable combination of inputs before using a
read-out layer to extract information from this representational layer (to predict,
classify, generate, ...).
We see now that this “reservoir of computation” do not have to be ﬁxed, it can
change and adapt over time, for example with homeostatic rules. More importantly
it does not need to be computer-based.
3.3
Some equations
There can be diﬀerent kinds of units in a reservoir: spiking or non-spiking (average
ﬁring rate) neurons. There are diﬀerent kinds of equations for both. I will not speak
about spiking version of reservoirs, because I am less familiar with their dynamics5.
One of the general ways to deﬁne ESN is as follows. The state transition of the
ESN is computed as follows:
x(t) = (1 −↵)x(t −1) + ↵tanh(Winu(t) + Wx(t −1))
(3.1)
where u(t) 2 RNU is the input vector at time t, x(t) 2 RNR is the reservoir state,
Win 2 RNR⇥NU is the input matrix, W 2 RNR⇥NR is the recurrent matrix, ↵2 [0, 1]
4This is particularly useful for online version of SVMs to save memory and computation time.
5However, I look forward to compare dynamics of spiking and rate-coding neurons as we plan
to include spiking neurons inside ReservoirPy (see Chapter 4 Subsection 4.4).
22
Chapter 3.
Diving into Reservoirs
Figure 3.3: An example of Echo State Network (ESN) architecture (with-
out feedback). Image from [Pedrelli & Hinaut 2020].
is the leaking rate – more often called the leak-rate – and tanh is the element-wise
hyperbolic tangent. NU is the number of input units and NR the number of units
in the reservoir. The leak-rate is equivalent to the inverse of a time constant, it is a
simpliﬁcation of writing:
↵= dt
⌧
(3.2)
with ⌧the time constant of neurons and dt the time step discretisation (which equals
1 by default)6.
The values of matrix W are randomly initialized, for instance using a uniform
distribution and then rescaled.
This rescaling of W is done in order to obtain
a spectral radius7 ⇢equal to the one set by the user as hyperparameter (HP)8.
The values in matrix Win are randomly initialized, for instance from a uniform
distribution and then rescaled in order to have an input scaling of σ, which is the
one set by the user as hyperparameter. Usually, W and Win matrices are sparse: my
recommendation is to use a percentage of non-zero connection of about 10 −20%,
but the inﬂuence of the sparseness on the performance is often weak.
A sparse
reservoir enables faster computations.
The output of the ESN is computed as follows:
y(t) = Wout[1; x(t)]
(3.3)
where y(t) 2 RNY is the output at time t, Wout is the output matrix, and [.; .] stands
for the concatenation of two vectors. NY is the number of output (read-out) units.
6We showed in [Hinaut & Dominey 2013] that changing dt does not aﬀect much the performance
on a language task as soon as the sampling rate of inputs are changed accordingly.
7The spectral radius is the maximum absolute eigenvalue of the matrix W.
8A hyperparameter is a parameter that need to be predeﬁned and which is not optimized by
the learning algorithm.
3.3.
Some equations
23
The output weights are learned using an equivalent of linear regression. The
most common practice is to use a regularized version, like the ridge regression:
Wout = Y XT (XXT + βI)−1
(3.4)
where X is the concatenation of the reservoir activities at all time steps with a bias
vector at 1, each row corresponding to a time step. Similarly, Y is the concate-
nation of desired outputs and β is the regularization parameter (often called ridge
parameter).
A few more details. The spectral radius ⇢controls the internal dynamics: more
stable dynamics will be obtained for low values and more chaotic ones with high
values. I will not talk about the Echo State Property (ESP) as it is a theoretical
recommendation from Jaeger [Jaeger 2001] (derived from principles of linear net-
works) but not a rule that should be followed blindly. In practice spectral radii
higher than one should be always tried when exploring hyperparameters because an
ESN is a non-linear system that depends on its inputs. Especially, in the case of
the leaky ESN where the eﬀective spectral radius is diﬀerent from the one deﬁned
by the user [Jaeger et al. 2007]9.
You can ﬁnd a tutorial to explore the hyperparameters of reservoirs in the
GitHub repository of our ReservoirPy library10. We illustrate plots to show how
the internal dynamics of the network change with respect to the changes of hyper-
parameters such as the spectral radius, the input scaling and the leak-rate.
In most our studies, we are use ESNs as deﬁned by Jaeger11 [Jaeger 2001,
Jaeger et al. 2007], where the state of each unit also corresponds to its output (i.e.
the activation function applies to the states directly). One may argue that it is
less biologically plausible, but it has the advantage of having bounded states which
prevents the states to take inﬁnite values – which would stop the program because
Not A Number (NAN) values are encountered. Of course bounded states are ob-
tained with a bounded activation function: e.g. hyperbolic tangent (tanh). This is
one of the reasons why we use Jaeger’s deﬁnition of ESNs. To my knowledge, they
seem to be the most used type of reservoir since two decades. Another reason is
that it enables to compare our models with many other published papers. For a
detailed explanation of the various version of ESNs, David Verstraeten provides a
clear explanation in his PhD thesis [Verstraeten 2009].
9I have unpublished results showing that one can have very high values of spectral radius (e.g. a
million) that still work for a given task as soon as one also decrease the leak-rate. Hyperparameters
such as the spectral radius, the leak-rate and the input scaling are linked, that is why we suggest
to ﬁx at least one of them when doing hyperparameter exploration [Hinaut & Trouvain 2021].
10https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/4-Understand_
and_optimize_hyperparameters.ipynb
11In particular we often use the “leaky” version of ESNs.
