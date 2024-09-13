Reservoir Computing Approaches to Recurrent Neural Network Training
Mantas Lukoˇseviˇcius∗, Herbert Jaeger
School of Engineering and Science, Jacobs University Bremen gGmbH, P.O. Box 750 561, 28725 Bremen, Germany
Abstract
Echo State Networks and Liquid State Machines introduced a new paradigm in artiﬁcial recurrent neural
network (RNN) training, where an RNN (the reservoir) is generated randomly and only a readout is trained.
The paradigm, becoming known as reservoir computing, greatly facilitated the practical application of RNNs
and outperformed classical fully trained RNNs in many tasks. It has lately become a vivid research ﬁeld with
numerous extensions of the basic idea, including reservoir adaptation, thus broadening the initial paradigm
to using diﬀerent methods for training the reservoir and the readout. This review systematically surveys
both current ways of generating/adapting the reservoirs and training diﬀerent types of readouts. It oﬀers
a natural conceptual classiﬁcation of the techniques, which transcends boundaries of the current “brand-
names” of reservoir methods, and thus aims to help in unifying the ﬁeld and providing the reader with a
detailed “map” of it.
Key words:
Computational Intelligence, Machine Learning, Connectionist, Recurrent Neural Network,
Echo State Network, Liquid State Machine
1. Introduction
Artiﬁcial recurrent neural networks (RNNs) represent a large and varied class of computational models
that are designed by more or less detailed analogy with biological brain modules. In an RNN numerous
abstract neurons (also called units or processing elements) are interconnected by likewise abstracted synaptic
connections (or links), which enable activations to propagate through the network.
The characteristic
feature of RNNs that distinguishes them from the more widely used feedforward neural networks is that the
connection topology possesses cycles. The existence of cycles has a profound impact:
• An RNN may develop a self-sustained temporal activation dynamics along its recurrent connection
pathways, even in the absence of input. Mathematically, this renders an RNN to be a dynamical
system, while feedforward networks are functions.
• If driven by an input signal, an RNN preserves in its internal state a nonlinear transformation of the
input history — in other words, it has a dynamical memory, and is able to process temporal context
information.
This review article concerns a particular subset of RNN-based research in two aspects:
• RNNs are used for a variety of scientiﬁc purposes, and at least two major classes of RNN models
exist: they can be used for purposes of modeling biological brains, or as engineering tools for technical
applications. The ﬁrst usage belongs to the ﬁeld of computational neuroscience, while the second
∗Corresponding author.
Email addresses: m.lukosevicius@jacobs-university.de (Mantas Lukoˇseviˇcius), h.jaeger@jacobs-university.de
(Herbert Jaeger)
Preprint submitted to Computer Science Review
January 18, 2010
frames RNNs in the realms of machine learning, the theory of computation, and nonlinear signal
processing and control. While there are interesting connections between the two attitudes, this survey
focuses on the latter, with occasional borrowings from the ﬁrst.
• From a dynamical systems perspective, there are two main classes of RNNs. Models from the ﬁrst
class are characterized by an energy-minimizing stochastic dynamics and symmetric connections. The
best known instantiations are Hopﬁeld networks [1, 2], Boltzmann machines [3, 4], and the recently
emerging Deep Belief Networks [5]. These networks are mostly trained in some unsupervised learning
scheme. Typical targeted network functionalities in this ﬁeld are associative memories, data com-
pression, the unsupervised modeling of data distributions, and static pattern classiﬁcation, where the
model is run for multiple time steps per single input instance to reach some type of convergence or
equilibrium (but see e.g., [6] for extension to temporal data). The mathematical background is rooted
in statistical physics. In contrast, the second big class of RNN models typically features a determin-
istic update dynamics and directed connections. Systems from this class implement nonlinear ﬁlters,
which transform an input time series into an output time series. The mathematical background here
is nonlinear dynamical systems. The standard training mode is supervised. This survey is concerned
only with RNNs of this second type, and when we speak of RNNs later on, we will exclusively refer
to such systems.1
RNNs (of the second type) appear as highly promising and fascinating tools for nonlinear time series
processing applications, mainly for two reasons. First, it can be shown that under fairly mild and general
assumptions, such RNNs are universal approximators of dynamical systems [7]. Second, biological brain
modules almost universally exhibit recurrent connection pathways too.
Both observations indicate that
RNNs should potentially be powerful tools for engineering applications.
Despite this widely acknowledged potential, and despite a number of successful academic and practical
applications, the impact of RNNs in nonlinear modeling has remained limited for a long time. The main
reason for this lies in the fact that RNNs are diﬃcult to train by gradient-descent-based methods, which
aim at iteratively reducing the training error. While a number of training algorithms have been proposed
(a brief overview in Section 2.5), these all suﬀer from the following shortcomings:
• The gradual change of network parameters during learning drives the network dynamics through
bifurcations [8]. At such points, the gradient information degenerates and may become ill-deﬁned. As
a consequence, convergence cannot be guaranteed.
• A single parameter update can be computationally expensive, and many update cycles may be neces-
sary. This results in long training times, and renders RNN training feasible only for relatively small
networks (in the order of tens of units).
• It is intrinsically hard to learn dependences requiring long-range memory, because the necessary gradi-
ent information exponentially dissolves over time [9] (but see the Long Short-Term Memory networks
[10] for a possible escape).
• Advanced training algorithms are mathematically involved and need to be parameterized by a number
of global control parameters, which are not easily optimized.
As a result, such algorithms need
substantial skill and experience to be successfully applied.
In this situation of slow and diﬃcult progress, in 2001 a fundamentally new approach to RNN design and
training was proposed independently by Wolfgang Maass under the name of Liquid State Machines [11] and
by Herbert Jaeger under the name of Echo State Networks [12]. This approach, which had predecessors in
computational neuroscience [13] and subsequent ramiﬁcations in machine learning as the Backpropagation-
Decorrelation [14] learning rule, is now increasingly often collectively referred to as Reservoir Computing
(RC). The RC paradigm avoids the shortcomings of gradient-descent RNN training listed above, by setting
up RNNs in the following way:
1However, they can also be used in a converging mode, as shown at the end of Section 8.6.
2
• A recurrent neural network is randomly created and remains unchanged during training. This RNN is
called the reservoir. It is passively excited by the input signal and maintains in its state a nonlinear
transformation of the input history.
• The desired output signal is generated as a linear combination of the neuron’s signals from the input-
excited reservoir. This linear combination is obtained by linear regression, using the teacher signal as
a target.
Figure 1 graphically contrasts previous methods of RNN training with the RC approach.
...
...
...
target
error
output
input
A.
...
...
...
B.
Figure 1:
A. Traditional gradient-descent-based RNN training methods adapt all connection weights (bold arrows), including
input-to-RNN, RNN-internal, and RNN-to-output weights. B. In Reservoir Computing, only the RNN-to-output weights are
adapted.
Reservoir Computing methods have quickly become popular, as witnessed for instance by a theme issue
of Neural Networks [15], and today constitute one of the basic paradigms of RNN modeling [16]. The main
reasons for this development are the following:
Modeling accuracy. RC has starkly outperformed previous methods of nonlinear system identiﬁcation,
prediction and classiﬁcation, for instance in predicting chaotic dynamics (three orders of magnitude
improved accuracy [17]), nonlinear wireless channel equalization (two orders of magnitude improve-
ment [17]), the Japanese Vowel benchmark (zero test error rate, previous best: 1.8% [18]), ﬁnancial
forecasting (winner of the international forecasting competition NN32 ), and in isolated spoken digits
recognition (improvement of word error rate on benchmark from 0.6% of previous best system to 0.2%
[19], and further to 0% test error in recent unpublished work).
Modeling capacity. RC is computationally universal for continuous-time, continuous-value real-time sys-
tems modeled with bounded resources (including time and value resolution) [20, 21].
Biological plausibility. Numerous connections of RC principles to architectural and dynamical properties
of mammalian brains have been established. RC (or closely related models) provides explanations of
why biological brains can carry out accurate computations with an “inaccurate” and noisy physical
substrate [22, 23], especially accurate timing [24]; of the way in which visual information is super-
imposed and processed in primary visual cortex [25, 26]; of how cortico-basal pathways support the
representation of sequential information; and RC oﬀers a functional interpretation of the cerebellar
circuitry [27, 28].
A central role is assigned to an RC circuit in a series of models explaining se-
quential information processing in human and primate brains, most importantly of speech signals
[13, 29, 30, 31].
2http://www.neural-forecasting-competition.com/NN3/index.htm
3
Extensibility and parsimony. A notorious conundrum of neural network research is how to extend previ-
ously learned models by new items without impairing or destroying previously learned representations
(catastrophic interference [32]). RC oﬀers a simple and principled solution: new items are represented
by new output units, which are appended to the previously established output units of a given reser-
voir. Since the output weights of diﬀerent output units are independent of each other, catastrophic
interference is a non-issue.
These encouraging observations should not mask the fact that RC is still in its infancy, and signiﬁcant
further improvements and extensions are desirable. Speciﬁcally, just simply creating a reservoir at random
is unsatisfactory. It seems obvious that when addressing a speciﬁc modeling task, a speciﬁc reservoir design
that is adapted to the task will lead to better results than a naive random creation. Thus, the main stream
of research in the ﬁeld is today directed at understanding the eﬀects of reservoir characteristics on task
performance, and at developing suitable reservoir design and adaptation methods. Also, new ways of reading
out from the reservoirs, including combining them into larger structures, are devised and investigated. While
shifting from the initial idea of having a ﬁxed randomly created reservoir and training only the readout,
the current paradigm of reservoir computing remains (and diﬀerentiates itself from other RNN training
approaches) as producing/training the reservoir and the readout separately and diﬀerently.
This review oﬀers a conceptual classiﬁcation and a comprehensive survey of this research.
As is true for many areas of machine learning, methods in reservoir computing converge from diﬀerent
ﬁelds and come with diﬀerent names. We would like to make a distinction here between these diﬀerently
named “tradition lines”, which we like to call brands, and the actual ﬁner-grained ideas on producing good
reservoirs, which we will call recipes. Since recipes can be useful and mixed across diﬀerent brands, this
review focuses on classifying and surveying them. To be fair, it has to be said that the authors of this survey
associate themselves mostly with the Echo State Networks brand, and thus, willingly or not, are inﬂuenced
by its mindset.
Overview. We start by introducing a generic notational framework in Section 2. More speciﬁcally,
we deﬁne what we mean by problem or task in the context of machine learning in Section 2.1.
Then
we deﬁne a general notation for expansion (or kernel) methods for both non-temporal (Section 2.2) and
temporal (Section 2.3) tasks, introduce our notation for recurrent neural networks in Section 2.4, and outline
classical training methods in Section 2.5. In Section 3 we detail the foundations of Reservoir Computing
and proceed by naming the most prominent brands. In Section 4 we introduce our classiﬁcation of the
reservoir generation/adaptation recipes, which transcends the boundaries between the brands. Following
this classiﬁcation we then review universal (Section 5), unsupervised (Section 6), and supervised (Section 7)
reservoir generation/adaptation recipes. In Section 8 we provide a classiﬁcation and review the techniques
for reading the outputs from the reservoirs reported in literature, together with discussing various practical
issues of readout training. A ﬁnal discussion (Section 9) wraps up the entire picture.
2. Formalism
2.1. Formulation of the problem
Let a problem or a task in our context of machine learning be deﬁned as a problem of learning a functional
relation between a given input u(n) ∈RNu and a desired output ytarget(n) ∈RNy, where n = 1, . . . , T, and
T is the number of data points in the training dataset {(u(n), ytarget(n))}. A non-temporal task is where
the data points are independent of each other and the goal is to learn a function y(n) = y(u(n)) such that
E(y, ytarget) is minimized, where E is an error measure, for instance, the normalized root-mean-square error
(NRMSE)
E(y, ytarget) =
v
u
u
u
t
D
∥y(n) −ytarget(n)∥2E
D
∥ytarget(n) −⟨ytarget(n)⟩∥2E,
(1)
where ∥·∥stands for the Euclidean distance (or norm).
4
A temporal task is where u and ytarget are signals in a discrete time domain n = 1, . . . , T, and the goal
is to learn a function y(n) = y(. . . , u(n −1), u(n)) such that E(y, ytarget) is minimized. Thus the diﬀerence
between the temporal and non-temporal task is that the function y(·) we are trying to learn has memory in
the ﬁrst case and is memoryless in the second. In both cases the underlying assumption is, of course, that
the functional dependence we are trying to learn actually exists in the data. For the temporal case this spells
out as data adhering to an additive noise model of the form ytarget(n) = ytarget(. . . , u(n −1), u(n)) + θ(n),
where ytarget(·) is the relation to be learned by y(·) and θ(n) ∈RNy is a noise term, limiting the learning
precision, i.e., the precision of matching the learned y(n) to ytarget(n).
Whenever we say that the task or the problem is learned well, or with good accuracy or precision, we
mean that E(y, ytarget) is small. Normally one part of the T data points is used for training the model and
another part (unseen during the training) for testing it. When speaking about output errors and performance
or precision we will have testing errors in mind (if not explicitly speciﬁed otherwise). Also n, denoting the
discrete time, will often be used omitting its range 1, . . . , T.
2.2. Expansions and kernels in non-temporal tasks
Many tasks cannot be accurately solved by a simple linear relation between the u and ytarget, i.e., a
linear model y(n) = Wu(n) (where W ∈RNy×Nu) gives big errors E(y, ytarget) regardless of W. In such
situations one has to resort to nonlinear models.
A number of generic and widely used approaches to
nonlinear modeling are based on the idea of nonlinearly expanding the input u(n) into a high-dimensional
feature vector x(n) ∈RNx, and then utilizing those features using linear methods, for instance by linear
regression or computing for a linear separation hyperplane, to get a reasonable y. Solutions of this kind can
be expressed in the form
y(n) = Woutx(n) = Woutx(u(n)),
(2)
where Wout ∈RNy×Nx are the trained output weights. Typically Nx ≫Nu, and we will often consider u(n)
as included in x(n). There is also typically a constant bias value added to (2), which is omitted here and in
other equations for brevity. The bias can be easily implemented, having one of the features in x(n) constant
(e.g., = 1) and a corresponding column in Wout. Some models extend (2) to
y(n) = fout(Woutx[u(n)]),
(3)
where fout(·) is some nonlinear function (e.g., a sigmoid applied element-wise). For the sake of simplicity
we will consider this deﬁnition as equivalent to (2), since fout(·) can be eliminated from y by redeﬁning the
target as y′
target = fout
−1(ytarget) (and the error function E(y, y′
target), if desired). Note that (2) is a special
case of (3), with fout(·) being the identity.
Functions x(u(n)) that transform an input u(n) into a (higher-dimensional) vector x(n) are often called
kernels (and traditionally denoted φ(u(n))) in this context. Methods using kernels often employ the ker-
nel trick, which refers to the option aﬀorded by many kernels of computing inner products in the (high-
dimensional, hence expensive) feature space of x more cheaply in the original space populated by u. The
term kernel function has acquired a close association with the kernel trick. Since here we will not exploit
the kernel trick, in order to avoid confusion we will use the more neutral term of an expansion function
for x(u(n)), and refer to methods using such functions as expansion methods. These methods then include
Support Vector Machines (which standardly do use the kernel trick), Feedforward Neural Networks, Radial
Basis Function approximators, Slow Feature Analysis, and various Probability Mixture models, among many
others. Feedforward neural networks are also often referred to as (multilayer) perceptrons in the literature.
While training the output Wout is a well deﬁned and understood problem, producing a good expansion
function x(·) generally involves more creativity. In many expansion methods, e.g., Support Vector Machines,
the function is chosen “by hand” (most often through trial-and-error) and is ﬁxed.
2.3. Expansions in temporal tasks
Many temporal methods are based on the same principle. The diﬀerence is that in a temporal task
the function to be learned depends also on the history of the input, as discussed in Section 2.1. Thus, the
5
expansion function has memory: x(n) = x(. . . , u(n −1), u(n)), i.e., it is an expansion of the current input
and its (potentially inﬁnite) history. Since this function has an unbounded number of parameters, practical
implementations often take an alternative, recursive, deﬁnition:
x(n) = x(x(n −1), u(n)).
(4)
The output y(n) is typically produced in the same way as for non-temporal methods by (2) or (3).
In addition to the nonlinear expansion, as in the non-temporal tasks, such x(n) could be seen as a type of
a spatial embedding of the temporal information of . . . , u(n−1), u(n). This, for example, enables capturing
higher-dimensional dynamical attractors y(n) = ytarget(. . . , u(n −1), u(n)) = u(n + 1) of the system being
modeled by y(·) from a series of lower-dimensional observations u(n) the system is emitting, which is shown
to be possible by Takens’s theorem [33].
2.4. Recurrent neural networks
The type of recurrent neural networks that we will consider most of the time in this review is a straight-
forward implementation of (4). The nonlinear expansion with memory here leads to a state vector of the
form
x(n) = f(Winu(n) + Wx(n −1)),
n = 1, . . . , T,
(5)
where x(n) ∈RNx is a vector of reservoir neuron activations at a time step n, f(·) is the neuron activation
function, usually the symmetric tanh(·), or the positive logistic (or Fermi) sigmoid, applied element-wise,
Win ∈RNx×Nu is the input weight matrix and W ∈RNx×Nx is a weight matrix of internal network
connections. The network is usually started with the initial state x(0) = 0. Bias values are again omitted
in (5) in the same way as in (2). The readout y(n) of the network is implemented as in (3).
Some models of RNNs extend (5) as
x(n) = f(Winu(n) + Wx(n −1) + Wofby(n −1)),
n = 1, . . . , T,
(6)
where Wofb ∈RNx×Ny is an optional output feedback weight matrix.
2.5. Classical training of RNNs
The classical approach to supervised training of RNNs, known as gradient descent, is by iteratively
adapting all weights Wout, W, Win, and possibly Wofb (which as a whole we denote Wall for brevity)
according to their estimated gradients ∂E/∂Wall, in order to minimize the output error E = E(y, ytarget).
A classical example of such methods is Real-Time Recurrent Learning [34], where the estimation of ∂E/∂Wall
is done recurrently, forward in time. Conversely, error backpropagation (BP) methods for training RNNs,
which are derived as extensions of the BP method for feedforward neural networks [35], estimate ∂E/∂Wall
by propagating E(y, ytarget) backwards through network connections and time. The BP group of methods
is arguably the most prominent in classical RNN training, with the classical example in this group being
Backpropagation Through Time [36]. It has a runtime complexity of O(Nx
2) per weight update per time
step for a single output Ny = 1, compared to O(Nx
4) for Real-Time Recurrent Learning.
A systematic unifying overview of many classical gradient descent RNN training methods is presented
in [37]. The same contribution also proposes a new approach, often referred to by others as Atiya-Parlos
Recurrent Learning (APRL). It estimates gradients with respect to neuron activations ∂E/∂x (instead of
weights directly) and gradually adapts the weights Wall to move the activations x into the desired directions.
The method is shown to converge faster than previous ones. See Section 3.4 for more implications of APRL
and bridging the gap between the classical gradient descent and the reservoir computing methods.
There are also other versions of supervised RNN training, formulating the training problem diﬀerently,
such as using Extended Kalman Filters [38] or the Expectation-Maximization algorithm [39], as well as
dealing with special types of RNNs, such as Long Short-Term Memory [40] modular networks capable of
learning long-term dependences.
6
There are many more, arguably less prominent, methods and their modiﬁcations for RNN training that
are not mentioned here, as this would lead us beyond the scope of this review.
The very fact of their
multiplicity suggests that there is no clear winner in all aspects. Despite many advances that the methods
cited above have introduced, they still have multiple common shortcomings as pointed out in Section 1.
3. Reservoir methods
Reservoir computing methods diﬀer from the “traditional” designs and learning techniques listed above
in that they make a conceptual and computational separation between a dynamic reservoir — an RNN as
a nonlinear temporal expansion function — and a recurrence-free (usually linear) readout that produces the
desired output from the expansion.
This separation is based on the understanding (common with kernel methods) that x(·) and y(·) serve
diﬀerent purposes: x(·) expands the input history u(n), u(n −1), . . . into a rich enough reservoir state space
x(n) ∈RNx, while y(·) combines the neuron signals x(n) into the desired output signal ytarget(n). In the
linear readout case (2), for each dimension yi of y an output weight vector (Wout)i in the same space RNx
is found such that
(Wout)ix(n) = yi(n) ≈ytargeti(n),
(7)
while the “purpose” of x(n) is to contain a rich enough representation to make this possible.
Since the expansion and the readout serve diﬀerent purposes, training/generating them separately and
even with diﬀerent goal functions makes sense. The readout y(n) = y(x(n)) is essentially a non-temporal
function, learning which is relatively simple. On the other hand, setting up the reservoir such that a “good”
state expansion x(n) emerges is an ill-understood challenge in many respects.
The “traditional” RNN
training methods do not make the conceptual separation of a reservoir vs. a readout, and train both reservoir-
internal and output weights in technically the same fashion. Nonetheless, even in traditional methods the
ways of deﬁning the error gradients for the output y(n) and the internal units x(n) are inevitably diﬀerent,
reﬂecting that an explicit target ytarget(n) is available only for the output units. Analyses of traditional
training algorithms have furthermore revealed that the learning dynamics of internal vs. output weights
exhibit systematic and striking diﬀerences. This theme will be expanded in Section 3.4.
Currently, reservoir computing is a vivid fresh RNN research stream, which has recently gained wide
attention due to the reasons pointed out in Section 1.
We proceed to review the most prominent “named” reservoir methods, which we call here brands. Each
of them has its own history, a speciﬁc mindset, speciﬁc types of reservoirs, and speciﬁc insights.
3.1. Echo State Networks
Echo State Networks (ESNs) [16] represent one of the two pioneering reservoir computing methods. The
approach is based on the observation that if a random RNN possesses certain algebraic properties, training
only a linear readout from it is often suﬃcient to achieve excellent performance in practical applications.
The untrained RNN part of an ESN is called a dynamical reservoir, and the resulting states x(n) are termed
echoes of its input history [12] — this is where reservoir computing draws its name from.
ESNs standardly use simple sigmoid neurons, i.e., reservoir states are computed by (5) or (6), where
the nonlinear function f(·) is a sigmoid, usually the tanh(·) function.
Leaky integrator neuron models
represent another frequent option for ESNs, which is discussed in depth in Section 5.5. Classical recipes of
producing the ESN reservoir (which is in essence Win and W) are outlined in Section 5.1, together with
input-independent properties of the reservoir. Input-dependent measures of the quality of the activations
x(n) in the reservoir are presented in Section 6.1.
The readout from the reservoir is usually linear (3), where u(n) is included as part of x(n), which can
also be spelled out in (3) explicitly as
y(n) = fout(Wout[u(n)|x(n)]),
(8)
where Wout ∈RNy×(Nu+Nx) is the learned output weight matrix, fout(·) is the output neuron activation
function (usually the identity) applied component-wise, and ·|· stands for a vertical concatenation of vectors.
7
The original and most popular batch training method to compute Wout is linear regression, discussed in
Section 8.1.1, or a computationally cheap online training discussed in Section 8.1.2.
The initial ESN publications [12, 41, 42, 43, 17] were framed in settings of machine learning and nonlinear
signal processing applications. The original theoretical contributions of early ESN research concerned alge-
braic properties of the reservoir that make this approach work in the ﬁrst place (the echo state property [12]
discussed in Section 5.1) and analytical results characterizing the dynamical short-term memory capacity of
reservoirs [41] discussed in Section 6.1.
3.2. Liquid State Machines
Liquid State Machines (LSMs) [11] are the other pioneering reservoir method, developed independently
from and simultaneously with ESNs. LSMs were developed from a computational neuroscience background,
aiming at elucidating the principal computational properties of neural microcircuits [11, 20, 44, 45]. Thus
LSMs use more sophisticated and biologically realistic models of spiking integrate-and-ﬁre neurons and
dynamic synaptic connection models in the reservoir. The connectivity among the neurons often follows
topological and metric constraints that are biologically motivated. In the LSM literature, the reservoir is
often referred to as the liquid, following an intuitive metaphor of the excited states as ripples on the surface
of a pool of water. Inputs to LSMs also usually consist of spike trains. In their readouts LSMs originally used
multilayer feedforward neural networks (of either spiking or sigmoid neurons), or linear readouts similar to
ESNs [11]. Additional mechanisms for averaging spike trains to get real-valued outputs are often employed.
RNNs of the LSM-type with spiking neurons and more sophisticated synaptic models are usually more
diﬃcult to implement, to correctly set up and tune, and typically more expensive to emulate on digital
computers3 than simple ESN-type “weighted sum and nonlinearity” RNNs. Thus they are less widespread
for engineering applications of RNNs than the latter. However, while the ESN-type neurons only emulate
mean ﬁring rates of biological neurons, spiking neurons are able to perform more complicated information
processing, due to the time coding of the information in their signals (i.e., the exact timing of each ﬁring
also matters). Also ﬁndings on various mechanisms in natural neural circuits are more easily transferable
to these more biologically-realistic models (there is more on this in Section 6.2).
The main theoretical contributions of the LSM brand to Reservoir Computing consist in analytical
characterizations of the computational power of such systems [11, 21] discussed in Sections 6.1 and 7.4.
3.3. Evolino
Evolino [46] transfers the idea of ESNs from an RNN of simple sigmoidal units to a Long Short-Term
Memory type of RNNs [40] constructed from units capable of preserving memory for long periods of time.
In Evolino the weights of the reservoir are trained using evolutionary methods, as is also done in some
extensions of ESNs, both discussed in Section 7.2.
3.4. Backpropagation-Decorrelation
The idea of separation between a reservoir and a readout function has also been arrived at from the point
of view of optimizing the performance of the RNN training algorithms that use error backpropagation, as
already indicated in Section 2.5. In an analysis of the weight dynamics of an RNN trained using the APRL
learning algorithm [47], it was revealed that the output weights Win of the network being trained change
quickly, while the hidden weights W change slowly and in the case of a single output Ny = 1 the changes
are column-wise coupled. Thus in eﬀect APRL decouples the RNN into a quickly adapting output and a
slowly adapting reservoir. Inspired by these ﬁndings a new iterative/online RNN training method, called
BackPropagation-DeCorrelation (BPDC), was introduced [14]. It approximates and signiﬁcantly simpliﬁes
the APRL method, and applies it only to the output weights Wout, turning it into an online RC method.
BPDC uses the reservoir update equation deﬁned in (6), where output feedbacks Wofb are essential, with
the same type of units as ESNs. BPDC learning is claimed to be insensitive to the parameters of ﬁxed
3With a possible exception of event-driven spiking NN simulations, where the computational load varies depending on the
amount of activity in the NN.
8
reservoir weights W. BPDC boasts fast learning times and thus is capable of tracking quickly changing
signals. As a downside of this feature, the trained network quickly forgets the previously seen data and
is highly biased by the recent data. Some remedies for reducing this eﬀect are reported in [48]. Most of
applications of BPDC in the literature are for tasks having one-dimensional outputs Ny = 1; however BPDC
is also successfully applied to Ny > 1, as recently demonstrated in [49].
From a conceptual perspective we can deﬁne a range of RNN training methods that gradually bridge the
gap between the classical BP and reservoir methods:
1. Classical BP methods, such as Backpropagation Through Time (BPTT) [36];
2. Atiya-Parlos recurrent learning (APRL) [37];
3. BackPropagation-DeCorrelation (BPDC) [14];
4. Echo State Networks (ESNs) [16].
In each method of this list the focus of training gradually moves from the entire network towards the output,
and convergence of the training is faster in terms of iterations, with only a single “iteration” in case 4. At
the same time the potential expressiveness of the RNN, as per the same number of units in the NN, becomes
weaker. All methods in the list primarily use the same type of simple sigmoid neuron model.
3.5. Temporal Recurrent Networks
This summary of RC brands would be incomplete without a spotlight directed at Peter F. Dominey’s
decade-long research suite on cortico-striatal circuits in the human brain (e.g., [13, 29, 31], and many
more). Although this research is rooted in empirical cognitive neuroscience and functional neuroanatomy
and aims at elucidating complex neural structures rather than theoretical computational principles, it is
probably Dominey who ﬁrst clearly spelled out the RC principle: “(. . . ) there is no learning in the recurrent
connections [within a subnetwork corresponding to a reservoir], only between the State [i.e., reservoir] units
and the Output units. Second, adaptation is based on a simple associative learning mechanism (. . . )” [50]. It
is also in this article where Dominey brands the neural reservoir module as a Temporal Recurrent Network.
The learning algorithm, to which Dominey alludes, can be seen as a version of the Least Mean Squares
discussed in Section 8.1.2. At other places, Dominey emphasizes the randomness of the connectivity in
the reservoir: “It is worth noting that the simulated recurrent prefrontal network relies on ﬁxed randomized
recurrent connections, (. . . )” [51]. Only in early 2008 did Dominey and “computational” RC researchers
become aware of each other.
3.6. Other (exotic) types of reservoirs
As is clear from the discussion of the diﬀerent reservoir methods so far, a variety of neuron models can
be used for the reservoirs. Using diﬀerent activation functions inside a single reservoir might also improve
the richness of the echo states, as is illustrated, for example, by inserting some neurons with wavelet-shaped
activation functions into the reservoir of ESNs [52]. A hardware implementation friendly version of reservoirs
composed of stochastic bitstream neurons was proposed in [53].
In fact the reservoirs do not necessarily need to be neural networks, governed by dynamics similar
to (5).
Other types of high-dimensional dynamical systems that can take an input u(n) and have an
observable state x(n) (which does not necessarily fully describe the state of the system) can be used as
well.
In particular this makes the reservoir paradigm suitable for harnessing the computational power
of unconventional hardware, such as analog electronics [54, 55], biological neural tissue [26], optical [56],
quantum, or physical “computers”. The last of these was demonstrated (taking the “reservoir” and “liquid”
idea quite literally) by feeding the input via mechanical actuators into a reservoir full of water, recording the
state of its surface optically, and successfully training a readout multilayer perceptron on several classiﬁcation
tasks [57]. An idea of treating a computer-simulated gene regulation network of Escherichia Coli bacteria
as the reservoir, a sequence of chemical stimuli as an input, and measures of protein levels and mRNAs as
an output is explored in [58].
9
3.7. Other overviews of reservoir methods
An experimental comparison of LSM, ESN, and BPDC reservoir methods with diﬀerent neuron models,
even beyond the standard ones used for the respective methods, and diﬀerent parameter settings is presented
in [59].
A brief and broad overview of reservoir computing is presented in [60], with an emphasis on
applications and hardware implementations of reservoir methods. The editorial in the “Neural Networks”
journal special issue on ESNs and LSMs [15] oﬀers a short introduction to the topic and an overview of the
articles in the issue (most of which are also surveyed here). An older and much shorter part of this overview,
covering only reservoir adaptation techniques, is available as a technical report [61].
4. Our classiﬁcation of reservoir recipes
The successes of applying RC methods to benchmarks (see the listing in Section 1) outperforming classical
fully trained RNNs do not imply that randomly generated reservoirs are optimal and cannot be improved.
In fact, “random” is almost by deﬁnition an antonym to “optimal”. The results rather indicate the need for
some novel methods of training/generating the reservoirs that are very probably not a direct extension of
the way the output is trained (as in BP). Thus besides application studies (which are not surveyed here),
the bulk of current RC research on reservoir methods is devoted to optimal reservoir design, or reservoir
optimization algorithms.
It is worth mentioning at this point that the general “no free lunch” principle in supervised machine
learning [62] states that there can exist no bias of a model which would universally improve the accuracy of
the model for all possible problems. In our context this can be translated into a claim that no single type
of reservoir can be optimal for all types of problems.
In this review we will try to survey all currently investigated ideas that help producing “good” reservoirs.
We will classify those ideas into three major groups based on their universality:
• Generic guidelines/methods of producing good reservoirs irrespective of the task (both the input u(n)
and the desired output ytarget(n));
• Unsupervised pre-training of the reservoir with respect to the given input u(n), but not the target
ytarget(n);
• Supervised pre-training of the reservoir with respect to both the given input u(n) and the desired
output ytarget(n).
These three classes of methods are discussed in the following three sections. Note that many of the methods
to some extend transcend the boundaries of these three classes, but will be classiﬁed according to their main
principle.
5. Generic reservoir recipes
The most classical methods of producing reservoirs all fall into this category.
All of them generate
reservoirs randomly, with topology and weight characteristics depending on some preset parameters. Even
though they are not optimized for a particular input u(n) or target ytarget(n), a good manual selection of the
parameters is to some extent task-dependent, complying with the “no free lunch” principle just mentioned.
5.1. Classical ESN approach
Some of the most generic guidelines of producing good reservoirs were presented in the papers that
introduced ESNs [12, 42]. Motivated by an intuitive goal of producing a “rich” set of dynamics, the recipe
is to generate a (i) big, (ii) sparsely and (iii) randomly connected, reservoir. This means that (i) Nx is
suﬃciently large, with order ranging from tens to thousands, (ii) the weight matrix W is sparse, with several
to 20 per cent of possible connections, and (iii) the weights of the connections are usually generated randomly
from a uniform distribution symmetric around the zero value. This design rationale aims at obtaining many,
due to (i), reservoir activation signals, which are only loosely coupled, due to (ii), and diﬀerent, due to (iii).
10
The input weights Win and the optional output feedback weights Wofb are usually dense (they can also
be sparse like W) and generated randomly from a uniform distribution. The exact scaling of both matrices
and an optional shift of the input (a constant value added to u(n)) are the few other free parameters that
one has to choose when “baking” an ESN. The rules of thumb for them are the following. The scaling of
Win and shifting of the input depends on how much nonlinearity of the processing unit the task needs:
if the inputs are close to 0, the tanh neurons tend to operate with activations close to 0, where they are
essentially linear, while inputs far from 0 tend to drive them more towards saturation where they exhibit
more nonlinearity. The shift of the input may help to overcome undesired consequences of the symmetry
around 0 of the tanh neurons with respect to the sign of the signals. Similar eﬀects are produced by scaling
the bias inputs to the neurons (i.e., the column of Win corresponding to constant input, which often has a
diﬀerent scaling factor than the rest of Win). The scaling of Wofb is in practice limited by a threshold at
which the ESN starts to exhibit an unstable behavior, i.e., the output feedback loop starts to amplify (the
errors of) the output and thus enters a diverging generative mode. In [42], these and related pieces of advice
are given without a formal justiﬁcation.
An important element for ESNs to work is that the reservoir should have the echo state property [12].
This condition in essence states that the eﬀect of a previous state x(n) and a previous input u(n) on a
future state x(n + k) should vanish gradually as time passes (i.e., k →∞), and not persist or even get
ampliﬁed. For most practical purposes, the echo state property is assured if the reservoir weight matrix
W is scaled so that its spectral radius ρ(W) (i.e., the largest absolute eigenvalue) satisﬁes ρ(W) < 1 [12].
Or, using another term, W is contractive. The fact that ρ(W) < 1 almost always ensures the echo state
property has led to an unfortunate misconception which is expressed in many RC publications, namely,
that ρ(W) < 1 amounts to a necessary and suﬃcient condition for the echo state property. This is wrong.
The mathematically correct connection between the spectral radius and the echo state property is that the
latter is violated if ρ(W) > 1 in reservoirs using the tanh function as neuron nonlinearity, and for zero
input. Contrary to widespread misconceptions, the echo state property can be obtained even if ρ(W) > 1
for non-zero input (including bias inputs to neurons), and it may be lost even if ρ(W) < 1, although it is
hard to construct systems where this occurs (unless f ′(0) > 1 for the nonlinearity f), and in practice this
does not happen.
The optimal value of ρ(W) should be set depending on the amount of memory and nonlinearity that the
given task requires. A rule of thumb, likewise discussed in [12], is that ρ(W) should be close to 1 for tasks
that require long memory and accordingly smaller for the tasks where a too long memory might in fact be
harmful. Larger ρ(W) also have the eﬀect of driving signals x(n) into more nonlinear regions of tanh units
(further from 0) similarly to Win. Thus scalings of both Win and W have a similar eﬀect on nonlinearity
of the ESN, while their diﬀerence determines the amount of memory.
A rather conservative rigorous suﬃcient condition of the echo state property for any kind of inputs
u(n) (including zero) and states x(n) (with tanh nonlinearity) being σmax(W) < 1, where σmax(W)
is the largest singular value of W, was proved in [12].
Recently, a less restrictive suﬃcient condition,
namely, infD∈D σmax(DWD−1) < 1, where D is an arbitrary matrix, minimizing the so-called D-norm
σmax(DWD−1), from a set D ⊂RNx×Nx of diagonal matrices, has been derived in [63]. This suﬃcient
condition approaches the necessary infD∈D σmax(DWD−1) →ρ(W)−, ρ(W) < 1, e.g., when W is a nor-
mal or a triangular (permuted) matrix. A rigorous suﬃcient condition for the echo state property is rarely
ensured in practice, with a possible exception being critical control tasks, where provable stability under
any conditions is required.
5.2. Diﬀerent topologies of the reservoir
There have been attempts to ﬁnd topologies of the ESN reservoir diﬀerent from sparsely randomly
connected ones. Speciﬁcally, small-world [64], scale-free [65], and biologically inspired connection topologies
generated by spatial growth [66] were tested for this purpose in a careful study [67], which we point out
here due to its relevance although it was obtained only as a BSc thesis. The NRMS error (1) of y(n) as well
as the eigenvalue spread of the cross-correlation matrix of the activations x(n) (necessary for a fast online
learning described in Section 8.1.2; see Section 6.1 for details) were used as the performance measures of the
topologies. This work also explored an exhaustive brute-force search of topologies of tiny networks (motifs)
11
of four units, and then combining successful motives (in terms of the eigenvalue spread) into larger networks.
The investigation, unfortunately, concludes that “(. . . ) none of the investigated network topologies was able
to perform signiﬁcantly better than simple random networks, both in terms of eigenvalue spread as well as
testing error” [67]. This, however, does not serve as a proof that similar approaches are futile. An indication
of this is the substantial variation in ESN performance observed among randomly created reservoirs, which
is, naturally, more pronounced in smaller reservoirs (e.g., [68]).
In contrast, LSMs often use a biologically plausible connectivity structure and weight settings. In the
original form they model a single cortical microcolumn [11]. Since the model of both the connections and the
neurons themselves is quite sophisticated, it has a large number of free parameters to be set, which is done
manually, guided by biologically observed parameter ranges, e.g., as found in the rat somatosensory cortex
[69]. This type of model also delivers good performance for practical applications of speech recognition [69],
[70] (and many similar publications by the latter authors). Since LSMs aim at accuracy of modeling natural
neural structures, less biologically plausible connectivity patterns are usually not explored.
It has been demonstrated that much more detailed biological neural circuit models, which use anatomical
and neurophysiological data-based laminar (i.e., cortical layer) connectivity structures and Hodgkin-Huxley
model neurons, improve the information-processing capabilities of the models [23]. Such highly realistic
(for present-day standards) models “perform signiﬁcantly better than control circuits (which are lacking
the laminar structures but are otherwise identical with regard to their components and overall connection
statistics) for a wide variety of fundamental information-processing tasks” [23].
Diﬀerent from this direction of research, there are also explorations of using even simpler topologies of the
reservoir than the classical ESN. It has been demonstrated that the reservoir can even be an unstructured
feed-forward network with time-delayed connections if the ﬁnite limited memory window that it oﬀers is
suﬃcient for the task at hand [71].
A degenerate case of a “reservoir” composed of linear units and a
diagonalized W and unitary inputs Win was considered in [72]. A one-dimensional lattice (ring) topology
was used for a reservoir, together with an adaptation of the reservoir discussed in Section 6.2, in [73]. A
special kind of excitatory and inhibitory neurons connected in a one-dimensional spatial arrangement was
shown to produce interesting chaotic behavior in [74].
A tendency that higher ranks of the connectivity matrix Wmask (where wmaski,j = 1 if wi,j ̸= 0, and = 0
otherwise, for i, j = 1, . . . , Nx) correlate with lower ESN output errors was observed in [75]. Connectivity
patterns of W such that W∞≡limk→∞Wk (Wk standing for “W to the power k” and approximating
weights of the cumulative indirect connections by paths of length k among the reservoir units) is neither
fully connected, nor all-zero, are claimed to give a broader distribution of ESN prediction performances, thus
including best performing reservoirs, than random sparse connectivities in [76]. A permutation matrix with
a medium number and diﬀerent lengths of connected cycles, or a general orthogonal matrix, are suggested
as candidates for such Ws.
5.3. Modular reservoirs
One of the shortcomings of conventional ESN reservoirs is that even though they are sparse, the activa-
tions are still coupled so strongly that the ESN is poor in dealing with diﬀerent time scales simultaneously,
e.g., predicting several superimposed generators. This problem was successfully tackled by dividing the
reservoir into decoupled sub-reservoirs and introducing inhibitory connections among all the sub-reservoirs
[77]. For the approach to be eﬀective, the inhibitory connections must predict the activations of the sub-
reservoirs one time step ahead. To achieve this the inhibitory connections are heuristically computed from
(the rest of) W and Wofb, or the sub-reservoirs are updated in a sequence and the real activations of the
already updated sub-reservoirs are used.
The Evolino approach introduced in Section 3.3 can also be classiﬁed as belonging to this group, as
the LSTM RNN used for its reservoir consists of speciﬁc small memory-holding modules (which could
alternatively be regarded as more complicated units of the network).
Approaches relying on combining outputs from several separate reservoirs will be discussed in Section
8.8.
12
5.4. Time-delayed vs. instantaneous connections
Another time-related limitation of the classical ESNs pointed out in [78] is that no matter how many
neurons are contained in the reservoir, it (like any other fully recurrent network with all connections having a
time delay) has only a single layer of neurons (Figure 2). This makes it intrinsically unsuitable for some types
of problems. Consider a problem where the mapping from u(n) to ytarget(n) is a very complex, nonlinear
one, and the data in neighboring time steps are almost independent (i.e., little memory is required), as e.g.,
the “meta-learning” task in [79] 4. Consider a single time step n: signals from the input u(n) propagate
only through one untrained layer of weights Win, through the nonlinearity f inﬂuence the activations x(n),
and reach the output y(n) through the trained weights Wout (Figure 2). Thus ESNs are not capable of
producing a very complex instantaneous mapping from u(n) to y(n) using a realistic number of neurons,
which could (only) be eﬀectively done by a multilayer FFNN (not counting some non-NN-based methods).
Delaying the target ytarget by k time steps would in fact make the signals coming from u(n) “cross” the
nonlinearities k + 1 times before reaching y(n + k), but would mix the information from diﬀerent time steps
in x(n), . . . , x(n + k), breaking the required virtually independent mapping u(n) →ytarget(n + k), if no
special structure of W is imposed.
u(n)
y(n)
Win
Wout
W
z-1 
x(n)
f
Figure 2: Signal ﬂow diagram of the standard ESN.
As a possible remedy Layered ESNs were introduced in [78], where a part (up to almost half) of the
reservoir connections can be instantaneous and the rest take one time step for the signals to propagate as
in normal ESNs. Randomly generated Layered ESNs, however, do not oﬀer a consistent improvement for
large classes of tasks, and pre-training methods of such reservoirs have not yet been investigated.
The issue of standard ESNs not having enough trained layers is also discussed and addressed in a broader
context in Section 8.8.
5.5. Leaky integrator neurons and speed of dynamics
In addition to the basic sigmoid units, leaky integrator neurons were suggested to be used in ESNs from
the point of their introduction [12]. This type of neuron performs a leaky integration of its activation from
previous time steps. Today a number of versions of leaky integrator neurons are often used in ESNs, which
we will call here leaky integrator ESNs (LI-ESNs) where the distinction is needed. The main two groups are
those using leaky integration before application of the activation function f(·), and after. One example of
the latter (in the discretized time case) has reservoir dynamics governed by
x(n) = (1 −a∆t)x(n −1) + ∆tf(Winu(n) + Wx(n −1)),
(9)
where ∆t is a compound time gap between two consecutive time steps divided by the time constant of the
system and a is the decay (or leakage) rate [81]. Another popular (and we believe, preferable) design can be
seen as setting a = 1 and redeﬁning ∆t in (9) as the leaking rate a to control the “speed” of the dynamics,
x(n) = (1 −a)x(n −1) + af(Winu(n) + Wx(n −1)),
(10)
which in eﬀect is an exponential moving average, has only one additional parameter and the desirable
property that neuron activations x(n) never go outside the boundaries deﬁned by f(·). Note that the simple
ESN (5) is a special case of LI-ESNs (9) or (10) with a = 1 and ∆t = 1. As a corollary, an LI-ESN with a
4ESNs have been shown to perform well in a (signiﬁcantly) simpler version of the “meta-learning” in [80].
13
good choice of the parameters can always perform at least as well as a corresponding simple ESN. With the
introduction of the new parameter a (and ∆t), the condition for the echo state property is redeﬁned [12].
A natural constraint on the two new parameters is a∆t ∈[0, 1] in (9), and a ∈[0, 1] in (10) — a neuron
should neither retain, nor leak, more activation than it had. The eﬀect of these parameters on the ﬁnal
performance of ESNs was investigated in [18] and [82]. The latter contribution also considers applying the
leaky integrator in diﬀerent places of the model and resampling the signals as an alternative.
The additional parameters of the LI-ESN control the “speed” of the reservoir dynamics. Small values of
a and ∆t result in reservoirs that react slowly to the input. By changing these parameters it is possible to
shift the eﬀective interval of frequencies in which the reservoir is working. Along these lines, time warping
invariant ESNs (TWIESNs) — an architecture that can deal with strongly time-warped signals — were
outlined in [81, 18]. This architecture varies ∆t on-the-ﬂy in (9), directly depending on the speed at which
the input u(n) is changing.
From a signal processing point of view, the exponential moving average on the neuron activation (10)
does a simple low-pass ﬁltering of its activations with the cutoﬀfrequency
fc =
a
2π(1 −a)∆t,
(11)
where ∆t is the discretization time step. This makes the neurons average out the frequencies above fc and
enables tuning the reservoirs for particular frequencies. Elaborating further on this idea, high-pass neurons,
that produce their activations by subtracting from the unﬁltered activation (5) the low-pass ﬁltered one (10),
and band-pass neurons, that combine the low-pass and high-pass ones, were introduced [83]. The authors
also suggested mixing neurons with diﬀerent passbands inside a single ESN reservoir, and reported that a
single reservoir of such kind is able to predict/generate signals having structure on diﬀerent timescales.
Following this line of thought, Inﬁnite Impulse Response (IIR) band-pass ﬁlters having sharper cutoﬀ
characteristics were tried on neuron activations in ESNs with success in several types of signals [84]. Since
the ﬁlters often introduce an undesired phase shift to the signals, a time delay for the activation of each
neuron was learned and applied before the linear readout from the reservoir. A successful application of
Butterworth band-pass ﬁlters in ESNs is reported in [85].
Connections between neurons that have diﬀerent time delays (more than one time step) can actually also
be used inside the recurrent part, which enables the network to operate on diﬀerent timescales simultaneously
and learn longer-term dependences [86]. This idea has been tried for RNNs trained by error backpropagation,
but could also be useful for multi-timescale reservoirs. Long-term dependences can also be learned using
the reservoirs mentioned in Section 3.3.
6. Unsupervised reservoir adaptation
In this section we describe reservoir training/generation methods that try to optimize some measure
deﬁned on the activations x(n) of the reservoir, for a given input u(n), but regardless of the desired output
ytarget(n). In Section 6.1 we survey measures that are used to estimate the quality of the reservoir, irrespec-
tive of the methods optimizing them. Then local, Section 6.2, and global, Section 6.3 unsupervised reservoir
training methods are surveyed.
6.1. “Goodness” measures of the reservoir activations
The classical feature that reservoirs should possess is the echo state property, deﬁned in Section 5.1. Even
though this property depends on the concrete input u(n), usually in practice its existence is not measured
explicitly, and only the spectral radius ρ(W) is selected to be < 1 irrespective of u(n), or just tuned for the
ﬁnal performance. A measure of short-term memory capacity, evaluating how well u(n) can be reconstructed
by the reservoir as y(n + k) after various delays k, was introduced in [41].
The two necessary and suﬃcient conditions for LSMs to work were introduced in [11]. A separation
property measures the distance between diﬀerent states x caused by diﬀerent input sequences u.
The
measure is reﬁned for binary ESN-type reservoirs in [87] with a generalization in [88]. An approximation
14
property measures the capability of the readout to produce a desired output ytarget from x, and thus is not
an unsupervised measure, but is included here for completeness.
Methods for estimating the computational power and generalization capability of neural reservoirs were
presented in [89]. The proposed measure for computational power, or kernel quality, is obtained in the
following way. Take k diﬀerent input sequences (or segments of the same signal) ui(n), where i = 1, . . . , k,
and n = 1, . . . , Tk. For each input i take the resulting reservoir state xi(n0), and collect them into a matrix
M ∈Rk×Nx, where n0 is some ﬁxed time after the appearance of ui(n) in the input. Then the rank r of the
matrix M is the measure. If r = k, this means that all the presented inputs can be separated by a linear
readout from the reservoir, and thus the reservoir is said to have a linear separation property. For estimating
the generalization capability of the reservoir, the same procedure can be performed with s (s ≫k) inputs
uj(n), j = 1, . . . , s, that represent the set of all possible inputs. If the resultant rank r is substantially
smaller than the size s of the training set, the reservoir generalizes well. These two measures are more
targeted to tasks of time series classiﬁcation, but can also be revealing in predicting the performance of
regression [90].
A much-desired measure to minimize is the eigenvalue spread (EVS, the ratio of the maximal eigenvalue
to the minimal eigenvalue) of the cross-correlation matrix of the activations x(n). A small EVS is necessary
for an online training of the ESN output by a computationally cheap and stable stochastic gradient descent
algorithm outlined in Section 8.1.2 (see, e.g., [91], chapter 5.3, for the mathematical reasons that render this
mandatory). In classical ESNs the EVS sometimes reaches 1012 or even higher [92], which makes the use
of stochastic gradient descent training unfeasible. Other commonly desirable features of the reservoir are
small pairwise correlation of the reservoir activations xi(n), or a large entropy of the x(n) distribution (e.g.,
[92]). The latter is a rather popular measure, as discussed later in this review. A criterion for maximizing
the local information transmission of each individual neuron was investigated in [93] (more in Section 6.2).
The so-called edge of chaos is a region of parameters of a dynamical system at which it operates at the
boundary between the chaotic and non-chaotic behavior. It is often claimed (but not undisputed; see, e.g.,
[94]) that at the edge of chaos many types of dynamical systems, including binary systems and reservoirs,
possess high computational power [87, 95]. It is intuitively clear that the edge of chaos in reservoirs can
only arise when the eﬀect of inputs on the reservoir state does not die out quickly; thus such reservoirs
can potentially have high memory capacity, which is also demonstrated in [95]. However, this does not
universally imply that such reservoirs are optimal [90]. The edge of chaos can be empirically detected (even
for biological networks) by measuring Lyapunov exponents [95], even though such measurements are not
trivial (and often involve a degree of expert judgment) for high-dimensional noisy systems. For reservoirs of
simple binary threshold units this can be done more simply by computing the Hamming distances between
trajectories of the states [87]. There is also an empirical observation that, while changing diﬀerent parameter
settings of a reservoir, the best performance in a given task correlates with a Lyapunov exponent speciﬁc to
that task [59]. The optimal exponent is related to the amount of memory needed for the task as discussed in
Section 5.1. It was observed in ESNs with no input that when ρ(W) is slightly greater than 1, the internally
generated signals are periodic oscillations, whereas for larger values of ρ(W), the signals are more irregular
and even chaotic [96]. Even though stronger inputs u(n) can push the dynamics of the reservoirs out of the
chaotic regime and thus make them useful for computation, no reliable beneﬁt of such a mode of operation
was found in the last contribution.
In contrast to ESN-type reservoirs of real-valued units, simple binary threshold units exhibit a more
immediate transition from damped to chaotic behavior without intermediate periodic oscillations [87]. This
diﬀerence between the two types of activation functions, including intermediate quantized ones, in ESN-type
reservoirs was investigated more closely in [88]. The investigation showed that reservoirs of binary units
are more sensitive to the topology and the connection weight parameters of the network in their transition
between damped and chaotic behavior, and computational performance, than the real-valued ones. This
diﬀerence can be related to the similar apparent diﬀerence in sensitivity of the ESNs and LSM-type reservoirs
of ﬁring units, discussed in Section 5.2.
15
6.2. Unsupervised local methods
A natural strategy for improving reservoirs is to mimic biology (at a high level of abstraction) and count
on local adaptation rules. “Local” here means that parameters pertaining to some neuron i are adapted on
the basis of no other information than the activations of neurons directly connected with neuron i. In fact
all local methods are almost exclusively unsupervised, since the information on the performance E at the
output is unreachable in the reservoir.
First attempts to decrease the eigenvalue spread in ESNs by classical Hebbian [97] (inspired by synaptic
plasticity in biological brains) or Anti-Hebbian learning gave no success [92]. A modiﬁcation of Anti-Hebbian
learning, called Anti-Oja learning is reported to improve the performance of ESNs in [98].
On the more biologically realistic side of the RC research with spiking neurons, local unsupervised
adaptations are very natural to use. In fact, LSMs had used synaptic connections with realistic short-term
dynamic adaptation, as proposed by [99], in their reservoirs from the very beginning [11].
The Hebbian learning principle is usually implemented in spiking NNs as spike-time-dependent plasticity
(STDP) of synapses. STDP is shown to improve the separation property of LSMs for real-world speech data,
but not for random inputs u, in [100]. The authors however were uncertain whether manually optimizing
the parameters of the STDP adaptation (which they did) or the ones for generating the reservoir would
result in a larger performance gain for the same eﬀort spent. STDP is shown to work well with time-coded
readouts from the reservoir in [101].
Biological neurons are widely observed to adapt their intrinsic excitability, which often results in expo-
nential distributions of ﬁring rates, as observed in visual cortex (e.g., [102]). This homeostatic adaptation
mechanism, called intrinsic plasticity (IP) has recently attracted a wide attention in the reservoir computing
community. Mathematically, the exponential distribution maximizes the entropy of a non-negative random
variable with a ﬁxed mean; thus it enables the neurons to transmit maximal information for a ﬁxed metabolic
cost of ﬁring. An IP learning rule for spiking model neurons aimed at this goal was ﬁrst presented in [103].
For a more abstract model of the neuron, having a continuous Fermi sigmoid activation function f :
R →(0, 1), the IP rule was derived as a proportional control that changes the steepness and oﬀset of the
sigmoid to get an exponential-like output distribution in [104]. A more elegant gradient IP learning rule
for the same purpose was presented in [93], which is similar to the information maximization approach in
[105]. Applying IP with Fermi neurons in reservoir computing signiﬁcantly improves the performance of
BPDC-trained networks [106, 107], and is shown to have a positive eﬀect on oﬄine trained ESNs, but can
cause stability problems for larger reservoirs [106]. An ESN reservoir with IP-adapted Fermi neurons is also
shown to enable predicting several superimposed oscillators [108].
An adaptation of the IP rule to tanh neurons (f : R →(−1, 1)) that results in a zero-mean Gaussian-like
distribution of activations was ﬁrst presented in [73] and investigated more in [109]. The IP-adapted ESNs
were compared with classical ones, both having Fermi and tanh neurons, in the latter contribution. IP
was shown to (modestly) improve the performance in all cases. It was also revealed that ESNs with Fermi
neurons have signiﬁcantly smaller short-term memory capacity (as in Section 6.1) and worse performance in
a synthetic NARMA prediction task, while having a slightly better performance in a speech recognition task,
compared to tanh neurons. The same type of tanh neurons adapted by IP aimed at Laplacian distributions
are investigated in [110]. In general, IP gives more control on the working points of the reservoir nonlinearity
sigmoids. The slope (ﬁrst derivative) and the curvature (second derivative) of the sigmoid at the point around
which the activations are centered by the IP rule aﬀect the eﬀective spectral radius and the nonlinearity of
the reservoir, respectively. Thus, for example, centering tanh activations around points other than 0 is a
good idea if no quasi-linear behavior is desired. IP has recently become employed in reservoirs as a standard
practice by several research groups.
Overall, an information-theoretic view on adaptation of spiking neurons has a long history in compu-
tational neuroscience.
Even better than maximizing just any information in the output of a neuron is
maximizing relevant information. In other words, in its output the neuron should encode the inputs in
such a way as to preserve maximal information about some (local) target signal. This is addressed in a
general information-theoretical setting by the Information Bottleneck (IB) method [111]. A learning rule
for a spiking neuron that maximizes mutual information between its inputs and its output is presented in
16
[112]. A more general IB learning rule, transferring the general ideas of IB method to spiking neurons is
introduced in [113] and [114]. Two semi-local training scenarios are presented in these two contributions. In
the ﬁrst, a neuron optimizes the mutual information of its output with outputs of some neighboring neurons,
while minimizing the mutual information with its inputs. In the second, two neurons reading from the same
signals maximize their information throughput, while keeping their inputs statistically independent, in eﬀect
performing Independent Component Analysis (ICA). A simpliﬁed online version of the IB training rule with
a variation capable of performing Principle Component Analysis (PCA) was recently introduced in [115].
In addition, it assumes slow semi-local target signals, which is more biologically plausible. The approaches
described in this paragraph are still waiting to be tested in the reservoir computing setting.
It is also of great interest to understand how diﬀerent types of plasticity observed in biological brains
interact when applied together and what eﬀect this has on the quality of reservoirs. The interaction of the
IP with Hebbian synaptic plasticity in a single Fermi neuron is investigated in [104] and further in [116].
The synergy of the two plasticities is shown to result in a better specialization of the neuron that ﬁnds
heavy-tail directions in the input. An interaction of IP with a neighborhood-based Hebbian learning in a
layer of such neurons was also shown to maximize information transmission, perform nonlinear ICA, and
result in an emergence of orientational Gabor-like receptive ﬁelds in [117]. The interaction of STDP with
IP in an LSM-like reservoir of simple sparsely spiking neurons was investigated in [118]. The interaction
turned out to be a non-trivial one, resulting in networks more robust to perturbations of the state x(n) and
having a better short-time memory and time series prediction performance.
A recent approach of combining STDP with a biologically plausible reinforcement signal is discussed in
Section 7.5, as it is not unsupervised.
6.3. Unsupervised global methods
Here we review unsupervised methods that optimize reservoirs based on global information of the reservoir
activations induced by the given input u(x), but irrespective of the target ytarget(n), like for example the
measures discussed in Section 6.1. The intuitive goal of such methods is to produce good representations of
(the history of) u(n) in x(n) for any (and possibly several) ytarget(n).
A biologically inspired unsupervised approach with a reservoir trying to predict itself is proposed in
[119].
An additional output z(n) ∈RNx, z(n) = Wzx(n) from the reservoir is trained on the target
ztarget(n) = x′(n + 1), where x′(n) are the activations of the reservoir before applying the neuron transfer
function tanh(·), i.e., x(n) = tanh(x′(n)).
Then, in the application phase of the trained networks, the
original activations x′(n), which result from u(n), Win, and W, are mixed with the self-predictions z(n−1)
obtained from Wz, with a certain mixing ratio (1 −α) : α. The coeﬃcient α determines how much the
reservoir is relying on the external input u(n) and how much on the internal self-prediction z(n). With
α = 0 we have the classical ESN and with α = 1 we have an “autistic” reservoir that does not react to the
input. Intermediate values of α close to 1 were shown to enable reservoirs to generate slow, highly nonlinear
signals that are hard to get otherwise.
An algebraic unsupervised way of generating ESN reservoirs was proposed in [120].
The idea is to
linearize the ESN update equation (5) locally around its current state x(n) at every time step n to get a
linear approximation of (5) as x(n + 1) = Ax(n) + Bu(n), where A and B are time (n)-dependent matrices
corresponding to W and Win respectively.
The approach aims at distributing the predeﬁned complex
eigenvalues of A uniformly within the unit circle on the C plane. The reservoir matrix W is obtained
analytically from the set of these predeﬁned eigenvalues and a given input u(n). The motivation for this
is, as for Kautz ﬁlters [121] in linear systems, that if the target ytarget(n) is unknown, it is best to have
something like an orthogonal basis in x(n), from which any ytarget(n) could, on average, be constructed
well. The spectral radius of the reservoir is suggested to be set by hand (according to the correlation time
of u(n), which is an indication of a memory span needed for the task), or by adapting the bias value of the
reservoir units to minimize the output error (which actually renders this method supervised, as in Section
7). Reservoirs generated this way are shown to yield higher average entropy of x(n) distribution, higher
short-term memory capacity (both measures mentioned in Section 6.1), and a smaller output error on a
number of synthetic problems, using relatively small reservoirs (Nx = 20, 30). However, a more extensive
empirical comparison of this type of reservoir with the classical ESN one is still lacking.
17
7. Supervised reservoir pre-training
In this section we discuss methods for training reservoirs to perform a speciﬁc given task, i.e., not only
the concrete input u(n), but also the desired output ytarget(n) is taken into account. Since a linear readout
from a reservoir is quickly trained, the suitability of a candidate reservoir for a particular task (e.g., in terms
of NRMSE (1)) is inexpensive to check. Notice that even for most methods of this class the explicit target
signal ytarget(n) is not technically required for training the reservoir itself, but only for evaluating it in an
outer loop of the adaptation process.
7.1. Optimization of global reservoir parameters
In Section 5.1 we discussed guidelines for the manual choice of global parameters for reservoirs of ESNs.
This approach works well only with experience and a good intuitive grasp on nonlinear dynamics.
A
systematic gradient descent method of optimizing the global parameters of LI-ESNs (recalled from Section
5.5) to ﬁt them to a given task is presented in [18]. The investigation shows that the error surfaces in
the combined global parameter and Wout spaces may have very high curvature and multiple local minima.
Thus, gradient descent methods are not always practical.
7.2. Evolutionary methods
As one can see from the previous sections of this review, optimizing reservoirs is generally challenging,
and breakthrough methods remain to be found. On the other hand checking the performance of a resulting
ESN is relatively inexpensive, as said. This brings in evolutionary methods for the reservoir pre-training as
a natural strategy.
Recall that the classical method generates a reservoir randomly; thus the performance of the resulting
ESN varies slightly (and for small reservoirs not so slightly) from one instance to another. Then indeed,
an “evolutionary” method as naive as “generate k reservoirs, pick the best” will outperform the classical
method (“generate a reservoir”) with probability (k −1)/k, even though the improvement might be not
striking.
Several evolutionary approaches on optimizing reservoirs of ESNs are presented in [122].
The ﬁrst
approach was to carry out an evolutionary search on the parameters for generating W: Nx, ρ(W), and the
connection density of W. Then an evolutionary algorithm [123] was used on individuals consisting of all the
weight matrices (Win, W, Wofb) of small (Nx = 5) reservoirs. A variant with a reduced search space was
also tried where the weights, but not the topology, of W were explored, i.e., elements of W that were zero
initially always stayed zero. The empirical results of modeling the motion of an underwater robot showed
superiority of the methods over other state-of-art methods, and that the topology-restricted adaptation of
W is almost as eﬀective as the full one.
Another approach of optimizing the reservoir W by a greedy evolutionary search is presented in [75].
Here the same idea of separating the topology and weight sizes of W to reduce the search space was
independently used, but the search was, conversely, restricted to the connection topology. This approach also
was demonstrated to yield on average 50% smaller (and much more stable) error in predicting the behavior
of a mass–spring–damper system with small (Nx = 20) reservoirs than without the genetic optimization.
Yet another way of reducing the search space of the reservoir parameters is constructing a big reservoir
weight matrix W in a fractal fashion by repeatedly applying Kronecker self-multiplication to an initial small
matrix, called the Kronecker kernel [124]. This contribution showed that among Ws constructed in this
way some yield ESN performance similar to the best unconstrained Ws; thus only the good weights of the
small Kronecker kernel need to be found by evolutionary search for producing a well-performing reservoir.
Evolino [46], introduced in Section 3.3, is another example of adapting a reservoir (in this case an LSTM
network) using a genetic search.
It has been recently demonstrated that by adapting only the slopes of the reservoir unit activation
functions f(·) by a state-of-art evolutionary algorithm, and having Wout random and ﬁxed, a prediction
performance of an ESN can be achieved close to the best of classical ESNs [68].
In addition to (or instead of) adapting the reservoirs, an evolutionary search can also be applied in
training the readouts, such as readouts with no explicit ytarget(n) as discussed in Section 8.4.
18
7.3. Other types of supervised reservoir tuning
A greedy pruning of neurons from a big reservoir has been shown in a recent initial attempt [125] to often
give a (bit) better classiﬁcation performance for the same ﬁnal Nx than just a randomly created reservoir
of the same size. The eﬀect of neuron removal to the reservoir dynamics, however, has not been addressed
yet.
7.4. Trained auxiliary feedbacks
While reservoirs have a natural capability of performing complex real-time analog computations with
fading memory [11], an analytical investigation has shown that they can approximate any k-order diﬀerential
equation (with persistent memory) if extended with k trained feedbacks [21, 126]. This is equivalent to
simulating any Turing machine, and thus also means universal digital computing. In the presence of noise
(or ﬁnite precision) the memory becomes limited in such models, but they still can simulate Turing machines
with ﬁnite tapes.
This theory has direct implications for reservoir computing; thus diﬀerent ideas on how the power of
ESNs could be improved along its lines are explored in [78]. It is done by deﬁning auxiliary targets, training
additional outputs of ESNs on these targets, and feeding the outputs back to the reservoir. Note that this
can be implemented in the usual model with feedback connections (6) by extending the original output
y(n) with additional dimensions that are trained before training the original (ﬁnal) output. The auxiliary
targets are constructed from ytarget(n) and/or u(n) or some additional knowledge of the modeled process.
The intuition is that the feedbacks could shift the internal dynamics of x(n) in the directions that would
make them better linearly combinable into ytarget(n).
The investigation showed that for some types of
tasks there are natural candidates for such auxiliary targets, which improve the performance signiﬁcantly.
Unfortunately, no universally applicable methods for producing auxiliary targets are known such that the
targets would be both easy to learn and improve the accuracy of the ﬁnal output y(n). In addition, training
multiple outputs with feedback connections Wofb makes the whole procedure more complicated, as cyclical
dependences between the trained outputs (one must take care of the order in which the outputs are trained)
as well as stability issues discussed in Section 8.2 arise. Despite these obstacles, we perceive this line of
research as having a big potential.
7.5. Reinforcement learning
In the line of biologically inspired local unsupervised adaptation methods discussed in Section 6.2, an
STDP modulated by a reinforcement signal has recently emerged as a powerful learning mechanism, capable
of explaining some famous ﬁndings in neuroscience (biofeedback in monkeys), as demonstrated in [127,
128] and references thereof. The learning mechanism is also well biologically motivated as it uses a local
unsupervised STDP rule and a reinforcement (i.e., reward) feedback, which is present in biological brains
in a form of chemical signaling, e.g., by the level of dopamine. In the RC framework this learning rule has
been successfully applied for training readouts from the reservoirs so far in [128], but could in principle be
applied inside the reservoir too.
Overall the authors of this review believe that reinforcement learning methods are natural candidates
for reservoir adaptation, as they can immediately exploit the knowledge of how well the output is learned
inside the reservoir without the problems of error backpropagation. They can also be used in settings where
no explicit target ytarget(n) is available. We expect to see more applications of reinforcement learning in
reservoir computing in the future.
8. Readouts from the reservoirs
Conceptually, training a readout from a reservoir is a common supervised non-temporal task of mapping
x(n) to ytarget(n). This is a well investigated domain in machine learning, much more so than learning
temporal mappings with memory. A large choice of methods is available, and in principle any of them can
be applied. Thus we will only brieﬂy go through the ones reported to be successful in the literature.
19
8.1. Single-layer readout
By far the most popular readout method from the ESN reservoirs is the originally proposed [12] simple
linear readout, as in (3) (we will consider it as equivalent to (8), i.e., u(n) being part of x(n)). It is shown to
be often suﬃcient, as reservoirs provide a rich enough pool of signals for solving many application-relevant
and benchmark tasks, and is very eﬃcient to train, since optimal solutions can be found analytically.
8.1.1. Linear regression
In batch mode, learning of the output weights Wout (2) can be phrased as solving a system of linear
equations
WoutX = Ytarget
(12)
with respect to Wout, where X ∈RN×T are all x(n) produced by presenting the reservoir with u(n),
and Ytarget ∈RNy×T are all ytarget(n), both collected into respective matrices over the training period
n = 1, . . . , T. Usually x(n) data from the beginning of the training run are discarded (they come before
n = 1), since they are contaminated by initial transients.
Since typically the goal is minimizing a quadratic error E(Ytarget, WoutX) as in (1) and T > N, to
solve (12) one usually employs methods for ﬁnding least square solutions of overdetermined systems of linear
equations (e.g., [129]), the problem also known as linear regression. One direct method is calculating the
Moore-Penrose pseudoinverse X+ of X, and Wout as
Wout = YtargetX+.
(13)
Direct pseudoinverse calculations exhibit high numerical stability, but are expensive memory-wise for large
state-collecting matrices X ∈RN×T , thereby limiting the size of the reservoir N and/or the number of
training samples T.
This issue is resolved in the normal equations formulation of the problem:5
WoutXX
T = YtargetX
T.
(14)
A naive solution of it would be
Wout = YtargetX
T(XX
T)−1.
(15)
Note that in this case YtargetX
T ∈RNy×N and XX
T ∈RN×N do not depend on the length T of the training
sequence, and can be calculated incrementally while the training data are passed through the reservoir.
Thus, having these two matrices collected, the solution complexity of (15) does not depend on T either in
time or in space. Also, intermediate values of Wout can be calculated in the middle of running through the
training data, e.g., for an early assessment of the performance, making this a “semi-online” training method.
The method (15) has lower numerical stability, compared to (13), but the problem can be mitigated by
using the pseudoinverse (XX
T)+ instead of the real inverse (XX
T)−1 (which usually also works faster). In
addition, this method enables one to introduce ridge, or Tikhonov, regularization elegantly:
Wout = YtargetX
T(XX
T + α2I)−1,
(16)
where I ∈RN×N is the identity matrix and α is a regularization factor. In addition to improving the numeri-
cal stability, the regularization in eﬀect reduces the magnitudes of entries in Wout, thus mitigating sensitivity
to noise and overﬁtting; see Section 8.2 for more details. All this makes (16) a highly recommendable choice
for learning outputs from the reservoirs.
Another alternative for solving (14) is decomposing the matrix XX
T into a product of two triangular
matrices via Cholesky or LU decomposition, and solving (14) by two steps of substitution, avoiding (pseudo-
)inverses completely. The Cholesky decomposition is the more numerically stable of the two.
5Note that our matrices are transposed compared to the conventional notation.
20
Weighted regression can be used for training linear readouts by multiplying both x(n) and the corre-
sponding ytarget(n) by diﬀerent weights over time, thus emphasizing some time steps n over others. Multi-
plying certain recorded x(n) and corresponding ytarget(n) by
√
k has the same emphasizing eﬀect as if they
appeared in the training sequence k times.
When the reservoir is made from spiking neurons and thus x(n) becomes a collection of spike trains,
smoothing by low-pass ﬁltering may be applied to it before doing the linear regression, or it can be done
directly on x(n) [11]. For more on linear regression based on spike train data, see [130].
Evolutionary search for training linear readouts can also be employed. State-of-art evolutionary methods
are demonstrated to be able to achieve the same record levels of precision for supervised tasks as with the
best applications of linear regression in ESN training [68]. Their much higher computational cost is justiﬁable
in settings where no explicit ytarget(n) is available, discussed in Section 8.4.
8.1.2. Online adaptive output weight training
Some applications require online model adaptation, e.g., in online adaptive channel equalization [17]. In
such cases one typically minimizes an error that is exponentially discounted going back in time. Wout here
acts as an adaptive linear combiner. The simplest way to train Wout is to use stochastic gradient descent.
The method is familiar as the Least Mean Squares (LMS) algorithm in linear signal processing [91], and
has many extensions and modiﬁcations. Its convergence performance is unfortunately severely impaired by
large eigenvalue spreads of XX
T, as mentioned in Section 6.1.
An alternative to LMS, known in linear signal processing as the Recursive Least Squares (RLS) algorithm,
is insensitive to the detrimental eﬀects of eigenvalue spread and boasts a much faster convergence because it
is a second-order method. The downside is that RLS is computationally more expensive (order O(N2) per
time step instead of O(N) for LMS, for Ny = 1) and notorious for numerical stability issues. Demonstrations
of RLS are presented in [17, 43]. A careful and comprehensive comparison of variants of RLS is carried out
in a Master’s thesis [131], which we mention here because it will be helpful for practitioners.
The BackPropagation-DeCorrelation (BPDC) algorithm discussed in Section 3.4 is another powerful
method for online training of single-layer readouts with feedback connections from the reservoirs.
Simple forms of adaptive online learning, such as LMS, are also more biologically plausible than batch-
mode training. From spiking neurons a ﬁring time-coded (instead of a more common ﬁring rate-coded)
output for classiﬁcation can also be trained by only adapting the delays of the output connections [101].
And ﬁring rate-coded readouts can be trained by a biologically-realistic reward-modulated STDP [128],
mentioned in Section 6.2.
8.1.3. SVM-style readout
Continuing the analogy between the temporal and non-temporal expansion methods, discussed in Section
2, the reservoir can be considered a temporal kernel, and the standard linear readout Wout from it can be
trained using the same loss functions and regularizations as in Support Vector Machines (SVMs) or Support
Vector Regression (SVR). Diﬀerent versions of this approach are proposed and investigated in [132].
A standard SVM (having its own kernel) can also be used as a readout from a continuous-value reservoir
[133]. Similarly, special kernel types could be applied in reading out from spiking (LSM-type) reservoirs
[134] (and references therein).
8.2. Feedbacks and stability issues
Stability issues (with reservoirs having the echo state property) usually only occur in generative setups
where a model trained on (one step) signal prediction is later run in a generative mode, looping its output
y(n) back into the input as u(n + 1). Note that this is equivalent to a model with output feedbacks Wofb
(6) and no input at all (Nu = 0), which is usually trained using teacher forcing (i.e., feeding ytarget(n) as
y(n) for the feedbacks during the training run) and later is run freely to generate signals as y(n). Win in
the ﬁrst case is equivalent to Wofb in the second one. Models having feedbacks Wofb may also suﬀer from
instability while driven with external input u(n), i.e., not in a purely generative mode.
21
The reason for these instabilities is that even if the model can predict the signal quite accurately, going
through the feedback loop of connections Wout and Wofb (or Win) small errors get ampliﬁed, making y(n)
diverge from the intended ytarget(n).
One way to look at this for trained linear outputs is to consider the feedback loop connections Wout and
Wofb as part of the reservoir W. Putting (6) and (2) together we get
x(n) = f(Winu(n) + [W + WofbWout]x(n −1)),
(17)
where W + WofbWout forms the “extended reservoir” connections, which we will call W∗for brevity (as
in [78] Section 3.2). If the spectral radius of the extended reservoir ρ(W∗) is very large we can expect
unstable behavior. A more detailed analysis using Laplace transformations and a suﬃcient condition for
stability is presented in [135]. On the other hand, for purely generative tasks, ρ(W∗) < 1 would mean that
the generated signal would die out, which is not desirable in most cases. Thus producing a generator with
stable dynamics is often not trivial.
Quite generally, models trained with clean (noise-free) data for the best one-time-step prediction diverge
fast in the generative mode, as they are too “sharp” and not noise-robust. A classical remedy is adding some
noise to reservoir states x(n) [12] during the training. This way the generator forms a stable attractor by
learning how to come to the desired next output ytarget(n) from a neighborhood of the current state x(n),
having seen it perturbed by noise during training. Setting the right amount of noise is a delicate balance
between the sharpness (of the prediction) and the stability of the generator. Alternatively, adding noise to
x(n) can be seen as a form of regularization in training, as it in eﬀect also emphasizes the diagonal of matrix
XX
T in (16). A similar eﬀect can be achieved using ridge regression (16) [136], or to some extent even
pruning of Wout [137]. Ridge regression (16) is the least computationally expensive to do of the three, since
the reservoir does not need to be rerun with the data to test diﬀerent values of the regularization factor α.
Using diﬀerent modiﬁcations of signals for teacher forcing, like mixing ytarget(n) with noise, or in some
cases using pure strong noise, during the training also has an eﬀect on the ﬁnal performance and stability,
as discussed in Section 5.4 of [78].
8.3. Readouts for classiﬁcation/recognition
The time series classiﬁcation or temporal pattern detection tasks that need a category indicator (as
opposed to real values) as an output can be implemented in two main ways.
The most common and
straightforward way is having a real-valued output for each class (or a single output and a threshold for the
two-class classiﬁer), and interpreting the strengths of the outputs as votes for the corresponding classes, or
even class probabilities (several options are discussed in [18]). Often the most probable class is taken as the
decision. A simple target ytarget for this approach is a constant ytargeti(n) = 1 signal for the right class i
and 0 for the others in the range of n where the indicating output is expected. More elaborate shapes of
ytarget(n) can improve classiﬁcation performance, depending on the task (e.g., [81]). With spiking neurons
the direct classiﬁcation based on time coding can be learned and done, e.g., the class is assigned depending
on which output ﬁres ﬁrst [101].
The main alternative to direct class indications is to use predictive classiﬁers, i.e., train diﬀerent predictors
to predict diﬀerent classes and assign a class to a new example corresponding to the predictor that predicts
it best. Here the quality of each predictor serves as the output strength for the corresponding class. The
method is quite popular in automated speech recognition (e.g., Section 6 in [138] for an overview). However,
in Section 6.5 of [138] the author argues against this approach, at least in its straightforward form, pointing
out some weaknesses, like the lack of speciﬁcity, and negative practical experience.
For both approaches a weighting scheme can be used for both training (like in weighted regression) and
integrating the class votes, e.g., putting more emphasis on the end of the pattern when suﬃcient information
has reached the classiﬁer to make the decision.
An advanced version of ESN-based predictive classiﬁer, where for each class there is a set of competitively
trained predictors and dynamic programming is used to ﬁnd the optimal sequence of them, is reported to
be much more noise robust than a standard Hidden Markov Model in spoken word recognition [139].
22
8.4. Readouts beyond supervised learning
Even though most of the readout types from reservoirs reported in the literature are trained in a purely
supervised manner, i.e., making y(n) match an explicitly given ytarget(n), the reservoir computing paradigm
lends itself to settings where no ytarget(n) is available. A typical such setting is reinforcement learning where
only a feedback on the model’s performance is available. Note that an explicit ytarget(n) is not required for
the reservoir adaptation methods discussed in Sections 5 and 6 of this survey by deﬁnition. Even most of
the adaptation methods classiﬁed as supervised in Section 7 do not need an explicit ytarget(n), as long as
one can evaluate the performance of the reservoir. Thus they can be used without modiﬁcation, provided
that unsupervised training and evaluation of the output is not prohibitively expensive or can be done
simultaneously with reservoir adaptation. In this section we will give some pointers on training readouts
using reinforcement learning.
A biologically inspired learning rule of Spike-Time-Dependent Plasticity (STDP) modulated by a rein-
forcement signal has been successfully applied for training a readout of ﬁring neurons from the reservoirs of
the same LSTM-type in [128].
Evolutionary algorithms are a natural candidate for training outputs in a non-supervised manner. Using
a genetic search with crossover and mutation to ﬁnd optimal output weights Wout of an ESN is reported in
[140]. Such an ESN is successfully applied for a hard reinforcement learning task of direct adaptive control,
replacing a classical indirect controller.
ESNs trained with a simple “(1+1)” evolution strategy for an unsupervised artiﬁcial embryogeny (the,
so-called, “ﬂag”) problem are shown to perform very well in [141].
An ESN trained with a state-of-art evolutionary continuous parameter optimization method (CMA-ES)
shows comparable performance in a benchmark double pole balancing problem to the best RNN topology-
learning methods in [68, 142]. For this problem the best results are obtained when the spectral radius ρ(W)
is adapted together with Wout. The same contributions also validate the CMA-ES readout training method
on a standard supervised prediction task, achieving the same excellent precision (MSE of the order 10−15)
as the state-of-art with linear regression. Conversely, the best results for this task were achieved with ρ(W)
ﬁxed and training only Wout. An even more curious ﬁnding is that almost as good results were achieved
by only adapting slopes of the reservoir activation functions f(·) and having Wout ﬁxed, as mentioned in
Section 7.2.
8.5. Multilayer readouts
Multilayer perceptrons (MLPs) as readouts, trained by error backpropagation, were used from the very
beginnings of LSMs [11] and ESNs (unpublished). They are theoretically more powerful and expressive in
their instantaneous mappings from x(n) to y(n) than linear readouts, and are thus suitable for particularly
nonlinear outputs, e.g., in [143, 144]. In both cases the MLP readouts are trained by error backpropagation.
On the other hand they are signiﬁcantly harder to train than an optimal single-layer linear regression, thus
often giving inferior results compared to the latter in practice.
Some experience in training MLPs as ESN readouts, including network initialization, using stochastic,
batch, and semi-batch gradients, adapting learning rates, and combining with regression-training of the last
layer of the MLP, is presented in Section 5.3 of [78].
8.6. Readouts with delays
While the readouts from reservoirs are usually recurrence-free, it does not mean that they may not have
memory. In some approaches they do, or rather some memory is inserted between the reservoir and the
readout.
Learning a delay for each neuron in an ESN reservoir x(n) in addition to the output weight from it is
investigated in [84]. Cross-correlation (simple or generalized) is used to optimally align activations of each
neuron in x(n) with ytarget(n), and then activations with the delays xdelayed(n) are used to ﬁnd Wout in
a usual way. This approach potentially enables utilizing the computational power of the reservoir more
eﬃciently. In a time-coded output from a spiking reservoir the output connection delays can actually be the
only thing that is learned [101].
23
For time series classiﬁcation tasks the decision can be based on a readout from a joined reservoir state
xjoined = [x(n1), x(n2), . . . , x(nk)] that is a concatenation of the reservoir states from diﬀerent moments
n1, n2, . . . , nk in time during the time series [18]. This approach, compared to only using the last state of
the given time series, moves the emphasis away from the ending of the series, depending on how the support
times ni are spread. It is also more expressive, since it has k times more trainable parameters in Wout for
the same size of the reservoir N. As a consequence, it is also more prone to overﬁtting. It is also possible to
integrate intervals of states in some way, e.g., use x∗(n1) =
1
n1−n0+1
Pn1
m=n0 x(m) instead of using a single
snapshot of the states x(n1).
An approach of treating a ﬁnite history of reservoir activations x(n) (similar to X in (12)) as a two-
dimensional image, and training a minimum average correlations energy ﬁlter as the readout for dynamical
pattern recognition is presented in [145].
Even though in Section 1 we stated that the RNNs considered in this survey are used as nonlinear ﬁlters,
which transform an input time series into an output time series, ESNs can also be utilized for non-temporal
(deﬁned in Section 2.1) tasks {(u(n), ytarget(n))} by presenting an ESN with the same input u(n) for many
time steps letting the ESN converge to a ﬁxed-point attractor state xu(n)(∞) (which it does if it possesses
echo state property) and reading the output from the attractor state y(n) = y(xu(n)(∞)) [146, 147].
8.7. Combining several readouts
Segmenting of the spatially embedded trajectory of x(n) by k-means clustering and assigning a separate
“responsible” linear readout for each cluster is investigated in [148]. This approach increases the expres-
siveness of the ESN by having k linear readouts trained and an online switching mechanism among them.
Bigger values of k are shown to compensate for smaller sizes Nx of the reservoirs to get the same level of
performance.
A benchmark-record-breaking approach of taking an average of outputs from many (1000) diﬀerent
instances of tiny (N = 4) trained ESNs is presented in Section 5.2.2 of [18]. The approach is also combined
with reading from diﬀerent support times as discussed in Section 8.6 of this survey. Averaging outputs over
20 instances of ESNs was also shown to reﬁne the prediction of chaotic time series in supporting online
material of [17].
Using dynamic programing to ﬁnd sequences in multiple sets of predicting readouts for classiﬁcation
[139] was already mentioned at the end of Section 8.3.
8.8. Hierarchies
Following the analogy between the ESNs and non-temporal kernel methods, ESNs would be called “type-1
shallow architectures” according to the classiﬁcation proposed in [149]. The reservoir adaptation techniques
reviewed in our article would make ESNs “type-3 shallow architectures”, which are more expressive. How-
ever, the authors in [149] argue that any type of shallow (i.e., non-hierarchical) architectures is incapable of
learning really complex intelligent tasks. This suggests that for demandingly complex tasks the adaptation
of a single reservoir might not be enough and a hierarchical architecture of ESNs might be needed, e.g.,
such as presented in [150]. Here the outputs of a higher level in the hierarchy serve as coeﬃcients of mixing
(or voting on) outputs from a lower one. The structure can have an arbitrary number of layers. Only the
outputs from the reservoirs of each layer are trained simultaneously, using stochastic gradient descent and
error backpropagation through the layers. The structure is demonstrated to discover features on diﬀerent
timescales in an unsupervised way when being trained for predicting a synthetic time series of interchang-
ing generators. On the downside, such hierarchies require many epochs to train, and suﬀer from a similar
problem of vanishing gradients, as deep feedforward neural networks or gradient-descent methods for fully
trained RNNs. They also do not scale-up yet to real-world demanding problems. Research on hierarchically
structured RC models has only just begun.
9. Discussion
The striking success of the original RC methods in outperforming fully trained RNNs in many (though
not all) tasks, established an important milestone, or even a turning point, in the research of RNN training.
24
The fact that a randomly generated ﬁxed RNN with only a linear readout trained consistently outperforms
state-of-art RNN training methods had several consequences:
• First of all it revealed that we do not really know how to train RNNs well, and something new is
needed. The error backpropagation methods, which had caused a breakthrough in feedforward neural
network training (up to a certain depth), and had also become the most popular training methods for
RNNs, are hardly unleashing their full potential.
• Neither are the classical RC methods yet exploiting the full potential of RNNs, since they use a random
RNN, which is unlikely to be optimal, and a linear readout, which is quite limited by the quality of
the signals it is combining. But they give a quite tough performance reference for more sophisticated
methods.
• The separation between the RNN reservoir and the readout provides a good platform to try out all
kinds of RNN adaptation methods in the reservoir and see how much they can actually improve the
performance over randomly created RNNs. This is particularly well suited for testing various biology-
inspired RNN adaptation mechanisms, which are almost exclusively local and unsupervised, in how
they can improve learning of a supervised task.
• In parallel, it enables all types of powerful non-temporal methods to be applied for reading out of the
reservoir.
This platform is the current paradigm of RC: using diﬀerent methods for (i) producing/adapting the reser-
voir, and (ii) training diﬀerent types of readouts. It enables looking for good (i) and (ii) methods inde-
pendently, and combining the best practices from both research directions. The platform has been actively
used by many researchers, ever since the ﬁrst ESNs and LSMs appeared. This research in both (i) and
(ii) directions, together with theoretical insights, like what characterizes a “good” reservoir, constitutes the
modern ﬁeld of RC.
In this review, together with motivating the new paradigm, we have provided a comprehensive survey of
all this RC research. We introduced a natural taxonomy of the reservoir generation/adaptation techniques
(i) with three big classes of methods (generic, unsupervised, and supervised), depending on their universality
with respect to the input and desired output of the task. Inside each class, methods are also grouped into
major directions of approaches, taking diﬀerent inspirations. We have also surveyed all types of readouts
from the reservoirs (ii) reported in the literature, including the ones containing several layers of nonlinearities,
combining several time steps, or several reservoirs, among others. We also brieﬂy discussed some practical
issues of training the most popular types of readouts in a tutorial-like fashion.
The survey is transcending the boundaries among several traditional methods that fall under the umbrella
of RC, generalizing the results to the whole RC ﬁeld and pointing out relations, where applicable.
Even though this review is quite extensive, we tried to keep it concise, outlining only the basic ideas
of each contribution. We did not try to include every contribution relating to RC in this survey, but only
the ones highlighting the main research directions. Publications only reporting applications of reservoir
methods, but not proposing any interesting modiﬁcations of them, were left out. Since this review is aimed
at a (fast) moving target, which RC is, some (especially very new) contributions might have been missed
unintentionally.
In general, the RC ﬁeld is still very young, but very active and quickly expanding. While the original
ﬁrst RC methods made an impact that could be called a small revolution, current RC research is more in a
phase of a (rapid) evolution. The multiple new modiﬁcations of the original idea are gradually increasing the
performance of the methods. While with no striking breakthroughs lately, the progress is steady, establishing
some of the extensions as common practices to build on further. There are still many promising directions
to be explored, hopefully leading to breakthroughs in the near future.
While the tasks for which RNNs are applied nowadays often are quite complex, hardly any of them could
yet be called truly intelligent, as compared to the human level of intelligence. The fact that RC methods
perform well in many of these simple tasks by no means indicates that there is little space left for their
25
improvement. More complex tasks and adequate solutions are still to meet each other in RC. We further
provide some of our (subjective, or even speculative) outlooks on the future of RC.
The elegant simplicity of the classical ESNs gives many beneﬁts in these simple applications, but it also
has some intrinsic limitations (as, for example, discussed in Section 5.4) that must be overcome in some way
or other. Since the RNN model is by itself biologically inspired, looking at real brains is a natural (literally)
source of inspiration on how to do that. RC models may reasonably explain some aspects of how small
portions of the brain work, but if we look at the bigger picture, the brain is far from being just a big blob
of randomly connected neurons. It has a complex structure that is largely predeﬁned before even starting
to learn. In addition, there are many learning mechanisms observed in the real brain, as brieﬂy outlined
in Section 6.2. It is very probable that there is no single easily implementable underlying rule which can
explain all learning.
The required complexity in the context of RC can be achieved in two basic ways: either (i) by giving
the reservoir a more complex internal structure, like that discussed in Section 5.3 or (ii) externally building
structures combining several reservoirs and readouts, like those discussed in Section 8.8. Note that the two
ways correspond to the above-mentioned dichotomy of the RC research and are not mutually exclusive. An
“externally” (ii) built structure can also be regarded as a single complex reservoir (i) and a readout from it
all can be trained.
An internal auto-structuring of the reservoir (i) through an (unsupervised) training would be conceptually
appealing and nature-like, but not yet quite feasible at the current state of knowledge. A robust realization
of such a learning algorithm would signify a breakthrough in the generation/training of artiﬁcial NNs. Most
probably such an approach would combine several competing learning mechanisms and goals, and require a
careful parameter selection to balance them, and thus would not be easy to successfully apply. In addition,
changing the structure of the RNN during the adaptive training would lead to bifurcations in the training
process, as in [8], which makes learning very diﬃcult.
Constructing external architectures or several reservoirs can be approached as more of an engineering
task. The structures can be hand-crafted, based on the speciﬁcs of the application, and, in some cases,
trained entirely supervised, each reservoir having a predeﬁned function and a target signal for its readout.
While such approaches are successfully being applied in practice, they are very case-speciﬁc, and not quite
in the scope of the research reviewed here, since in essence they are just applications of (several instances
of) the classical RC methods in bigger settings.
However, generic structures of multiple reservoirs (ii) that can be trained with no additional information,
such as discussed in Section 8.8, are of high interest. Despite their current state being still an “embryo”,
and the diﬃculties pointed out earlier, the authors of this review see this direction as highly promising.
Biological inspiration and progress of neuroscience in understanding how real brains work are beneﬁcial
for both (i) and (ii) approaches. Well understood natural principles of local neural adaptation and devel-
opment can be relatively easily transfered to artiﬁcial reservoirs (i), and reservoirs internally structured to
more closely resemble cortical microcolumns in the brain have been shown to perform better [23]. Under-
standing how diﬀerent brain areas interact could also help in building external structures of reservoirs (ii)
better suited for nature-like tasks.
In addition to processing and “understanding” multiple scales of time and abstraction in the data, which
hierarchical models promise to solve, other features still lacking in the current RC (and overall RNN) methods
include robustness and stability of pattern generation. A possible solution to this could be a homeostasis-like
self-regulation in the RNNs. Other intelligence-tending features as selective longer-term memory or active
attention are also not yet well incorporated.
In short, RC is not the end, but an important stepping-stone in the big journey of developing RNNs,
ultimately leading towards building artiﬁcial and comprehending natural intelligence.
Acknowledgments
This work is partially supported by Planet Intelligent Systems GmbH, a private company with an in-
spiring interest in fundamental research. The authors are also thankful to Benjamin Schrauwen, Michael
Thon, and an anonymous reviewer of this journal for their helpful constructive feedback.
26
References
[1] John J. Hopﬁeld. Hopﬁeld network. Scholarpedia, 2(5):1977, 2007.
[2] John J. Hopﬁeld. Neural networks and physical systems with emergent collective computational abilities. Proceedings of
National Academy of Sciences USA, 79:2554–2558, 1982.
[3] Geoﬀrey E. Hinton. Boltzmann machine. Scholarpedia, 2(5):1668, 2007.
[4] David H. Ackley, Geoﬀrey E. Hinton, and Terrence J. Sejnowski. A learning algorithm for Boltzmann machines. Cognitive
Science, 9:147–169, 1985.
[5] Geoﬀrey E. Hinton and Ruslan Salakhutdinov.
Reducing the dimensionality of data with neural networks.
Science,
313(5786):504–507, 2006.
[6] Graham W. Taylor, Geoﬀrey E. Hinton, and Sam Roweis. Modeling human motion using binary latent variables. In
Advances in Neural Information Processing Systems 19 (NIPS 2006), pages 1345–1352. MIT Press, Cambridge, MA,
2007.
[7] Ken-ichi Funahashi and Yuichi Nakamura. Approximation of dynamical systems by continuous time recurrent neural
networks. Neural Networks, 6:801–806, 1993.
[8] Kenji Doya. Bifurcations in the learning of recurrent neural networks. In Proceedings of IEEE International Symposium
on Circuits and Systems 1992, volume 6, pages 2777–2780, 1992.
[9] Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is diﬃcult.
IEEE Transactions on Neural Networks, 5(2):157–166, 1994.
[10] Felix A. Gers, J¨urgen Schmidhuber, and Fred A. Cummins. Learning to forget: continual prediction with LSTM. Neural
Computation, 12(10):2451–2471, 2000.
[11] Wolfgang Maass, Thomas Natschl¨ager, and Henry Markram. Real-time computing without stable states: a new framework
for neural computation based on perturbations. Neural Computation, 14(11):2531–2560, 2002.
[12] Herbert Jaeger. The “echo state” approach to analysing and training recurrent neural networks. Technical Report GMD
Report 148, German National Research Center for Information Technology, 2001.
[13] Peter F. Dominey. Complex sensory-motor sequence learning based on recurrent state representation and reinforcement
learning. Biological Cybernetics, 73:265–274, 1995.
[14] Jochen J. Steil. Backpropagation-decorrelation: recurrent learning with O(N) complexity. In Proceedings of the IEEE
International Joint Conference on Neural Networks, 2004 (IJCNN 2004), volume 2, pages 843–848, 2004.
[15] Herbert Jaeger, Wolfgang Maass, and Jos´e C. Pr´ıncipe. Special issue on echo state networks and liquid state machines
— Editorial. Neural Networks, 20(3):287–289, 2007.
[16] Herbert Jaeger. Echo state network. Scholarpedia, 2(9):2330, 2007.
[17] Herbert Jaeger and Harald Haas.
Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless
communication. Science, pages 78–80, 2004.
[18] Herbert Jaeger, Mantas Lukoˇseviˇcius, Dan Popovici, and Udo Siewert.
Optimization and applications of echo state
networks with leaky-integrator neurons. Neural Networks, 20(3):335–352, 2007.
[19] David Verstraeten, Benjamin Schrauwen, and Dirk Stroobandt. Reservoir-based techniques for speech recognition. In
Proceedings of the IEEE International Joint Conference on Neural Networks, 2006 (IJCNN 2006), pages 1050 – 1053,
2006.
[20] Wolfgang Maass, Thomas Natschl¨ager, and Henry Markram.
A model for real-time computation in generic neural
microcircuits.
In Advances in Neural Information Processing Systems 15 (NIPS 2002), pages 213–220. MIT Press,
Cambridge, MA, 2003.
[21] Wolfgang Maass, Prashant Joshi, and Eduardo D. Sontag. Principles of real-time computing with feedback applied to
cortical microcircuit models. In Advances in Neural Information Processing Systems 18 (NIPS 2005), pages 835–842.
MIT Press, Cambridge, MA, 2006.
[22] Dean V. Buonomano and Michael M. Merzenich. Temporal information transformed into a spatial code by a neural
network with realistic properties. Science, 267:1028–1030, 1995.
[23] Stefan Haeusler and Wolfgang Maass. A statistical analysis of information-processing properties of lamina-speciﬁc cortical
microcircuit models. Cerebral Cortex, 17(1):149–162, 2007.
[24] Uma R. Karmarkar and Dean V. Buonomano. Timing in the absence of clocks: encoding time in neural network states.
Neuron, 53(3):427–438, 2007.
[25] Garrett B. Stanley, Fei F. Li, and Yang Dan. Reconstruction of natural scenes from ensemble responses in the lateral
genicualate nucleus. Journal of Neuroscience, 19(18):8036–8042, 1999.
[26] Danko Nikoli´c, Stefan Haeusler, Wolf Singer, and Wolfgang Maass. Temporal dynamics of information content carried
by neurons in the primary visual cortex. In Advances in Neural Information Processing Systems 19 (NIPS 2006), pages
1041–1048. MIT Press, Cambridge, MA, 2007.
[27] Werner M. Kistler and Chris I. De Zeeuw. Dynamical working memory and timed responses: the role of reverberating
loops in the olivo-cerebellar system. Neural Computation, 14:2597–2626, 2002.
[28] Tadashi Yamazaki and Shigeru Tanaka. The cerebellum as a liquid state machine. Neural Networks, 20(3):290–297, 2007.
[29] Peter F. Dominey, Michel Hoen, Jean-Marc Blanc, and Ta¨ıssia Lelekov-Boissard. Neurological basis of language and
sequential cognition: evidence from simulation, aphasia, and ERP studies. Brain and Language, 86:207–225, 2003.
[30] Jean-Marc Blanc and Peter F. Dominey. Identiﬁcationof prosodic attitudes by atemporal recurrent network. Cognitive
Brain Research, 17:693–699, 2003.
[31] Peter F. Dominey, Michel Hoen, and Toshio Inui.
A neurolinguistic model of grammatical construction processing.
Journal of Cognitive Neuroscience, 18(12):2088–2107, 2006.
27
[32] Robert M. French. Catastrophic interference in connectionist networks. In L. Nadel, editor, Encyclopedia of Cognitive
Science, volume 1, pages 431–435. Nature Publishing Group, 2003.
[33] Floris Takens. Detecting strange attractors in turbulence. In Proceedings of a Symposium on Dynamical Systems and
Turbulence, volume 898 of LNM, pages 366–381. Springer, 1981.
[34] Ronald J. Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks.
Neural Computation, 1:270–280, 1989.
[35] David E. Rumelhart, Geoﬀrey E. Hinton, and Ronald J. Williams. Learning internal representations by error propagation.
In Neurocomputing: Foundations of research, pages 673–695. MIT Press, Cambridge, MA, USA, 1988.
[36] Paul J. Werbos. Backpropagation through time: what it does and how to do it. Proceedings of the IEEE, 78(10):1550–
1560, 1990.
[37] Amir F. Atiya and Alexander G. Parlos.
New results on recurrent network training:
unifying the algorithms and
accelerating convergence. IEEE Transactions on Neural Networks, 11(3):697–709, 2000.
[38] Gintaras V. Puˇskorius and Lee A. Feldkamp. Neurocontrol of nonlinear dynamical systems with Kalman ﬁlter trained
recurrent networks. IEEE Transactions on Neural Networks, 5(2):279–297, 1994.
[39] Sheng Ma and Chuanyi Ji. Fast training of recurrent networks based on the EM algorithm. IEEE Transactions on Neural
Networks, 9(1):11–26, 1998.
[40] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997.
[41] Herbert Jaeger.
Short term memory in echo state networks.
Technical Report GMD Report 152, German National
Research Center for Information Technology, 2002.
[42] Herbert Jaeger.
Tutorial on training recurrent neural networks, covering BPTT, RTRL, EKF and the “echo state
network” approach. Technical Report GMD Report 159, German National Research Center for Information Technology,
2002.
[43] Herbert Jaeger. Adaptive nonlinear system identiﬁcation with echo state networks. In Advances in Neural Information
Processing Systems 15 (NIPS 2002), pages 593–600. MIT Press, Cambridge, MA, 2003.
[44] Thomas Natschl¨ager, Henry Markram, and Wolfgang Maass. Computer models and analysis tools for neural microcircuits.
In R. K¨otter, editor, A Practical Guide to Neuroscience Databases and Associated Tools, chapter 9. Kluver Academic
Publishers (Boston), 2002.
[45] Wolfgang Maass, Thomas Natschl¨ager, and Henry Markram. Computational models for generic cortical microcircuits.
In J. Feng, editor, Computational Neuroscience: A Comprehensive Approach. CRC-Press, 2002.
[46] J¨urgen Schmidhuber, Daan Wierstra, Matteo Gagliolo, and Faustino J. Gomez. Training recurrent networks by Evolino.
Neural Computation, 19(3):757–779, 2007.
[47] Ulf D. Schiller and Jochen J. Steil. Analyzing the weight dynamics of recurrent learning algorithms. Neurocomputing,
63C:5–23, 2005.
[48] Jochen J. Steil. Memory in backpropagation-decorrelation O(N) eﬃcient online recurrent learning. In Proceedings of
the 15th International Conference on Artiﬁcial Neural Networks (ICANN 2005), volume 3697 of LNCS, pages 649–654.
Springer, 2005.
[49] Felix R. Reinhart and Jochen J. Steil. Recurrent neural autoassociative learning of forward and inverse kinematics for
movement generation of the redundant PA-10 robot. In Proceedings of the ECSIS Symposium on Learning and Adaptive
Behaviors for Robotic Systems (LAB-RS), volume 1, pages 35–40. IEEE Computer Society Press, 2008.
[50] Peter F. Dominey and Franck Ramus. Neural network processing of natural language: I. sensitivity to serial, temporal
and abstract structure of language in the infant. Language and Cognitive Processes, 15(1):87–127, 2000.
[51] Peter F. Dominey. From sensorimotor sequence to grammatical construction: evidence from simulation and neurophysi-
ology. Adaptive Behaviour, 13(4):347–361, 2005.
[52] Se Wang, Xiao-Jian Yang, and Cheng-Jian Wei. Harnessing non-linearity by sigmoid-wavelet hybrid echo state networks
(SWHESN). The 6th World Congress on Intelligent Control and Automation (WCICA 2006), 1:3014– 3018, 2006.
[53] David Verstraeten, Benjamin Schrauwen, and Dirk Stroobandt. Reservoir computing with stochastic bitstream neurons.
In Proceedings of the 16th Annual ProRISC Workshop, pages 454–459, Veldhoven, The Netherlands, 2005.
[54] Felix Sch¨urmann, Karlheinz Meier, and Johannes Schemmel. Edge of chaos computation in mixed-mode VLSI - a hard
liquid. In Advances in Neural Information Processing Systems 17 (NIPS 2004), pages 1201–1208. MIT Press, Cambridge,
MA, 2005.
[55] Benjamin Schrauwen, Michiel D‘Haene, Davﬁd Verstraeten, and Dirk Stroobandt. Compact hardware liquid state ma-
chines on FPGA for real-time speech recognition. Neural Networks, 21(2-3):511–523, 2008.
[56] Kristof Vandoorne, Wouter Dierckx, Benjamin Schrauwen, David Verstraeten, Roel Baets, Peter Bienstman, and Jan Van
Campenhout. Toward optical signal processing using photonic reservoir computing. Optics Express, 16(15):11182–11192,
2008.
[57] Chrisantha Fernando and Sampsa Sojakka. Pattern recognition in a bucket. In Proceedings of the 7th European Conference
on Advances in Artiﬁcial Life (ECAL 2003), volume 2801 of LNCS, pages 588–597. Springer, 2003.
[58] Ben Jones, Dov Stekelo, Jon Rowe, and Chrisantha Fernando. Is there a liquid state machine in the bacterium Escherichia
coli? In Procedings of the 1st IEEE Symposium on Artiﬁcial Life (ALIFE 2007), pages 187–191, 2007.
[59] David Verstraeten, Benjamin Schrauwen, Michiel D’Haene, and Dirk Stroobandt. An experimental uniﬁcation of reservoir
computing methods. Neural Networks, 20(3):391–403, 2007.
[60] Benjamin Schrauwen, David Verstraeten, and Jan Van Campenhout. An overview of reservoir computing: theory, appli-
cations and implementations. In Proceedings of the 15th European Symposium on Artiﬁcial Neural Networks (ESANN
2007), pages 471–482, 2007.
[61] Mantas Lukoˇseviˇcius and Herbert Jaeger. Overview of reservoir recipes. Technical Report No. 11, Jacobs University
28
Bremen, 2007.
[62] David H. Wolpert. The supervised learning no-free-lunch theorems. In Proceedings of the 6th Online World Conference
on Soft Computing in Industrial Applications (WSC 2006), pages 25–42, 2001.
[63] Michael Buehner and Peter Young. A tighter bound for the echo state property. IEEE Transactions on Neural Networks,
17(3):820–824, 2006.
[64] Duncan J. Watts and Steven H. Strogatz. Collective dynamics of ’small-world’ networks. Nature, 393:440–442, 1998.
[65] Albert-Laszlo Barabasi and Reka Albert. Emergence of scaling in random networks. Science, 286:509, 1999.
[66] Marcus Kaiser and Claus C. Hilgetag. Spatial growth of real-world networks. Physical Review E, 69:036103, 2004.
[67] Benjamin Liebald. Exploration of eﬀects of diﬀerent network topologies on the ESN signal crosscorrelation matrix spec-
trum. Bachelor’s thesis, Jacobs University Bremen, 2004. http://www.eecs.jacobs-university.de/archive/bsc-2004/
liebald.pdf.
[68] Fei Jiang, Hugues Berry, and Marc Schoenauer.
Supervised and evolutionary learning of echo state networks.
In
Proceedings of 10th International Conference on Parallel Problem Solving from Nature (PPSN 2008), volume 5199 of
LNCS, pages 215–224. Springer, 2008.
[69] Wolfgang Maass, Thomas Natschl¨ager, and Henry Markram. Computational models for generic cortical microcircuits.
In Computational Neuroscience: A Comprehensive Approach, pages 575–605. Chapman & Hall/CRC, 2004.
[70] David Verstraeten, Benjamin Schrauwen, Dirk Stroobandt, and Jan Van Campenhout. Isolated word recognition with
the liquid state machine: a case study. Information Processing Letters, 95(6):521–528, 2005.
[71] Michal ˇCerˇnansk´y and Matej Makula. Feed-forward echo state networks. In Proceedings of the IEEE International Joint
Conference on Neural Networks, 2005 (IJCNN 2005), volume 3, pages 1479–1482, 2005.
[72] Georg Fette and Julian Eggert. Short term memory and pattern matching with simple echo state network. In Proceedings
of the 15th International Conference on Artiﬁcial Neural Networks (ICANN 2005), volume 3696 of LNCS, pages 13–18.
Springer, 2005.
[73] David Verstraeten, Benjamin Schrauwen, and Dirk Stroobandt. Adapting reservoirs to get Gaussian distributions. In
Proceedings of the 15th European Symposium on Artiﬁcial Neural Networks (ESANN 2007), pages 495–500, 2007.
[74] Carlos Louren¸co. Dynamical reservoir properties as network eﬀects. In Proceedings of the 14th European Symposium on
Artiﬁcial Neural Networks (ESANN 2006), pages 503–508, 2006.
[75] Keith Bush and Batsukh Tsendjav. Improving the richness of echo state features using next ascent local search. In
Proceedings of the Artiﬁcial Neural Networks In Engineering Conference, pages 227–232, St. Louis, MO, 2005.
[76] M´arton Albert Hajnal and Andr´as L˝orincz.
Critical echo state networks.
In Proceedings of the 16th International
Conference on Artiﬁcial Neural Networks (ICANN 2006), volume 4131 of LNCS, pages 658–667. Springer, 2006.
[77] Yanbo Xue, Le Yang, and Simon Haykin.
Decoupled echo state networks with lateral inhibition.
Neural Networks,
20(3):365–376, 2007.
[78] Mantas Lukoˇseviˇcius. Echo state networks with trained feedbacks. Technical Report No. 4, Jacobs University Bremen,
2007.
[79] Danil V. Prokhorov, Lee A. Feldkamp, and Ivan Yu. Tyukin. Adaptive behavior with ﬁxed weights in RNN: an overview.
In Proceedings of the IEEE International Joint Conference on Neural Networks, 2002 (IJCNN 2002), pages 2018–2023,
2002.
[80] Mohamed Oubbati, Paul Levi, and Michael Schanz. Meta-learning for adaptive identiﬁcation of non-linear dynamical
systems. In Proceedings of the IEEE International Joint Symposium on Intelligent Control, pages 473–478, 2005.
[81] Mantas Lukoˇseviˇcius, Dan Popovici, Herbert Jaeger, and Udo Siewert. Time warping invariant echo state networks.
Technical Report No. 2, Jacobs University Bremen, 2006.
[82] Benjamin Schrauwen, Jeroen Defour, David Verstraeten, and Jan M. Van Campenhout. The introduction of time-scales
in reservoir computing, applied to isolated digits recognition. In Proceedings of the 17th International Conference on
Artiﬁcial Neural Networks (ICANN 2007), volume 4668 of LNCS, pages 471–479. Springer, 2007.
[83] Udo Siewert and Welf Wustlich. Echo-state networks with band-pass neurons: towards generic time-scale-independent
reservoir structures. Internal status report, PLANET intelligent systems GmbH, 2007. Available online at http://snn.
elis.ugent.be/.
[84] Georg Holzmann.
Echo state networks with ﬁlter neurons and a delay&sum readout.
Internal status report, Graz
University of Technology, 2007. Available online at http://grh.mur.at/data/misc.html.
[85] Francis wyﬀels, Benjamin Schrauwen, David Verstraeten, and Stroobandt Dirk.
Band-pass reservoir computing.
In
Z. Hou and N. Zhang, editors, Proceedings of the IEEE International Joint Conference on Neural Networks, 2008
(IJCNN 2008), pages 3204–3209, Hong Kong, 2008.
[86] Salah El Hihi and Yoshua Bengio. Hierarchical recurrent neural networks for long-term dependencies. In Advances in
Neural Information Processing Systems 8 (NIPS 1995), pages 493–499. MIT Press, Cambridge, MA, 1996.
[87] Nils Bertschinger and Thomas Natschl¨ager. Real-time computation at the edge of chaos in recurrent neural networks.
Neural Computation, 16(7):1413–1436, 2004.
[88] Benjamin Schrauwen, Lars Buesing, and Robert Legenstein. On computational power and the order-chaos phase transition
in reservoir computing. In Advances in Neural Information Processing Systems 21 (NIPS 2008), pages 1425–1432. MIT
Press, Cambridge, MA, 2009.
[89] Wolfgang Maass, Robert A. Legenstein, and Nils Bertschinger. Methods for estimating the computational power and
generalization capability of neural microcircuits. In Advances in Neural Information Processing Systems 17 (NIPS 2004),
pages 865–872. MIT Press, Cambridge, MA, 2005.
[90] Robert A. Legenstein and Wolfgang Maass. Edge of chaos and prediction of computational performance for neural circuit
models. Neural Networks, 20(3):323–334, 2007.
29
[91] Behrouz Farhang-Boroujeny. Adaptive Filters: Theory and Applications. Wiley, 1998.
[92] Herbert Jaeger. Reservoir riddles: suggestions for echo state network research. In Proceedings of the IEEE International
Joint Conference on Neural Networks, 2005 (IJCNN 2005), volume 3, pages 1460–1462, 2005.
[93] Jochen Triesch. A gradient rule for the plasticity of a neuron’s intrinsic excitability. In Proceedings of the 15th In-
ternational Conference on Artiﬁcial Neural Networks (ICANN 2005), volume 3696 of LNCS, pages 65–70. Springer,
2005.
[94] Melanie Mitchell, James P. Crutchﬁeld, and Peter T. Hraber.
Dynamics, computation, and the “edge of chaos”: a
re-examination. In G. Cowan, D. Pines, and D. Melzner, editors, Complexity: Metaphors, Models, and Reality, pages
497–513. Addison-Wesley, Reading, MA, 1994.
[95] Robert Legenstein and Wolfgang Maass. What makes a dynamical system computationally powerful?
In S. Haykin,
J. Pr´ıncipe, T. Sejnowski, and J. McWhirter, editors, New Directions in Statistical Signal Processing: From Systems to
Brain, pages 127–154. MIT Press, 2007.
[96] Mustafa C. Ozturk and Jos´e C. Pr´ıncipe. Computing with transiently stable states. In Proceedings of the IEEE Inter-
national Joint Conference on Neural Networks, 2005 (IJCNN 2005), volume 3, pages 1467–1472, 2005.
[97] Donald O. Hebb. The Organization of Behavior: A Neuropsychological Theory. Wiley, New York, 1949.
[98] ˇStefan Babinec and Jiˇr´ı Posp´ıchal. Improving the prediction accuracy of echo state neural networks by anti-Oja’s learning.
In Proceedings of the 17th International Conference on Artiﬁcial Neural Networks (ICANN 2007), volume 4668 of LNCS,
pages 19–28. Springer, 2007.
[99] Henry Markram, Yun Wang, and Misha Tsodyks.
Diﬀerential signaling via the same axon of neocortical pyramidal
neurons. Proceedings of National Academy of Sciences USA, 95(9):5323–5328, 1998.
[100] David Norton and Dan Ventura. Preparing more eﬀective liquid state machines using hebbian learning. Proceedings of
the IEEE International Joint Conference on Neural Networks, 2006 (IJCNN 2006), pages 4243–4248, 2006.
[101] H´el`ene Paugam-Moisy, Regis Martinez, and Samy Bengio. Delay learning and polychronization for reservoir computing.
Neurocomputing, 71(7-9):1143–1158, 2008.
[102] Roland Baddeley, Larry F. Abbott, Michael C. A. Booth, Frank Sengpeil, Toby Freeman, Edward A. Wakeman, and
Edmund T. Rolls. Responses of neurons in primary and inferior temporal visual cortices to natural scenes. Proc. R. Soc.
Lond. B, 264:1775–1783, 1997.
[103] Martin Stemmler and Christof Koch.
How voltage-dependent conductances can adapt to maximize the information
encoded by neuronal ﬁring rate. Nature Neuroscience, 2(6):521–527, 1999.
[104] Jochen Triesch. Synergies between intrinsic and synaptic plasticity in individual model neurons. In Advances in Neural
Information Processing Systems 17 (NIPS 2004), pages 1417–1424. MIT Press, Cambridge, MA, 2005.
[105] Anthony J. Bell and Terrence J. Sejnowski.
An information-maximization approach to blind separation and blind
deconvolution. Neural Computation, 7(6):1129–1159, 1995.
[106] Jochen J. Steil.
Online reservoir adaptation by intrinsic plasticity for backpropagation-decorrelation and echo state
learning. Neural Networks, 20(3):353–364, 2007.
[107] Marion Wardermann and Jochen J. Steil. Intrinsic plasticity for reservoir learning algorithms. In Proceedings of the 15th
European Symposium on Artiﬁcial Neural Networks (ESANN 2007), pages 513–518, 2007.
[108] Jochen J. Steil. Several ways to solve the MSO problem. In Proceedings of the 15th European Symposium on Artiﬁcial
Neural Networks (ESANN 2007), pages 489–494, 2007.
[109] Benjamin Schrauwen, Marion Wardermann, David Verstraeten, Jochen J. Steil, and Dirk Stroobandt. Improving reser-
voirs using intrinsic plasticity. Neurocomputing, 71(7-9):1159–1171, 2008.
[110] Joschka Boedecker, Oliver Obst, Norbert Michael Mayer, and Minoru Asada.
Studies on reservoir initialization and
dynamics shaping in echo state networks. In Proceedings of the 17th European Symposium on Artiﬁcial Neural Networks
(ESANN 2009), 2009. To appear.
[111] Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In Proceedings of the
37th Annual Allerton Conference on Communication, Control and Computing, pages 368–377, 1999.
[112] Taro Toyoizumi, Jean-Pascal Pﬁster, Kazuyuki Aihara, and Wulfram Gerstner. Generalized Bienenstock-Cooper-Munro
rule for spiking neurons that maximizes information transmission. Procedings of National Academy of Sciences USA,
102:5239–5244, 2005.
[113] Stefan Klampﬂ, Robert Legenstein, and Wolfgang Maass. Information bottleneck optimization and independent compo-
nent extraction with spiking neurons. In Advances in Neural Information Processing Systems 19 (NIPS 2006), pages
713–720. MIT Press, Cambridge, MA, 2007.
[114] Stefan Klampﬂ, Robert Legenstein, and Wolfgang Maass. Spiking neurons can learn to solve information bottleneck
problems and to extract independent components. Neural Computation, 21(4):911–959, 2008.
[115] Lars Buesing and Wolfgang Maass. Simpliﬁed rules and theoretical analysis for information bottleneck optimization and
PCA with spiking neurons. In Advances in Neural Information Processing Systems 20 (NIPS 2007), pages 193–200.
MIT Press, Cambridge, MA, 2008.
[116] Jochen Triesch. Synergies between intrinsic and synaptic plasticity mechanisms. Neural Computation, 19(4):885–909,
2007.
[117] Nicholas J. Butko and Jochen Triesch.
Learning sensory representations with intrinsic plasticity.
Neurocomputing,
70(7-9):1130–1138, 2007.
[118] Andreea Lazar, Gordon Pipa, and Jochen Triesch. Fading memory and time series prediction in recurrent networks with
diﬀerent forms of plasticity. Neural Networks, 20(3):312–322, 2007.
[119] Norbert M. Mayer and Matthew Browne.
Echo state networks and self-prediction.
In Revised Selected Papers of
Biologically Inspired Approaches to Advanced Information Technology (BioADIT 2004), pages 40–48, 2004.
30
[120] Mustafa C. Ozturk, Dongming Xu, and Jos´e C. Pr´ıncipe. Analysis and design of echo state networks. Neural Computation,
19(1):111–138, 2007.
[121] William H. Kautz. Transient synthesis in the time domain. IRE Transactions on Circuit Theory, 1(3):29–39, 1954.
[122] Kazuo Ishii, Tijn van der Zant, Vlatko Beˇcanovi´c, and Paul Pl¨oger. Identiﬁcation of motion with echo state network.
In Proceedings of the OCEANS 2004 MTS/IEEE – TECHNO-OCEAN 2004 Conference, volume 3, pages 1205–1210,
2004.
[123] John H. Holland. Adaptation in Natural and Artiﬁcial Systems: An Introductory Analysis with Applications to Biology,
Control and Artiﬁcial Intelligence. MIT Press, Cambridge, MA, USA, 1992.
[124] Ali Ajdari Rad, Mahdi Jalili, and Martin Hasler. Reservoir optimization in recurrent neural networks using Kronecker
kernels. In Proceedings of IEEE International Symposium on Circuits and Systems 2008, pages 868–871. IEEE, 2008.
[125] Xavier Dutoit, Hendrik Van Brussel, and Marnix Nutti. A ﬁrst attempt of reservoir pruning for classiﬁcation problems.
In Proceedings of the 15th European Symposium on Artiﬁcial Neural Networks (ESANN 2007), pages 507–512, 2007.
[126] Wolfgang Maass, Prashant Joshi, and Eduardo D. Sontag. Computational aspects of feedback in neural circuits. PLoS
Computational Biology, 3(1):e165+, 2007.
[127] Robert Legenstein, Dejan Pecevski, and Wolfgang Maass. Theoretical analysis of learning with reward-modulated spike-
timing-dependent plasticity. In Advances in Neural Information Processing Systems 20 (NIPS 2007), pages 881–888.
MIT Press, Cambridge, MA, 2008.
[128] Robert Legenstein, Dejan Pecevski, and Wolfgang Maass. A learning theory for reward-modulated spike-timing-dependent
plasticity with application to biofeedback. PLoS Computational Biology, 4(10):e1000180, 2008.
[129] ˚Ake Bj¨orck. Numerical Method for Least Squares Problems. SIAM, Philadelphia, PA, USA, 1996.
[130] Andrew Carnell and Daniel Richardson. Linear algebra for time series of spikes. In Proceedings of the 13th European
Symposium on Artiﬁcial Neural Networks (ESANN 2005), pages 363–368, 2005.
[131] Ali U. K¨u¸c¨ukemre. Echo state networks for adaptive ﬁltering. Master’s thesis, University of Applied Sciences Bohn-
Rhein-Sieg, Germany, 2006. http://www.faculty.jacobs-university.de/hjaeger/pubs/Kucukemre.pdf.
[132] Zhinwei Shi and Min Han. Support vector echo-state machine for chaotic time-series prediction. IEEE Transactions on
Neural Networks, 18(2):359–72, 2007.
[133] J¨urgen Schmidhuber, Matteo Gagliolo, Daan Wierstra, and Faustino J. Gomez. Evolino for recurrent support vector
machines. In Proceedings of the 14th European Symposium on Artiﬁcial Neural Networks (ESANN 2006), pages 593–598,
2006.
[134] Benjamin Schrauwen and Jan Van Campenhout. Linking non-binned spike train kernels to several existing spike train
metrics. In M. Verleysen, editor, Proceedings of the 14th European Symposium on Artiﬁcial Neural Networks (ESANN
2006), pages 41–46, Evere, 2006. d-side publications.
[135] Jochen J. Steil. Stability of backpropagation-decorrelation eﬃcient O(N) recurrent learning. In Proceedings of the 13th
European Symposium on Artiﬁcial Neural Networks (ESANN 2005), pages 43–48, 2005.
[136] Francis wyﬀels, Benjamin Schrauwen, and Dirk Stroobandt. Stable output feedback in reservoir computing using ridge
regression. In Proceedings of the 18th International Conference on Artiﬁcial Neural Networks (ICANN 2008), volume
5163 of LNCS, pages 808–817. Springer, 2008.
[137] Xavier Dutoit, Benjamin Schrauwen, Jan Van Campenhout, Dirk Stroobandt, Hendrik Van Brussel, and Marnix Nuttin.
Pruning and regularization in reservoir computing: a ﬁrst insight. In Proceedings of the 16th European Symposium on
Artiﬁcial Neural Networks (ESANN 2008), pages 1–6, 2008.
[138] Joe Tebelskis. Speech Recognition using Neural Networks. PhD thesis, School of Computer Science, Carnegie Mellon
University, Pittsburgh, Pennsylvania, 1995.
[139] Mark D. Skowronski and John G. Harris. Automatic speech recognition using a predictive echo state network classiﬁer.
Neural Networks, 20(3):414–423, 2007.
[140] Dongming Xu, Jing Lan, and Jos´e C. Pr´ıncipe. Direct adaptive control: an echo state network and genetic algorithm
approach. In Proceedings of the IEEE International Joint Conference on Neural Networks, 2005 (IJCNN 2005), volume 3,
pages 1483–1486, 2005.
[141] Alexandre Devert, Nicolas Bredeche, and Marc Schoenauer. Unsupervised learning of echo state networks: a case study
in artiﬁcial embryogeny. In Proceedings of the 8th International Conference on Artiﬁcial Evolution (EA 2007), volume
4926 of LNCS, pages 278–290. Springer, 2008.
[142] Fei Jiang, Hugues Berry, and Marc Schoenauer. Unsupervised learning of echo state networks: balancing the double pole.
In Proceedings of the 10th Genetic and Evolutionary Computation Conference (GECCO 2008), pages 869–870. ACM,
2008.
[143] Keith Bush and Charles Anderson. Modeling reward functions for incomplete state representations via echo state net-
works. In Proceedings of the IEEE International Joint Conference on Neural Networks, 2005 (IJCNN 2005), volume 5,
pages 2995–3000, 2005.
[144] ˇStefan Babinec and Jiˇr´ı Posp´ıchal. Merging echo state and feedforward neural networks for time series forecasting. In
Proceedings of the 16th International Conference on Artiﬁcial Neural Networks (ICANN 2006), volume 4131 of LNCS,
pages 367–375. Springer, 2006.
[145] Mustafa C. Ozturk and Jos´e C. Pr´ıncipe.
An associative memory readout for ESNs with applications to dynamical
pattern recognition. Neural Networks, 20(3):377–390, 2007.
[146] Mark Embrechts, Luis Alexandre, and Jonathan Linton. Reservoir computing for static pattern recognition. In Proceed-
ings of the 17th European Symposium on Artiﬁcial Neural Networks (ESANN 2009), 2009. To appear.
[147] Felix R. Reinhart and Jochen J. Steil. Attractor-based computation with reservoirs for online learning of inverse kine-
matics. In Proceedings of the 17th European Symposium on Artiﬁcial Neural Networks (ESANN 2009), 2009. To appear.
31
[148] Keith Bush and Charles Anderson.
Exploiting iso-error pathways in the N,k-plane to improve echo state network
performance, 2006.
[149] Yoshua Bengio and Yann LeCun. Scaling learning algorithms toward AI. In L. Bottou, O. Chapelle, D. DeCoste, and
J. Weston, editors, Large Scale Kernel Machines. MIT Press, Cambridge, MA, 2007.
[150] Herbert Jaeger. Discovering multiscale dynamical features with hierarchical echo state networks. Technical Report No.
9, Jacobs University Bremen, 2007.
32
