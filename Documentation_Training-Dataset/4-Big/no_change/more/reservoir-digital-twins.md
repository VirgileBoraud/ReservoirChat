Chaos 33, 033111 (2023); https://doi.org/10.1063/5.0138661
33, 033111
© 2023 Author(s).
Reservoir computing as digital twins for
nonlinear dynamical systems
Cite as: Chaos 33, 033111 (2023); https://doi.org/10.1063/5.0138661
Submitted: 13 December 2022 • Accepted: 13 February 2023 • Published Online: 07 March 2023
 Ling-Wei Kong, Yang Weng, Bryan Glaz, et al.
Chaos
ARTICLE
scitation.org/journal/cha
Reservoir computing as digital twins for nonlinear
dynamical systems
Cite as: Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
Submitted: 13 December 2022 · Accepted: 13 February 2023 ·
Published Online: 7 March 2023
View Online
Export Citation
CrossMark
Ling-Wei Kong,1
Yang Weng,1 Bryan Glaz,2 Mulugeta Haile,2
and Ying-Cheng Lai1,3,a)
AFFILIATIONS
1School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, Arizona 85287, USA
2Vehicle Technology Directorate, CCDC Army Research Laboratory, 2800 Powder Mill Road, Adelphi, Maryland 20783-1138, USA
3Department of Physics, Arizona State University, Tempe, Arizona 85287, USA
a)Author to whom correspondence should be addressed: Ying-Cheng.Lai@asu.edu
ABSTRACT
We articulate the design imperatives for machine learning based digital twins for nonlinear dynamical systems, which can be used to monitor
the “health” of the system and anticipate future collapse. The fundamental requirement for digital twins of nonlinear dynamical systems
is dynamical evolution: the digital twin must be able to evolve its dynamical state at the present time to the next time step without further
state input—a requirement that reservoir computing naturally meets. We conduct extensive tests using prototypical systems from optics,
ecology, and climate, where the respective specific examples are a chaotic CO2 laser system, a model of phytoplankton subject to seasonality,
and the Lorenz-96 climate network. We demonstrate that, with a single or parallel reservoir computer, the digital twins are capable of a
variety of challenging forecasting and monitoring tasks. Our digital twin has the following capabilities: (1) extrapolating the dynamics of the
target system to predict how it may respond to a changing dynamical environment, e.g., a driving signal that it has never experienced before,
(2) making continual forecasting and monitoring with sparse real-time updates under non-stationary external driving, (3) inferring hidden
variables in the target system and accurately reproducing/predicting their dynamical evolution, (4) adapting to external driving of different
waveform, and (5) extrapolating the global bifurcation behaviors to network systems of different sizes. These features make our digital twins
appealing in applications, such as monitoring the health of critical systems and forecasting their potential collapse induced by environmental
changes or perturbations. Such systems can be an infrastructure, an ecosystem, or a regional climate system.
Published under an exclusive license by AIP Publishing. https://doi.org/10.1063/5.0138661
Digital twins have attracted much attention recently in many
fields. This article develops machine learning based digital twins
for nonlinear dynamical systems subject to external forcing.
There are two assumptions underlying the situation of interest.
First, the governing equations are not known, and only measured
time series from the system are available. Second, none of the cur-
rently available sparse optimization methods for discovering the
system equations from data is applicable. The goal is to generate a
replica or twin of the system with reservoir computing machines,
a class of recurrent neural networks. Considering that, in nonlin-
ear dynamical systems, various bifurcations leading to chaos and
system collapse can take place, a basic requirement for the digital
twin is its ability to generate statistically correct evolution of the
system and accurate prediction of the bifurcations, especially the
critical bifurcations that are associated with catastrophic behav-
iors. A digital twin so designed can have significant applications,
such as monitoring the “health” of the target system in real time
and providing early warnings of critical bifurcations or events. In
terms of predictive problem solving, successful forecasting of a
system collapse in the future by the digital twin makes it possi-
ble to devise an optimal control strategy as an early intervention
to prevent the collapse. What machine-learning scheme can be
exploited for constructing digital twins for nonlinear dynami-
cal systems? Recall the basic property of any dynamical system:
its state naturally evolves forward in time according to a set of
rules which, mathematically, can be described by a set of differ-
ential equations or discrete-time maps. The basic requirement
for a digital twin is then that it must be able to evolve forward
in time without any state input. Reservoir computing, because
of its ability to execute closed-loop, dynamical self-evolution
with memory, stands out as the suitable choice for meeting this
requirement. This article demonstrates that, with a single or par-
allel reservoir computer, the digital twin is capable of challenging
forecasting and monitoring tasks for prototypical systems from
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-1
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
climate, optics, and ecology. It is also shown that the digital twin
is capable of the following tasks: extrapolating the dynamics of the
target system to external driving signals that it has never expe-
rienced before, making continual forecasting/monitoring with
sparse real-time updates under nonstationary external driving,
inferring hidden variables and accurately predicting their dynam-
ical evolution, adapting to different forms of external driving,
and extrapolating the global bifurcations to systems of different
sizes. These features make the reservoir computing based digi-
tal twins appealing in significant applications, such as monitor-
ing the health of critical systems and forecasting their potential
collapse induced by environmental changes.
I. INTRODUCTION
The concept of digital twins originated from aerospace engi-
neering for aircraft structural life prediction.1 In general, a digital
twin can be used for predicting and monitoring dynamical systems
and generating solutions of emergent behaviors that can poten-
tially be catastrophic.2 Digital twins have attracted much attention
from a wide range of fields,3 including medicine and health care.4,5
For example, the idea of developing medical digital twins in viral
infection through a combination of mechanistic knowledge, obser-
vational data, medical histories, and artificial intelligence has been
proposed recently,6 which can potentially lead to a powerful addition
to the existing tools to combat future pandemics. In a more dramatic
development, the European Union plans to fund the development of
digital twins of Earth for its green transition.7,8
The physical world is nonlinear. Many engineering systems,
such as complex infrastructural systems, are governed by nonlin-
ear dynamical rules too. In nonlinear dynamics, various bifurcations
leading to chaos and system collapse can take place.9 For example, in
ecology, environmental deterioration caused by global warming can
lead to slow parameter drift toward chaos and species extinction.10,11
In an electrical power system, voltage collapse can occur after a
parameter shift that lands the system in transient chaos.12 Various
climate systems in different geographic regions of the world are also
nonlinear and the emergent catastrophic behaviors as a result of
increasing human activities are of grave concern. In all these cases,
it is of interest to develop a digital twin of the system of interest to
monitor its “health” in real time as well as for predictive problem
solving in the sense that, if the digital twin indicates a possible sys-
tem collapse in the future, proper control strategies should and can
be devised and executed in time to prevent the collapse.
What does it take to create a digital twin for a nonlinear dynam-
ical system? For natural and engineering systems, there are two
general approaches: one is based on mechanistic knowledge and
another is based on observational data. In principle, if the detailed
physics of the system is well understood, it should be possible to
construct a digital twin through mathematical modeling. However,
there are two difficulties associated with this modeling approach.
First, a real-world system can be high-dimensional and complex,
preventing the rules governing its dynamical evolution from being
known at a sufficiently detailed level. Second, the hallmark of chaos
is sensitive dependence on initial conditions. Because no mathe-
matical model of the underlying physical system can be perfect,
the small deviations and high dimensionality of the system coupled
with environmental disturbances can cause the model predictions
of the future state of the system to be inaccurate and completely
irrelevant.13,14 These difficulties motivate the proposition that the
data-based approach can have advantages in many realistic scenarios
and a viable method to develop a digital twin is through data. While
in certain cases, approximate system equations can be found from
data through sparse optimization,15–17 the same difficulties with the
modeling approach arise. These considerations have led us to exploit
machine learning to create digital twins for nonlinear dynamical
systems.
What machine-learning scheme is suitable for digital twins of
nonlinear dynamical systems? The very basic characteristic of any
dynamical system is its ability to evolve over time. That is, from an
initial state, the system evolves in time by generating the state at the
next time step according to a set of dynamical rules. Mathemati-
cally, these rules can be described by a set of differential equations in
continuous time or maps in discrete time. Given a nonlinear dynam-
ical system, the basic requirement is then that its digital twin must
also be a dynamical system capable of self-evolution with memory.
Reservoir computing,18–20 a kind of recurrent neural network, nat-
urally stands out as a suitable candidate, which in recent years has
been extensively studied for predicting chaotic systems.21–42 More
specifically, with observational data as input, a reservoir computer
can be trained. The training phase corresponds to open-loop oper-
ation because of the external input. After training, the output of the
reservoir computer is connected to its input, creating a closed loop
that enables the neural network to update or evolve its state in time,
without input data. A properly trained reservoir computer can then
follow the evolution of the target dynamical system, acting as its
digital twin. Another advantage of reservoir computing is that no
backpropagation is needed for optimizing the parameters—only a
linear regression is required in the training, so it is computationally
efficient. A common situation is that the target system is subject to
external driving, such as a driven laser, a regional climate system, or
an ecosystem under external environmental disturbances. Accord-
ingly, the digital twin must accommodate a mechanism to control
or steer the dynamics of the neural network to account for external
driving. Introducing a control mechanism distinguishes our work
from existing ones in the literature of reservoir computing as applied
to nonlinear dynamical systems. Of particular interest is whether the
collapse of the target chaotic system can be anticipated from the dig-
ital twin. The purpose of this paper is to demonstrate that the digital
twin so created can accurately produce the bifurcation diagram of
the target system and faithfully mimic its dynamical evolution from
a statistical point of view. The digital twin can then be used to moni-
tor the present and future “health” of the system. More importantly,
with proper training from observational data, the twin can reliably
anticipate system collapses, providing early warnings of potentially
catastrophic failures of the system.
More specifically, using three prototypical systems from optics,
ecology, and climate, respectively, we demonstrate that the reser-
voir computing based digital twins developed in this paper solve the
following challenging problems: (1) extrapolation of the dynamical
evolution of the target system into certain “uncharted territories”
of the environmental condition with driving signals that it has never
experienced before, (2) long-term continual forecasting of nonlinear
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-2
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
dynamical systems subject to non-stationary external driving with
sparse state updates, (3) inference of hidden variables in the sys-
tem and accurate prediction of their dynamical evolution into the
future, (4) adaptation to external driving of different waveform, and
(5) extrapolation of the global bifurcation behaviors of network sys-
tems to some different sizes. These features make our digital twins
appealing in applications.
II. PRINCIPLE OF RESERVOIR COMPUTING BASED
DIGITAL TWIN
The basic construction of the digital twin of a nonlinear
dynamical system44 is illustrated in Fig. 1. It is essentially a recur-
rent reservoir computing neural network with a control mechanism,
which requires two types of input signals: the observational time
series for training and the control signal f(t) that remains in both the
training and self-evolving phase. The hidden layer hosts a random or
complex network of artificial neurons. During the training, the hid-
den recurrent layer is driven by both the input signal u(t) and the
control signal f(t). The neurons in the hidden layer generate a high-
dimensional nonlinear response signal. Linearly combining all the
responses of these hidden neurons with a set of trainable and opti-
mizable parameters yields the output signal. Specifically, the digital
twin consists of four components: (i) an input subsystem that maps
the low-dimensional (Din) input signal into a (high) Dr-dimensional
signal through the weighted Dr × Din matrix Win, (ii) a reservoir
network of Dr neurons characterized by Wr, a weighted network
matrix of dimension Dr × Dr, where Dr ≫Din, (iii) a readout sub-
system that converts the Dr-dimensional signal from the reservoir
network into a Dout-dimensional signal through the output weighted
matrix Wout, and (iv) a controller with the matrix Wc. The matrix Wr
defines the structure of the reservoir neural network in the hidden
layer, where the dynamics of each node are described by an internal
state and a nonlinear hyperbolic tangent activation function.
The matrices Win, Wc, and Wr are generated randomly prior to
training, whereas all elements of Wout are to be determined through
training. Specifically, the state updating equations for the training
and self-evolving phases are, respectively,
r(t + 1t) = (1 −α)r(t)
+ α tanh [Wrr(t) + Winu(t) + Wcf(t)],
(1)
r(t + 1t) = (1 −α)r(t)
+ α tanh [Wrr(t) + WinWoutr′(t) + Wcf(t)],
(2)
where r(t) is the hidden state, u(t) is the vector of input train-
ing data, 1t is the time step, the vector tanh (p) is defined to be
[tanh (p1), tanh (p2), . . .]T for a vector p = [p1, p2, . . .]T, and α is the
leakage factor. During the training, several trials of data are typi-
cally used under different driving signals so that the digital twin
can “sense, learn, and mingle” the responses of the target system
to gain the ability to extrapolate a response to a new driving sig-
nal that has never been encountered before. We input these trials of
training data, i.e., a few pairs of u(t) and the associated f(t), through
the matrices Win and Wc sequentially. Then, we record the state
FIG. 1. Basic structure of the digital twin of a chaotic system. It consists of three
layers: the input layer, the hidden recurrent layer, an output layer, as well as a con-
troller component. The input matrix Win maps the Din-dimensional input chaotic
data to a vector of much higher dimension Dr, where Dr ≫Din. The recurrent
hidden layer is characterized by the Dr × Dr weighted matrix Wr. The dynamical
state of the ith neuron in the reservoir is ri, for i = 1, . . . , Dr. The hidden-layer
state vector is r(t), which is an embedding of the input.43 The output matrix Wout
readout the hidden state into the Dout-dimensional output vector. The controller
provides an external driving signal f(t) to the neural network. During training, vec-
tor u(t) is the input data, and the blue arrow exists during the training phase only.
In the predicting phase, the output vector v(t) is directly fed back to the input layer,
generating a closed-loop, self-evolving dynamical system, as indicated by the red
arrow connecting v(t) to u(t). The controller remains on in both the training and
predicting phases.
vector r(t) of the neural network during the entire training phase
as a matrix R. We also record all the desired output, which is the
one-step prediction result v(t) = u(t + 1t), as matrix V. To make
the readout nonlinear and to avoid unnecessary symmetries in the
system,24,45 we change matrix R into R′ by squaring the entries of
even dimensions in the states of the hidden layer. [The vector (r′(t)
in Eq. (2) is defined in a similar way.] We carry out a linear regres-
sion between V and R′, with a ℓ-2 regularization coefficient β, to
determine the readout matrix,
Wout = V · R′T(R′ · R′T + βI)
−1.
(3)
To achieve acceptable learning performance, optimization of hyper-
parameters is necessary. The four widely used global optimization
methods are genetic algorithm,46–48 particle swarm optimization,49,50
Bayesian optimization,51,52 and surrogate optimization.53–55 We use
the surrogate optimization (the algorithm surrogateopt in Matlab).
The hyperparameters that are optimized include d—the average
degree of the recurrent network in the hidden layer, λ—the spec-
tral radius of the recurrent network, kin—the scaling factor of Win,
kc—the scaling of Wc, α—the leakage factor, and β—the ℓ-2 regu-
larization coefficient. The neural network is validated using the same
driving f(t) as in the training phase, but driving signals with differ-
ent amplitudes and frequencies are used in the testing phase. Prior to
making predictions, the neural network is initialized using random
short segments of the training data, so no data from the target sys-
tem under the testing driving signals f(t) are required. To produce
the bifurcation diagram, sufficiently long transients in the dynamical
evolution of the neural network are disregarded.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-3
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
III. RESULTS
For clarity, we present results on the digital twin for a pro-
totypical nonlinear dynamical systems with adjustable phase-space
dimension: the Lorenz-96 climate network model.56 In Appendixes,
we present two additional examples: a chaotic laser (Appendix A)
and a driven ecological system (Appendix B), together with a num-
ber of pertinent issues.
A. Low-dimensional Lorenz-96 climate network and
its digital twin
The Lorenz-96 system56 is an idealized atmospheric climate
model. Mathematically, the toy climate system is described by m
coupled first-order nonlinear differential equations subject to exter-
nal periodic driving f(t),
dxi
dt = xi−1(xi+1 −xi−2) −xi + f(t),
(4)
where i = 1, . . . , m is the spatial index. Under the periodic bound-
ary condition, the m nodes constitute a ring network, where each
node is coupled to three neighboring nodes. To be concrete, we set
m = 6 (more complex high-dimensional cases are treated below).
The driving force is sinusoidal with a bias F: f(t) = A sin(ωt) + F.
We fix ω = 2 and F = 2, and use the forcing amplitude A as the
bifurcation parameter. For relatively large values of A, the system
exhibits chaotic behaviors, as exemplified in Fig. 2(a1) for A = 2.2.
Quasi-periodic dynamics arise for smaller values of A, as exemplified
in Fig. 2(a2). As A decreases from a large value, a critical transition
from chaos to quasi-periodicity occurs at Ac ≈1.9. We train the dig-
ital twin with time series from four values of A, all in the chaotic
regime: A = 2.2, 2.6, 3.0, and 3.4. The size of the random reservoir
network is Dr = 1, 200. For each value of A in the training set, the
training and validation lengths are t = 2500 and t = 12, respectively,
where the latter corresponds to approximately five Lyapunov times.
The warming-up length is t = 20 and the time step of the reser-
voir dynamical evolution is 1t = 0.025. The hyperparameter values
(please refer to Sec. II for their meanings) are optimized to be d =
843, λ = 0.48, kin = 0.29, kc = 0.113, α = 0.41, and β = 1 × 10−10.
Our computations reveal that, for the deterministic version of the
Lorenz-96 model, it is difficult to reduce the validation error below
a small threshold. However, adding an appropriate amount of noise
into the training time series18 can lead to smaller validation errors.
We add an additive Gaussian noise with standard deviation σnoise
to each input data channel to the reservoir network [including the
driving channel f(t)]. The noise amplitude σnoise is treated as an addi-
tional hyperparameter to be optimized. For the toy climate system,
we test several noise levels and find the optimal noise level giving the
best validating performance: σnoise ≈10−3.
Figures 2(b1) and 2(b2) show the dynamical behaviors gener-
ated by the digital twin for the same values of A as in Figs. 2(a1) and
2(a2), respectively. It can be seen that not only does the digital twin
produce the correct dynamical behavior in the same chaotic regime
where the training is carried out, it can also extrapolate beyond the
training parameter regime to correctly predict the unseen system
dynamics there (quasiperiodicity in this case). To provide support
in a broader parameter range, we calculate a true bifurcation dia-
gram, as shown in Fig. 2(c), where the four vertical dashed lines
indicate the four values of the training parameter. The bifurcation
diagram generated by the digital twin is shown in Fig. 2(d), which
agrees reasonably well with the true diagram. Note that the digi-
tal twin fails to predict the periodic window about A = 3.2, due to
its high period (period-21—see Appendix H for a discussion). To
quantify the prediction performance, we examine the smallest sim-
ple connected region that encloses the entire attractor—the spanned
region, and calculate the overlapping ratio of the true to the pre-
dicted spanned regions. Figure 2(e) shows the relative error of the
spanned regions (RESR) vs A, where the spanned regions are calcu-
lated from a two-dimensional projection of the attractor. Except for
the locations of two periodic windows, RESR is within 4%. When the
testing values of A are further away from the training values, RESR
tends to increase.
Previously, it was suggested that reservoir computing can have
a certain degree of extrapolability.34–39 Figure 2 represents an exam-
ple where the target system’s response is extrapolated to external
sinusoidal driving with unseen amplitudes. In general, extrapolation
is a difficult problem. Some limitations of the extrapolability with
respect to the external driving signal are discussed in Appendix A,
where the digital twin can predict the crisis point but cannot extrap-
olate the asymptotic behavior after the crisis.
In the following, we systematically study the applicability of the
digital twin in solving forecasting problems in more complicated sit-
uations than the basic settings demonstrated in Fig. 2. The issues to
be addressed are high dimensionality, the effect of the waveform of
the driving on forecasting, and the generalizability across Lorenz-
96 networks of different sizes. Results of continual forecasting and
inferring hidden dynamical variables using only rare updates of the
observable are presented in Appendixes C and D.
B. Digital twins of parallel reservoir-computing neural
networks for high-dimensional Lorenz-96 climate
networks
We extend the methodology of digital twin to high-
dimensional Lorenz-96 climate networks, e.g., m = 20. To deal with
such a high-dimensional target system, if a single reservoir system
is used, the required size of the neural network in the hidden layer
will be too large to be computationally efficient. We, thus, turn to
the parallel configuration25 that consists of many small-size neural
networks, each “responsible” for a small part of the target system.
For the Lorenz-96 network with m = 20 coupled nodes, our digi-
tal twin consists of ten parallel neural networks, each monitoring
and forecasting the dynamical evolution of two nodes (Dout = 2).
Because each node in the Lorenz-96 network is coupled to three
nearby nodes, we set Din = Dout + Dcouple = 2 + 3 = 5 to ensure
that sufficient information is supplied to each neural network.
The specific parameters of the digital twin are as follows. The
size of the recurrent layer is Dr = 1, 200. For each training value
of the forcing amplitude A, the training and validation lengths are
t = 3500 and t = 100, respectively. The “warming-up” length is
t = 20 and the time step of the dynamical evolution of the digi-
tal twin is 1t = 0.025. The optimized hyperparameter values are
d = 31, λ = 0.75, kin = 0.16, kc = 0.16, α = 0.33, β = 1 × 10−12,
and σnoise = 10−2.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-4
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 2. Digital twin of the Lorenz-96 climate system. The toy climate system is described by six coupled ﬁrst-order nonlinear differential equations (phase-space dimension
m = 6), which is driven by a sinusoidal signal f(t) = A sin(ωt) + F. (a1) and (a2) Ground truth: chaotic and quasi-periodic dynamics in the system for A = 2.2 and A = 1.6,
respectively, for ω = 2 and F = 2. The sinusoidal driving signals f(t) are schematically illustrated. (b1) and (b2) The corresponding dynamics of the digital twin under the
same driving signal f(t). Training of the digital twin is conducted using time series from the chaotic regime. The result in (b2) indicates that the digital twin is able to extrapolate
outside the chaotic regime to generate the unseen quasi-periodic behavior. (c) and (d) True and digital twin generated bifurcation diagrams of the toy climate system, where
the four vertical red dashed lines indicate the values of driving amplitudes A, from which the training time series data are obtained. The reasonable agreement between the
two bifurcation diagrams attests to the ability of the digital twin to reproduce the distinct dynamical behaviors of the target climate system in different parameter regimes, even
with training data only in the chaotic regime. (e) Relative error of the spanned regions (RESR) vs A. The error is within 4%, except for the locations of two periodic windows
at which the large errors are due to long transients see Appendix H.
The periodic signal used to drive the Lorenz-96 climate net-
work of 20 nodes is f(t) = A sin(ωt) + F with ω = 2, and F = 2. The
structure of the digital twin consists of 20 small neural networks
as illustrated in Fig. 3(a). Figures 3(b1) and 3(b2) show a chaotic
and a periodic attractor for A = 1.8 and A = 1.6, respectively, in the
(x1, x2) plane. Training of the digital twin is conducted by using four
time series from four different values of A, all in the chaotic regime.
The attractors generated by the digital twin for A = 1.8 and A = 1.6
are shown in Figs. 3(c1) and 3(c2), respectively, which agree well
with the ground truth. Figure 3(d) shows the bifurcation diagram
of the target system (the ground truth), where the four values of A:
A = 1.8, 2.2, 2.6, and 3.0, from which the training chaotic time series
are obtained, are indicated by the four respective vertical dashed
lines. The bifurcation diagram generated by the digital twin is shown
in Fig. 3(e), which agrees well with the ground truth in Fig. 3(d).
Figure 3(f) shows the relative error RESR vs A, where a peak occurs
at A ≈1.1 due to the mismatched ending point of the large periodic
window.
C. Digital twins under external driving with varied
waveform
The external driving signal is an essential ingredient in our
articulation of the digital twin, which is particularly relevant to crit-
ical systems of interest such as the climate systems. In applications,
the mathematical form of the driving signal may change with time.
Can a digital twin produce the correct system behavior under a driv-
ing signal that is different than the one it has “seen” during the
training phase? Note that, in the examples treated so far, it has been
demonstrated that our digital twin can extrapolate the dynamical
behavior of a target system under a driving signal of the same math-
ematical form but with a different amplitude. Here, the task is more
challenging as the form of the driving signal has changed.
As a concrete example, we consider the Lorenz-96 climate net-
work of m = 6 nodes, where a digital twin is trained with a purely
sinusoidal signal f(t) = A sin(ωt) + F, as illustrated in the left col-
umn of Fig. 4(a). During the testing phase, the driving signal has
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-5
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 3. Digital twin consisting of a number of parallel reservoir-computing neural networks for high-dimensional chaotic systems. The target system is the Lorenz-96 climate
network of m = 20 nodes, subject to a global periodic driving f(t) = A sin(ωt) + F. (a) The structure of the digital twin, where each ﬁlled green circle represents a small
neural network with the input dimension Din = 5 and output dimension Dout = 2. (b1) and (b2) A chaotic and periodic attractor in a two-dimensional subspace of the target
system for A = 1.8 and A = 1.6, respectively, for ω = 2 and F = 2. (c1) and (c2) The attractors generated by the digital twin corresponding to those in (b1) and (b2),
respectively, where the training is done using four time series from four different values of forcing amplitude A, all in the chaotic regime. The digital twin with a parallel
structure is able to successfully extrapolate the unseen periodic behavior with completely chaotic training data. (d) and (e) The true and digital twin generated bifurcation
diagrams, respectively, where the four vertical dashed lines in (c) specify the four values of A from which the training time series are obtained. (f) RESR vs A, where the peak
at A ≈1.1 is due to the mismatched ending point of the wide periodic window for A ∈(1.2, 1.7).
the form of the sum of two sinusoidal signals with different fre-
quencies: f(t) = A1 sin(ω1t) + A2 sin(ω2t + 1φ) + F, as illustrated
in the right panel of Fig. 4(a). We set A1 = 2, A2 = 1, ω1 = 2,
ω2 = 1, F = 2, and use 1φ as the bifurcation parameter. The param-
eter setting for reservoir computing is the same as that in Fig. 2.
The training and validating lengths for each driving amplitude A
value are t = 3000 and t = 12, respectively. We find that this setting
prevents the digital twin from generating an accurate bifurcation
diagram, but a small amount of dynamical noise to the target system
can improve the performance of the digital twin. To demonstrate
this, we apply an additive noise term to the driving signal f(t) in
the training phase: df(t)/dt = ωA cos(ωt) + δDNξ(t), where ξ(t) is
a Gaussian white noise of zero mean and unit variance, and δDN
is the noise amplitude (e.g., δDN = 3 × 10−3). We use the second-
order Heun method57 to solve the stochastic differential equations
describing the target Lorenz-96 system. Intuitively, the noise serves
to excite different modes of the target system to instill richer infor-
mation into the training time series, making the process of learning
the target dynamics more effective. Figures 4(b) and 4(c) show the
actual and digital twin generated bifurcation diagrams. Although the
digital twin encountered driving signals in a completely “uncharted
territory,” it is still able to generate the bifurcation diagram with
reasonable accuracy. The added dynamical noise is creating small
fluctuations in the driving signal f(t). This may yield richer excited
dynamical features of the target system in the training data set,
which can be learned by the neural network. This should be ben-
eficial for the neural network to adapt to different waveforms in
the testing. Additional results with varied testing waves f(t) are
presented in Appendix E.
D. Extrapolability of digital twin with respect to
system size
In the examples studied so far, it has been demonstrated that
our reservoir computing based digital twin has a strong extrapola-
bility in certain dimensions of the parameter space. Specifically, the
digital twin trained with time series data from one parameter region
can follow the dynamical evolution of the target system in a different
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-6
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 4. Effects of waveform change in the external driving on the performance
of the digital twin. The time series used to train the digital twin are from the
target system subject to external driving of a particular waveform. A change in
the waveform occurs subsequently, leading to a different driving signal during
the testing phase. (a) During the training phase, the driving signal is of the form
f(t) = A sin(ωt) + F and time series from four different values of A are used for
training the digital twin. The right panel illustrates an example of the changed driv-
ing signal during the testing phase. (b) The true bifurcation diagram of the target
system under a testing driving signal. (c) The bifurcation diagram generated by
the digital twin, facilitated by an optimal level of training noise determined through
hyperparameter optimization.
parameter regime. One question is whether the digital twin possesses
certain extrapolability in the system size. For example, consider the
Lorenz-96 climate network of size m. In Fig. 3, we use an array of
parallel neural networks to construct a digital twin for the climate
FIG. 5. Demonstration of extrapolability of digital twin in the system size. (a) The
digital twin is trained using time series from the Lorenz-96 climate networks of size
m = 6 and m = 10. The target climate system is subject to a sinusoidal driving
f(t) = A sin(ωt) + F, and the training time series data are from the A values
marked by the eight vertical orange dashed lines. (b) The true bifurcation diagrams
of the target climate network of size m = 4 and m = 12. (c) The corresponding
digital twin generated bifurcation diagrams, where the twin consists of m/2 parallel
neural networks, each taking input from two nodes in the target system and from
the nodes in the network that are coupled to the two nodes.
network of a fixed size m, where the number of parallel reservoir
computers is m/2 (assuming that m is even), and training and test-
ing/monitoring are carried out for the same system size. We ask, if
a digital twin is trained for climate networks of certain sizes, will it
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-7
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
have the ability to generate the correct dynamical behaviors for cli-
mate networks of different sizes? If yes, we say that the digital twin
has extrapolability with respect to system size.
As an example, we create a digital twin with time series data
from Lorenz-96 climate networks of sizes m = 6 and m = 10, as
shown in Fig. 5(a). For each system size, four values of the forcing
amplitude A are used to generate the training time series: A =1.5,
2.0, 2.5, and 3.0, as marked by the vertical orange dashed lines
in Figs. 5(a) and 5(b). As in Fig. 3, the digital twin consists of
m/2 parallel neural networks, each of size Dr = 1500. The opti-
mized hyperparameter values are determined to be d = 927, λ =
0.71, kin = 0.076, kc = 0.078, α = 0.27, β = 1 × 10−11, and σnoise =
3 × 10−3. Then, we consider climate networks of two different sizes:
m = 4 and m = 12, and test if the trained digital twin can be adapted
to the new systems. For the network of size m = 4, we keep only
two parallel neural networks for the digital twin. For m = 12, we
add one additional neural network to the trained digital twin for
m = 10, so the new twin consists of six parallel neural networks with
identical matrices. The true bifurcation diagrams for the climate
system of sizes m = 4 and m = 12 are shown in Fig. 5(b) (the left
and right panels, respectively). The corresponding bifurcation dia-
grams generated by the adapted digital twins are shown in Fig. 5(c),
which agree with the ground truth reasonably well, demonstrating
that our reservoir computing based digital twin possesses certain
extrapolability in system size.
IV. DISCUSSION
We have articulated the principle of creating digital twins for
nonlinear dynamical systems based on reservoir computers that
are recurrent neural networks. The basic consideration leading us
to choose reservoir computing as the suitable machine-learning
scheme is its ability to execute dynamical self-evolution in closed-
loop operation. In general, reservoir computing is a powerful neural
network framework that does not require backpropagation during
training but only a linear regression is needed. This feature makes
the development of digital twins based on reservoir computing com-
putationally efficient. We have demonstrated that a well-trained
reservoir computer is able to serve as a digital twin for systems
subject to external, time-varying driving. The twin can be used to
anticipate possible critical transitions or regime shifts in the target
system as the driving force changes, thereby providing early warn-
ings for potential catastrophic collapse of the system. We have used
a variety of examples from different fields to demonstrate the work-
ings and the anticipating power of the digital twin, which include
the Lorenz-96 climate network of different sizes (in the main text),
a driven chaotic CO2 laser system (Appendix A), and an ecologi-
cal system (Appendix B). For low-dimensional nonlinear dynamical
systems, a single neural network is sufficient for the digital twin. For
high-dimensional systems such as the climate network of a relatively
large size, parallel neural networks can be integrated to construct the
digital twin. At the level of the detailed state evolution, our recur-
rent neural network based digital twin is essentially a dynamical
twin system that evolves in parallel to the real system, and the evo-
lution of the digital twin can be corrected from time to time using
sparse feedback of data from the target system (Appendix C). In
certain circumstances, continual forecasting in the presence of hid-
den dynamical variables is possible (Appendix D), and digital twins
under external driving with varied waveforms can be constructed
(Sec. III C and Appendix E). In applications where direct measure-
ments of the target system are not feasible or are too costly, the
digital twin provides a way to assess the dynamical evolution of
the target system. Qualitatively, the digital twin can faithfully repro-
duce the attractors of the target system, e.g., chaotic, periodic, or
quasiperiodic, without the need for state updating. In addition, we
show that the digital twin is able to accurately predict a critical bifur-
cation point and the average lifetime of transient chaos that occurs
after the bifurcation, even under a driving signal that is different
from that during the training (Appendix F). The issues of robust-
ness against dynamical and observational noises in the training data
(Appendix G) and the effect of long transients in periodic windows
of a high period (Appendix H) have also been treated.
To summarize, our reservoir computing based digital twins are
capable of performing the following tasks: (1) extrapolating cer-
tain dynamical evolution of the target system under external driving
conditions unseen during training, (2) making long-term contin-
ual forecasting of nonlinear dynamical systems under nonstationary
external driving with sparse state updates, (3) inferring the existence
of hidden variables in the system and reproducing/predicting their
dynamical evolution, (4) adapting to external driving of different
waveform, and (5) extrapolating the global bifurcation behaviors to
systems of different sizes.
Our design of the digital twins for nonlinear dynamical systems
can be extended in a number of ways.
A. Online learning
Online or continual learning is a recent trend in machine-
learning research. Unlike the approach of batch learning, where
one gathers all the training data in one place and does the train-
ing on the entire data set (the way by which training is conducted
for our work), in an online learning environment, one evolves the
machine-learning model incrementally with the flow of data. For
each training step, only the newest inputted training data are used
to update the machine-learning model. When a new data set is
available, it is not necessary to train the model over again on the
entire data set accumulated so far, but only on the new set. This can
result in a significant reduction in computational complexity. Pre-
viously, an online learning approach to reservoir computing known
as the FORCE learning was developed.58 An attempt to deal with
the key problem of online learning termed “catastrophic forgetting”
was made in the context of reservoir computing.59 Further investi-
gation is required to determine if these methods can be exploited for
creating digital twins through online learning.
B. Beyond reservoir computing
Second, the potential power of recurrent neural network based
digital twin may be further enhanced by using more sophisticated
recurrent neural network models depending on the target prob-
lem. We use reservoir computing because it is relatively simple
yet powerful enough for both low- and high-dimensional dynam-
ical systems. Schemes, such as knowledge-based hybrid reservoir
computing60 or ODE-nets,61 are worth investigating.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-8
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
C. Reinforcement learning
Is it possible to use digital twins to make reinforcement
learning feasible in situations where the target system cannot be
“disturbed”? Particularly, reinforcement learning requires constant
interaction with the target system during training so that the
machine can learn from its mistakes and successes. However, for
a real-world system, these interactions may be harmful, uncontrol-
lable, and irreversible. As a result, reinforcement learning algorithms
are rarely applied to safety-critical systems.62 In this case, digital
twins can be beneficial. By building a digital twin, the reinforcement
learning model does not need to interact with the real system, but
with its simulated replica for efficient training. This area of research
is called model-based reinforcement learning.63
D. Potential beneﬁts of noise
A phenomenon uncovered in our study is the beneficial role of
dynamical noise in the target system. As briefly discussed in Fig. 4,
adding dynamic noise in the training dataset enhances the digital
twin’s ability to extrapolate the dynamics of the target system with
different waveforms of driving. Intuitively, noise can facilitate the
exploration of the phase space of the target nonlinear system. A
systematic study of the interplay between dynamical noise and the
performance of the digital twin is worthy.
E. Extrapolability
The demonstrated extrapolability of our digital twin, albeit lim-
ited, may open the door to forecasting the behavior of large systems
using twins trained on small systems. Much research is needed to
address this issue.
F. Spatiotemporal dynamical systems with
multistability
We have considered digital twins for a class of coupled dynam-
ical systems: the Lorenz-96 climate model. When developing digital
twins for spatiotemporal dynamical systems, two issues can arise.
One is the computational complexity associated with such high-
dimensional systems. We have demonstrated that parallel reservoir
computing provides a viable solution. Another issue is multistabil-
ity. Spatiotemporal dynamical systems, in general, exhibit extremely
rich dynamical behaviors such as chimera states.64–72 Developing
digital twins of spatiotemporal dynamical systems with multiple
coexisting states requires that the underlying recurrent neural net-
works possess certain memory capabilities. Developing methods to
incorporate memories into digital twins is a problem of current
interest.
ACKNOWLEDGMENTS
We thank Z.-M. Zhai and A. Flynn for discussions. This work
was supported by the Army Research Office through Grant No.
W911NF-21-2-0055 and by the U.S.-Israel Energy Center managed
by the Israel-U.S. Binational Industrial Research and Development
(BIRD) Foundation.
AUTHOR DECLARATIONS
Conﬂict of Interest
The authors have no conflicts to disclose.
Author Contributions
Ling-Wei Kong: Conceptualization (equal); Data curation (equal);
Formal analysis (equal); Investigation (equal); Validation (equal);
Visualization (equal); Writing – original draft (equal). Yang Weng:
Conceptualization (supporting); Funding acquisition (supporting);
Writing – review & editing (supporting). Bryan Glaz: Conceptu-
alization (supporting); Writing – review & editing (supporting).
Mulugeta Haile: Conceptualization (supporting); Writing – review
& editing (supporting). Ying-Cheng Lai: Conceptualization (equal);
Funding acquisition (equal); Investigation (equal); Methodology
(equal); Project administration (equal); Supervision (equal); Writing
– original draft (equal); Writing – review & editing (equal).
DATA AVAILABILITY
The data that support the findings of this study are available
from the corresponding author upon reasonable request.
APPENDIX A: A DRIVEN CHAOTIC LASER SYSTEM
We consider the single-mode, class B, driven chaotic CO2 laser
system73–76 described by
du
dt = −u[f(t) −z],
(A1)
dz
dt = ϵ1z −u −ϵ2zu + 1,
(A2)
where the dynamical variables u and z are proportional to the
normalized intensity and the population inversion, f(t) = A cos(t
+ φ) is the external sinusoidal driving signal of amplitude A and fre-
quency , and ϵ1 and ϵ2 are two parameters. Chaos is common in
this laser system.73,74,76 For example, for ϵ1 = 0.09, ϵ2 = 0.003, and
A = 1.8, there is a chaotic attractor for  < c ≈0.912, as shown
by a sustained chaotic time series in Fig. 6(a1). The chaotic attrac-
tor is destroyed by a boundary crisis77 at c. For  > c, there is
transient chaos, after which the system settles into periodic oscilla-
tions, as exemplified in Fig. 6(a2). Suppose chaotic motion is desired.
The crisis bifurcation at c can then be regarded as a kind of system
collapse.
To build a digital twin for the chaotic laser system, we use the
external driving signal as the natural control signal for the reservoir-
computing neural network. Different from the examples in the main
text, here the driving frequency , instead of the driving amplitude
A, serves as the bifurcation parameter. Assuming observational data
in the form of time series are available for several values of  in
the regime of a chaotic attractor, we train the neural network using
chaotic time series collected from four values of  < c:  = 0.81,
0.84, 0.87, and 0.90. The training parameter setting is as follows.
For each  value in the training set, the training and validation
lengths are t = 2000 and t = 83, respectively, where the latter cor-
responds to approximately five Lyapunov times. The “warming-up”
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-9
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 6. Performance of digital twin of a driven CO2 laser system to extrapolate system dynamics under different driving frequencies. (a1) and (a2) True sustained and
transient chaotic time series of log10 u(t) of the target system, for driving frequencies  = 0.905 < c and  = 0.925 > c, respectively. The sinusoidal driving signal
f(t) is schematically illustrated. In (a1), the system exhibits sustained chaos. In (a2), the system settles into a periodic state after transient chaos. (b1) and (b2) The
corresponding time series generated by the digital twin. In both cases, the dynamical behaviors generated by the digital twin agree with the ground truth in (a1) and (a2):
sustained chaos in (b1) and transient chaos to a periodic attractor in (b2). (c1) and (c2) The return maps constructed from the local minima of u(t) from the true dynamics,
where the green dashed square deﬁnes an interval that contains the chaotic attractor in (c1) or a non-attracting chaotic set due to the escaping region (marked by the brown
arrow) leading to transient chaos in (c2). (d1) and (d2) The return maps generated by the digital twin for the same values of  as in (c1) and (c2), respectively, which agree
with the ground truth. The escaping region is successfully predicted in (d2).
length is t = 0.5. The time step of the reservoir system is 1t = 0.05.
The size of the random neural network is Dr = 800. The optimal
hyperparameter values are determined to be d = 151, λ = 0.0276,
kin = 1.18, kc = 0.113, α = 0.33, and β = 2 × 10−4.
Figures 6(a1) and 6(a2) show two representative time series
from the laser model (the ground truth) for  = 0.905 < c and
 = 0.925 > c, respectively. The one in panel (a1) is associated
with sustained chaos (pre-critical) and the other in panel (a2) is
characteristic of transient chaos with a final periodic attractor (post-
critical). The corresponding time series generated by the digital twin
are shown in Figs. 6(b1) and 6(b2), respectively. It can be seen that
the training aided by the control signal enables the digital twin to
correctly capture the dynamical climate of the target system, e.g.,
sustained or transient chaos. The true return maps in the pre-critical
and post-critical regimes are shown in Figs. 6(c1) and 6(c2), respec-
tively, and the corresponding maps generated by the digital twin
are shown in Figs. 6(d1) and 6(d2). In the pre-critical regime, an
invariant region (the green dashed square) exists on the return map
in which the trajectories are confined, leading to sustained chaotic
motion, as shown in Figs. 6(c1) and 6(d1). Within the invariant
region in which the chaotic attractor lives, the digital twin captures
the essential dynamical features of the attractor. Because the training
data are from the chaotic attractor of the target system, the digital
twin fails to generate the portion of the real return map that lies
outside the invariant region, which is expected because the digital
twin has never been exposed to the dynamical behaviors that are not
on the chaotic attractor. In the post-critical regime, a “leaky” region
emerges, as indicated by the red arrows in Figs. 6(c2) and 6(d2),
which destroys the invariant region and leads to transient chaos. The
remarkable feature is that the digital twin correctly assesses the exis-
tence of the leaky region, even when no such information is fed into
the twin during training. From the point of view of predicting system
collapse, the digital twin is able to anticipate the occurrence of the
crisis and transient chaos. A quantitative result of these predictions
is demonstrated in Appendix F.
As indicated by the predicted return maps in Figs. 6(d1)
and 6(d2), the digital twin is unable to give the final state after
the transient, because such a state must necessarily lie outside the
invariant region from which the training data are originated. In par-
ticular, the digital twin is trained with time series data from the
chaotic attractors prior to the crisis. With respect to Figs. 6(d1) and
6(d2), the digital twin can learn the dynamics within the dashed
green box in the plotted return maps but is unable to predict the
dynamics outside the box, as it has never been exposed to these
dynamics.
A comparison of the real and predicted bifurcation diagram is
demonstrated in Fig. 7. The strong resemblance between them indi-
cates the power of the digital twin in extrapolating the correct global
behavior of the target system. Moreover, this demonstrates that not
only can this approach extrapolate with various driving amplitudes
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-10
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 7. Comparison of the real (a) and predicted (b) bifurcation diagrams of the
driven laser system with varying driving frequencies . The four vertical gray
dashed lines indicate the values of driving frequencies  used for training the
neural network. The strong resemblance between the two bifurcation diagrams
indicates the power of the digital twin in extrapolating the correct global behavior
of the target system and demonstrates that not only can this approach extrapo-
late system dynamics to various driving amplitudes A but also to varying driving
frequency .
A (as demonstrated in the main text) but the approach can also work
with varying the driving frequency .
APPENDIX B: A DRIVEN CHAOTIC ECOLOGICAL
SYSTEM
We study a chaotic driven ecological system that models the
annual blooms of phytoplankton under seasonal driving.78 Seasonal-
ity plays a crucial role in ecological systems and epidemic spreading
of infectious diseases,79 which is usually modeled as a simple peri-
odic driving force on the system. The dynamical equations of this
model in the dimensionless form are78
dN
dt = I −f(t)NP −qN,
(B1)
dP
dt = f(t)NP −P,
(B2)
where N represents the level of the nutrients, P is the biomass of
the phytoplankton, the Lotka–Volterra term NP models the phyto-
plankton uptake of the nutrients, I represents a small and constant
nutrient flow from external sources, q is the sinking rate of the nutri-
ents to the lower level of the water unavailable to the phytoplankton,
and f(t) is the seasonality term: f(t) = A sin(ωecot). The parameter
values are:78 I = 0.02, q = 0.0012, and ωeco = 0.19.
Climate change can dramatically alter the dynamics of this
ecosystem.80 We consider the task of forecasting how the system
behaves if the climate change causes the seasonal fluctuation to be
more extreme. In particular, suppose the training data are measured
from the system when it behaves normally under a driving signal
of relatively small amplitude, and we wish to predict the dynami-
cal behaviors of the system in the future when the amplitude of the
driving signal becomes larger (due to climate change). The train-
ing parameter setting is as follows. The size of the neural network
is Dr = 600 with Din = Dout = 2. The time step of the evolution
of the network dynamics is 1t = 0.1. The training and validation
lengths for each value of the driving amplitude A in the training
are t = 1500 and t = 500, respectively. The optimized hyperparam-
eters of the reservoir computer are d = 350, λ = 0.42, kin = 0.39,
kc = 1.59, α = 0.131, β = 1 × 10−7.5, and σnoise = 0.
Figure 8 shows the results of our digital twin approach to this
ecological model to learn from the dynamics under a few differ-
ent values of the driving amplitude to generate the correct response
of the system to a driving signal of a larger amplitude. In par-
ticular, the training data are collected with the driving amplitude
A = 0.35, 0.4, 0.45, and 0.5, all in the chaotic regions. Figures 8(a1)
and 8(a2) show the true attractors of the system for A = 0.45 and
0.56, respectively, where the attractor is chaotic in the former case
(within the training parameter regime) and periodic in the latter
(outside the training regime). The corresponding attractors gener-
ated by the digital twin are shown in Figs. 8(b1) and 8(b2). The dig-
ital twin can not only replicate the chaotic behavior in the training
data [Fig. 8(b1)] but also predict the transition to a periodic attractor
under a driving signal with larger amplitudes (more extreme season-
ality), as shown in Fig. 8(b2). In fact, the digital twin can faithfully
produce the global dynamical behavior of the system, both inside
and outside the training regime, as can be seen from the nice agree-
ment between the ground-truth bifurcation diagram in Fig. 8(c) and
the diagram generated by the digital twin in Fig. 8(d).
APPENDIX C: CONTINUAL FORECASTING UNDER
NON-STATIONARY EXTERNAL DRIVING WITH SPARSE
REAL-TIME DATA
The three examples (Lorenz-96 climate network in the main
text, the driven CO2 laser, and the ecological system) have demon-
strated that our reservoir computing based digital twin is capable
of extrapolating and generating the correct statistical features of the
dynamical trajectories of the target system such as the attractor and
bifurcation diagram. That is, the digital twin can be regarded as a
“twin” of the target system only in a statistical sense. In particu-
lar, from random initial conditions, the digital twin can generate
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-11
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 8. Performance of the digital twin of an ecological model about blooms of phytoplankton with seasonality. The effect of seasonality is modeled by a sinusoidal driving
signal f(t) = A sin(ωecot). (a1) and (a2) Chaotic and periodic attractors of this system in the (N, log10 P) plane for A = 0.45 and A = 0.56, respectively. (b1) and (b2) The
corresponding attractors generated by the digital twin under the same driving signals f(t) as in (a1) and (a2). The digital twin has successfully extrapolated the periodical
behavior outside the chaotic training region. (c) The ground-truth bifurcation diagram of the target system. (d) The digital twin generated bifurcation diagram. In (c) and (d),
the four vertical gray dashed lines indicate the values of driving amplitudes A used for training the neural network. The strong resemblance between the two bifurcation
diagrams indicates the power of the digital twin in extrapolating the correct global behavior of the target system.
an ensemble of trajectories, and the statistics calculated from the
ensemble agree with those of the original system. At the level of indi-
vidual trajectories, if a target system and its digital twin start from
the same initial condition, the trajectory generated by the twin can
stay close to the true trajectory only for a short period of time (due to
chaos). However, with infrequent state updates, the trajectory gen-
erated by the twin can shadow the true trajectory (in principle) for
an arbitrarily long period of time,32 realizing continual forecasting of
the state evolution of the target system.
In data assimilation for numerical weather forecasting, the state
of the model system needs to be updated from time to time.81–83
This idea has recently been exploited to realize long-term pre-
diction of the state evolution of chaotic systems using reservoir
computing.32 Here, we demonstrate that, even when the driving
signal is non-stationary, the digital twin can still generate the cor-
rect state evolution of the target system. As a specific example,
we use the chaotic ecosystem in Eqs. (B1) and (B2) with the
same neural network trained in Appendix B. Figure 9(a) shows the
non-stationary external driving f(t) = A(t) sin(ωecot) whose ampli-
tude A(t) increases linearly from A(t = 0) = 0.4 to A(t = 2500)
= 0.6 in the time interval [0, 2500]. Figure 9(b) shows the true (blue)
and digital twin generated (red) time evolution of the nutrient abun-
dance. Due to chaos, without state updates, the two trajectories
diverge from each other after a few cycles of oscillation. However,
even with rare state updates, the two trajectories can stay close to
each other for any arbitrarily long time, as shown in Fig. 9(c). In
particular, there are 800 time steps involved in the time interval
[0, 2500] and the state of the digital twin is updated 20 times, i.e.,
2.5% of the available time series data. We will discuss the results
further in Appendix D.
APPENDIX D: CONTINUAL FORECASTING WITH
HIDDEN DYNAMICAL VARIABLES
In real-world scenarios, usually not all the dynamical variables
of a target system are accessible. It is often the case that only a subset
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-12
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 9. Continual forecasting of the chaotic ecological system under non-stationary external driving f(t) and with sparse updates of the dynamical variables. (a) A nonsta-
tionary sinusoidal driving signal f(t) whose amplitude increases with time. The task for the digital twin is to forecast the response of the chaotic target system under this
driving signal for a relatively long term. (b) The trajectory generated by the digital twin (red) in comparison with the true trajectory (blue). For t ∈[0, 400], the two trajectories
match each other with small errors, but the digital twin generated trajectory begins to deviate from the true trajectory at t ∼400 (due to chaos). (c) With only sparse updates
from real data at times indicated by the vertical lines (2.5% of the time steps in the given time interval), the digital twin can make relatively accurate predictions for the long
term, demonstrating the ability to perform continual forecasting.
of the dynamical variables can be measured and the remaining vari-
ables are inaccessible or hidden from the outside world. Can a digital
twin still make continual forecasting in the presence of hidden vari-
ables based on the time series data from the accessible variables?
Also, can the digital twin do this without knowing that there exist
some hidden variables before training? In general, when there are
hidden variables, the reservoir network needs to sense their exis-
tence, encode them in the hidden state of the recurrent layer, and
constantly update them. As such, the recurrent structure of reser-
voir computing is necessary, because there must be a place for the
machine to store and restore the implicit information that it has
learned from the data. Compared with the cases where complete
information about the dynamical evolution of all the observable is
available, when there are hidden variables, it is significantly more
challenging to predict the evolution of a target system driven by
a non-stationary external signal using sparse observations of the
accessible variables.
As an illustrative example, we again consider the ecosystem
described by Eqs. (B1) and (B2). We assume that the dynamical
variable N (the abundance of the nutrients) is hidden and P(t), the
biomass of the phytoplankton, is externally accessible. Despite the
accessibility to P(t), we assume that it can be measured only occa-
sionally. That is, only sparsely updated data of the variable P(t) are
available. It is necessary that the digital twin is able to learn some
equivalent of N(t) as the time evolution of P(t) also depends on the
value N(t) and to encode the equivalent in the reservoir network. In
an actual application, when the digital twin is deployed, knowledge
about the existence of such a hidden variable is not required.
Figure 10 presents a representative resulting trial, where
Fig. 10(a) shows the non-stationary external driving signal f(t) [the
same as the one in Fig. 9(a)]. Figure 10(b) shows, when the observ-
able P(t) is not updated with the real data, the predicted time series
(red) P(t) diverges from the true time series (blue) after about a
dozen oscillations. However, if P(t) is updated to the digital twin
with the true values at the times indicated by the purple vertical lines
in Fig. 10(c), the predicted time series P(t) matches the ground truth
for a much longer time. The results suggest that the existence of the
hidden variable does not significantly impede the performance of
continual forecasting.
The results in Fig. 10 motivate the following questions. First,
has the reservoir network encoded information about the hidden
variable? Second, suppose it is known that there is a hidden vari-
able and the training dataset contains this variable, can its evolu-
tion be inferred with only rare updates of the observable during
continual forecasting? Previous results24,28,84 suggested that reser-
voir computing can be used to infer the hidden variables in a
nonlinear dynamical system. Here, we show that, with a segment
of the time series of N(t) used only for training an additional
readout layer, our digital twin can forecast N(t) with only occa-
sional inputs of the observable time series P(t). In particular, the
additional readout layer for N(t) is used only for extracting infor-
mation about N(t) from the reservoir network and its output is
never injected back into the reservoir. Consequently, whether this
additional task of inferring N(t) is included or not, the trained
output layer for P(t) and the forecasting results of P(t) are not
altered.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-13
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 10. Continual forecasting and monitoring of a hidden dynamical variable in the chaotic ecological system under non-stationary external driving with sparse updates
from the observable. The system is described by Eqs. (B1) and (B2). The dynamical variable N(t) is hidden, and the other variable P(t) is externally accessible but only
sparsely sampled measurements of it can be performed. (a) The non-stationary sinusoidal driving signal f(t) with a time-varying amplitude. (b) Digital twin generated time
evolution of the accessible variable P(t) (red) in comparison with the ground truth (blue) in the absence of any state update of P(t). The predicted time evolution quickly
diverges from the true behavior. (c) With sparse updates of P(t) at the times indicated by the purple vertical lines (10% of the time steps), the digital twin is able to make an
accurate forecast of P(t). (d) Digital twin generated time evolution of the hidden variable N(t) (red) in comparison with the ground truth (blue) in the absence of any state
update of P(t). (e) Accurate forecasting of the hidden variable N(t) with sparse updates of P(t).
Figure 10(d) shows that, when the observable P(t) is not
updated with the real data, the digital twin can infer the hidden
variable N(t) for several oscillations. If P(t) is updated with the true
value at the times indicated by the purple vertical lines in Fig. 10(c),
the dynamical evolution of the hidden variable N(t) can also be
accurately predicted for a much longer period of time, as shown in
Fig. 10(e). It is worth emphasizing that during the whole process of
forecasting and monitoring, no information about the hidden vari-
able N(t) is required—only sparse data points of the observable P(t)
are used.
The training and testing settings of the digital twin for the task
involving a hidden variable are as follows. The input dimension of
the reservoir is Din = 1 because there is a single observable log10 P(t).
The output dimension is Dout = 2 with one dimension of the observ-
able log10 P(t + 1t) in addition to one dimension of the hidden
variable N(t + 1t). Because of the higher memory requirement in
dealing with a hidden variable, a somewhat larger reservoir network
is needed, so we use Dr = 1000. The time step of the dynamical evo-
lution of the neural network is 1t = 0.1. The training and validating
lengths for each value of the driving amplitude in the training are
t = 3500 and t = 350, respectively. Other optimized hyperparam-
eters of the reservoir are d = 450, λ = 1.15, kin = 0.32, kc = 3.1,
α = 0.077, β = 1 × 10−8.3, and σnoise = 10−3.0.
It is also worth noting that Figs. 9 and 10 have demon-
strated the ability of the digital twin to extrapolate beyond
the parameter regime of the target system from which the
training data are obtained. In particular, the digital twin was
trained only with time series under stationary external driving
of the amplitude A = 0.35, 0.4, 0.45, and 0.5. During the testing
phase associated with both Figs. 9 and 10, the external driv-
ing is non-stationary with its amplitude linearly increasing from
A = 0.4 to A = 0.6. The second half of the time series P(t) and
N(t) in Figs. 9 and 10 are, thus, beyond the training parameter
regime.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-14
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 11. Comparison of the prediction performance between the noiseless (left) and noisy (right) cases on the task of predicting under external driving with different waveforms.
The target system is a six-dimensional Lorenz-96 system. Panel (a) shows the true bifurcation diagram. Panels (b1)–(b3) show the prediction results without any dynamical
noise in the training data with three realizations of the reservoir network. Panels (c1)–(c3) show the prediction results with dynamical noise of a strength δDB = 3 × 10−3 in
the training data. The settings are the same as that in Fig. 4.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-15
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 12. Performance of the digital twin with the ecological model under driving signals with waveform different from the training set. The testing driving signals are described
by Eq. (E1) while the training driving signals are sinusoidal waves with small dynamical noise. (a1) The real bifurcation diagram for Atest = 0.3. (a2) and (a3) Predicted
bifurcation diagrams for Atest = 0.3 with two random realizations of the reservoir networks. (b1)–(b3) Same as (a1)–(a3) but with Atest = 0.4.
The results in Figs. 9 and 10 help legitimize the terminology
“digital twin,” as the reservoir computers subject to the external
driving are dynamical twin systems that evolve “in parallel” to the
corresponding real systems. Even when the target system is only
partially observable, the digital twin contains both the observable
and hidden variables whose dynamical evolution is encoded in the
recurrent neural network in the hidden layer. The dynamical evo-
lution of the output is constantly (albeit infrequently) corrected
by sparse feedback from the real system, so the output trajec-
tory of the digital twin shadows the true trajectory of the target
system. Suppose one wishes to monitor a variable in the tar-
get system, it is only necessary to read it from the digital twin
instead of making more (possibly costly) measurements on the real
system.
APPENDIX E: DIGITAL TWINS UNDER EXTERNAL
DRIVING WITH VARIED WAVEFORM
In the main text, it is demonstrated that dynamical noise added
to the driving signal during the training can be beneficial. Figure 11
presents a comparison between the noiseless training and the train-
ing with dynamical noise of a strength δDB = 3 × 10−3. The ground-
truth bifurcation diagram is shown in Fig. 11(a) and three examples
with different reservoir neural networks for the noiseless (b1)–(b3)
and noisy (c1)–(c3) training schemes are shown. All the settings
other than the noise level are the same as that in Fig. 4. Though
there is still a fluctuation in predicted results, adding dynamical
noise into the training data can produce bifurcation diagrams that
are, in general, closer to the ground truth than without noise.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-16
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 13. Quantitative performance of the digital twin for a chaotic driven laser
system. (A) Distribution of the predicted values of the crisis bifurcation point ˆc,
at which a chaotic attractor is destroyed and replaced by a non-attracting chaotic
invariant set leading to transient chaos. The blue and red vertical dashed lines
denote the true value c ≈0.912 and the average predicted value ⟨ˆc⟩, respec-
tively, where 200 random realizations of the reservoir neural network are used to
generate this distribution. Despite the ﬂuctuations in the predicted crisis point, the
ensemble average value of the prediction is quite close to the ground truth. (B)
Exponential distribution of the lifetime of transient chaos slightly beyond the cri-
sis point: true (blue) and predicted (red) behaviors. The predicted distribution is
generated using 100 random reservoir realizations, each with 200 random initial
“warming-up” data.
To further demonstrate the beneficial role of noise, we test
the additive training noise scheme using the ecological system. The
training process and hyperparameter values of the digital twin are
identical to those in Appendix B. A dynamical noise of amplitude
δDB = 3 × 10−4 is added to the driving signal f(t) during training
in the same way as in Fig. 4. During testing, the driving signals are
altered to
ftest(t) = Atest sin(ωecot) + Atest
2
sin
ωeco
2 t + 1φ

,
(E1)
where ωeco = 0.19. Two sets of testing signals ftest(t) are used, with
Atest = 0.3 and 0.4, respectively. Figure 12 shows the true and pre-
dicted bifurcation diagrams of log10 Pmax vs 1φ for Atest = 0.3 (left
column) and Atest = 0.4 (right column). It can be seen that the
bifurcation diagrams generated by the digital twin with the aid of
training noise are remarkably accurate. We also find that, for this
ecological system, the amplitude δDB of the dynamical noise during
training does not have a significant effect on the predicted bifurca-
tion diagram. A plausible reason is that the driving signal f(t) is a
multiplicative term in the system equations. Further investigation is
required to determine the role of dynamical noise in these tasks with
varied driving waveforms.
APPENDIX F: QUANTITATIVE CHARACTERIZATION OF
THE PERFORMANCE OF DIGITAL TWINS
In the main text, a quantitative measure of the overlapping rate
between the target and predicted spanning regions is introduced
to measure the performance of the digital twins, where a span-
ning region is the smallest simply connected region that encloses
the entire attractor. In a two-dimensional projection, we divide a
large reference plane into pixels of size 0.05 × 0.05. All the pixels
through which the system trajectory crosses and those surrounded
by the trajectory belong to the spanning region, and the regions cov-
ering the true attractor of the target system and predicted attractor
can be compared. In particular, all the pixels that belong to one
spanned region but not to the other are counted and the number
is divided by the total number of pixels in the spanned region of
the true attractor. This gives RESR, the relative error of the spanned
regions, as described in the main text. While this measure is effec-
tive in most cases, near a bifurcation (e.g., near the boundary of a
periodic window), large errors can arise because a small parame-
ter mismatch can lead to a characteristically different attractor. To
reduce the error, we test the attractors at three nearby parameter val-
ues, e.g., A and A ± 1A with 1A = 0.005 and choose the smallest
RESR values among the three.
A direct comparison between the predicted bifurcation dia-
gram with the ground truth is difficult given the rich information
a bifurcation diagram can provide. To better quantify, here we pro-
vide another measure to quantify the performance of digital twins,
and we employ another measure (besides RESR). In particular, for
a bifurcation diagram, the parameter values at which the various
bifurcations occur are of great interest, as they define the critical
points at which characteristic changes in the system can occur. To
be concrete, we focus on the crisis point at which sustained chaotic
motion on an attractor is destroyed and replaced by transient chaos.
To characterize the performance of the digital twins in extrapolat-
ing the dynamics of the target system, we examine the errors in the
predicted critical bifurcation point and in the average lifetime of the
chaotic transient after the bifurcation.
As an illustrative example, we take the driven chaotic laser sys-
tem in Appendix A, where a crisis bifurcation occurs at the critical
driving frequency c ≈0.912 at which the chaotic attractor of the
system is destroyed and replaced by a non-attracting chaotic invari-
ant set leading to transient chaos. We test to determine if the digital
twin can faithfully predict the crisis point based only on training data
from the parameter regime of a chaotic attractor. Let ˆc be the digi-
tal twin predicted critical point. Figure 13(a) shows the distribution
of ˆc obtained from 200 random realizations of the reservoir neural
network. Despite the fluctuations in the predicted ˆc, their average
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-17
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 14. Robustness of digital twin against combined dynamical and observational noises. The setting is the same as that in Fig. 2, except with additional noises in the
training data. (a) A true bifurcation diagram of the six-dimensional Lorenz-96 system. (b1) and (b2) Two examples of the bifurcation diagram predicted by the digital twin with
training data under dynamical noise of amplitude σdyn = 10−2 and observational noise of amplitude σob = 10−2. (c1) and (c2) Two examples of the predicted bifurcation
diagrams under the two kinds of noise with σdyn = 10−1 and σob = 10−1. Both the dynamical and observational noises are additive Gaussian processes. It can be seen that
though larger additional noises make the predicted details less accurate, the general shapes of the predicted results are not harmed signiﬁcantly. The settings of the training
data and reservoir neural networks are the same as those in Fig. 3. The dynamical noises are added to the dynamical equations of the state variables. There is no noise in
the sinusoidal external driving.
value is ⟨ˆc⟩= 0.914, which is close to the true value c = 0.912. A
relative error ε of ˆc can be defined as
ε =
|c −ˆc|
D(c, {train}),
(F1)
where D(c, {train}) denotes the minimal distance from c to the
set of training parameter points {train}, i.e., the difference between
c and the closest training point. For the driven laser system, we
have D(c, {train}) ≈10%.
The second quantity is the lifetime τtransient of transient chaos
after the crisis bifurcation,35,39 as shown in Fig. 13(b). The aver-
age transient lifetime is the inverse of the slope of the lin-
ear regression of predicted data points in Fig. 13(b), which is
⟨τ⟩≈0.8 × 103. Compared with the true value ⟨τ⟩≈1.2 × 103, we
see that the digital twin is able to predict the average chaotic tran-
sient lifetime to within the same order of magnitude. Considering
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-18
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
FIG. 15. Origin of the failure of the digital twin in predicting the periodic window
in Fig. 2(c). (a) A two-dimensional portrait of the periodic attractor of period-21 in
the Lorenz-96 system for A = 3.2. (b) The digital twin predicted chaotic attrac-
tor. (c) The transient behavior of the target Lorenz-96 system for A = 3.2. The
remarkable resemblance between (b) and (c) suggests that the trained digital twin
has faithfully captured the dynamical climate of the target system.
that key to the transient dynamics is the small escaping region in
Fig. 6(d2), which is sensitive to the inevitable training errors, the
performance can be deemed as satisfactory.
APPENDIX G: ROBUSTNESS OF DIGITAL TWIN
AGAINST COMBINED DYNAMICAL/OBSERVATIONAL
NOISES
Can our reservoir computing based digital twins withstand the
influences of different types of noises? To address this question,
we introduce dynamical and observational noises in the training
data, which are modeled as additive Gaussian noises. Take the six-
dimensional Lorenz-96 system as an example. Figure 14(a) shows
the true bifurcation diagram under different amplitudes of external
driving, where the vertical dashed lines specify the training points.
Figures 14(b1) and 14(b2) show two realizations of the bifurca-
tion diagram generated by the digital twin under both dynamical
and observational noises of amplitudes σdyn = 10−2 and σob = 10−2.
Two bifurcation diagrams for noise amplitudes of an order of mag-
nitude larger: σdyn = 10−1 and σob = 10−1 are shown in Figs. 14(c1)
and 14(c2). It can be seen that the additional noises have little effect
on the performance of the digital twin in generating the bifurcation
diagram.
APPENDIX H: PERIODIC WINDOWS OF A HIGH
PERIOD: EFFECT OF LONG TRANSIENTS
Figure 2 demonstrates that the digital twin is able to predict
many details of a bifurcation diagram but it fails to generate a rela-
tively large periodic window of about A = 3.2. A closer examination
of the dynamics of the target Lorenz-96 system reveals that the peri-
odic attractor in the window has period 21 with a rather complicated
structure, as shown in Fig. 15(a) in a two-dimensional projection.
The digital twin predicts a chaotic attractor, as shown in Fig. 15(b).
The reason that the digital twin fails to predict the periodic attractor
lies in the long transient of the trajectory before it reaches the final
attractor, as shown in Fig. 15(c). A comparison between Figs. 15(b)
and 15(c) indicates that what the digital twin has predicted is, in fact,
the transient behavior in the periodic window. The implication is
that the digital twin has, in fact, faithfully captured the dynamical
climate of the target system.
REFERENCES
1E. J. Tuegel, A. R. Ingraffea, T. G. Eason, and S. M. Spottswood, “Reengineering
aircraft structural life prediction using a digital twin,” Int. J. Aerospace Eng. 2011,
154798 (2011).
2F. Tao and Q. Qi, “Make more digital twins,” Nature 573, 274–277 (2019).
3A. Rasheed, O. San, and T. Kvamsdal, “Digital twin: Values, challenges and
enablers from a modeling perspective,” IEEE Access 8, 21980–22012 (2020).
4K. Bruynseels, F. S. de Sio, and J. van den Hoven, “Digital twins in health care:
Ethical implications of an emerging engineering paradigm,” Front. Gene. 9, 31
(2018).
5S. M. Schwartz, K. Wildenhaus, A. Bucher, and B. Byrd, “Digital twins and the
emerging science of self: Implications for digital health experience design and
“small” data,” Front. Comp. Sci. 2, 31 (2020).
6R. Laubenbacher, J. P. Sluka, and J. A. Glazier, “Using digital twins in viral
infection,” Science 371, 1105–1106 (2021).
7P. Voosen, “Europe builds ‘digital twin’ of earth to hone climate forecasts,”
Science 370, 16–17 (2020).
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-19
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
8P. Bauer, B. Stevens, and W. Hazeleger, “A digital twin of earth for the green
transition,” Nat. Clim. Change 11, 80–83 (2021).
9Y.-C. Lai and T. Tél, Transient Chaos—Complex Dynamics on Finite Time Scales
(Springer, New York, 2011).
10K. McCann and P. Yodzis, “Nonlinear dynamics and population disappear-
ances,” Am. Naturalist 144, 873–879 (1994).
11A. Hastings, K. C. Abbott, K. Cuddington, T. Francis, G. Gellner, Y.-C. Lai, A.
Morozov, S. Petrivskii, K. Scranton, and M. L. Zeeman, “Transient phenomena in
ecology,” Science 361, eaat6412 (2018).
12M. Dhamala and Y.-C. Lai, “Controlling transient chaos in deterministic flows
with applications to electrical power systems and ecology,” Phys. Rev. E 59,
1646–1655 (1999).
13Y.-C. Lai, C. Grebogi, and J. Kurths, “Modeling of deterministic chaotic sys-
tems,” Phys. Rev. E 59, 2907–2910 (1999).
14Y.-C. Lai and C. Grebogi, “Modeling of coupled chaotic oscillators,” Phys. Rev.
Lett. 82, 4803–4806 (1999).
15W.-X. Wang, R. Yang, Y.-C. Lai, V. Kovanis, and C. Grebogi, “Predicting catas-
trophes in nonlinear dynamical systems by compressive sensing,” Phys. Rev. Lett.
106, 154101 (2011).
16W.-X. Wang, Y.-C. Lai, and C. Grebogi, “Data based identification and pre-
diction of nonlinear and complex dynamical systems,” Phys. Rep. 644, 1–76
(2016).
17Y.-C. Lai, “Finding nonlinear system equations and complex network structures
from data: A sparse optimization approach,” Chaos 31, 082101 (2021).
18H. Jaeger, “The “echo state” approach to analysing and training recurrent
neural networks-with an erratum note,” German National Research Center for
Information Technology GMD Technical Report (2001), Vol. 148, p. 13.
19W. Mass, T. Nachtschlaeger, and H. Markram, “Real-time computing without
stable states: A new framework for neural computation based on perturbations,”
Neur. Comp. 14, 2531–2560 (2002).
20H. Jaeger and H. Haas, “Harnessing nonlinearity: Predicting chaotic systems
and saving energy in wireless communication,” Science 304, 78–80 (2004).
21N. D. Haynes, M. C. Soriano, D. P. Rosin, I. Fischer, and D. J. Gauthier, “Reser-
voir computing with a single time-delay autonomous Boolean node,” Phys. Rev. E
91, 020801 (2015).
22L. Larger, A. Baylón-Fuentes, R. Martinenghi, V. S. Udaltsov, Y. K. Chembo,
and M. Jacquot, “High-speed photonic reservoir computing using a time-delay-
based architecture: Million words per second classification,” Phys. Rev. X 7,
011015 (2017).
23J. Pathak, Z. Lu, B. Hunt, M. Girvan, and E. Ott, “Using machine learning to
replicate chaotic attractors and calculate Lyapunov exponents from data,” Chaos
27, 121102 (2017).
24Z. Lu, J. Pathak, B. Hunt, M. Girvan, R. Brockett, and E. Ott, “Reservoir
observers: Model-free inference of unmeasured variables in chaotic systems,”
Chaos 27, 041102 (2017).
25J. Pathak, B. Hunt, M. Girvan, Z. Lu, and E. Ott, “Model-free prediction of large
spatiotemporally chaotic systems from data: A reservoir computing approach,”
Phys. Rev. Lett. 120, 024102 (2018).
26T. L. Carroll, “Using reservoir computers to distinguish chaotic signals,” Phys.
Rev. E 98, 052209 (2018).
27K. Nakai and Y. Saiki, “Machine-learning inference of fluid variables from data
using reservoir computing,” Phys. Rev. E 98, 023111 (2018).
28Z. S. Roland and U. Parlitz, “Observing spatio-temporal dynamics of excitable
media using reservoir computing,” Chaos 28, 043118 (2018).
29A. Griffith, A. Pomerance, and D. J. Gauthier, “Forecasting chaotic systems with
very low connectivity reservoir computers,” Chaos 29, 123108 (2019).
30J. Jiang and Y.-C. Lai, “Model-free prediction of spatiotemporal dynamical sys-
tems with recurrent neural networks: Role of network spectral radius,” Phys. Rev.
Res. 1, 033056 (2019).
31G. Tanaka, T. Yamane, J. B. Héroux, R. Nakane, N. Kanazawa, S. Takeda,
H. Numata, D. Nakano, and A. Hirose, “Recent advances in physical reservoir
computing: A review,” Neu. Net. 115, 100–123 (2019).
32H. Fan, J. Jiang, C. Zhang, X. Wang, and Y.-C. Lai, “Long-term prediction of
chaotic systems with machine learning,” Phys. Rev. Res. 2, 012080 (2020).
33C. Zhang, J. Jiang, S.-X. Qu, and Y.-C. Lai, “Predicting phase and sensing phase
coherence in chaotic systems with machine learning,” Chaos 30, 083114 (2020).
34C. Klos, Y. F. K. Kossio, S. Goedeke, A. Gilra, and R.-M. Memmesheimer,
“Dynamical learning of dynamics,” Phys. Rev. Lett. 125, 088103 (2020).
35L.-W. Kong, H.-W. Fan, C. Grebogi, and Y.-C. Lai, “Machine learning pre-
diction of critical transition and system collapse,” Phys. Rev. Res. 3, 013090
(2021).
36D. Patel, D. Canaday, M. Girvan, A. Pomerance, and E. Ott, “Using machine
learning to predict statistical properties of non-stationary dynamical processes:
System climate, regime transitions, and the effect of stochasticity,” Chaos 31,
033149 (2021).
37J. Z. Kim, Z. Lu, E. Nozari, G. J. Pappas, and D. S. Bassett, “Teaching recur-
rent neural networks to infer global temporal structure from local examples,” Nat.
Machine Intell. 3, 316–323 (2021).
38H. Fan, L.-W. Kong, Y.-C. Lai, and X. Wang, “Anticipating synchronization
with machine learning,” Phys. Rev. Res. 3, 023237 (2021).
39L.-W. Kong, H. Fan, C. Grebogi, and Y.-C. Lai, “Emergence of transient
chaos and intermittency in machine learning,” J. Phys. Complexity 2, 035014
(2021).
40E. Bollt, “On explaining the surprising success of reservoir computing forecaster
of chaos? The universal machine learning dynamical system with contrast to VAR
and DMD,” Chaos 31, 013108 (2021).
41D. J. Gauthier, E. Bollt, A. Griffith, and W. A. Barbosa, “Next generation
reservoir computing,” Nat. Commun. 12, 1–8 (2021).
42T. L. Carroll, “Optimizing memory in reservoir computers,” Chaos 32, 023123
(2022).
43A. Hart, J. Hook, and J. Dawes, “Embedding and approximation theorems for
echo state networks,” Neu. Net. 128, 234–247 (2020).
44The codes of this work are shared at github.com/lw-kong/Digital_Twin_2022.
45J. Herteux and C. Räth, “Breaking symmetries of the reservoir equations in echo
state networks,” Chaos 30, 123142 (2020).
46D. E. Goldberg, Genetic Algorithms (Pearson Education India, 2006).
47A. R. Conn, N. I. Gould, and P. Toint, “A globally convergent augmented
Lagrangian algorithm for optimization with general constraints and simple
bounds,” SIAM J. Numer. Anal. 28, 545–572 (1991).
48A. Conn, N. Gould, and P. Toint, “A globally convergent Lagrangian bar-
rier algorithm for optimization with general inequality constraints and simple
bounds,” Math. Comput. 66, 261–288 (1997).
49J. Kennedy and R. Eberhart, “Particle swarm optimization,” in Proceedings of
ICNN’95-International Conference on Neural Networks (IEEE, 1995), Vol. 4, pp.
1942–1948.
50E. Mezura-Montes and C. A. C. Coello, “Constraint-handling in nature-inspired
numerical optimization: Past, present and future,” Swarm Evol. Comput. 1,
173–194 (2011).
51M. A. Gelbart, J. Snoek, and R. P. Adams, “Bayesian optimization with unknown
constraints,” arXiv:1403.5607 (2014).
52J. Snoek, H. Larochelle, and R. P. Adams, “Practical Bayesian optimization of
machine learning algorithms,” in NeurIPS (Curran Associates, Inc., 2012), pp.
2951–2959, available at https://proceedings.neurips.cc/paper/2012/file/05311655a
15b75fab86956663e1819cd-Paper.pdf.
53H.-M. Gutmann, “A radial basis function method for global optimization,” J.
Global Optim. 19, 201–227 (2001).
54R. G. Regis and C. A. Shoemaker, “A stochastic radial basis function method
for the global optimization of expensive functions,” INFORMS J. Comput. 19,
497–509 (2007).
55Y. Wang and C. A. Shoemaker, “A general stochastic algorithmic framework for
minimizing expensive black box objective functions based on surrogate models
and sensitivity analysis,” arXiv:1410.6271 (2014).
56E. N. Lorenz, “Predictability: A problem partly solved,” in Proceedings Seminar
on Predictability (ECMWF, 1996), Vol. 1.
57C. Van den Broeck, J. Parrondo, R. Toral, and R. Kawai, “Nonequilib-
rium phase transitions induced by multiplicative noise,” Phys. Rev. E 55, 4084
(1997).
58D. Sussillo and L. F. Abbott, “Generating coherent patterns of activity from
chaotic neural networks,” Neuron 63, 544–557 (2009).
59T. Kobayashi and T. Sugino, “Continual learning exploiting structure of fractal
reservoir computing,” in International Conference on Artificial Neural Networks
(Springer, 2019), pp. 35–47.
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-20
Published under an exclusive license by AIP Publishing
Chaos
ARTICLE
scitation.org/journal/cha
60J. Pathak, A. Wikner, R. Fussell, S. Chandra, B. R. Hunt, M. Girvan, and E. Ott,
“Hybrid forecasting of chaotic processes: Using machine learning in conjunction
with a knowledge-based model,” Chaos 28, 041101 (2018).
61R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary
differential equations,” Adv. Neural Inform. Process. Syst. 31 (2018).
62F. Berkenkamp, M. Turchetta, A. P. Schoellig, and A. Krause, “Safe model-based
reinforcement learning with stability guarantees,” arXiv:1705.08551 (2017).
63T. M. Moerland, J. Broekens, and C. M. Jonker, “Model-based reinforcement
learning: A survey,” arXiv:2006.16712 (2020).
64Y. Kuramoto and D. Battogtokh, “Coexistence of coherence and incoherence in
nonlocally coupled phase oscillators,” Nonlin. Phenom. Complex Syst. 5, 380–385
(2002).
65D. M. Abrams and S. H. Strogatz, “Chimera states for coupled oscillators,” Phys.
Rev. Lett. 93, 174102 (2004).
66I. Omelchenko, Y. Maistrenko, P. Hövel, and E. Schöll, “Loss of coherence
in dynamical networks: Spatial chaos and chimera states,” Phys. Rev. Lett. 106,
234102 (2011).
67M. R. Tinsley, S. Nkomo, and K. Showalter, “Chimera and phase-cluster states
in populations of coupled chemical oscillators,” Nat. Phys. 8, 662 (2012).
68A. M. Hagerstrom, T. E. Murphy, R. Roy, P. Hövel, I. Omelchenko, and E.
Schöll, “Experimental observation of chimeras in coupled-map lattices,” Nat.
Phys. 8, 658 (2012).
69I. Omelchenko, O. E. Omel’chenko, P. Hövel, and E. Schöll, “When nonlocal
coupling between oscillators becomes stronger: Patched synchrony or multi-
chimera states,” Phys. Rev. Lett. 110, 224101 (2013).
70I. Omelchenko, A. Zakharova, P. Hövel, J. Siebert, and E. Schöll, “Nonlinearity
of local dynamics promotes multi-chimeras,” Chaos 25, 083104 (2015).
71I. Omelchenko, O. E. Omel’chenko, A. Zakharova, and E. Schöll, “Optimal
design of tweezer control for chimera states,” Phys. Rev. E 97, 012216 (2018).
72L.-W. Kong and Y.-C. Lai, “Scaling law of transient lifetime of chimera states
under dimension-augmenting perturbations,” Phys. Rev. Res. 2, 023196 (2020).
73D. Dangoisse, P. Glorieux, and D. Hennequin, “Laser chaotic attractors in
crisis,” Phys. Rev. Lett. 57, 2657 (1986).
74D. Dangoisse, P. Glorieux, and D. Hennequin, “Chaos in a CO2 laser with mod-
ulated parameters: Experiments and numerical simulations,” Phys. Rev. A 36,
4775 (1987).
75H. G. Solari, E. Eschenazi, R. Gilmore, and J. R. Tredicce, “Influence of coex-
isting attractors on the dynamics of a laser system,” Opt. Commun. 64, 49–53
(1987).
76I. B. Schwartz, “Sequential horseshoe formation in the birth and death of chaotic
attractors,” Phys. Rev. Lett. 60, 1359 (1988).
77C. Grebogi, E. Ott, and J. A. Yorke, “Crises, sudden changes in chaotic attractors
and chaotic transients,” Physica D 7, 181–200 (1983).
78A. Huppert, B. Blasius, R. Olinky, and L. Stone, “A model for seasonal phyto-
plankton blooms,” J. Theoret. Biol. 236, 276–290 (2005).
79L. Stone, R. Olinky, and A. Huppert, “Seasonal dynamics of recurrent epi-
demics,” Nature 446, 533–536 (2007).
80M. Winder and U. Sommer, “Phytoplankton response to a changing climate,”
Hydrobiologia 698, 5–16 (2012).
81E. Kalnay, Atmospheric Modeling, Data Assimilation and Predictability (Cam-
bridge University Press, 2003).
82M. Asch, M. Bocquet, and M. Nodet, Data Assimilation: Methods, Algorithms,
and Applications (SIAM, 2016).
83A. Wikner, J. Pathak, B. R. Hunt, I. Szunyogh, M. Girvan, and E. Ott, “Using
data assimilation to train a hybrid forecast system that combines machine-
learning and knowledge-based components,” Chaos 31, 053114 (2021).
84T. Weng, H. Yang, C. Gu, J. Zhang, and M. Small, “Synchronization of chaotic
systems and their machine-learning models,” Phys. Rev. E 99, 042203 (2019).
Chaos 33, 033111 (2023); doi: 10.1063/5.0138661
33, 033111-21
Published under an exclusive license by AIP Publishing
