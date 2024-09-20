Digital twins of nonlinear dynamical systems
Ling-Wei Kong,1 Yang Weng,1 Bryan Glaz,2 Mulugeta Haile,2 and Ying-Cheng Lai1, 3, ∗
1School of Electrical, Computer and Energy Engineering,
Arizona State University, Tempe, Arizona 85287, USA
2Vehicle Technology Directorate, CCDC Army Research Laboratory,
2800 Powder Mill Road, Adelphi, MD 20783-1138, USA
3Department of Physics, Arizona State University, Tempe, Arizona 85287, USA
(Dated: October 13, 2022)
We articulate the design imperatives for machine-learning based digital twins for nonlinear dy-
namical systems subject to external driving, which can be used to monitor the “health” of the target
system and anticipate its future collapse. We demonstrate that, with single or parallel reservoir com-
puting conﬁgurations, the digital twins are capable of challenging forecasting and monitoring tasks.
Employing prototypical systems from climate, optics and ecology, we show that the digital twins
can extrapolate the dynamics of the target system to certain parameter regimes never experienced
before, make continual forecasting/monitoring with sparse real-time updates under non-stationary
external driving, infer hidden variables and accurately predict their dynamical evolution, adapt to
diﬀerent forms of external driving, and extrapolate the global bifurcation behaviors to systems of
some diﬀerent sizes.
These features make our digital twins appealing in signiﬁcant applications
such as monitoring the health of critical systems and forecasting their potential collapse induced by
environmental changes.
I.
INTRODUCTION
The concept of digital twins originated from aerospace
engineering for aircraft structural life prediction [1]. In
general, a digital twin can be used for predicting dynam-
ical systems and generating solutions of emergent behav-
iors that can potentially be catastrophic [2]. Digital twins
have attracted a great deal of attention from a wide range
of ﬁelds [3] including medicine and health care [4, 5].
For example, the idea of developing medical digital twins
in viral infection through a combination of mechanistic
knowledge, observational data, medical histories, and ar-
tiﬁcial intelligence has been proposed recently [6], which
can potentially lead to a powerful addition to the exist-
ing tools to combat future pandemics. In a more dra-
matic development, the European Union plans to fund
the development of digital twins of Earth for its green
transition [7, 8].
The physical world is nonlinear. Many engineering sys-
tems, such as complex infrastructural systems, are gov-
erned by nonlinear dynamical rules, too. In nonlinear dy-
namics, various bifurcations leading to chaos and system
collapse can take place [9]. For example, in ecology, en-
vironmental deterioration caused by global warming can
lead to slow parameter drift towards chaos and species
extinction [10, 11]. In an electrical power system, volt-
age collapse can occur after a parameter shift that lands
the system in transient chaos [12]. The various climate
systems in diﬀerent geographic regions of the world are
also nonlinear and the emergent catastrophic behaviors
as the result of increasing human activities are of grave
concern. In all these cases, it is of interest to develop
∗Ying-Cheng.Lai@asu.edu
a digital twin of the system of interest to monitor its
“health” in real time as well as for predictive problem
solving in the sense that, if the digital twin indicates
a possible system collapse in the future, proper control
strategies should and can be devised and executed in time
to prevent the collapse.
What does it take to create a digital twin for a non-
linear dynamical system? For natural and engineering
systems, there are two general approaches: one is based
on mechanistic knowledge and another is based on ob-
servational data. In principle, if the detailed physics of
the system is well understood, it should be possible to
construct a digital twin through mathematical modeling.
However, there are two diﬃculties associated with this
modeling approach.
First, a real-world system can be
high-dimensional and complex, preventing the rules gov-
erning its dynamical evolution from being known at a
suﬃciently detailed level. Second, the hallmark of chaos
is sensitive dependence on initial conditions.
Because
no mathematical model of the underlying physical sys-
tem can be perfect, the small deviations and high di-
mensionality of the system coupled with environmental
disturbances can cause the model predictions of the fu-
ture state of the system to be inaccurate and completely
irrelevant [13, 14]. These diﬃculties motivate the propo-
sition that data-based approach can have advantages in
many realistic scenarios and a viable method to develop
a digital twin is through data. While in certain cases,
approximate system equations can be found from data
through sparse optimization [15–17], the same diﬃculties
with the modeling approach arise. These considerations
have led us to exploit machine learning to create digital
twins for nonlinear dynamical systems.
Given a nonlinear dynamical system, its digital twin is
also a dynamical system, rendering appropriate exploita-
tion of recurrent neural networks that can be designed
arXiv:2210.06144v1  [nlin.AO]  5 Oct 2022
2
to generate self-dynamical evolution with memory.
In
this regard, reservoir computers (RC) [18–20] that have
been extensively studied in recent years [21–43] provide a
starting point, which can be trained from observational
data to generate closed-loop dynamical evolution that
follows the evolution of the target system for a ﬁnite
amount of time.
Another advantage of RC is that no
back-propagation is needed for optimizing the parame-
ters - only a linear regression is required in the training
so it is computationally eﬃcient. A common situation is
that the target system is subject to external driving, such
as a driven laser, a regional climate system, or an ecosys-
tem under external environmental disturbances. Accord-
ingly, the digital twin must accommodate a mechanism
to control or steer the dynamics of the RC neural net-
work to account for the external driving. Introducing a
control mechanism into the RC structure with an exoge-
nous control signal acting directly onto the RC network
distinguishes our work from existing ones in the litera-
ture of RC as applied to nonlinear dynamical systems.
Of particular interest is whether the collapse of the tar-
get chaotic system can be anticipated from the digital
twin. The purpose of this paper is to demonstrate that
the digital twin so created can accurately produce the
bifurcation diagram of the target system and faithfully
mimic its dynamical evolution from a statistical point of
view. The digital twin can then be used to monitor the
present and future “health” of the system. More impor-
tantly, with proper training from observational data the
twin can reliably anticipate system collapses, providing
early warnings of potentially catastrophic failures of the
system.
More speciﬁcally, using three prototypical systems
from optics,
ecology,
and climate,
respectively,
we
demonstrate that the RC based digital twins developed
in this paper solve the following challenging problems:
(1) extrapolation of the dynamical evolution of the target
system into certain “uncharted territories” in the param-
eter space, (2) long-term continual forecasting of nonlin-
ear dynamical systems subject to non-stationary external
driving with sparse state updates, (3) inference of hidden
variables in the system and accurate prediction of their
dynamical evolution into the future, (4) adaptation to
external driving of diﬀerent waveform, and (5) extrapo-
lation of the global bifurcation behaviors of network sys-
tems to some diﬀerent sizes. These features make our
digital twins appealing in applications.
II.
METHODS
The basic construction of the digital twin of a nonlin-
ear dynamical system [45] is illustrated in Fig. 1. It is
essentially a recurrent RC neural network with a control
mechanism, which requires two types of input signals:
the observational time series for training and the con-
trol signal f(t) that remains in both the training and
self-evolving phase. The hidden layer hosts a random or
complex network of artiﬁcial neurons. During the train-
ing, the hidden recurrent layer is driven by both the in-
put signal u(t) and the control signal f(t). The neurons
in the hidden layer generate a high-dimensional nonlin-
ear response signal. Linearly combining all the responses
of these hidden neurons with a set of trainable and opti-
mizable parameters yields the output signal. Speciﬁcally,
the digital twin consists of four components: (i) an input
subsystem that maps the low-dimensional (Din) input
signal into a (high) Dr-dimensional signal through the
weighted Dr × Din matrix Win, (ii) a reservoir network
of N neurons characterized by Wr, a weighted network
matrix of dimension Dr × Dr, where Dr ≫Din, (iii) an
readout subsystem that converts the Dr-dimensional sig-
nal from the reservoir network into an Dout-dimensional
signal through the output weighted matrix Wout, and (iv)
a controller with the matrix Wc. The matrix Wr deﬁnes
the structure of the reservoir neural network in the hid-
den layer, where the dynamics of each node are described
by an internal state and a nonlinear hyperbolic tangent
activation function.
The matrices Win, Wc, and Wr are generated ran-
domly prior to training, whereas all elements of Wout
are to be determined through training.
Speciﬁcally,
the state updating equations for the training and self-
evolving phases are, respectively,
r(t+∆t) = (1 −α)r(t)
+ α tanh [Wrr(t) + Winu(t) + Wcf(t)],
(1)
r(t+∆t) = (1 −α)r(t)
+ α tanh [Wrr(t) + WinWoutr′(t) + Wcf(t)],
(2)
where r(t) is the hidden state, u(t) is the vector of input
training data, ∆t is the time step, the vector tanh (p)
is deﬁned to be [tanh (p1), tanh (p2), . . .]T for a vector
p = [p1, p2, ...]T , and α is the leakage factor.
During
the training, several trials of data are typically used un-
der diﬀerent driving signals so that the digital twin can
“sense, learn, and mingle” the responses of the target sys-
tem to gain the ability to extrapolate a response to a new
driving signal that has never been encountered before.
We input these trials of training data, i.e., a few pairs of
u(t) and the associated f(t), through the matrices Win
and Wc sequentially. Then we record the state vector r(t)
of the neural network during the entire training phase as
a matrix R. We also record all the desired output, which
is the one-step prediction result v(t) = u(t + ∆t), as
the matrix V.
To make the readout nonlinear and to
avoid unnecessary symmetries in the system [24, 46], we
change the matrix R into R′ by squaring the entries of
even dimensions in the states of the hidden layer. [The
vector (r′(t) in Eq. (2) is deﬁned in a similar way.] We
carry out a linear regression between V and R′, with a
ℓ-2 regularization coeﬃcient β, to determine the readout
matrix:
Wout = V · R′T (R′ · R′T + βI)−1.
(3)
To achieve acceptable learning performance, optimiza-
tion of hyperparameters is necessary. The four widely
3
v(t)
𝒲in
r(t)
𝒲out
Input layer
Hidden layer
Output layer
𝒲r
u(t)
Controller 𝑓(𝑡)
Closed loop operation: 
a self-evolving 
dynamical system 
during predicting
Open loop operation 
for training
FIG. 1. Basic structure of the digital twin of a chaotic system. It consists of three layers: the input layer, the hidden recurrent
layer, an output layer, as well as a controller component. The input matrix Win maps the Din-dimensional input chaotic data to
a vector of much higher dimension Dr, where Dr ≫Din. The recurrent hidden layer is characterized by the Dr × Dr weighted
matrix Wr. The dynamical state of the ith neuron in the reservoir is ri, for i = 1, . . . , Dr. The hidden-layer state vector is r(t),
which is an embedding of the input [44]. The output matrix Wout readout the hidden state into the Dout-dimensional output
vector. The controller provides an external driving signal f(t) to the neural network. During training, the vector u(t) is the
input data, and the blue arrow exists during the training phase only. In the predicting phase, the output vector v(t) is directly
fed back to the input layer, generating a closed-loop, self-evolving dynamical system, as indicated by the red arrow connecting
v(t) to u(t). The controller remains on in both the training and predicting phases.
used global optimization methods are genetic algo-
rithm [47–49], particle swarm optimization [50, 51],
Bayesian optimization [52, 53], and surrogate optimiza-
tion [54–56]. We use the surrogate optimization (the al-
gorithm surrogateopt in Matlab). The hyperparameters
that are optimized include d - the average degree of the
recurrent network in the hidden layer, λ - the spectral
radius of the recurrent network, kin - the scaling factor
of Win, kc - the scaling of Wc, c0 - the bias in Eq. (1) and
(2), α - the leakage factor, and β - the ℓ-2 regularization
coeﬃcient. In this paper, the validation of the RC net-
works are done with the same driving signals f(t) as in
the training data. We test driving signals f(t) that are
diﬀerent from those generating the training data (e.g.,
with diﬀerent amplitude, frequency, or waveform).
To
generate the predicted bifurcation diagrams, we let the
RC networks make predictions for long enough periods to
approach the asymptotic behavior. During the warming-
up process to initialize the RC networks prior to making
the predictions, we feed randomly chosen short segments
of the training time series to feed into the RC network.
That is, no data from the target system under the testing
driving signals f(t) are required for making the predic-
tions.
III.
RESULTS
For clarity, we present results on the digital twin for
a prototypical nonlinear dynamical systems with ad-
justable phase-space dimension: the Lorenz-96 climate
network model [57]. In the appendix, we present two ad-
ditional examples: a chaotic laser (Appendix A) and a
driven ecological system (Appendix B), together with a
number of pertinent issues.
A.
A low-dimensional Lorenz-96 climate network
and its digital twin
The Lorenz-96 system [57] is an idealized atmospheric
climate model. Mathematically, the toy climate system is
described by m coupled ﬁrst-order nonlinear diﬀerential
equations subject to external periodic driving f(t):
dxi
dt = xi−1(xi+1 −xi−2) −xi + f(t),
(4)
where i = 1, . . . , m, is the spatial index. Under the peri-
odic boundary condition, the m nodes constitute a ring
network, where each node is coupled to three neighboring
nodes. To be concrete, we set m = 6 (more complex high-
dimensional cases are treated below). The driving force
is sinusoidal with a bias F: f(t) = A sin(ωt) + F. We ﬁx
ω = 2 and F = 2, and use the forcing amplitude A as
4
Driving the
real system
Driving the
real system
Driving the
digital twin
Driving the
digital twin
FIG. 2. Digital twin of the Lorenz-96 climate system. The toy climate system is described by six coupled ﬁrst-order nonlinear
diﬀerential equations (phase-space dimension m = 6), which is driven by a sinusoidal signal f(t) = A sin(ωt) + F. (A1,A2)
Ground truth: chaotic and quasi-periodic dynamics in the system for A = 2.2 and A = 1.6, respectively, for ω = 2 and F = 2.
The sinusoidal driving signals f(t) are schematically illustrated. (B1, B2) The corresponding dynamics of the digital twin under
the same driving signal f(t). Training of the digital twin is conducted using time series from the chaotic regime. The result
in (B2) indicates that the digital twin is able to extrapolate outside the chaotic regime to generate the unseen quasi-periodic
behavior. (C, D) True and digital-twin generated bifurcation diagrams of the toy climate system, where the four vertical red
dashed lines indicate the values of driving amplitudes A, from which the training time series data are obtained. The remarkable
agreement between the two bifurcation diagrams attests to the strong ability of the digital twin to reproduce the distinct
dynamical behaviors of the target climate system in diﬀerent parameter regimes, even with training data only in the chaotic
regime. Note that there are mismatches in the details such as the positions of some periodic windows.
the bifurcation parameter. For relatively large values of
A, the system exhibits chaotic behaviors, as exempliﬁed
in Fig. 2(A1) for A = 2.2. Quasi-periodic dynamics arise
for smaller values of A, as exempliﬁed in Fig. 2(A2). As
A decreases from a large value, a critical transition from
chaos to quasi-periodicity occurs at Ac ≈1.9. We train
the digital twin with time series from four values of A, all
in the chaotic regime: A = 2.2, 2.6, 3.0, and 3.4. The size
of the random reservoir network is Dr = 1, 200. For each
value of A in the training set, the training and validation
lengths are t = 2, 500 and t = 12, respectively, where
the latter corresponds to approximately ﬁve Lyapunov
times. The warming-up length is t = 20 and the time
step of the reservoir dynamical evolution is ∆t = 0.025.
The hyperparameter values (See Sec. II for their mean-
ings) are optimized to be d = 843, λ = 0.48, kin = 0.29,
kc = 0.113, α = 0.41, and β = 1 × 10−10. Our compu-
tations reveal that, for the deterministic version of the
Lorenz-96 model, it is diﬃcult to reduce the validation
error below a small threshold. However, adding an appro-
priate amount of noise into the training time series [18]
can lead to smaller validation errors. We add an additive
Gaussian noise with standard deviation σnoise to each
input data channel to the reservoir network [including
the driving channel f(t)]. The noise amplitude σnoise is
treated as an additional hyperparameter to be optimized.
For the toy climate system, we test several noise levels
and ﬁnd the optimal noise level giving the best validating
performance: σnoise ≈10−3.
Figures 2(B1) and 2(B2) show the dynamical behav-
iors generated by the digital twin for the same values
of A as in Figs. 2(A1) and 2(A2), respectively. It can
be seen that not only does the digital twin produce the
correct dynamical behavior in the same chaotic regime
where the training is carried out, it can also extrapolate
beyond the training parameter regime to correctly pre-
dict the unseen system dynamics there (quasiperiodicity
in this case). To provide support in a broader parameter
range, we calculate true bifurcation diagram, as shown in
Fig. 2(C), where the four vertical dashed lines indicate
5
the four values of the training parameter. The bifurca-
tion generated by the digital twin is shown in Fig. 2(D),
which agrees remarkably well with the true diagram even
at a detailed level. Note that there are mismatches in the
details such as the positions of some periodic windows
in Figs. 2(C) and 2(D). To predict all the features in a
bifurcation diagram requires extensive interpolation and
extrapolation of the system dynamics in the phase space.
Previously, it was suggested that RC can have a cer-
tain degree of extrapolability [34–39].
Figure 2 repre-
sents an example where the target system’s response is
extrapolated to external sinusoidal driving with unseen
amplitudes. In general, extrapolation is a diﬃcult prob-
lem. Some limitations of the extrapolability with respect
to the external driving signal is discussed in Appendix
A, where the digital twin can predict the crisis point but
cannot extrapolate the asymptotic behavior after the cri-
sis.
In the following, we systematically study the applica-
bility of the digital twin in solving forecasting problems
in more complicated situations than the basic settings
demonstrated in Fig. 2. The issues to be addressed are
high dimensionality, the eﬀect of the waveform of the
driving on forecasting, and the generalizability across
Lorenz-96 networks of diﬀerent sizes. Results of contin-
ual forecasting and inferring hidden dynamical variables
using only rare updates of the observable are presented
in Appendices C and D, respectively.
B.
Digital twins of parallel RC neural networks for
high-dimensional Lorenz-96 climate networks
We extend the methodology of digital twin to high-
dimensional Lorenz-96 climate networks, e.g., m = 20.
To deal with such a high-dimensional target system, if a
single reservoir system is used, the required size of the
neural network in the hidden layer will be too large to
be computationally eﬃcient. We thus turn to the par-
allel conﬁguration [25] that consists of many small-size
RC networks, each “responsible” for a small part of the
target system. For the Lorenz-96 network with m = 20
coupled nodes, our digital twin consists of ten parallel RC
networks, each monitoring and forecasting the dynamical
evolution of two nodes (Dout = 2). Because each node in
the Lorenz-96 network is coupled to three nearby nodes,
we set Din = Dout + Dcouple = 2 + 3 = 5 to ensure that
suﬃcient information is supplied to each RC network.
The speciﬁc parameters of the digital twin are as fol-
lows. The size of the recurrent layer is Dr = 1, 200. For
each training value of the forcing amplitude A, the train-
ing and validation lengths are t = 3, 500 and t = 100,
respectively. The “warming up” length is t = 20 and the
time step of the dynamical evolution of the digital twin
is ∆t = 0.025.
The optimized hyperparameter values
are d = 31, λ = 0.75, kin = 0.16, kc = 0.16, α = 0.33,
β = 1 × 10−12, and σnoise = 10−2.
The periodic signal used to drive the Lorenz-96 cli-
mate network of 20 nodes is f(t) = A sin(ωt) + F with
ω = 2, and F = 2. The structure of the digital twin con-
sists of 20 small RC networks as illustrated in Fig. 3(A).
Figures 3(B1) and 3(B2) show a chaotic and a periodic
attractor for A = 1.8 and A = 1.6, respectively, in the
(x1, x2) plane. Training of the digital twin is conducted
by using four time series from four diﬀerent values of
A, all in the chaotic regime. The attractors generated
by the digital twin for A = 1.8 and A = 1.6 are shown
in Figs. 3(C1) and 3(C2), respectively, which agree well
with the ground truth. Figure 3(D) shows the bifurca-
tion diagram of the target system (the ground truth),
where the four values of A: A = 1.8, 2.2, 2.6, and 3.0,
from which the training chaotic time series are obtained,
are indicated by the four respective vertical dashed lines.
The bifurcation diagram generated by the digital twin is
shown in Fig. 3(E), which agrees well with the ground
truth in Fig. 3(D).
C.
Digital twins under external driving with varied
waveform
The external driving signal is an essential ingredient
in our articulation of the digital twin, which is particu-
larly relevant to critical systems of interest such as the
climate systems. In applications, the mathematical form
of the driving signal may change with time. Can a dig-
ital twin produce the correct system behavior under a
driving signal that is diﬀerent than the one it has “seen”
during the training phase? Note that, in the examples
treated so far, it has been demonstrated that our digital
twin can extrapolate the dynamical behavior of a target
system under a driving signal of the same mathematical
form but with a diﬀerent amplitude. Here, the task is
more challenging as the form of the driving signal has
changed.
As a concrete example, we consider the Lorenz-96 cli-
mate network of m = 6 nodes, where a digital twin is
trained with a purely sinusoidal signal f(t) = A sin(ωt)+
F, as illustrated in the left column of Fig. 4(A). During
the testing phase, the driving signal has the form of the
sum of two sinusoidal signals with diﬀerent frequencies:
f(t) = A1 sin(ω1t) + A2 sin(ω2t + ∆φ) + F, as illustrated
in the right panel of Fig. 4(A). We set A1 = 2, A2 = 1,
ω1 = 2, ω2 = 1, F = 2, and use ∆φ as the bifurca-
tion parameter. The RC parameter setting is the same
as that in Fig. 2. The training and validating lengths
for each driving amplitude A value are t = 3, 000 and
t = 12, respectively. We ﬁne that this setting prevents
the digital twin from generating an accurate bifurcation
diagram, but a small amount of dynamical noise to the
target system can improve the performance of the digital
twin. To demonstrate this, we apply an additive noise
term to the driving signal f(t) in the training phase:
df(t)/dt = ωA cos(ωt) + δDNξ(t), where ξ(t) is a Gaus-
sian white noise of zero mean and unit variance, and δDN
is the noise amplitude (e.g., δDN = 3 × 10−3). We use
6
FIG. 3. Digital twin consisting of a number of parallel RC neural networks for high-dimensional chaotic systems. The target
system is the Lorenz-96 climate network of m = 20 nodes, subject to a global periodic driving f(t) = A sin(ωt) + F. (A) The
structure of the digital twin, where each ﬁlled green circle represents a small RC network with the input dimension Din = 5 and
output dimension Dout = 2. (B1, B2) A chaotic and periodic attractor in a two-dimensional subspace of the target system for
A = 1.8 and A = 1.6, respectively, for ω = 2 and F = 2. (C1, C2) The attractors generated by the digital twin corresponding to
those in (B1, B2), respectively, where the training is done using four time series from four diﬀerent values of forcing amplitude
A, all in the chaotic regime. The digital twin with a parallel structure is able to successfully extrapolate the unseen periodic
behavior with completely chaotic training data. (D, E) The true and digital-twin generated bifurcation diagrams, respectively,
where the four vertical dashed lines in (c) specify the four values of A from which the training time series are obtained. The
remarkable agreement between the two bifurcation diagrams indicates that the digital twin so trained can faithfully generate
the dynamical behaviors of the high-dimensional target system.
the 2nd-order Heun method [58] to solve the stochas-
tic diﬀerential equations describing the target Lorenz-96
system. Intuitively, the noise serves to excite diﬀerent
modes of the target system to instill richer information
into the training time series, making the process of learn-
ing the target dynamics more eﬀective. Figures 4(B) and
4(C) show the actual and digital-twin generated bifur-
cation diagrams. Although the digital twin encountered
driving signals in a completely “uncharted territory,” it
is still able to generate the bifurcation diagram with a
reasonable accuracy. The added dynamical noise is cre-
ating small ﬂuctuations in the driving signal f(t). This
may yield richer excited dynamical features of the tar-
get system in the training data set, which can be learned
by the RC network.
This should be beneﬁcial for the
RC network to adapt to diﬀerent waveform in the test-
ing. Additional results with varying testing waves f(t)
are presented in Appendix E.
D.
Extrapolability of digital twin with respect to
system size
In the examples studied so far, it has been demon-
strated that our RC based digital twin has a strong
extrapolability in certain dimensions of the parameter
space.
Speciﬁcally, the digital twin trained with time
series data from one parameter region can follow the dy-
namical evolution of the target system in a diﬀerent pa-
rameter regime. One question is whether the digital twin
possesses certain extrapolability in the system size. For
example, consider the Lorenz-96 climate network of size
m. In Fig. 3, we use an array of parallel RC networks to
construct a digital twin for the climate network of a ﬁxed
size m, where the number of parallel RCs is m/2 (assum-
ing that m is even), and training and testing/monitoring
are carried out for the same system size. We ask, if a dig-
ital twin is trained for climate networks of certain sizes,
will it have the ability to generate the correct dynamical
behaviors for climate networks of diﬀerent sizes? If yes,
we say that the digital twin has the extrapolability with
respect to system size.
7
Driving Signals
for Training
Driving Signals
for Testing
0
0.5
1
1.5
2
2.5
3
0
2
4
6
0
0.5
1
1.5
2
2.5
3
0
2
4
6
A
B
C
FIG. 4. Eﬀects of waveform change in the external driving
on the performance of the digital twin. The time series used
to train the digital twin are from the target system subject
to external driving of a particular waveform.
A change in
the waveform occurs subsequently, leading to a diﬀerent driv-
ing signal during the testing phase. (A) During the training
phase, the driving signal is of the form f(t) = A sin(ωt) + F
and time series from four diﬀerent values of A are used for
training the digital twin. The right panel illustrates an ex-
ample of the changed driving signal during the testing phase.
(B) The true bifurcation diagram of the target system under a
testing driving signal. (C) The bifurcation diagram generated
by the digital twin, facilitated by an optimal level of training
noise determined through hyperparameter optimization.
As an example, we create a digital twin with a parallel
structure based on time series data from the Lorenz-96
climate networks of sizes m = 6 and m = 10, i.e., with
m/2 = 3 and m/2 = 5 numbers of identical RC networks
coupled in a parallel fashion. Testing is done with the
same individual RC networks that are coupled together
to simulate the target system of diﬀerent system sizes.
We also test if the digital twins can make predictions of
the system dynamics under driving signals with unseen
amplitudes.
The training data with m = 6 and m =
10 is shown in Fig. 5(A). For each system size in the
FIG. 5. Demonstration of extrapolability of digital twin in
system size. (A) The digital twin is trained using time se-
ries from the Lorenz-96 climate networks of size m = 6 and
m = 10. The target climate system is subject to a sinusoidal
driving f(t) = A sin(ωt)+F, and the training time series data
are from the A values marked by the eight vertical orange
dashed lines. (B) The true bifurcation diagrams of the target
climate network of size m = 4 and m = 12. (C) The corre-
sponding digital-twin generated bifurcation diagrams, where
the twin consists of m/2 parallel RC networks, each taking in-
put from two nodes in the target system and from the nodes
in the network that are coupled to the two nodes.
training set, four values of the forcing amplitude A are
used to generate the training time series: A =1.5, 2.0, 2.5,
and 3.0, as marked by the vertical orange dashed lines in
Figs. 5(A) and 5(B). As in Fig. 3, the digital twin consists
of m/2 parallel RC networks, each of size Dr = 1, 500.
The optimized hyperparameter values are determined to
be d = 927, λ = 0.71, kin = 0.076, kc = 0.078, α = 0.27,
β = 1 × 10−11, and σnoise = 3 × 10−3. Then we consider
8
climate networks of two diﬀerent sizes: m = 4 and m =
12, and test if the trained digital twin can be adapted to
the new systems. For the network of size m = 4, we keep
only two parallel RC networks for the digital twin. For
m = 12, we add one additional RC network to the trained
digital twin for m = 10, so the new twin consists of six
parallel RC networks of the same hyperparameter values.
The true bifurcation diagrams for the climate system of
sizes m = 4 and m = 12 are shown in Fig. 5(B) (the
left and right panels, respectively). The corresponding
bifurcation diagrams generated by the adapted digital
twins are shown in Fig. 5(C), which agree with the ground
truth reasonably well, demonstrating that our RC based
digital twin possesses certain extrapolability in system
size.
IV.
DISCUSSION
We have articulated the principle of creating digital
twins for nonlinear dynamical systems based on RCs that
are recurrent neural networks. In general, RC is a power-
ful neural network framework that does not require back-
propagation during training but only a linear regression
is needed. This feature makes the development of digital
twins based on RC computationally eﬃcient. We have
demonstrated that a well-trained RC network is able to
serve as a digital twin for systems subject to external,
time-varying driving. The twin can be used to anticipate
possible critical transitions or regime shifts in the target
system as the driving force changes, thereby providing
early warnings for potential catastrophic collapse of the
system. We have used a variety of examples from diﬀer-
ent ﬁelds to demonstrate the workings and the anticipat-
ing power of the digital twin, which include the Lorenz-
96 climate network of diﬀerent sizes (in the main text),
a driven chaotic CO2 laser system (Appendix A), and
an ecological system (Appendix B). For low-dimensional
nonlinear dynamical systems, a single RC network is suf-
ﬁcient for the digital twin. For high-dimensional systems
such as the climate network of a relatively large size,
parallel RC networks can be integrated to construct the
digital twin. At the level of the detailed state evolution,
our recurrent neural network based digital twin is essen-
tially a dynamical twin system that evolves in parallel
to the real system, and the evolution of the digital twin
can be corrected from time to time using sparse feedback
of data from the target system (Appendix C). In cases
where direct measurements of the target system are not
feasible or are too costly, the digital twin provides a way
to assess the dynamical evolution of the target system.
At the qualitative level, the digital twin can faithfully re-
produce the attractors of the target system, e.g., chaotic,
periodic, or quasiperiodic, without the need of state up-
dating. In addition, we show that the digital twin is able
to accurately predict a critical bifurcation point and the
average lifetime of transient chaos that occurs after the
bifurcation, even under a driving signal that is diﬀerent
from that during the training (Appendix F). The issue
of robustness against dynamical and observational noises
in the training data has also been treated (Appendix G).
To summarize, our RC based digital twins are capa-
ble of performing the following tasks: (1) extrapolating
certain dynamical evolution of the target system beyond
the training parameter regime, (2) making long-term con-
tinual forecasting of nonlinear dynamical systems under
nonstationary external driving with sparse state updates,
(3) inferring the existence of hidden variables in the sys-
tem and reproducing/predicting their dynamical evolu-
tion, (4) adapting to external driving of diﬀerent wave-
form, and (5) extrapolating the global bifurcation behav-
iors to systems of diﬀerent sizes.
Our design of the digital twins for nonlinear dynamical
systems can be extended in a number of ways.
1.
Online learning.
Online or continual learning is
a recent trend in machine-learning research. Unlike the
approach of batch learning, where one gathers all the
training data in one place and does the training on the
entire data set (the way by which training is conducted
for our work), in an online learning environment, one
evolves the machine learning model incrementally with
the ﬂow of data. For each training step, only the newest
inputted training data is used to update the machine
learning model. When a new data set is available, it is
not necessary to train the model over again on the en-
tire data set accumulated so far, but only on the new
set.
This can result in a signiﬁcant reduction in the
computational complexity. Previously, an online learn-
ing approach to RC known as the FORCE learning was
developed [59]. An attempt to deal with the key problem
of online learning termed “catastrophic forgetting” was
made in the context of RC [60]. Further investigation
is required to see if these methods can be exploited for
creating digital twins through online learning.
2. Beyond reservoir computing.
Second, the poten-
tial power of recurrent neural network based digital twin
may be further enhanced by using more sophisticated
recurrent neural network models depending on the tar-
get problem. We use the RC networks because they are
relatively simple yet powerful enough for both low- and
high-dimensional dynamical systems.
Schemes such as
knowledge-based hybrid RC [61] or ODE-nets [62] are
worth investigating.
3. Reinforcement learning.
Is it possible to use digi-
tal twins to make reinforcement learning feasible in sit-
uations where the target system cannot be “disturbed”?
Particularly, reinforcement learning requires constant in-
teraction with the target system during training so that
the machine can learn from its mistakes and successes.
However, for a real-world system, these interactions may
be harmful, uncontrollable, and irreversible. As a result,
reinforcement learning algorithms are rarely applied to
safety-critical systems [63].
In this case, digital twins
can be beneﬁcial. By building a digital twin, the rein-
forcement learning model does not need to interact with
the real system, but with its simulated replica for eﬃcient
9
training. This area of research is called model-based re-
inforcement learning [64].
4. Potential beneﬁts of noise.
A phenomenon uncov-
ered in our study is the beneﬁcial role of dynamical noise
in the target system.
As brieﬂy discussed in Fig. 4,
adding dynamic noise in the training dataset enhances
the digital twin’s ability to extrapolate the dynamics of
the target system with diﬀerent waveform of driving. In-
tuitively, noise can facilitate the exploration of the phase
space of the target nonlinear system. A systematic study
of the interplay between dynamical noise and the perfor-
mance of the digital twin is worthy.
5. Extrapolability.
The demonstrated extrapolability
of our digital twin, albeit limited, may open the door
to forecasting the behavior of large systems using twins
trained on small systems. Much research is needed to
address this issue.
6.
Spatiotemporal dynamical systems with multista-
bility.
We have considered digital twins for a class
of coupled dynamical systems:
the Lorenz-96 climate
model.
When developing digital twins for spatiotem-
poral dynamical systems, two issues can arise. One is
the computational complexity associated with such high-
dimensional systems. We have demonstrated that paral-
lel reservoir computing provides a viable solution. An-
other issue is multistability. Spatiotemporal dynamical
systems in general exhibit extremely rich dynamical be-
haviors such as chimera states [65–73]. To develop digital
twins of spatiotemporal dynamical systems with multi-
ple coexisting states requires that the underlying recur-
rent neural networks possess certain memory capabilities.
To develop methods to incorporate memories into digital
twins is a problem of current interest.
DATA AVAILABILITY
All relevant data are available from the authors upon
request.
CODE AVAILABILITY
All relevant computer codes are available from the au-
thors upon request.
ACKNOWLEDGMENT
We thank Z.-M. Zhai for discussions. This work was
supported by the Army Research Oﬃce through Grant
No. W911NF-21-2-0055 and by the U.S.-Israel Energy
Center managed by the Israel-U.S. Binational Industrial
Research and Development (BIRD) Foundation.
AUTHOR CONTRIBUTIONS
All authors designed the research project, the models,
and methods. L.-W.K. performed the computations. All
analyzed the data. L.-W.K. and Y.-C.L wrote the paper.
COMPETING INTERESTS
The authors declare no competing interests.
CORRESPONDENCE
To whom correspondence should be addressed. E-mail:
Ying-Cheng.Lai@asu.edu.
Appendix A: A driven chaotic laser system
We consider the single-mode, class B, driven chaotic
CO2 laser system [74–77] described by
du
dt = −u[f(t) −z],
(A1)
dz
dt = ϵ1z −u −ϵ2zu + 1,
(A2)
where the dynamical variables u and z are proportional
to the normalized intensity and the population inversion,
f(t) = A cos(Ωt + φ) is the external sinusoidal driving
signal of amplitude A and frequency Ω, ϵ1 and ϵ2 are two
parameters. Chaos is common in this laser system [74, 75,
77]. For example, for ϵ1 = 0.09, ϵ2 = 0.003, and A = 1.8,
there is a chaotic attractor for Ω< Ωc ≈0.912, as shown
by a sustained chaotic time series in Fig. 6(a1).
The
chaotic attractor is destroyed by a boundary crisis [78] at
Ωc. For Ω> Ωc, there is transient chaos, after which the
system settles into periodic oscillations, as exempliﬁed
in Fig. 6(a2). Suppose chaotic motion is desired. The
crisis bifurcation at Ωc can then be regarded as a kind of
system collapse.
To build a digital twin for the chaotic laser system, we
use the external driving signal as the natural control sig-
nal for the RC network. Diﬀerent from the examples in
the main text, here the driving frequency Ω, instead of
the driving amplitude A, serves as the bifurcation param-
eter. Assuming observational data in the form of time
series are available for several values of Ωin the regime
of a chaotic attractor, we train the RC network using
chaotic time series collected from four values of Ω< Ωc:
Ω= 0.81, 0.84, 0.87, and 0.90. The training parameter
setting is as follows. For each Ωvalue in the training
set, the training and validation lengths are t = 2, 000
and t = 83, respectively, where the latter corresponds to
approximately ﬁve Lyapunov times. The “warming up”
length is t = 0.5.
The time step of the reservoir sys-
tem is ∆t = 0.05. The size of the random RC network is
10
driving the
real system
driving the
real system
driving the
digital twin
driving the
digital twin
FIG. 6. Performance of the digital twin of the driven CO2 laser system to extrapolate system dynamics under diﬀerent driving frequencies.
(A1, A2) True sustained and transient chaotic time series of log10 u(t) of the target system, for driving frequencies Ω= 0.905 < Ωc and
Ω= 0.925 > Ωc, respectively, where the sinusoidal driving signal f(t) is schematically illustrated. In (A1), the system exhibits sustained
chaos. In (A2), the system settles into a periodic state after transient chaos. (B1, B2) The corresponding time series generated by the
digital twin. In both cases, the dynamical behaviors generated by the digital twin agree with the ground truth in (A1, A2): sustained
chaos in (B1) and transient chaos to a periodic attractor in (B2). (C1, C2) The return maps constructed from the local minima of u(t)
from the true dynamics, where the green dashed square deﬁnes an interval that contains the chaotic attractor (C1) or a nonattracting
chaotic set due to the escaping region (marked by the red arrow) leading to transient chaos (C2). (D1, D2) The return maps generated
by the digital twin for the same values of Ωas in (C1, C2), respectively, which agree with the ground truth.
Dr = 800. The optimal hyperparameter values are deter-
mined to be d = 151, λ = 0.0276, kin = 1.18, kc = 0.113,
α = 0.33, and β = 2 × 10−4.
Figures 6(A1) and 6(A2) show two representative time
series from the laser model (the ground truth) for Ω=
0.905 < Ωc and Ω= 0.925 > Ωc, respectively. The one
in panel (A1) is associated with sustained chaos (pre-
critical) and the other in panel (A2) is characteristic
of transient chaos with a ﬁnal periodic attractor (post-
critical). The corresponding time series generated by the
digital twin are shown in Figs. 6(B1) and 6(B2), respec-
tively.
It can be seen that the training aided by the
control signal enables the digital twin to correctly cap-
ture the dynamical climate of the target system, e.g.,
sustained or transient chaos. The true return maps in
the pre-critical and post-critical regimes are shown in
Figs. 6(C1) and 6(C2), respectively, and the correspond-
ing maps generated by the digital twin are shown in
Figs. 6(D1) and 6(D2).
In the pre-critical regime, an
invariant region (the green dashed square) exists on the
return map in which the trajectories are conﬁned, leading
to sustained chaotic motion, as shown in Figs. 6(C1) and
6(D1). Within the invariant region in which the chaotic
attractor lives, the digital twin captures the essential dy-
namical features of the attractor. Because the training
data are from the chaotic attractor of the target system,
the digital twin fails to generate the portion of the real
return map that lies outside the invariant region, which is
expected because the digital twin has never been exposed
to the dynamical behaviors that are not on the chaotic
attractor. In the post-critical regime, a “leaky” region
emerges, as indicated by the red arrows in Figs. 6(C2)
and 6(D2), which destroys the invariant region and leads
to transient chaos. The remarkable feature is that the
digital twin correctly assesses the existence of the leaky
region, even when no such information is fed into the
twin during training. From the point of view of predict-
ing system collapse, the digital twin is able to anticipate
the occurrence of the crisis and transient chaos. A quan-
titative result of these predictions are demonstrated in
F.
As
indicated
by
the
predicted
return
maps
in
Figs. 6(D1) and 6(D2), the digital twin is unable to give
the ﬁnal state after the transient, because such state must
necessarily lie outside the invariant region from which the
training data are originated.
In particular, the digital
twin is trained with time series data from the chaotic at-
tractors prior to the crisis. With respect to Figs. 6(D1)
and 6(D2), the digital twin can learn the dynamics within
the dash green box in the plotted return maps, but is un-
able to predict the dynamics outside the box, as it has
never been exposed to these dynamics.
A comparison of the real and predicted bifurcation di-
agram is demonstrated in Fig. 7. The strong resemblance
between them indicate the power of the digital twin in ex-
trapolating the correct global behavior of the target sys-
11
FIG. 7. Comparison of the real (A) and predicted (B) bifurcation diagrams of the driven laser system with varying driving frequencies.
The four vertical grey dashed lines indicate the values of driving frequencies Ωused for training the RC neural network. The strong
resemblance between the two bifurcation diagrams indicates the power of the digital twin in extrapolating the correct global behavior of
the target system, and demonstrates that not only can this approach extrapolate system dynamics to various driving amplitudes A, but
also to varying driving frequency Ω.
tem. Moreover, this demonstrates that not only can this
approach extrapolate with various driving amplitudes A
(as demonstrated in the main text), but the approach
can also work with varying driving frequencies Ω.
Appendix B: A driven chaotic ecological system
We study a chaotic driven ecological system that mod-
els the annual blooms of phytoplankton under seasonal
driving [79]. Seasonality plays a crucial role in ecolog-
ical systems and epidemic spreading of infectious dis-
eases [80], which is usually modeled as a simple periodic
driving force on the system. The dynamical equations of
this model in the dimensionless form are [79]:
dN
dt = I −f(t)NP −qN,
(B1)
dP
dt = f(t)NP −P,
(B2)
where N represents the level of the nutrients, P is the
biomass of the phytoplankton, the Lotka-Volterra term
NP models the phytoplankton uptake of the nutrients, I
represents a small and constant nutrient ﬂow from exter-
nal sources, q is the sinking rate of the nutrients to the
lower level of the water unavailable to the phytoplank-
ton, and f(t) is the seasonality term: f(t) = A sin(ωecot).
The parameter values are [79]: I = 0.02, q = 0.0012, and
ωeco = 0.19.
Climate change can dramatically alter the dynamics
of this ecosystem [81]. We consider the task of forecast-
ing how the system behaves if the climate change causes
the seasonal ﬂuctuation to be more extreme. In partic-
ular, suppose the training data are measured from the
system when it behaves normally under a driving signal
of relatively small amplitude, and we wish to predict the
dynamical behaviors of the system in the future when
the amplitude of the driving signal becomes larger (due
to climate change). The training parameter setting is as
follows. The size of the RC network is Dr = 600 with
Din = Dout = 2. The time step of the evolution of the
network dynamics is ∆t = 0.1.
The training and val-
idation lengths for each value of the driving amplitude
A in the training are t = 1, 500 and t = 500, respec-
tively.
The optimized hyperparameters of the RC are
d = 350, λ = 0.42, kin = 0.39, kc = 1.59, α = 0.131, and
12
FIG. 8. Performance of the digital twin of an ecological system of the blooms of phytoplankton with seasonality. The eﬀect of seasonality
is modeled by a sinusoidal driving signal f(t) = A sin(ωecot). (A1, A2) Chaotic and periodic attractors of this system in the (N, log10 P)
plane for A = 0.45 and A = 0.56, respectively. (B1, B2) The corresponding attractors generated by the digital twin under the same driving
signals f(t) as in (A1, A2). The digital twin has successfully extrapolated the periodical behavior outside the chaotic training region. (C)
The ground-truth bifurcation diagram of the target system. (D) The digital-twin generated bifurcation diagram. In (C) and (D), the four
vertical grey dashed lines indicate the values of driving amplitudes A used for training the RC network. The strong resemblance between
the two bifurcation diagrams indicates the power of the digital twin in extrapolating the correct global behavior of the target system.
β = 1 × 10−7.5.
Figure 8 shows the results of our digital twin approach
on this ecological model to learn from the dynamics under
a few diﬀerent values of the driving amplitude to gener-
ate the correct response of the system to a driving signal
of larger amplitude. In particular, the training data are
collected with the driving amplitude A = 0.35, 0.4, 0.45
and 0.5, all in the chaotic regions. Figures 8(A1) and
8(A2) show the true attractors of the system for A = 0.45
and 0.56, respectively, where the attractor is chaotic in
the former case (within the training parameter regime)
and periodic in the latter (outside the training regime).
The corresponding attractors generated by the digital
twin are shown in Figs. 8(B1) and 8(B2). The digital
twin can not only replicate the chaotic behavior in the
training data [Fig. 8(B1)] but also predict the transi-
tion to a periodic attractor under a driving signal with
larger amplitudes (more extreme seasonality), as shown
in Fig. 8(B2). In fact, the digital twin can faithfully pro-
duce the global dynamical behavior of the system, both
inside and outside the training regime, as can be seen
from the nice agreement between the ground-truth bifur-
cation diagram in Fig. 8(C) and the diagram generated
by the digital twin in Fig. 8(D).
Appendix C: Continual forecasting under
non-stationary external driving with sparse
real-time data
The three examples (Lorenz-96 climate network in the
main text, the driven CO2 laser and the ecological sys-
tem) have demonstrated that our RC based digital twin
is capable of extrapolating and generating the correct
statistical features of the dynamical trajectories of the
target system such as the attractor and bifurcation di-
agram. That is, the digital twin can be regarded as a
“twin” of the target system only on a statistical sense.
In particular, from random initial conditions the digital
twin can generate an ensemble of trajectories, and the
statistics calculated from the ensemble agree with those
of the original system. At the level of individual trajec-
tories, if a target system and its digital twin start from
the same initial condition, the trajectory generated by
the twin can stay close to the true trajectory only for a
short period of time (due to chaos). However, with in-
frequent state updates, the trajectory generated by the
twin can shadow the true trajectory (in principle) for an
arbitrarily long period of time [32], realizing continual
forecasting of the state evolution of the target system.
13
FIG. 9. Continual forecasting of the chaotic ecological system under non-stationary external driving with sparse updates. (A) A non-
stationary sinusoidal driving signal f(t) whose amplitude increases with time. The task for the digital twin is to forecast the response of
the chaotic target system under this driving signal for a relatively long term (B) The trajectory generated by the digital twin (red) in
comparison with the true trajectory (blue). For 0 ≤t <
∼400, the two trajectories match each other with small errors, but the digital-twin
generated trajectory begins to deviate from the true trajectory at t ∼400 (due to chaos). (C) With only sparse updates from real data
at times indicated by the vertical lines (2.5% of the time steps in the given time interval), the digital twin can make relatively accurate
predictions for a long term, demonstrating the ability to perform continual forecasting.
In data assimilation for numerical weather forecast-
ing, the state of the model system needs to be up-
dated from time to time [82–84].
This idea has re-
cently been exploited to realize long-term prediction of
the state evolution of chaotic systems using RC [32].
Here we demonstrate that, even when the driving sig-
nal is non-stationary, the digital twin can still gener-
ate the correct state evolution of the target system.
As a speciﬁc example, we use the chaotic ecosystem
in Eqs. (B1-B2) with the same RC network trained in
Sec. B. Figure 9(A) shows the non-stationary external
driving f(t) = A(t) sin(ωecot) whose amplitude A(t) in-
creases linearly from A(t = 0) = 0.4 to A(t = 2500) = 0.6
in the time interval [0, 2500]. Figure 9(B) shows the true
(blue) and digital-twin generated (red) time evolution of
the nutrient abundance. Due to chaos, without state up-
dates, the two trajectories diverge from each other after
a few cycles of oscillation. However, even with rare state
updates, the two trajectories can stay close to each other
for any arbitrarily long time, as shown in Fig. 9(C). In
particular, there are 800 time steps involved in the time
interval [0, 2500] and the state of the digital twin is up-
dated 20 times, i.e., 2.5% of the available time series data.
We will discuss the results further discussion in the next
section.
Appendix D: Continual forecasting with hidden
dynamical variables
In real-world scenarios, usually not all the dynamical
variables of a target system are accessible. It is often the
case that only a subset of the dynamical variables can
be measured and the remaining variables are inaccessi-
ble or hidden from the outside world. Can a digital twin
still make continual forecasting in the presence of hidden
variables based on the time series data from the accessi-
ble variables? Also, Can the digital twin do this without
knowing that there exists some hidden variables before
training? In general, when there are hidden variables, the
reservoir network needs to sense their existence, encode
them in the hidden state of the recurrent layer, and con-
stantly update them. As such, the recurrent structure of
reservoir computing is necessary, because there must be
a place for the machine to store and restore the implicit
information that it has learned from the data.
Com-
pared with the cases where complete information about
the dynamical evolution of all the observable is available,
when there are hidden variables, it is signiﬁcantly more
challenging to predict the evolution of a target system
driven by an non-stationary external signal using sparse
observations of the accessible variables.
As an illustrative example, we again consider the
ecosystem described by Eqs. (B1) and (B2). We assume
that the dynamical variable N (the abundance of the
nutrients) is hidden and P(t), the biomass of the phy-
toplankton, is externally accessible. Despite the accessi-
bility to P(t), we assume that it can be measured only
occasionally. That is, only sparsely updated data of the
variable P(t) is available. It is necessary that the digi-
tal twin is able to learn some equivalent of N(t) as the
time evolution of P(t) also depends on the value N(t),
and to encode the equivalent in the reservoir network. In
an actual application, when the digital twin is deployed,
14
FIG. 10.
Continual forecasting and monitoring of a hidden dynamical variable in the chaotic ecological system under non-stationary
external driving with sparse updates from the observable. The system is described by Eqs. (B1) and (B2). The dynamical variable N(t)
is hidden, and the other variable P(t) is externally accessible but only sparsely sampled measurement of it can be performed. (A) The
non-stationary sinusoidal driving signal f(t) with a time-varying amplitude. (B) Digital-twin generated time evolution of the accessible
variable P(t) (red) in comparison with the ground truth (blue) in the absence of any state update of P(t). The predicted time evolution
quickly diverges from the true behavior. (C) With sparse updates of P(t) at the times indicated by the purple vertical lines (10% of the
times steps), the digital twin is able to make an accurate forecast of P(t). (D) Digital-twin generated time evolution of the hidden variable
N(t) (red) in comparison with the ground truth (blue) in the absence of any state update of P(t). (E) Accurate forecasting of the hidden
variable N(t) with sparse updates of P(t).
knowledge about the existence of such a hidden variable
is not required.
Figure 10 presents a representative resulting trial,
where Fig. 10(A) shows the non-stationary external driv-
ing signal f(t) (the same as the one in Fig. 9(A)). Fig-
ure 10(B) shows, when the observable P(t) is not up-
dated with the real data, the predicted time series (red)
P(t) diverges from the true time series (blue) after about
a dozen oscillations. However, if P(t) is updated to the
digital twin with the true values at the times indicated by
the purple vertical lines in Fig. 10(C), the predicted time
series P(t) matches the ground truth for a much longer
time. The results suggest that the existence of the hidden
variable does not signiﬁcantly impede the performance of
continual forecasting.
The results in Fig. 10 motivate the following ques-
tions.
First, has the reservoir network encoded infor-
mation about the hidden variable? Second, suppose it
is known that there is a hidden variable and the train-
ing dataset contains this variable, can its evolution be
inferred with only rare updates of the observable during
continual forecasting? Previous results [24, 28, 85] sug-
gested that reservoir computing can be used to infer the
hidden variables in a nonlinear dynamical system. Here
we show that, with a segment of the time series of N(t)
used only for training an additional readout layer, our
digital twin can forecast N(t) with only occasional inputs
of the observable time series P(t). In particular, the ad-
ditional readout layer for N(t) is used only for extracting
information about N(t) from the reservoir network and
its output is never injected back to the reservoir. Con-
sequently, whether this additional task of inferring N(t)
is included or not, the trained output layer for P(t) and
the forecasting results of P(t) are not altered.
Figure 10(D) shows that, when the observable P(t) is
not updated with the real data, the digital twin can to
infer the hidden variable N(t) for several oscillations. If
P(t) is updated with the true value at the times indi-
cated by the purple vertical lines in Fig. 10(C), the dy-
namical evolution of the hidden variable N(t) can also be
accurately predicted for a much longer period of time, as
shown in Fig. 10(E). It is worth emphasizing that dur-
ing the whole process of forecasting and monitoring, no
information about the hidden variable N(t) is required -
15
only sparse data points of the observable P(t) are used.
The training and testing settings of the digital twin
for the task involving a hidden variable are as follows.
The input dimension of the reservoir is Din = 1 because
there is a single observable log10 P(t). The output di-
mension is Dout = 2 with one dimension of the observable
log10 P(t+∆t) in addition to one dimension of the hidden
variable N(t + ∆t). Because of the higher memory re-
quirement in dealing with a hidden variable, a somewhat
larger reservoir network is needed, so we use Dr = 1, 000.
The times step of the dynamical evolution of the neural
network is ∆t = 0.1. The training and validating lengths
for each value of the driving amplitude in the training
are t = 3, 500 and t = 350, respectively. Other optimized
hyperparameters of the reservoir are d = 450, λ = 1.15,
kin = 0.32, kc = 3.1, α = 0.077, β = 1 × 10−8.3, and
σnoise = 10−3.0.
It is also worth noting that Figs. 9 and 10 have
demonstrated the ability of the digital twin to extrap-
olate beyond the parameter regime of the target system
from which the training data are obtained.
In partic-
ular, the digital twin was trained only with time se-
ries under stationary external driving of the amplitude
A = 0.35, 0.4, 0.45, and 0.5.
During the testing phase
associated with both Figs. 9 and 10, the external driving
is non-stationary with its amplitude linearly increasing
from A = 0.4 to A = 0.6. The second half of the time
series P(t) and N(t) in Figs. 9 and 10 are thus beyond
the training parameter regime.
The results in Figs. 9 and 10 help legitimize the termi-
nology “digital twin,” as the reservoir computers subject
to the external driving are dynamical twin systems that
evolve “in parallel” to the corresponding real systems.
Even when the target system is only partially observ-
able, the digital twin contains both the observable and
hidden variables whose dynamical evolution is encoded
in the recurrent neural network in the hidden layer. The
dynamical evolution of the output is constantly (albeit
infrequently) corrected by sparse feedback from the real
system, so the output trajectory of the digital twin shad-
ows the true trajectory of the target system. Suppose
one wishes to monitor a variable in the target system, it
is only necessary to read it from the digital twin instead
of making more (possibly costly) measurements on the
real system.
Appendix E: Digital twins under external driving
with varied waveform
In the main text, it is demonstrated that dynamical
noise added to the driving signal during the training can
be beneﬁcial. Figure 11 presents a comparison between
the noiseless training and the training with dynamical
noise of a strength δDB = 3 × 10−3 (as in the main
text).
The ground-truth bifurcation diagram is shown
in Fig. 11(A) and three examples with diﬀerent reservoir
neural networks for the noiseless (B1, B2, B3) and noisy
(C1, C2, C3) training schemes are shown. All the settings
other than the noise level are the same as that in Fig. 4
in the main text. Though there are still ﬂuctuations in
the predicted results, adding dynamical noise into the
training data can produce bifurcation diagrams that are
in general closer to the ground truth than without noise.
The results shown in Fig. 11 also raises the issue of
performance ﬂuctuations in the predicted results among
diﬀerent randomly generated RC networks [46]. It is nec-
essary to train an ensemble of RC networks to obtain a
statistical quantiﬁcation of the performance. An example
is presented in Appendix F, where it is shown that the
ensemble average of the predicted crisis point is accurate.
To further demonstrate the beneﬁcial role of noise, we
test the additive training noise scheme using the ecolog-
ical system. The training process and hyperparameter
values of the digital twin are identical to these in B. A
dynamical noise of amplitude δDB = 3×10−4 is added to
the driving signal f(t) during training in the same way
as in Fig. 4 in the main text. During testing, the driving
signals is altered to
ftest(t) = Atest sin(ωecot) + Atest
2
sin(ωeco
2 t + ∆φ) (E1)
where ωeco = 0.19 as in B. Two sets of testing signals
ftest(t) are used, with Atest = 0.3 and 0.4, respectively.
Figure 12 show the true and predicted bifurcation dia-
grams of log10 Pmax versus ∆φ for Atest = 0.3 (left col-
umn) and Atest = 0.4 (right column). It can be seen that
the bifurcation diagrams generated by the digital twin
with the aid of training noise are remarkably accurate.
We also ﬁnd that, for this ecological system, the ampli-
tude δDB of the dynamical noise during training does not
have a signiﬁcant eﬀect on the predicted bifurcation di-
agram. A plausible reason is that the driving signal f(t)
is a multiplicative term in the system equations.
Appendix F: Quantitative characterization of
performance of digital twin
In the main text, we demonstrate the performance of
the digital twin qualitatively based on visually compar-
ing the predicted bifurcation diagram with the ground
truth. Given the rich bifurcation structure, to quantify
the similarities between two bifurcation diagrams is dif-
ﬁcult. However, for a bifurcation diagram, the parame-
ter values at which the various bifurcations occur are of
great interest, as they deﬁne the critical points at which
characteristic changes in the system can occur. In this
section we focus on the crisis point at which sustained
chaotic motion on an attractor is destroyed and replaced
by transient chaos. And, accordingly, we use two quan-
tities to characterize the performance of the digital twin
in extrapolating the dynamics of the target system: the
errors in the predicted critical bifurcation point and aver-
age lifetime of the chaotic transient after the bifurcation.
16
FIG. 11. Comparisons of the prediction performance between the noiseless (left) and noisy (right) cases on the task of predicting under
external driving with diﬀerent waveform. The target system is a six-dimensional Lorenz-96 system. Panel (A) shows the true bifurcation
diagram. Panels (B1-B3) show the prediction results without any dynamical noise in the training data with three realizations of the
reservoir network. Panels (C1-C3) show the prediction results with dynamical noise of a strength δDB = 3 × 10−3 in the training data.
The settings are the same as that in Fig. 4 in the main text.
As an illustrative example, we take the driven chaotic
laser system in Appendix A, where a crisis bifurcation
occurs at the critical driving frequency Ωc ≈0.912 at
which the chaotic attractor of the system is destroyed
and replaced by a non-attracting chaotic invariant set
leading to transient chaos. We test to determine if the
digital twin can faithfully predict the crisis point based
only on training data from the parameter regime of a
chaotic attractor. Let ˆΩc be the digital-twin predicted
critical point. Figure 13(A) shows the distribution of ˆΩc
obtained from 200 random realizations of the reservoir
neural network. Despite the ﬂuctuations in the predicted
17
FIG. 12. Performance of the digital twin with the ecological model under driving signals with waveform diﬀerent from the training set.
The testing driving signals are described by Eq. E1 while the training driving signals are sinusoidal waves with small dynamical noise.
(A1) The real bifurcation diagram for Atest = 0.3. (A2, A3) Predicted bifurcation diagrams for Atest = 0.3 with two random realizations
of the reservoir networks. (B1-B3) Same as (A1-A3) but with Atest = 0.4.
ˆΩc, their average value is ⟨ˆΩc⟩= 0.914, which is close to
the true value Ωc = 0.912. A relative error εΩof ˆΩc can
then be deﬁned as
εΩ=
|Ωc −ˆΩc|
D(Ωc, {Ωtrain}),
(F1)
where D(Ωc, {Ωtrain}) denotes the minimal distance from
Ωc to the set of training parameter points {Ωtrain}, i.e.,
the diﬀerence between Ωc and the closest training point.
For the driven laser system, we have D(Ωc, {Ωtrain}) ≈
10%.
The second quantity is the lifetime τtransient of tran-
sient chaos after the crisis bifurcation [35, 39], as shown
in Fig. 13(B). The average transient lifetime is the in-
verse of the slope of the linear regression of predicted
data points in Fig. 13(B), which is ⟨τ⟩≈0.8×103. Com-
pared with the true value ⟨τ⟩≈1.2 × 103, we see that
the digital twin is able to predict the average chaotic
transient lifetime to within the same order of magnitude.
Considering that key to the transient dynamics is the
small escaping region in Fig. 6(D2), which is sensitive
to the inevitable training errors, the performance can be
deemed as satisfactory.
Appendix G: Robustness of digital twin against
combined dynamical/observational noises
Can our RC based digital twins withstand the inﬂu-
ences of diﬀerent types of noises? To address this ques-
tion, we introduce dynamical and observational noises in
the training data, which are modeled as additive Gaus-
sian noises.
Take the six-dimensional Lorenz-96 sys-
tem in Sec. IIA in the main text as an example. Fig-
ure 14(A) shows the true bifurcation diagram under dif-
ferent amplitudes of external driving, where the vertical
18
0.9
0.905
0.91
0.915
0.92
0.925
0.93
0
0.05
0.1
0.15
0.2
0.25
500
1000
1500
2000
2500
3000
-2
-1.5
-1
-0.5
real
predicted
A
B
FIG. 13.
Quantitative performance of the digital twin for a chaotic driven laser system. (A) Distribution of the predicted values of
the crisis bifurcation point ˆΩc, at which a chaotic attractor is destroyed and replaced by a non-attracting chaotic invariant set leading
to transient chaos.
The blue and red vertical dashed lines denote the true value Ωc ≈0.912 and the average predicted value ⟨ˆΩc⟩,
respectively, where 200 random realizations of the reservoir neural network are used to generate this distribution. Despite the ﬂuctuations
in the predicted crisis point, the ensemble average value of the prediction is quite close to the ground truth. (B) Exponential distribution
of the lifetime of transient chaos slightly beyond the crisis point: true (blue) and predicted (red) behaviors. The predicted distribution is
generated using 100 random reservoir realizations, each with 200 random initial ‘warming up” data.
dashed lines specify the training points. Figures 14(B1)
and 14(B2) show two realizations of the bifurcation dia-
gram generated by the digital twin under both dynamical
and observational noises of amplitudes σdyn = 10−2 and
σob = 10−2. Two bifurcation diagrams for noise ampli-
tudes of an order of magnitude larger: σdyn = 10−1 and
σob = 10−1, are shown in Figs. 14(C1) and 14(C2). It
can be seen that the additional noises have little eﬀect
on the performance of the digital twin in generating the
bifurcation diagram.
[1] E. J. Eric J. Tuegel, A. R. Ingraﬀea, T. G. Eason, and
S. M. Spottswood, Reengineering aircraft structural life
prediction using a digital twin, Int. J. Aerospace Eng.
2011, 154798 (2011).
[2] F. Tao and Q. Qi, Make more digital twins, Nature 573,
274 (2019).
[3] A. Rasheed, O. San, and T. Kvamsdal, Digital twin: Val-
ues, challenges and enablers from a modeling perspective,
IEEE Access 8, 21980 (2020).
[4] K. Bruynseels, F. S. de Sio, and J. van den Hoven, Digital
twins in health care: Ethical implications of an emerging
engineering paradigm, Front. Gene. 9, 31 (2018).
[5] S. M. Schwartz, K. Wildenhaus, A. Bucher, and B. Byrd,
Digital twins and the emerging science of self: Impli-
cations for digital health experience design and “small”
data, Front. Comp. Sci. 2, 31 (2020).
[6] R. Laubenbacher, J. P. Sluka, and J. A. Glazier, Using
digital twins in viral infection, Science 371, 1105 (2021).
[7] P. Voosen, Europe builds ‘digital twin’ of earth to hone
climate forecasts, Science 370, 16 (2020).
[8] P. Bauer, B. Stevens, and W. Hazeleger, A digital twin
of earth for the green transition, Nat. Clim. Change 11,
80 (2021).
[9] Y.-C. Lai and T. T´el, Transient Chaos - Complex Dynam-
ics on Finite Time Scales (Springer, New York, 2011).
[10] K. McCann and P. Yodzis, Nonlinear dynamics and pop-
ulation disappearances, Ame. Naturalist 144, 873 (1994).
19
FIG. 14. Robustness of digital twin against combined dynamical and observational noises. The setting is the same as that in Fig. 2 in the
main text, except with additional noises in the training data. (A) A true bifurcation diagram of the six-dimensional Lorenz-96 system.
(B1, B2) Two examples of the bifurcation diagram predicted by the digital twin with training data under dynamical noise of amplitude
σdyn = 10−2 and observational noise of amplitude σob = 10−2. (C1, C2) Two examples of the predicted bifurcation diagrams under the
two kinds of noise with σdyn = 10−1 and σob = 10−1. Both the dynamical and observational noises are additive Gaussian processes. It
can be seen that though larger additional noises make the predicted details less accurate, the general shapes of the predicted results are
not harmed signiﬁcantly. The settings of the training data and reservoir neural networks are the same as those in Fig. 2 in the main text.
The dynamical noises are added to the dynamical equations of the state variables. There is no noise in the sinusoidal external driving.
[11] A. Hastings, K. C. Abbott, K. Cuddington, T. Fran-
cis, G. Gellner, Y.-C. Lai, A. Morozov, S. Petrivskii,
K. Scranton, and M. L. Zeeman, Transient phenomena
in ecology, Science 361, eaat6412 (2018).
[12] M. Dhamala and Y.-C. Lai, Controlling transient chaos in
deterministic ﬂows with applications to electrical power
systems and ecology, Phys. Rev. E 59, 1646 (1999).
[13] Y.-C. Lai, C. Grebogi, and J. Kurths, Modeling of deter-
ministic chaotic systems, Phys. Rev. E 59, 2907 (1999).
[14] Y.-C. Lai and C. Grebogi, Modeling of coupled chaotic
oscillators, Phys. Rev. Lett. 82, 4803 (1999).
[15] W.-X. Wang, R. Yang, Y.-C. Lai, V. Kovanis, and
C. Grebogi, Predicting catastrophes in nonlinear dynami-
cal systems by compressive sensing, Phys. Rev. Lett. 106,
154101 (2011).
[16] W.-X. Wang, Y.-C. Lai, and C. Grebogi, Data based
identiﬁcation and prediction of nonlinear and complex
dynamical systems, Phys. Rep. 644, 1 (2016).
[17] Y.-C. Lai, Finding nonlinear system equations and com-
plex network structures from data: A sparse optimization
approach, Chaos 31, 082101 (2021).
[18] H. Jaeger, The “echo state” approach to analysing and
training recurrent neural networks-with an erratum note,
German National Research Center for Information Tech-
nology GMD Technical Report 148, 13 (2001).
20
[19] W. Mass, T. Nachtschlaeger, and H. Markram, Real-time
computing without stable states: A new framework for
neural computation based on perturbations, Neur. Comp.
14, 2531 (2002).
[20] H. Jaeger and H. Haas, Harnessing nonlinearity: Predict-
ing chaotic systems and saving energy in wireless com-
munication, Science 304, 78 (2004).
[21] N. D. Haynes, M. C. Soriano, D. P. Rosin, I. Fischer,
and D. J. Gauthier, Reservoir computing with a single
time-delay autonomous Boolean node, Phys. Rev. E 91,
020801 (2015).
[22] L. Larger, A. Bayl´on-Fuentes, R. Martinenghi, V. S.
Udaltsov, Y. K. Chembo, and M. Jacquot, High-speed
photonic reservoir computing using a time-delay-based
architecture:
Million words per second classiﬁcation,
Phys. Rev. X 7, 011015 (2017).
[23] J. Pathak, Z. Lu, B. Hunt, M. Girvan, and E. Ott, Using
machine learning to replicate chaotic attractors and cal-
culate Lyapunov exponents from data, Chaos 27, 121102
(2017).
[24] Z. Lu, J. Pathak, B. Hunt, M. Girvan, R. Brockett, and
E. Ott, Reservoir observers: Model-free inference of un-
measured variables in chaotic systems, Chaos 27, 041102
(2017).
[25] J. Pathak, B. Hunt, M. Girvan, Z. Lu, and E. Ott, Model-
free prediction of large spatiotemporally chaotic systems
from data: A reservoir computing approach, Phys. Rev.
Lett. 120, 024102 (2018).
[26] T. L. Carroll, Using reservoir computers to distinguish
chaotic signals, Phys. Rev. E 98, 052209 (2018).
[27] K. Nakai and Y. Saiki, Machine-learning inference of
ﬂuid variables from data using reservoir computing, Phys.
Rev. E 98, 023111 (2018).
[28] Z. S. Roland and U. Parlitz, Observing spatio-temporal
dynamics of excitable media using reservoir computing,
Chaos 28, 043118 (2018).
[29] A. Griﬃth, A. Pomerance, and D. J. Gauthier, Forecast-
ing chaotic systems with very low connectivity reservoir
computers, Chaos 29, 123108 (2019).
[30] J. Jiang and Y.-C. Lai, Model-free prediction of spa-
tiotemporal dynamical systems with recurrent neural
networks: Role of network spectral radius, Phys. Rev.
Research 1, 033056 (2019).
[31] G. Tanaka, T. Yamane, J. B. H´eroux, R. Nakane,
N. Kanazawa, S. Takeda, H. Numata, D. Nakano, and
A. Hirose, Recent advances in physical reservoir comput-
ing: A review, Neu. Net. 115, 100 (2019).
[32] H. Fan, J. Jiang, C. Zhang, X. Wang, and Y.-C. Lai,
Long-term prediction of chaotic systems with machine
learning, Phys. Rev. Research 2, 012080 (2020).
[33] C. Zhang, J. Jiang, S.-X. Qu, and Y.-C. Lai, Predicting
phase and sensing phase coherence in chaotic systems
with machine learning, Chaos 30, 083114 (2020).
[34] C. Klos, Y. F. K. Kossio, S. Goedeke, A. Gilra, and R.-M.
Memmesheimer, Dynamical learning of dynamics, Phys.
Rev. Lett. 125, 088103 (2020).
[35] L.-W. Kong, H.-W. Fan, C. Grebogi, and Y.-C. Lai, Ma-
chine learning prediction of critical transition and system
collapse, Phys. Rev. Research 3, 013090 (2021).
[36] D. Patel, D. Canaday, M. Girvan, A. Pomerance, and
E. Ott, Using machine learning to predict statistical
properties of non-stationary dynamical processes: Sys-
tem climate, regime transitions, and the eﬀect of stochas-
ticity, Chaos 31, 033149 (2021).
[37] J. Z. Kim, Z. Lu, E. Nozari, G. J. Pappas, and D. S. Bas-
sett, Teaching recurrent neural networks to infer global
temporal structure from local examples, Nat. Machine
Intell. 3, 316 (2021).
[38] H. Fan, L.-W. Kong, Y.-C. Lai, and X. Wang, Anticipat-
ing synchronization with machine learning, Phys. Rev.
Resesearch 3, 023237 (2021).
[39] L.-W. Kong, H. Fan, C. Grebogi, and Y.-C. Lai, Emer-
gence of transient chaos and intermittency in machine
learning, J. Phys. Complexity 2, 035014 (2021).
[40] E. Bollt, On explaining the surprising success of reservoir
computing forecaster of chaos?
the universal machine
learning dynamical system with contrast to var and dmd,
Chaos 31, 013108 (2021).
[41] D. J. Gauthier, E. Bollt, A. Griﬃth, and W. A. Barbosa,
Next generation reservoir computing, Nat. Commun. 12,
1 (2021).
[42] A. Haluszczynski and C. R¨ath, Controlling nonlinear
dynamical systems into arbitrary states using machine
learning, Scientiﬁc reports 11, 1 (2021).
[43] T. L. Carroll, Optimizing memory in reservoir computers,
Chaos 32, 023123 (2022).
[44] A. Hart, J. Hook, and J. Dawes, Embedding and approx-
imation theorems for echo state networks, Neu. Net. 128,
234 (2020).
[45] The codes of this work are shared at github.com/
lw-kong/Digital_Twin_2021.
[46] J. Herteux and C. R¨ath, Breaking symmetries of the
reservoir equations in echo state networks, Chaos 30,
123142 (2020).
[47] D. E. Goldberg, Genetic Algorithms (Pearson Education
India, 2006).
[48] A. R. Conn, N. I. Gould, and P. Toint, A globally con-
vergent augmented lagrangian algorithm for optimization
with general constraints and simple bounds, SIAM J. Nu-
mer. Anal. 28, 545 (1991).
[49] A. Conn, N. Gould, and P. Toint, A globally conver-
gent lagrangian barrier algorithm for optimization with
general inequality constraints and simple bounds, Math.
Comput. 66, 261 (1997).
[50] J. Kennedy and R. Eberhart, Particle swarm optimiza-
tion, in Proceedings of ICNN’95-International Confer-
ence on Neural Networks, Vol. 4 (IEEE, 1995) pp. 1942–
1948.
[51] E. Mezura-Montes and C. A. C. Coello, Constraint-
handling
in
nature-inspired
numerical
optimization:
past, present and future, Swarm Evol. Comput. 1, 173
(2011).
[52] M. A. Gelbart, J. Snoek, and R. P. Adams, Bayesian
optimization with unknown constraints, arXiv preprint
arXiv:1403.5607 (2014).
[53] J. Snoek, H. Larochelle, and R. P. Adams, Practical
bayesian optimization of machine learning algorithms, in
NeurIPS (2012) pp. 2951–2959.
[54] H.-M. Gutmann, A radial basis function method for
global optimization, J. Global Optim. 19, 201 (2001).
[55] R. G. Regis and C. A. Shoemaker, A stochastic radial
basis function method for the global optimization of ex-
pensive functions, INFORMS J. Comput. 19, 497 (2007).
[56] Y. Wang and C. A. Shoemaker, A general stochastic al-
gorithmic framework for minimizing expensive black box
objective functions based on surrogate models and sensi-
tivity analysis, arXiv preprint arXiv:1410.6271 (2014).
21
[57] E. N. Lorenz, Predictability: A problem partly solved, in
Proc. Seminar on Predictability, Vol. 1 (1996).
[58] C. Van den Broeck, J. Parrondo, R. Toral, and R. Kawai,
Nonequilibrium phase transitions induced by multiplica-
tive noise, Phys. Rev. E 55, 4084 (1997).
[59] D. Sussillo and L. F. Abbott, Generating coherent pat-
terns of activity from chaotic neural networks, Neuron
63, 544 (2009).
[60] T. Kobayashi and T. Sugino, Continual learning ex-
ploiting structure of fractal reservoir computing, in In-
ternational Conference on Artiﬁcial Neural Networks
(Springer, 2019) pp. 35–47.
[61] J. Pathak, A. Wikner, R. Fussell, S. Chandra, B. R.
Hunt, M. Girvan, and E. Ott, Hybrid forecasting of
chaotic processes: Using machine learning in conjunc-
tion with a knowledge-based model, Chaos 28, 041101
(2018).
[62] R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K.
Duvenaud, Neural ordinary diﬀerential equations, Adv.
Neu. Info. Proc. Sys. 31 (2018).
[63] F. Berkenkamp, M. Turchetta, A. P. Schoellig, and
A. Krause, Safe model-based reinforcement learning with
stability guarantees, arXiv preprint arXiv:1705.08551
(2017).
[64] T. M. Moerland, J. Broekens, and C. M. Jonker, Model-
based reinforcement learning: A survey, arXiv preprint
arXiv:2006.16712 (2020).
[65] Y. Kuramoto and D. Battogtokh, Coexistence of coher-
ence and incoherence in nonlocally coupled phase oscil-
lators, Nonlin. Phenom. Complex Syst. 5, 380 (2002).
[66] D. M. Abrams and S. H. Strogatz, Chimera states for
coupled oscillators, Phys. Rev. Lett. 93, 174102 (2004).
[67] I. Omelchenko, Y. Maistrenko, P. H¨ovel, and E. Sch¨oll,
Loss of coherence in dynamical networks: Spatial chaos
and chimera states, Phys. Rev. Lett. 106, 234102 (2011).
[68] M. R. Tinsley, S. Nkomo, and K. Showalter, Chimera and
phase-cluster states in populations of coupled chemical
oscillators, Nat. Phys. 8, 662 (2012).
[69] A. M. Hagerstrom, T. E. Murphy, R. Roy, P. H¨ovel,
I. Omelchenko, and E. Sch¨oll, Experimental observation
of chimeras in coupled-map lattices, Nat. Phys. 8, 658
(2012).
[70] I. Omelchenko,
O. E. Omel’chenko,
P. H¨ovel, and
E. Sch¨oll, When nonlocal coupling between oscillators
becomes stronger: Patched synchrony or multichimera
states, Phys. Rev. Lett. 110, 224101 (2013).
[71] I. Omelchenko, A. Zakharova, P. H¨ovel, J. Siebert, and
E. Sch¨oll, Nonlinearity of local dynamics promotes multi-
chimeras, Chaos 25, 083104 (2015).
[72] I. Omelchenko, O. E. Omel’chenko, A. Zakharova, and
E. Sch¨oll, Optimal design of tweezer control for chimera
states, Phys. Rev. E 97, 012216 (2018).
[73] L.-W. Kong and Y.-C. Lai, Scaling law of transient life-
time of chimera states under dimension-augmenting per-
turbations, Phys. Rev. Research 2, 023196 (2020).
[74] D. Dangoisse, P. Glorieux, and D. Hennequin, Laser
chaotic attractors in crisis, Phys. Rev. Lett. 57, 2657
(1986).
[75] D. Dangoisse, P. Glorieux, and D. Hennequin, Chaos in a
CO2 laser with modulated parameters: experiments and
numerical simulations, Phys. Rev. A 36, 4775 (1987).
[76] H. G. Solari, E. Eschenazi, R. Gilmore, and J. R.
Tredicce, Inﬂuence of coexisting attractors on the dy-
namics of a laser system, Opt. Commun. 64, 49 (1987).
[77] I. B. Schwartz, Sequential horseshoe formation in the
birth and death of chaotic attractors, Phys. Rev. Lett.
60, 1359 (1988).
[78] C. Grebogi, E. Ott, and J. A. Yorke, Crises, sudden
changes in chaotic attractors and chaotic transients,
Physica D 7, 181 (1983).
[79] A. Huppert, B. Blasius, R. Olinky, and L. Stone, A model
for seasonal phytoplankton blooms, J. Theo. Biol. 236,
276 (2005).
[80] L. Stone, R. Olinky, and A. Huppert, Seasonal dynamics
of recurrent epidemics, Nature 446, 533 (2007).
[81] M. Winder and U. Sommer, Phytoplankton response to
a changing climate, Hydrobiologia 698, 5 (2012).
[82] E. Kalnay, Atmospheric Modeling, Data Assimilation and
Predictability (Cambridge university press, 2003).
[83] M. Asch, M. Bocquet, and M. Nodet, Data Assimilation:
Methods, Algorithms, and Applications (SIAM, 2016).
[84] A. Wikner, J. Pathak, B. R. Hunt, I. Szunyogh, M. Gir-
van, and E. Ott, Using data assimilation to train a hy-
brid forecast system that combines machine-learning and
knowledge-based components, Chaos 31, 053114 (2021).
[85] T. Weng, H. Yang, C. Gu, J. Zhang, and M. Small,
Synchronization of chaotic systems and their machine-
learning models, Phys. Rev. E 99, 042203 (2019).
