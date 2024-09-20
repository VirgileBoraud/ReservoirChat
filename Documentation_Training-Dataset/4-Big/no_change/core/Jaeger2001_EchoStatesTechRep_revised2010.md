The “echo state” approach to analysing and
training recurrent neural networks – with an
Erratum note1
Herbert Jaeger
Fraunhofer Institute for Autonomous Intelligent Systems
January 26, 2010
1This is a corrected version of the technical report H. Jaeger(2001): The ”echo
state” approach to analysing and training recurrent neural networks. GMD Report
148, German National Research Center for Information Technology, 2001. In this
report, one part of Deﬁnition 3 was ﬂawed, with repercussions in Proposition 1.
This error was pointed out to me by Tobias Strauss. It his corrected in this version.
For historical veracity, the original techreport wasn’t altered except that margin
notes point to the ﬂaws. The corrections are given in a new Erratum section which
is added as an Appendix.
Abstract. The report introduces a constructive learning algorithm for
recurrent neural networks, which modiﬁes only the weights to output units
in order to achieve the learning task.
key words: recurrent neural networks, supervised learning
Zusammenfassung. Der Report f¨uhrt ein konstruktives Lernverfahren
f¨ur rekurrente neuronale Netze ein, welches zum Erreichen des Lernzieles
lediglich die Gewichte der zu den Ausgabeneuronen f¨uhrenden Verbindungen
modiﬁziert.
Stichw¨orter: rekurrente neuronale Netze, ¨uberwachtes Lernen
3
1
Introduction
Under certain conditions, the activation state x(n) of a recurrent neural
network (RNN) is a function of the input history u(n), u(n−1), . . . presented
to the network, i.e., there exists a function E such that x(n) = E(u(n), u(n−
1), . . .). Metaphorically, the state x(n) can be understood as an “echo” of
the input history.
This article investigates what can be gained when RNN states are un-
derstood as echo states. Speciﬁcally, the article discusses under which con-
ditions echo states arise and describes how RNNs can be trained, exploiting
echo states.
The perspective taken is mainly that of mathematics and engineering,
where a recurrent network is seen as a computational device for realizing
a dynamical system. The RNNs considered here are mostly discrete-time,
sigmoid-unit networks. The basic idea of echo states has been independently
investigated in [11] in a complementary fashion. In that work, the emphasis
is on biological modeling and continuous-time (spiking) networks.
The article is organized as follows.
The second section deﬁnes echo states and provides several equivalent
characterizations.
The third section explains how echo state networks can be trained in a
supervised way.
The natural approach here is to adapt only the weights
of network-to-output connections. Essentially, this trains readout functions
which transform the echo state into the desired output signal. Technically,
this amounts to a linear regression task.
The fourth section gives two complementary, basic examples. Echo state
networks are trained to generate periodic sequences (mathematically: an m-
point attractor), and to function as multiple switches (mathematically: m
point attractors).
The ﬁfth section describes echo state networks with leaky integrator neu-
rons.
The sixth section demonstrates how leaky integrator echo state networks
master the quasi-benchmark task of learning the Mackey-Glass chaotic at-
tractor.
Section 7 concludes with a comparative discussion and points out open
problems.
4
2
Echo states
First we ﬁx our terminology. We consider discrete-time neural networks with
K input units, N internal network units and L output units. Activations of
input units at time step n are u(n) = (u1(n), . . . , uK(n)), of internal units are
x(n) = (x1(n), . . . , xN(n)), and of output units y(n) = (y1(n), . . . , yL(n)).
Real-valued connection weights are collected in a N × K weight matrix
Win = (win
ij) for the input weights, in an N × N matrix W = (wij) for
the internal connections, in an L × (K + N + L) matrix Wout = (wout
ij ) for
the connections to the output units, and in a N × L matrix Wback = (wback
ij
)
for the connections that project back from the output to the internal units.
Note that connections directly from the input to the output units and con-
nections between output units are allowed. No further conditions on the net-
work topology induced by the internal weights W are imposed (e.g., no layer
structure). We will also not formally require, but generally intend that the
internal connections W induce recurrent pathways between internal units.
Without further mention, we will always assume real-valued inputs, weights,
and activations. Figure 1 shows the basic network architecture considered
here.
...
...
K input
units
N internal units
L output
units
Figure 1: The basic network architecture assumed in this article. Dashed
arrows indicate connections that are possible but not required.
The activation of internal units is updated according to
x(n + 1) = f(Winu(n + 1) + Wx(n) + Wbacky(n)),
(1)
where f = (f1, . . . , fN) are the internal unit’s output functions (typically
sigmoid functions). The output is computed according to
y(n + 1) = f out(Wout(u(n + 1), x(n + 1), y(n)),
(2)
5
where f out = (f out
1
, . . . , f out
L ) are the output unit’s output functions and
(u(n + 1), x(n + 1), y(n)) is the concatenation of the input, internal, and
previous output activation vectors.
In this paper we consider input sequences (u(n))n∈J ∈U J. We require
that U be compact. We use shorthand ¯u±∞, ¯u+∞, ¯u−∞, ¯uh to denote input
sequences which are left-right-inﬁnite (J = Z), right-inﬁnite (J = k, k+1, . . .
for some k ∈Z), left-inﬁnite, and ﬁnite of length h, respectively. Similar
shorthands are used for state and output sequences. Other than compactness
we impose no restrictions on input sequences, i.e., every (u(n))n∈J ∈U J is
admissible. This implies shift-invariance of admissible inputs. We introduce
a network state update operator T and write x(n + h) = T(x(n), y(n), ¯uh)
to denote the network state that results from an iterated application of Eq.
(1) if the input sequence u(n+1), . . . , u(n+h) is fed into the network which
at time n is in state x(n) and had output y(n). In networks without output
feedback, this simpliﬁes to x(n + h) = T(x(n), ¯uh).
Recurrent networks (as all nonlinear dynamical systems) may exhibit dis-
joint regions A, B of their state space in which network states stay contained
regardless of input. We therefore discuss network dynamics always relative
to a compact set A ⊂RN of admissible states. Among other things, this
implies that if the network is started at some time n, the initial state x(n)
must be in A. We require that A is closed under network update, i.e. for
every u ∈U, x ∈A, it holds that T(x, u) ∈A.
Thus, our analysis will always rely on the following generic setup: (i) input
is drawn from a compact input space U; (ii) network states lie in a compact
set A. We will call these conditions standard compactness conditions.
We now deﬁne echo state networks for the case of networks with no output
feedback.
Deﬁnition 1 Assume standard compactness conditions.
Assume that the
network has no output feedback connections. Then, the network has echo
states, if the network state x(n) is uniquely determined by any left-inﬁnite
input sequence ¯u−∞. More precisely, this means that for every input sequence
. . . , u(n−1), u(n) ∈U −N, for all state sequences . . . , x(n−1), x(n) and x′(n−
1), x′(n) ∈A−N, where x(i) = T(x(i−1), u(i)) and x′(i) = T(x′(i−1), u(i)),
it holds that x(n) = x′(n).
An equivalent way of stating the echo state property is to say that there
exist input echo functions E = (e1, . . . , eN), where ei : U −N →R, such
that for all left-inﬁnite input histories . . . , u(n −1), u(n) ∈U −N the current
network state is
x(n) = E(. . . , u(n −1), u(n)).
(3)
6
We now provide a number of equivalent characterizations of echo states.
Deﬁnition 2 provides requisite terminology, deﬁnition 4 states the alternative
characterizations.
Deﬁnition 2 (a) A state sequence ¯x−∞= . . . , x(n−1), x(n) ∈A−N is called
compatible with an input sequence ¯u−∞= . . . , u(n −1), u(n), if ∀i < n :
T(x(i), u(i + 1)) = x(i + 1). (b) Similarly, a left-right-inﬁnite state sequence
¯x∞is called compatible with an input sequence ¯u∞, if ∀i : T(x(i), u(i+1)) =
x(i + 1). (c) A network state x ∈A is called end-compatible with an input
sequence ¯u−∞if there exists a state sequence . . . , x(n −1), x(n) such that
T(x(i), u(i + 1)) = x(i + 1), and x = x(n). (d) A network state x ∈A is
called end-compatible with a ﬁnite input sequence ¯uh if there exists a state
sequence x(n −h), . . . , x(n) ∈Ah+1 such that T(x(i), u(i + 1)) = x(i + 1),
and x = x(n).
Deﬁnition 3 Assume standard compactness conditions and a network with-
out output feedback.
1. The network is called state contracting
if for all right-inﬁnite input
sequences ¯u+∞there exists a null sequence (δh)h≥0 such that for all
states x, x′ ∈A, for all h ≥0, for all input sequence preﬁxes ¯uh =
u(n), . . . , u(n + h) it holds that d(T(x, ¯uh), T(x′, ¯uh)) < δh, where d is
the Euclidean distance on RN.1
EdNote(1)
2. The network is called state forgetting if for all left-inﬁnite input se-
quences . . . , u(n −1), u(n) ∈U −N there exists a null sequence (δh)h≥0
such that for all states x, x′ ∈A, for all h ≥0, for all input sequence
suﬃxes ¯uh = u(n −h), . . . , u(n) it holds that d(T(x, ¯uh), T(x′, ¯uh)) <
δh.
3. The network is called input forgetting if for all left-inﬁnite input se-
quences ¯u−∞there exists a null sequence (δh)h≥0 such that for all h ≥0,
for all input sequence suﬃxes ¯uh = u(n −h), . . . , u(n), for all left-
inﬁnite input sequences of the form ¯w−∞¯uh, ¯v−∞¯uh, for all states x
end-compatible with ¯w−∞¯uh and states x′ end-compatible with ¯v−∞¯uh
it holds that d(x, x′) < δh.
Proposition 1 Assume standard compactness conditions and a network with-
out output feedback. Assume that T is continuous in state and input. Then
the properties of being state contracting, state forgetting, and input forgetting
are all equivalent to the network having echo states.2
EdNote(2)
1EdNote: The state contracting property is ﬂawed – see Erratum note in Appendix E
2EdNote: See Erratum note in Appendix E.
7
The proof is given in the Appendix. Echo state networks have a desirable
continuity property. Intuitively, the following proposition states that in order
to know the current network output with a given precision, we only have to
know the most recent inputs with a similar precision:
Proposition 2 Assume standard compactness conditions, no output feed-
back, continuity of T in state and input, and echo states. Then it holds that
for all left-inﬁnite input sequences ¯u−∞, for all ε > 0 there exist δ > 0 and
h > 0 such that d(E(¯u−∞), E(¯u′−∞)) < ε for all input sequences ¯u′−∞which
satisfy d(u(k), u′(k)) < δ for all −h ≤k ≤0.
The proof is given in the Appendix. This continuity property was called
“fading memory” in [11].
The conditions of Def. 4 are hard to check in practice. Unfortunately,
I cannot oﬀer conditions for echo states that are both easy to check and
equivalent to echo states. The next proposition gives (a) a suﬃcient condition
for echo states, which depends on a Lipschitz property of the weight matrix,
and (b) a suﬃcient condition for the non-existence of echo states.
Proposition 3 Assume a sigmoid network with unit output functions fi =
tanh. (a) Let the weight matrix W satisfy σmax = Λ < 1, where σmax is its
largest singular value. Then d(T(x, u), T(x′, u)) < Λ d(x, x′) for all inputs
u, for all states x, x′ ∈[−1, 1]N. This implies echo states for all inputs u, for
all states x, x′ ∈[−1, 1]N. (b) Let the weight matrix have a spectral radius
|λmax | > 1, where λmax is an eigenvalue of W with the largest absolute value.
Then the network has an asymptotically unstable null state. This implies that
it has no echo states for any input set U containing 0 and admissible state
set A = [−1, 1]N.
The proof is given in the Appendix. Both conditions are easy to check and
mark the boundaries of an interesting scaling range for weight matrices, as
follows. In practice, a convenient strategy to obtain useful echo state net-
works is to start with some weight matrix ˜
W and try out global scalings α ˜
W
until one is satisﬁed with the properties of some ﬁnally ﬁxed weight matrix
W = αopt ˜
W. Let σmax(W) and |λmax | (W) denote the largest singular value
and the spectral radius of a matrix W. Observe that the maximal singular
value and the spectral radius of ˜
W scale with α, i.e. σmax(α ˜
W) = ασmax( ˜
W)
and | λmax | (α ˜
W) = α | λmax | ( ˜
W). Observe further that for every square
matrix W, | λmax | (W) ≤σmax(W). Thus, if one puts αmin = 1/σmax( ˜
W)
and αmax = 1/ | λmax | ( ˜
W), one obtains a scaling range αmin ≤α ≤αmax,
where below the smallest scaling factor αmin one would certainly have echo
8
states, and above αmax, certainly not. My (by now extensive) experience
with this scaling game indicates that one obtains echo states even when α is
only marginally smaller than αmax: the suﬃcient condition from Prop. 3(a)
apparently is very restrictive.
Note also that if a network has a weight matrix with | λmax | > 1, the
echo state property can hold nonetheless if input comes from a set U not
containing the null input. For a demonstration of this fact, consider a single-
unit network (N = 1) with a weight matrix W = (2). For null input, the
network has an instable null state and two stable states s, −s, where s is
deﬁned by tanh(2s) = s, so the network is clearly not echo state for an input
set U containing 0 and admissible state set A = [−1, 1]N. But if one restricts
the input to suﬃciently high values, e.g. to U = [3, 4], a Lipschitz condition
is met (exercise: use that tanh(x) < 1/2 for x ≥1).
More generally stated, non-echo-state networks sometimes can be turned
into echo-state networks by suitable “tonic” components in the input. This
might be an observation of interest for the analysis of biological neural net-
works, a matter that is outside the scope of this article.
3
How to compute with echo state networks,
and how to train them
In this section we describe the intuitions about computation in echo state
networks.
Generally speaking, RNNs are potentially powerful approximators of dy-
namics. There are many ways to make this statement more precise. For
instance, RNNs can be casted as representations of the vector ﬁeld of a dif-
ferential system [9], or they can be trained to embody some desired attractor
dynamics [5]. In those approaches, systems without input are modeled. Since
we wish to deal with input-driven systems, we adopt a standard perspective
of systems theory and view a (deterministic, discrete-time) dynamical system
as a function G which yields the next system output, given the input and
output history:
y(n + 1) = G(. . . , u(n), u(n + 1); . . . , y(n −1), y(n)).
(4)
In control engineering and signal processing, one typically uses rather re-
stricted versions of Eq. 4; speciﬁcally, one employs ﬁnite-history approxi-
mations of Eq. 4; furthermore, one often assumes that G is linear in the u
and/or y arguments. The most common approach to model such restricted
systems is to approximate G by a suitably powerful feedforward network and
9
iteratively feed it with u(n −h), . . . , u(n); y(n −h′), . . . , y(n −1) through a
sliding window (e.g., [14]).
Echo state networks aﬀord a direct way of modeling systems represented
by the fully general Equation 4, without the necessity to convert a time
series into static input patterns by sliding windows. For expository reasons,
we start by considering the restricted case where the output depends only on
the input history, i.e. systems without output feedback:
y(n + 1) = G(. . . , u(n), u(n + 1)).
(5)
The remainder of this section is organized as follows.
First we state the
intuitions in informal terms, using a toy example for illustration (Subsection
3.1); then give a formal account of the restricted case (Subsection 3.2); and
conclude with a formal description of the general case in the last Subsection.
3.1
The basic idea: informal statement and toy exam-
ple
Assume that we have some echo state network with many units, and assume
that the network’s internal connections are rather inhomogeneous. Assume
that a long input sequence is presented to the network. Due to the input
forgetting property of echo state networks, after some initial transient the
internal unit’s activation can be be written as follows (with some liberty of
notation)
xi(n)
≈
ei(. . . , u(n), u(n + 1)),
(6)
where ei is the echo function of the i-th unit.
If the network is suitably
inhomogeneous, the various echo functions will diﬀer from each other. For
an illustration of this fact, consider a single-channel sinusoid input u(n) =
sin(2πn/P) with period length P. The echo state property implies that the
activations xi(n) will also be periodic signals with the same period length;
but the network’s inhomogeneity will induce conspicuous deviations from
the input sinusoid form. Figure 2 gives an example of a 100-unit echo state
network. A period length of P = 10π was used, i.e. the input signal was
u(n) = sin(n/5). The ﬁrst six traces show the signals xi(n) of some randomly
selected units, the seventh trace shows the sinusoid input signal.
Now assume we want to train the network to produce a single-channel
output, which is an interesting transformation of the input, for instance
y(n)teach = 1/2 sin7(2πn/P). In terms of Eq. 5, this means we want to have
yteach(n + 1) = Gteach(. . . , u(n), u(n + 1)) = Gteach(n + 1) = 1/2u(n + 1)7.
10
20406080100
-1
-0.5
0.51
20406080100
-0.75
-0.5
-0.25
0.25
0.5
0.75
20406080100
-1
-0.5
0.51
20406080100
-0.4
-0.2
0.2
0.4
20406080100
-1
-0.5
0.51
20406080100
-0.75
-0.5
-0.25
0.25
0.5
0.75
20406080100
-0.2
-0.1
0.1
0.2
20406080100
-0.3
-0.2
-0.1
0.1
0.2
0.3
Figure 2: Traces of some selected units of a 100-unit echo state network
driven by sinusoid input. The input signal is shown in the last but one trace,
the output signal is shown in the last trace.
The idea is to combine Gteach from the echo functions ei in a mean square
error way. To this end, note that when there are no output feedbacks, the
network output is given by the following single-channel version of Eq. 2:
y(n) = f out(
X
i=1,...,N
wout
i
xi(n)),
(7)
where wout
i
is the weight of the i-th output connection. We use f out = tanh,
which is invertible, therefore (7) is equivalent to
(f out)−1y(n) =
X
i=1,...,N
wout
i
xi(n).
(8)
Inserting the echo functions ei yields
(f out)−1y(n)
=
X
i=1,...,N
wout
i
ei(. . . , u(n −1), u(n)).
(9)
Now we determine the weights wout
i
such that the error
ǫtrain(n)
=
(f out)−1yteach(n) −(f out)−1y(n)
=
(f out)−1yteach(n) −
X
i=1,...,N
wout
i
ei(. . . , u(n −1), u(n)) (10)
is minimized in the mean square error (MSE) sense, i.e. such that
msetrain = 1/(nmax −nmin)
X
i=nmin,...,nmax
ǫ2
train(n)
(11)
11
becomes minimal, where nmin refers to some data point of the training se-
quence after dismissal of an initial transient, and nmax is the last training
point. We will refer to msetrain as training error. Inspection of (10) reveals
that minimizing (11) is a simple task of computing a linear regression.
Concretely, (i) we let the network run for n = 0 to nmax = 300, starting
from a zero network state, (ii) dismiss an initial transient of 100 steps after
which the eﬀects of the initial state have died out (state forgetting property),
(iii) collect the network states x(n) for n = nmin = 101, . . . , nmax = 300, and
(iv) compute the weights wi oﬄine from these collected states, such that the
error (10) becomes minimal.
There is another statement of this task which is somewhat imprecise but
more intuitive. Rename (f out)−1Gteach to G′. Then, compute the weights
such that
G′ ≈
X
wiei
(12)
becomes a MSE approximation of G′ by a weighted combination of the echo
functions ei.
Actually, instead of minimizing the error (10) one would like to minimize
directly ˜ǫtrain = G(n)−y(n) in the MSE sense. However, our recipe minimizes
ǫtrain instead. But since ˜ǫtrain = G(n)−y(n) is strictly related to ǫtrain by f out,
minimizing ǫtrain will also yield a good minimization of ˜ǫtrain = G(n) −y(n).
(If we would use linear output functions f out = id, we would have ǫtrain =
˜ǫtrain = G(n) −y(n).)
In our sin-input, sin7-output example, with f out = tanh we found a train-
ing error msetrain ≈3.3 × 10−15. When the trained network was tested, a
test error msetest = ⟨(ytest −y)2⟩≈3.7 × 10−15 was obtained (⟨·, ·⟩denotes
expectation).
Two important points become apparent in this simple example:
1. The proposed learning procedure computes only the weights of con-
nections leading to the output units; all other connections remain un-
changed. This makes it possible to employ any of the many available
fast, constructive linear regression algorithms for the training. No spe-
cial, iterative gradient-descent procedure is needed.
2. In order to achive a good approximation G′ ≈P wiei, the echo func-
tions should provide a “rich” set of dynamics to combine from. The
network should be prepared in a suitably “inhomogeneous” way to meet
this demand. Metaphorically speaking, the echo state network should
provide a rich “reservoir” of dynamics which is “tapped” by the output
weights.
12
One simple method to prepare such a rich “reservoir” echo state network
is to supply a network which is sparsely and randomly connected. Sparse con-
nectivity provides for a relative decoupling of subnetworks, which encourages
the development of individual dynamics. The 100-unit network used in the
example was randomly connected; weights were set to values of 0, +0.4 and
−0.4 with probabilities 0.95, 0.025, 0.025 respectively. This means a sparse
connectivity of 5 %. The value of 0.4 for non-null weights resulted from a
global scaling such that |λmax |≈0.88 < 1 was obtained.
The input weights were set in an oﬀhand decision (without any optimiza-
tion) to values of +1, -1 with equal probability.
Calculations were done with the Mathematica software package; the linear
regression was done by calling Mathematica’s Fit procedure.
3.2
Formal statement of restricted case
We now present a more rigorous formulation of the training procedure for
the case with no output feedback.
Task. Given: a teacher I/O time series (uteach(n), yteach(n))n=0,...,nmax, where
the inputs come from a compact set U. Wanted: a RNN whose output
y(n) approximates yteach(n).
Procure an echo-state network. Construct a RNN that has echo states
in an admissible state set A with input from U. A convenient heuristic
to obtain such a network is by way of Proposition 3(b): in all experi-
ments carried out so far, every standard sigmoid network whose weight
matrix W satisﬁes |λmax | < 1 had echo states.
Choose input connection weights. Attach input units to the network.
If the original network satisﬁed the Lipschitz condition of Prop. 3(a),
input connections Win can be freely chosen without harming the echo
state property. Moreover, the experience accumulated so far indicates
that the echo state property remains intact with arbitrarily chosen
input connection weights, even if only the weaker condition |λmax | < 1
was ascertained in the previous step.
Run network with teacher input, dismiss initial transient. Start
with an arbitrary network state x(0) and update the network with the
training input for n = 0, . . . , nmax:
x(n + 1) = f(Win(uteach(n + 1)) + Wx(n)).
(13)
13
In agreement with the input forgetting property, choose an initial tran-
sient such that after the transient time nmin the internal network state
is determined by the preceding input history up to a negligible error.
Compute output weights which minimize the training error. Let
yteach(n) = (yteach,1(n), . . . , yteach,L(n)) and put g′
j(n) = (f out
j
)−1yteach,j(n).
For each j = 1, . . . , L compute output weights wout
ji
such that the MSE
msetrain,j = 1/(nmax −nmin)
nmax
X
n=nmin
 
g′
j(n) −
N
X
i=1
wout
ji xi(n)
!2
(14)
is minimized. Use your favorite linear regression algorithm for this.
With these output weights, the network is ready for use.
3.3
Formal statement of general case
In the terminology of signal processing, RNNs trained like above basically are
a “nonlinear moving average” (NMA) kind of models. However, the output
of many interesting dynamical systems depends not only on the input history,
but also of the output history. The most general form of such “nonlinear auto-
regressive moving average” (NARMA) models is stated in Eq. 4. We now
describe how the restricted training procedure from the previous subsection
can be generalized accordingly.
The key idea is that from the “perspective” of internal units, fed-back
output signals are just another kind of input. With this idea in mind, the
general training procedure becomes completely analogous to the restricted
case when the teacher output yteach(n) is written into the output units during
the training period (teacher forcing). We repeat the instructions from the
previous subsection with the appropriate modiﬁcations.
Task. Given: a teacher I/O time series (uteach(n), yteach(n))n=0,...,nmax, where
the inputs come from a compact set Uin and the desired outputs yteach(n)
from a compact set Uout. Wanted: a RNN whose output y(n) approx-
imates yteach(n).
Procure an echo-state network.
Using the theoretical results from
Section 2, construct a RNN that has echo states in an admissible
state set A with respect to input uteach from Uin and “pseudo”-input
yteach(n) from Uout. In formal terms, (i) re-interpret the N × L ma-
trix Wback as another input weight matrix and join it with N × K-
matrix Win into a N × (K + L)-matrix Win & back, (ii) join the input
14
uteach(n + 1) with the output yteach(n) into a compound pseudo-input
˜uteach(n+1) = (uteach(n+1), yteach(n)), and (iii) make sure that the re-
sulting network has the echo state property in an admissible state set A
with respect to input ˜uteach(n+1) from the compact set ˜U = Uin×Uout.
A convenient heuristic to obtain such a network is again provided by
Proposition 3(b): any standard sigmoid network whose weight matrix
W satisﬁes |λmax | < 1 will (– according to experience –) do. Experience
also suggests that the echo state property is independent of the input
connections Win, which can be freely chosen; furthermore, it appears
also to be independent of the output back projection connection weights
Wback, which can also be set arbitrarily.
Run network with teacher input and with teacher output forcing,
dismiss initial transient. Start with an arbitrary network state x(0)
and update the network with the training input and teacher-forced
output for n = 0, . . . , nmax:
x(n + 1) = f
 Win(uteach(n + 1)) + Wbackyteach(n) + Wx(n)

.
(15)
Dismiss from this run a suitably long initial transient up to nmin.
Compute output weights that minimize the output MSE. The rest
of the training is identical to the restricted case.
3.4
Example with output feedback
To illustrate that output feedback works similarly like ordinary input feed-in,
we reconsider the sin7 example. This time, we give the network no input; it
has to generate the desired waveform autonomously.
The same network as above was used, but without an input unit. The
output feedback weights were set to values of +1, -1 with equal probability.
The network was trained from a run of 300 steps, of which the ﬁrst 100
were discarded. Fig. 3 shows the results. The training error was msetrain ≈
1.3 × 10−9. In order to determine a test error, the network was run for 100
steps with teacher-forcing of the teacher signal in the output unit, then was
unlocked from teacher forcing.
During the ﬁrst free-running 100 steps, a
msetest ≈2.6 × 10−8 was found. After more free-running steps, the teacher
and and network signal diverge more strongly.
This is owed to the fact
that the frequency of the network signal is not completely the same as the
teacher’s, so a phase lag starts to build up. After 1000 steps, we observe
msetest ≈1.0 × 10−5; in plots the two signals are still undistinguishable.
15
20406080100
-0.75
-0.5
-0.25
0.25
0.5
0.75
20406080100
-0.2
-0.1
0.1
0.2
20406080100
-0.3
-0.2
-0.1
0.1
0.2
0.3
20406080100
-0.4
-0.2
0.2
0.4
20406080100
-0.4
-0.2
0.2
0.4
20406080100
-0.3
-0.2
-0.1
0.1
0.2
0.3
20406080100
-0.4
-0.2
0.2
0.4
20406080100
-0.2
-0.1
0.1
0.2
Figure 3: Traces of some selected units of a 100-unit echo state network
trained to generate a 1/2 sin7(n/5)-signal. The ﬁrst seven traces show arbi-
trarily selected internal units during a test run. The output signal is shown
in the last trace.
4
Two basic experiments
We now demonstrate how the echo state approach copes with two learn-
ing tasks that are mathematically fundamental: periodic sequence learning
and switchable point attractor learning. The ﬁrst is a primordeal form of
temporal memory, the second, of static pattern memory.
4.1
Discrete-periodic sequence learning
Here we investigate how our network learns a sequence which is periodic
with an integer period length. (Note that the sin7 examples are periodic
with a period that stands in no rational ratio with the update interval).
Mathematically speaking, we train the network to cycle through a periodic
attractor; in everyday language, it learns to play a melody.
4.1.1
House of the Rising Sun
In the ﬁrst task, the target signal was prepared from “The House of the Rising
Sun”. The notes of this melody were assigned numerical values ranging from
-1 (g#) to 14 (a’), with halftone intervals corresponding to unit increments.
Fig. 4 shows this melody. These values were divided by 28 to make the melody
range ﬁt into the network output unit’s range of [−1, 1] (an output function
f out = tanh was used). This resulted in a periodic sequence of period length
48. The target sequence (y(n))n≥0 is a concatenation of identical copies of
this melody.
16
10
20
30
40
2
4
6
8
10
12
14
Figure 4: The melody of “House of the Rising Sun” before squashing into
the network output interval [−1, 1].
4.1.2
Network preparation
A 400 unit sigmoid network was used. Internal connections were randomly as-
signed values of 0, 0.4, −0.4 with probabilities 0.9875, 0.00625, 0.00625. This
resulted in a weight matrix W with a sparse connectivity of 1.25%. The
maximal eigenvalue of W was | λmax |≈0.908. The fact that | λmax | is close
to 1 means that the network exhibits a long-lasting response to a unit impulse
input. Fig. 5 shows how the network (without output units, before training)
responds to a unit impulse input.
1020304050
-0.15
-0.1
-0.05
0.05
1020304050
-0.01
0.01
0.02
0.03
1020304050
-0.04
-0.02
0.02
1020304050
0.2
0.4
0.6
0.81
1020304050
-0.005
0.005
0.01
1020304050
-0.01
0.01
0.02
0.03
1020304050
-0.06
-0.04
-0.02
0.02
0.04
1020304050
-0.04
-0.02
0.02
Figure 5: Response of 7 arbitrarily selected units of the network to an impulse
input. The last trace shows the input.
Comment.
Generally, the closer | λmax | is to unity, the slower is the
decay of the network’s response to an impulse input.
A relatively long-
lasting “echoing” of inputs in the internal network dynamics is a requisite
for a sizable short-term memory performance of the network. A substantial
17
short-term memory is required for our present learning task, because the
target signal contains a subsequence of 8 consecutive 0’s (namely, the last 5
notes concatenated with the ﬁrst 3 notes of the subsequent instance of the
melody). This implies that the network must realize a memory span of at
least 9 update steps in order to correctly produce the ﬁrst non-0 output after
this sequence.
Since at time n the output y(n) depends on the previous outputs, output
feedback is required in this task. The output feedback weights were sampled
randomly from the uniform distribution in [−2, 2].
Figure 6 shows the setup of the network prepared for this learning task.
400 internal units
melody
output
Figure 6: Network setup of the “melody” periodic sequence learning task.
Solid arrows indicate weights that were ﬁxed and remained so during learning,
dashed arrows mark weights to be learnt.
4.1.3
Training and testing
The network was trained ﬁrst in a “naive” fashion, which fails in an inter-
esting way. We describe training and failure and show how the latter can be
remedied.
The network was run for 1500 steps with teacher-forced melody. The
inital 500 steps were discarded, and the output connections computed from
the network states that were collected during the remaining period.
The training error was msetrain ≈2.2 × 10−31, i.e. zero up to machine
precision. This zero training error can be explained as follows. After the ini-
tial 500 steps, the network states x(n) had converged to a periodic sequence,
i.e. for n ≥500, k = 1, 2, . . ., x(n) = x(48 k + n) up to machine precision.
According to Eq. (14), the 400 output weights Wout = (w1i)i=1,...,400 were
computed such that
18
msetrain = 1/1000
1500
X
n=501
 (f out)−1yteach(n) −Woutx(n)
2 .
(16)
was minimized. Since the network states x(n) are 48-periodic, (16) eﬀectively
reduces to
msetrain = 1/48
548
X
n=501
 (f out)−1yteach(n) −Woutx(n)
2 ,
(17)
which renders the computation of the weights Wout underdetermined, be-
cause the linear mapping Wout : R400 →R is determined by (17) only
through 48 linearly independent arguments.
Therefore, there exist many
perfect solutions. The Fit procedure of Mathematica selects one of them.
In order to check whether the network had indeed learnt its task, it was
tested whether it could stably continue to generate the desired ouputs after a
starting period with teacher forcing. The network was started from the null
state x(0) = 0, and the correct melody was written into the output units for
500 initial steps. After this initial period, the network state had developed
into the same periodic state set as during training. Then the network was
left running freely for further 300 steps.
Figure 7 shows what happens:
the network continues to produce the desired periodic output sequence for
some time but then degenerates into a novel dynamics which has nothing in
common with the desired target.
50
100
150
200
250
300
-0.6
-0.4
-0.2
0.2
0.4
0.6
0.8
Figure 7: Degeneration of network performance trained with the naive algo-
rithm. Solid line: target, dashed line: network output.
Obviously, the solution found in the “naive” training is unstable. This
can be stated more precisely as follows. Consider the state x(501) and the
48-step update operator T 48. Then x(501) should be a ﬁxed point of T 48.
19
Up to some very high precision this is true, as becomes evident from the ﬁrst
precise repetitions of the melody in Fig. 7. Unfortunately, x(501) is not a
stable ﬁxed point.
Likewise, the states x(502), . . . , x(548) should be stable ﬁxed points of
T 48. (Actually, stability of x(501) implies stability of these others).
How can this ﬁxed point be stabilized? Recall that the computation of
weights in (17) was underdetermined. Therefore, we have still the possibility
to ﬁnd alternative solutions of (17) which likewise render x(501) a ﬁxed point,
but a stable one. One trick to achieve this goal is to insert noise into the
training procedure.
Instead of updating the network according to Eq. (1), during training in
steps 1 to 1500 we use
x(n + 1) = f
 Wx(n) + Wback(y(n) + ν(n))

,
(18)
to update the network state, where ν(n) is a noise term sampled from a uni-
form distribution over [−0.001, 0.001]. This noise makes the network states
“wobble” around the perfect periodic state sequence used in the naive ap-
proach. Now the computation of weights minimizes
msewobbletrain = 1/1000
1500
X
n=501
 (f out)−1yteach(n) −Woutxwobble(n)
2 .
(19)
Since now we have truly 1000 arguments xwobble(n) (instead of eﬀectively
only 48), the linear regression is overdetermined and a nonzero training error
of msewobbletrain ≈1.3 × 10−9 is obtained.
The solution found with this wobbling trick is stable. Fig. 8 shows what
happens when the trained network is started from the null state with an
initial teacher-forced startup of 12 steps.
This short initialization brings
the network state close, but not very close to the proper state. Fig. 8(a)
shows the ﬁrst 100 free-running steps after this startup. The initial state
discrepancy leads to a visible initial discrepancy between target and network
output. After more free-running steps, the network eventually settles into
the desired attractor up to a residual msetest ≈10−7.
This “stabilization through wobbling” can be stably reproduced: it works
every time the training procedure is repeated. However, I cannot oﬀer a rigor-
ous explanation. Intuitively, what happens is that the weights are computed
such that small deviations from the target state/output sequence are coun-
teracted as good as possible, exploiting the remaining 400−48 = 352 degrees
of freedom in the weight determination.
20
(a)
20
40
60
80
100
0.1
0.2
0.3
0.4
0.5
(b)
20406080100
-0.6
-0.4
-0.2
20406080100
-0.4
-0.3
-0.2
-0.1
20406080100
0.2
0.4
0.6
0.8
20406080100
0.1
0.2
0.3
0.4
0.5
20406080100
-0.6
-0.4
-0.2
20406080100
-0.2
0.2
0.4
0.6
0.8
20406080100
-0.2
0.2
0.4
0.6
0.8
20406080100
-0.6
-0.4
-0.2
Figure 8: Network performance after “wobbling” stabilization. (a) First 100
steps after 30 step teacher forced initialization: a slight discrepancy between
target and network output in the beginning is visible, but then the network
gets caught in the target attractor. (b) Traces of some internal states. The
last trace is from the output unit.
4.1.4
Comments
We conclude this treatment of melody learning with miscellaneous comments.
• The choice of network parameters W is robust. No delicate parameter
tuning was necessary besides respecting that | λmax | be close to unity
in order to assure a suﬃciently long decay time of output feedback ef-
fects in the network state, which is required for the network to bridge
consecutive identical outputs. Using the same network, but with in-
ternal connection weights globally scaled such that | λmax |= 0.987 or
|λmax |= 0.79 did not aﬀect the success of stable learning. If, however,
the network was scaled to the smaller value | λmax |= 0.69, solutions
could not be stabilized through the wobbling trick any more.
The
maximal singular value of the weight matrix in this last (useless) case
was σmax ≈1.49, which is considerably greater than 1. Further down-
scaling of the weight matrix would be necessary to achieve σmax < 1,
which would lead even further into the “non-stabilizable” region. This
is an indication that the suﬃcient condition for echo states of Prop. 3(a)
21
is overly restrictive, at least in cases where “long short-term memory”
properties are requested from the network.
• The sizing of output feedback weights Wback is likewise robust. The
training went as well when the feedback weights were chosen from
[−0.5, 0.5] or from [−4, 4].
• The magnitude of noise is also uncritical.
The learning works with
no obvious diﬀerence whether noise is taken from [−0.01, 0.01] or from
[−0.000001, 0.000001].
• The same network was trained to generate periodic random sequences
of period length 96, which succeeded, and to periodic sequences of pe-
riod length 104, which didn’t. In the latter case, the training error was
still of the same (zero) size as in our length 48 example, but “stabiliza-
tion through wobbling” wouldn’t succeed any more. It appears that
somewhere between 96 and 104 there is a maximal sequence length
which this echo state network could stably reproduce. Stated some-
what imprecisely, the network needs to have enough “residual” degrees
of freedom to stabilize the ﬁxed point. This matter deserves a more
thorough investigation.
• Stable generation of a periodic sequence requires a nonlinear network.
It can be easily seen that if the network had linear units (i.e., it f and
f out were linear), one could only realize a linear (autoregressive, AR)
type of system. Such systems are inherently incapable of settling into
a stable attractor, because if y(n) is a periodic signal generated by an
AR system, then also Cy(n) would be a possible output signal for every
C ∈R.
• There is no obvious upper limit to the period length achievable with this
approach: longer periodic sequences should be learnable by accordingly
larger echo state networks.
4.2
Multiple attractor learning
We now describe an example which can be seen as dual to the previous one.
While there we trained a multipoint attractor, we will now train multiple
point attractors. Suitable input signal can switch between them.
We consider the following task. The network has m input units and m
output units. The input signal are rare spikes, which arrive randomly in
the input units. With each input unit we associate a point attractor of the
22
network. If a spike arrives at input unit i, the network is switched into its ith
point attractor, where it stays until another spike arrives at another input
unit. The output units indicate in which attractor the system is currently in.
The ith output unit has activation 0.5 when the system is in the ith attractor;
the remaining output units have then an activation of −0.5. Figure 9 provides
a sketch of the setup.
...
...
input:
channels for
rare spikes
m
output:
last
spike indicators
m
Figure 9: Setup of the switchable attractor network. Solid lines mark con-
nections whose weights are unchanged by training, dashed connections are
trained.
4.2.1
Task speciﬁcation
The input signal is a vector u(n) = (u1(n), . . . , um(n)), where ui(n) ∈
{0, 0.5}. At any time n, at most one of the input channels “spikes”, i.e.
takes a value of 0.5. Spikes are rare: the probability that at time n one of
the channels spikes is 0.02.
The output is a vector y(n) = (y1(n), . . . , ym(n)), where yi(n) ∈{−0.5, 0.5}.
The desired output is yi(n) = 0.5 if the ith input spiked at some time n −k
(where k ≥0) and no other input channel spiked during n, n−1, . . . , n−k+1.
We report here on a task where the number of channels/attractors m =
20.
4.2.2
Network preparation
A 100 unit standard sigmoid network was randomly connected (the same
network as in the sin7-example was used). Weights were scaled such that
| λmax |≈0.44 was obtained.
This value is much smaller than unity and
guarantees a fast input/state forgetting. The weight matrix had a maximal
23
singular value of σmax ≈0.89. Thus, in this example the suﬃcient condition
for echo states given in Prop. 3(a) was satisﬁed.
20 input units and 20 output units were attached. For each input chan-
nel, the input connection weights were randomly chosen to be 5 or −5 with
equal probability. The network had output feedback connections, which were
randomly set to either of the three values of 0, 0.1, −0.1 with probabilities
0.8, 0.1, 0.1.
4.2.3
Training and testing
The training data consisted of a 4050-step input/output sequence. Starting
with a spike at input channel i = 1 at time n = 1, every 200 steps one spike
was issued, cycling once through the input channels such that at time 4001
the ﬁrst input channel spiked again. The outputs were set to −0.5 or 0.5 as
speciﬁed above.
The network was run with these training data, where the output signals
were teacher-forced into the output units. For updating, Eq. (1) was used
(no extra noise insertion).
The ﬁrst 50 steps were discarded and the output weights were computed
from the network states collected from n = 51 through n = 4050. The twenty
training errors varied between a smallest value of msetrain,i ≈8 × 10−7 and a
largest value of ≈8 × 10−6.
The trained network was tested by feeding it with long input sequences
prepared according to the task speciﬁcation. The 20 test errors varied across
the output channels between values of 6 × 10−6 and 5 × 10−5. Fig. 10 gives
a demonstration of the trained network’s performance.
It is not diﬃcult to understand how and why the trained network func-
tions. There are two aspects to account for: (1), why the network locks into
stable states in the absence of input spikes, (2) why the network switches
states according to incoming spikes.
We ﬁrst discuss the ﬁrst point. During training, for extended periods of
time the network’s input and teacher-forced output feedback signals remain
ﬁxed, namely, during the intervals after some spike arrives and before the
next spike arrives. Because of the network’s fast input/state forgetting char-
acteristics, the network quickly converges to stationary states during these
training subintervals. A consequence of the learning algorithm (and again,
an underdetermination of the linear regression) is that these states become
ﬁxed points of the network’s dynamics (while there is no input spike). We
can observe in the testing phase that these ﬁxed points are stable. Like in
the periodic sequence example, I cannot oﬀer a rigorous explanation of this
fact. However, hypothetically a similar “wobbling” mechanism as above sta-
24
test inputs, first four of 20 channels
500
0.1
0.2
0.3
0.4
0.5
500
0.1
0.2
0.3
0.4
0.5
500
0.1
0.2
0.3
0.4
0.5
500
-1
-0.5
0.5
1
teacher outputs on same channels
500
-0.4
-0.2
0.2
0.4
500
-0.4
-0.2
0.2
0.4
500
-0.4
-0.2
0.2
0.4
500
-1
-0.8
-0.6
-0.4
-0.2
network outputs on same channels
500
-0.4
-0.2
0.2
0.4
500
-0.4
-0.2
0.2
0.4
500
-0.4
-0.2
0.2
0.4
500
-0.4
-0.2
0.2
0.4
traces of some randomly picked internal states
500
-0.3
-0.2
-0.1
0.1
500
0.16
0.18
0.22
0.24
500
-0.1
0.1
0.2
0.3
500
-0.2
-0.1
0.1
Figure 10: A 500-step section of a test run of the trained multi-attractor
network. The ﬁrst four (of twenty) inputs and outputs are shown. Note that
in the shown interval, spikes arrived at other input channels than the four
shown ones. For further comments see text.
bilizes these ﬁxed points. The role of noise from the previous example is ﬁlled
here by two other sources of deviations. The ﬁrst deviation from the ﬁxed
point states in the training material comes from the short transient dynam-
ics after input spikes (compare last sequence in Fig. 10). The second source
arises from the fact that an output value of −0.5 at output node i is trained
from 19 diﬀerent ﬁxed points, namely, the ﬁxed network states obtained in
the 19 interspike intervals. Both eﬀects together and the fact that there are
100 −20 = 80 freely disposable parameters in each output weight vector is
apparently suﬃcient to make these ﬁxed points stable.
The switching between diﬀerent stable states may be understood as fol-
lows. Due to the large absolute values (5.0) of the input connections, an
input spike that arrives at time n at channel i essentially erases all preceding
25
state information from the network and sets the states to xj(n) = 0.5 fjwin
ji.
Let’s call this the “signature state” of the ith channel. The linear regression
maximises the immediate (same time n) output response of the ith output
unit to this signature state, and minimizes the response of all others. There-
fore, in the trained network the immediate response of the ith output unit
will be greater than the response of all other output units. After time n, the
network evolves according to the trained ﬁxed point dynamics. For the short
transient after time n, the trained response also favors 0.5-valued responses in
the ith unit and −0.5-valued responses in the others. The resulting transient
dynamics in the test phase is apparently close enough to the corresponding
transient dynamics in the (teacher-forced) training phase to ensure that the
network settles into the desired state.
A closer inspection of the traces in Fig. 10 (third row) reveals that some
output units i “spike” ephemerically (without locking into their attractor)
also when input spikes arrive at unassociated channals i′. This must happen
when the signature states of the two channels have a high similarity in the
sense that ⟨win
i , win
i′ ⟩≫0 (here, ⟨·, ·⟩denotes inner product). This situation
has to be expected for some pairs i, i′ due to the random choice of the input
weights. If the input weight vectors would have been chosen orthogonally,
we may hypothesize that these erroneous ephemeral output spikes would not
occur. However, this has not been tested.
5
Echo state networks with leaky integrator
neurons
The units in standard sigmoid networks have no memory; their values at time
n + 1 depend only fractionally and indirectly on their previous value. Thus,
these networks are best suited for modeling intrinsically discrete-time systems
with a “computational”, “jumpy” ﬂavor. It is diﬃcult, for instance, to learn
slow dynamics like very slow sine waves. For learning slowly and continuously
changing systems, it is more adequate to use networks with a continuous
dynamics. We take a hybrid approach here and use discrete-time networks
whose dynamics is a coarse approximation to continuous networks with leaky
integrator units. The evolution of a continuous-time leaky integrator network
is
˙x = C
 −ax + f(Winu + Wx + Wbacky)

,
(20)
where C is a time constant and a the leaking decay rate. When this diﬀeren-
tial equation is turned into an approximate diﬀerence equation with stepsize
26
δ, one obtains
x(n + 1) = (1 −δCa)x(n) + δC
 f(Winu(n + 1) + Wx(n) + Wbacky(n))

.
(21)
If all internal unit output functions are standard sigmoids fi = tanh, we can
spell out an analog to Prop. 3 for a system of the kind (21).
Proposition 4 Let a network be updated according to (21), with teacher-
forced output. Let fi = tanh for all i = 1, . . . , N. (a) If | 1−δC(a −σmax) |=
L < 1 (where σmax is the maximal singular value of the weight matrix W),
the network has echo states, for every input and/or output feedback. (b) If
the matrix ˜
W = δCW + (1 −δCa)id (where id is the identity matrix) has a
spectral radius greater than 1, the network has no echo states for any input
set containing 0 and admissible state set [−1, 1]N.
The proof is sketched in the Appendix.
By choosing suﬃciently small time constants C, one can employ echo
state networks of type (21) to model slow systems. The training is done in a
completely analogous way as described in Section 3, with one optional modi-
ﬁcation: If a small stepsize δ is used, one can downsample the network states
collected in the training period before the linear regression computation is
performed. This reduces the computational load of the regression compu-
tation and does not markedly impair the model quality, because network
states between two sampled states will be almost linear interpolates of the
latter and thus do not contribute relevant new information to the regression
computation.
6
The Mackey-Glass system
A popular test for learning dynamical systems from data is chaotic attractor
learning. A particularly often used system is deﬁned by the Mackey-Glass
delay diﬀerential equation
˙y(t) = α y(t −τ)/(1 + y(t −τ)β) −γy(t),
(22)
where invariably in the chaotic systems modeling community the parameters
are set to α = 0.2, β = 10, γ = 0.1. We will use the same parametrization.
The system has a chaotic attractor if τ > 16.8. In the majority of studies,
τ = 17 is used, which yields a mildly chaotic attractor. A more rarely used
value is τ = 30, which leads to a wilder chaotic behavior. We will tackle both
the mild and the wild system.
27
6.1
Task speciﬁcation
Several discrete-time training sequences were prepared by ﬁrst approximating
(22) through
y(n + 1) = y(n) + δ

0.2 y(n −τ/δ)
(1 + y(n −τ/δ)10 −0.1 y(n)

(23)
with stepsize δ = 1/10 and then subsampling by 10. One step from n to
n + 1 in the resulting sequences corresponds to a unit time interval [t, t + 1]
of the original continuous system.
In this manner, four training sequences were generated, two of length 3000
and two of length 21000. For each length, one sequence was generated with a
delay of τ = 17 and the other with τ = 30. These sequences were then shifted
and squashed into a range of [−1, 1] by a transformation y 7→tanh(y −1).
The task is to train four echo-state networks from these training se-
quences, which after training should re-generate the corresponding chaotic
attractors.
Fig. 11 shows 500-step subsequences of these training sequences for τ = 17
and τ = 30.
100 200 300 400 500
-0.4
-0.2
0.2
100 200 300 400 500
-0.6
-0.4
-0.2
0.2
Figure 11: 500-step sections of the training sequences for delays τ = 17 (left)
and τ = 30 (right).
6.2
Network preparation
Essentially the same 400-unit network as in the periodic attractor learning
task was used. Its weight matrix W was globally rescaled such that |λmax |≈
0.79 resulted.
The network update was done according to Eq. (21) with stepsize δ = 1,
global time constant C = 0.44 and decay rate a = 0.9.
(This yields an
eﬀective spectral radius of |λmax | ( ˜
W) ≈0.95 according to Prop. 4(b)).
28
One input unit was attached which served to feed a constant bias signal
u(n) = 0.2 into the network. The input connections were randomly chosen
to be 0, 0.14, −0.14 with probabilities 0.5, 0.25, 0.25. The purpose of this bias
is to increase the variability of the individual units’ dynamics.
One output unit was attached with output feedback connections sampled
randomly from the uniform distribution over [−0.56, 0.56].
6.3
Training and testing for τ = 30
6.3.1
The length 3000 training sequence
The network was run from a zero starting state in teacher-forced mode with
the τ = 30, length 3000 training sequence. During this run, noise was inserted
to the network using the following variant of (21):
x(n + 1) =
(24)
(1 −δCa)x(n) + δC
 f(Winu(n + 1) + Wx(n) + Wbacky(n) + ν(n))

,
where the noise vector ν(n) was sampled from [−0.00001, 0.00001]N. The
ﬁrst 1000 steps were discarded and the output weights were computed by a
linear regression from the remaining 2000 network states.
The reason for discarding such a substantial number of initial states is
that | λmax | ( ˜
W) is close to unity, which implies a slow forgetting of the
starting state.
The network’s performance was ﬁrst visually judged by comparing the
network’s output with the original system. To this end, the trained network
was run for 4000 steps. The ﬁrst 1000 steps were teacher-forced with a newly
generated sequence from the original system. The network output of the
remaining free-running 3000 steps was re-transformed to the original coordi-
nates by y 7→arctanh(y) + 1. The resulting sequence y(n) was rendered as
a two-dimensional plot by plotting points (y(n), y(n + 20)) and connecting
subsequent points. A similar rendering of the original 3000 step training se-
quence was generated for comparison. Figure 12 shows the original attractor
plot (left) and the plot obtained from the trained network’s output (right).
It appears that the learnt model has captured the essential structure of the
original attractor. Not much more can be gleaned from these plots, besides
maybe the fact that the network stably remained in the desired attractor
state for at least 3000 steps.
In the literature one often ﬁnds plots where the original system’s evolu-
tion is plotted together with the evolution predicted from a learnt attractor
model. With chaotic systems, such plots are not very informative, because
29
0.20.40.60.8
1.21.4
0.2
0.4
0.6
0.8
1.2
1.4
0.20.40.60.8
1.21.4
0.2
0.4
0.6
0.8
1.2
1.4
0.20.40.60.8
1.21.4
0.2
0.4
0.6
0.8
1.2
1.4
Figure 12: Case τ = 30. Attractors of the original system (left), the network
trained from the 21000 step sequence (center) and the network trained from
the 3000 step sequence (right). For details see text.
the evolution of chaotic attractors sometimes goes through periods where
prediction is easy (in the sense that small deviations do not diverge quickly),
but at other times goes through very instable periods where small deviations
explode quickly. Figure 13 demonstrates this fact. The plots in this Figure
were obtained by teacher-forcing the network with the original attractor sig-
nal for 1500 steps and then letting it run freely for another 500 steps. The
free-running steps are plotted. The left diagram shows a case where predic-
tion is easy. The right shows another case where the original (and the learnt)
system go through a highly divergent period around n = 200.
100 200 300 400 500
-0.6
-0.4
-0.2
0.2
0.4
100 200 300 400 500
-0.6
-0.4
-0.2
0.2
Figure 13: Two test runs for τ = 30, model learnt from the 3000 step training
sequence. The free running network’s output (dashed) is plotted against the
original attractor (solid line).
A more informative measure of model quality, which is standardly given,
is the normalized root mean square error (NRMSE) of model predictions for
a given prediction horizon. A much-used prediction horizon is to compare the
model’s prediction 84 steps ahead, ˆy(n + 84) (after retransformation to the
original coordinates), with the original value y(n+84) (after retransformation
to the original coordinates). To obtain this NRMSE84, we ran the network 50
times. Each run consisted of a 1000 step initial period with teacher forcing
30
and a subsequent 84 step free run. The original evolutions were taken from 50
subsequent 1084-step evolutions of a 50 × 1084 run of the original attractor.
At each of the 50 runs, the values yi(n + 84) and ˆyi(n + 84) were collected
(i = 1, . . . , 50) and the NRMSE84 was computed by
NRMSE84 ≈
P
i(yi(n + 84) −ˆyi(n + 84))2
50σ2
1/2
,
(25)
where σ2 ≈0.067 is the variance of the original attractor signal. For the
model learnt from the 3000 step training sequence, a NRMSE84 ≈0.11 was
found.
The work reported in [5] contains the best modeling of the τ = 30 case I
am aware of. The authors use a multilayer perceptron with output feedback,
which is trained from 1880-step training sequences with a reﬁned version of
backpropagation through time, “extended Kalman ﬁlter multi-stream train-
ing”. In [5] another error criterium is used, namely the root mean square
error of 120-step predictions:
RMSE120 ≈(⟨(y(n + 84) −ˆyi(n + 84))2⟩)1/2,
(26)
where ⟨·, ·⟩denotes expectation. We estimated this value from 50 runs for
the 3000-step-trained model and found RMSE120 ≈0.048. The error given
in [5] is RMSE120 ≈0.040, a value similar in magnitude to our result.
6.3.2
The nuisance of initial state determination
In the echo state approach described here, a substantial amount of training
data is wasted for the discarded initial period of the training run. Likewise,
when the trained network is used to predict a sequence, a long initial run of
the sequence must be fed into the network before the actual prediction can
start. This is a nuisance because such long initial transients are in principle
unneccessary.
By Takens’ theorem [16], very few (actually, 4) successive
data points y(n), y(n + k), y(n + 2k), y(n + 3k) suﬃce to fully determine the
attractor’s state at time n + 3k. This is exploited in most approaches to
attractor modeling: typical prediction models found in the literature predict
a future data point y(n + 3k + l) from y(n), y(n + k), y(n + 2k), y(n + 3k)
(mostly k = 6 is used with the Mackey-Glass system). Such models allow
the user to start predictions from prior histories with a length of only 3k +1.
By contrast, a recurrent neural network such as our echo state network,
but also such as the networks used in [5] need long initial runs to “tune
in” to the to-be-predicted sequence. [5] tackle this problem by training a
second, auxiliary “initiator” network. This is a feedforward network which
31
computes an appropriate starting state of the recurrent model network from
a 3k + 1-long initial sequence.
This idea could be adapted to our echo state network approach in the
following way.
1. Run the network in teacher-forced mode on the training data, dis-
card initial transient as above.
Collect the network states x(n) =
(xi(n))i=1,...,N over the remaining steps nmin, . . . , nmax.
2. For each internal unit i of the echo state network, compute a separate
weight vector WstateInit,i of length N, in the following way.
(a) For m = nmin + 3k to m = nmax, run the network in from zero
starting state for 3k+1 steps, teacher-forcing it with a subsequence
of the original training subsequence taken from steps m −3k to
m. For each m, collect the network step xm obtained at the end
of this short run.
(b) By a linear regression, compute WstateInit,i such that the MSE
1/(nmax−(nmin+3k))
X
m=nmin+3k,...,nmax
(xi(m)−WstateInit,ixm) (27)
is minimized.
3. To compute an appropriate network starting state for the prediction
task from a prior history of short length 3k + 1, run the network from
zero starting state for 3k + 1 steps, teacher-forcing it with the short
initial history.
The resulting ﬁnal network state is ˜x.
Compute a
starting state x(3k + 1) for the prediction task by putting x(3k + 1) =
(WstateInit,1˜x, . . . , WstateInit,N ˜x).
The logic of this procedure is intuitively clear but remains to be imple-
mented and tested.
6.3.3
The length 21000 training sequence
The same experiment was done with the 21000-step training sequence. Due
to memory restrictions on my PC (the Mathematica Fit procedure is memory
intensive) the task had to be broken up into four tasks of length 6000 each
(splitting the 21000 training sequence into four with 1000 step overlaps).
From each of the four length 6000 training sequences, output weight vectors
32
were computed as above. These four vectors were then averaged to obtain a
ﬁnal weight vector. This averaging results in a poorer estimate than if a full
length 21000-step training could have been performed, but it still improves
over the single 6000 step weight vectors.
The noise inserted here was smaller than in the 3000-step trial; it was
sampled from [−0.00000001, 0.00000001]N.
Figure 12 (center) shows the resulting learnt attractor. The 84-step pre-
diction error (again averaged from 50 prediction runs) was NRMSE84 ≈
0.032. To my knowledge, this is the best prediction model for the τ = 30
Mackey-Glass attractor achieved with any method and any training data size
so far.
This experiment was carried out mainly to probe the representational
limits of the 400-unit echo state network. This trial demonstrates that echo
state networks are well-suited devices for chaotic attractor modeling, in the
sense that by learning a relatively small number of parameters (400 here) in a
generic model structure very close ﬁts to complex dynamics can be achieved.
6.3.4
Stability and noise insertion
Noise insertion was found to be crucial for stability in the τ = 30 task.
Trials with smaller amounts of noise than reported above resulted in unstable
trained networks. Their dynamics left the desired attractor typically after
about 100 steps and turned into ﬁxed points or oscillations. Also, it was
found that when noise was added to the output feedback as in the melody
learning task (instead to the network states) the resulting networks needed
much larger amounts of noise to become stable. The larger amount of noise
degraded the precision of the learnt model. It remains to be investigated
why noise insertion to the states was superior to noise insertion to output
feedback in this task.
While a rigorous theoretical understanding of the stabilizing function of
noise is lacking, it is worth mentioning that there appears to exist a stability-
precision tradeoﬀ. Larger noise was generally found to result in more stable
models (in the sense that they converged to the desired attractor from a
larger subspace of starting states) but at the same time also meant less
precise prediction.
6.3.5
A note on parameter selection
The general structure of the network was arbitrarily generated, including
connectivity and setting of the input and output feedback weights. In fact, I
33
have been re-using the very same basic 400-unit network for a large variety
of tasks since I started investigating echo states networks.
However, there are a handful of parameters which were optimized by hand
for this particular task: the global scaling of the raw echo state network
(which determines | λmax |), a global scaling factor for the input weights,
another global scaling factor for the output feedback weights, the global
time constant C and the leaking decay rate a. The optimiziation was done
in a crude fashion, one parameter after the other, testing a few settings of
each parameter on small test problems and choosing the best-performing
one. No iteration of the procedure was done after all parameters had been
optimized once. The overall time spent on this was about one hour of manual
experimentation, excluding PC running time.
A more systematic cross-optimization of the parameters would proba-
bly improve the training results, but much of the charm of the echo state
approach lies in its simplicity and robustness. The basic experience of work-
ing with echo state networks is that with a small eﬀort good results can be
achieved. However, it should be emphasized that a rough optimization of
these few parameters is crucial and task-speciﬁc.
6.4
Training and testing for τ = 17
The same experiments as reported above were repeated with the τ = 17
attractor. The network used in the τ = 30 trials was re-used without new
optimization of the few parameters mentioned in the previous subsection.
It turned out that the τ = 17 system is much simpler to cope with than
the τ = 30 one. It was found that no noise insertion was required to obtain
stable networks, and the achieved model precision was by far greater.
Figure 14 is the dual of Fig. 12. The attractor plots were generated with
plot points (y(n), y(n + 15)) here. The three plots are visually indistinguish-
able.
0.4 0.6 0.8
1.2
0.4
0.6
0.8
1.2
0.4 0.6 0.8
1.2
0.4
0.6
0.8
1.2
0.4 0.6 0.8
1.2
0.4
0.6
0.8
1.2
Figure 14: Case τ = 17. Attractors of the original system (left), the network
trained from the 21000 step sequence (right) and the network trained from
the 3000 step sequence (bottom).
34
The τ = 17 attractor seems not to have so diﬀerently divergent periods
as the τ = 30 attractor. In plots, predictions start to deviate perceptibly
from the original not earlier than after about 1200 steps for models learnt
from the 21000 step training sequence (Fig. 15).
200
400
600
800
1000
1200
1400
-0.4
-0.2
0.2
Figure 15: A 1500 step prediction with the model learnt from the 21000 step
training sequence.
The prediction errors were found to be NRMSE84 ≈0.00028 for the model
learnt from the 3000 step training sequence and NRMSE84 ≈0.00012 for
the model obtained from 21000 training data points. Since the variance of
prediction errors is much smaller here than in the τ = 30 case, only 20 trials
were averaged to estimate these errors.
For comparison we consider ﬁrst the work of [18], which contains the best
results from a survey given in [6]. In that approach, a local modelling tech-
nique was used (see [13] for an introduction to local modelling) in combination
with a self-organizing feature map (SOM) which was used to generate proto-
type vectors for choosing the best local model. [18] report on numerous exper-
iments. The best model has NRMSE84 ≈0.0088. It was trained from about
50,000 (or 10,000, this is not clear from the paper) training data points. The
SOM part of the model had 10,000 parameters (local models have no clear
number of parameters because in a certain sense, the training data them-
selves constitute the model). The best model obtained from 6000 (possibly
1200, which is also the number of SOM parameters) data points (comparable
to our model obtained from 3000 points) had NRMSE84 ≈0.032 and the best
model learnt from 28000 (or maybe 5600) points had NRMSE84 ≈0.016.
Another good model, not contained in the survey of [6], was achieved in
[12]. Unfortunately, NRMSE’s are only graphically rendered – and by rather
fat lines, so one can only guess an NRMSE84 ≈0.015 −0.02 for a model
learnt from 2000 data points. Again, the methods relied on local models.
All in all, it seems fair to say that the echo state model performs about
two orders of magnitude better than previous attempts to model the τ = 17
Mackey-Glass system, regardless of the method.
Furthermore, echo state
networks are far simpler and faster to compute than other kinds of models.
35
7
Discussion
Recurrent networks have usually been trained by gradient descent on an error
function. Several basic algorithms are known and have been reﬁned in various
directions (overviews in [3], [15], [2]). All of these methods are not completely
satisfactory, because of (i) local optima, (ii) slow convergence, (iii) disruption
and slowdown of the learning process when the network is driven through
bifurcations during learning, and (iv) diﬃculties to learn long-range temporal
dependencies due to vanishing or exploding gradient estimates. Furthermore,
these methods are relatively complex, an obstacle to their wide-spread use.
In practice, when a dynamical system has to be realized by a neural
network, one takes resort to a number of other solutions:
1. Train a feedforward network to predict the system output from network
input drawn from a few delayed instances of the system input and/or
output. This approach has its foundation in Takens embedding theorem
and is probably the most widely used solution.
2. Use restricted recurrent network architectures for which specialized
learning algorithms are known, like e.g. the well-known Elman net-
works (overview in [10]).
3. Use custom-designed network architectures to overcome some of the
listed problems. A conspicuous example is “Long Short Term Memory”
networks [7], which use specialized linear memory units with learnable
read and write gates to achieve very long memory spans.
One tantalizing motive to go beyond these working solutions is biological
neural networks. Biological RNNs are a proof of existence that eﬃcient and
versatile learning is possible with recurrent neural networks. While much is
known about temporal learning phenomena at the synapse level (overview:
[1]), not much is clear concerning the learning dynamics at the network level.
In fact, the basic idea of this article (viz., that a large network can serve
as a “reservoir of dynamics” which is “tapped” by trained output units)
has been independently discovered in a context of biologically motivated,
continuous-time, spiking neural networks [11]. That work assumes a larger
variety of readout functions (while here only linear readouts with an optional
subsequent transformation f out are considered), shows that under those as-
sumptions the class of considered networks is computationally universal in a
certain sense, and demonstrates a number of biologically motivated training
tasks.
36
I have been investigating echo state networks for about 1.5 years now and
many more results than the ones reported here have been obtained. They
will be reported in subsequent publications. Here is a preview:
• Echo state networks have been trained with similar ease and preci-
sion as in the above examples on a large variety of other tasks. They
include tuneable sinewave generators, waveform recognizers, word rec-
ognizers, dynamical pattern recognizers, frequency measuring devices,
controllers for nonlinear plants, excitable media, other chaotic attrac-
tors, and spectral analysis devices. In all cases (except excitable media
modeling), the very same two 100- and 400-unit networks also employed
in this article were used.
• A basic phenomenon in echo state networks is their short-term memory
capability. A quantitative measure of memory capacity has been de-
ﬁned. Several facts about this memory capacity have been proved, the
most important one being that for i.i.d. input, the memory capacity of
an echo state network is bounded by the number of network units.
• The main computation for training echo state networks is a linear re-
gression, or in other words, a mean square error minimization. This
was done here oﬄine. In the area of adaptive signal processing, numer-
ous online methods for the same purpose have been developed. They
can be used with echo state networks, too. Speciﬁcally, the LMS and
the RLS algorithm (see [4] for deﬁnitions) have been tested. It turns
out that LMS converges too slowly, and the reason has been elucidated.
RLS works very well.
One might object that echo state networks are a waste of units. It is
quite likely true that any task solved by a large, say 400-unit, echo state
network can also be solved by a much smaller, say 20-unit RNN (in both,
one has about 400 parameters to train). One problem, of course, is to ﬁnd
this Occamish small network – this is what previous learning techniques have
attempted. But the proper reply to the waste-of-units objection is that the
very same instance of a big echo state network can be re-used for a variety
of tasks. For each task, only the requisite number of input and output units
has to be added, and a roughly appropriate setting of the crucial global
scaling parameters has to be eﬀected for each task.
This sounds natural
from a biological perspective, where from all we know the same pieces of
brain matter participate in many tasks, and maybe sometimes are globally
modulated in task-speciﬁc ways.
37
Training echo state networks is both versatile (one basic network can be
trained on many diﬀerent tasks) and precise (training and test errors in the
order of 10−4 for diﬃcult tasks [like frequency measuring] to 10−15 for simple
tasks [like periodic signal generation] were found in the tasks addressed so
far). The pairing of versatility with precision springs from a single, funda-
mental fact. Consider Equation 12: G′ ≈P wiei. The ei on the rhs, written
out, spell ei = ei(. . . , u(n−1), u(n); . . . , y(n−2), y(n−1)). These functions
depend on the desired output signal y itself. By virtue of the echo state
property, in each particular task the signals ei will share certain properties
with the desired output and/or given input, but at the same time introduce
variations on the input/output theme. For instance, when the desired out-
put is a sine with period P, all ei will be oscillations of various forms but
with same period P. Fig. 8(b) and the last row of Fig. 10 provide other
striking demonstrations of this fact. So what happens is that the desired
output and given input shape the “basis” systems after themselves. The
task automatically shapes the tool used for its solution. Thus it is
no really amazing that the output can be so precisely recombined from these
ei. The true miracle (which remains to be analyzed) is not the precision of
the learning results, but their stability.
Mathematical analysis of echo state networks has just started. Here is a
list of questions, for which presently no rigorous answers are known:
1. Under which conditions are trained networks stable?
How, exactly,
does noise insertion during training improve stability?
2. How can the suﬃcient condition from Prop. 3 (a), which is too restric-
tive, be relaxed to obtain a more useful suﬃcient condition for echo
states?
3. Which task characteristics lead to which optimal values for the global
tuning parameters?
4. How is the “raw” echo state best prepared in order to obtain a “rich”
variety of internal dynamics?
5. What are the representational/computational limits of echo state net-
works? In [11] it is shown that they are computationally universal in a
certain sense, but there more general readout functions than here are
admitted.
Due to the basic simplicity of the approach (best expressed in the equation
G′ ≈P wiei), swift progress can reasonably be expected.
38
Acknowledgments I am greatly indebted to Thomas Christaller for his
conﬁdence and unfaltering support. Wolfgang Maass contributed inspiring
discussions and valuable references, and pointed out an error in Prop. 3(a) in
the ﬁrst printed version of this report. An international patent application
for the network architectures and training methods described in this paper
was ﬁled on October 13, 2000 (PCT/EP01/11490).
A
Proof of proposition 1
Let
D
=
{(x, x′) ∈A2 |∃¯u∞∈U Z, ∃¯x∞, ¯x′∞∈AZ, ∃n ∈Z :
¯x∞, ¯x′∞compatible with ¯u∞and x = ¯x(n) and x′ = ¯x′(n)}
denote the set of all state pairs that are compatible with some input sequence.
It is easy to see that the echo state property is equivalent to the condition
that D contain only identical pairs of the form (x, x).
We ﬁrst derive an alternative characterization of D. Consider the set
P +
=
{(x, x′, 1/h) ∈A × A × [0, 1] |
h ∈N, ∃¯uh ∈U h, x and x′ are end-compatible with ¯uh}.
(28)
Let D+ be the set of all points (x, x′) such that (x, x′, 0) is an accumu-
lation point of P + in the product topology of A × A × [0, 1]. Note that this
topology is compact and has a countable basis. We show that D+ = D.
D ⊆D+: If (x, x′) ∈D, then ∀h : (x, x′, 1/h) ∈P + due to input shift
invariance, hence (x, x′, 0) is an accumulation point of P + .
D+ ⊆D: (a) From continuity of T and compactness of A, a straight-
forward argument shows that D+ is closed under network update T, i.e., if
(x, x′) ∈D+, u ∈U, then (T(x, u), T(x′, u)) ∈D+. (b) Furthermore, it
holds that for every (x, x′) ∈D+, there exist u ∈U, (z, z′) ∈D+ such
that (T(z, u), T(z′, u)) = (x, x′).
To see this, let limi→∞(xi, x′
i, 1/hi) =
(x, x′, 0).
For each of the (xi, x′
i) there exist ui, (zi, z′
i) ∈A × A such
that (T(zi, ui), T(z′
i, ui)) = (xi, x′
i). Select from the sequence (zi, z′
i, ui) a
convergent subsequence (zj, z′
j, uj) (such a convergent subsequence must ex-
ist because A × A × U is compact and has a countable topological base).
Let (z, z′, u) be the limit of this subsequence. It holds that (z, z′) ∈D+
(compactness argument) and that (T(z, u), T(z′, u)) = (x, x′) (continuity ar-
gument about T). (c) Use (a) and (b) to conclude that for every (x, x′) ∈D+
39
there exists an input sequence ¯u∞, state sequences ¯x(n)∞, ¯x′(n)∞compatible
with ¯u∞, and n ∈Z such that x = x(n) and x′ = x′(n).
With this preparation we proceed to the proof of the statements of the
proposition.
“state contracting ⇒echo state”: Assume the network has no echo states,
i.e., ∃(x, x′) ∈D+, d(x, x′) > 2ε > 0. This implies that there exists a strictly
growing sequence (hi)i≥0 ∈NN, ﬁnite input sequences (¯uhi
i )i≥0, states xi, x′
i,
such that d(T(xi, ¯uhi
i ), x) < ε and d(T(x′
i, ¯uhi
i ), x′) < ε. Complete every uhi
i
on the right with an arbitrary right-inﬁnite input sequence ¯v+∞to obtain
a sequence (¯uhi
i ¯v+∞)i≥0 ∈(U N)N. By the theorem of Tychonov known in
topology, U N equipped with the product topology is compact. Furthermore,
this topology of U N has a countable basis. This implies that every sequence
in U N has a convergent subsequence. Use this to establish that there exists
a subsequence (hij)j≥0 of (hi)i≥0 such that (¯u
hij
ij ¯v+∞)j≥0 converges to an
input sequence ¯u+∞.
For yet a further suitable subsequence (hijk)k≥0 of
(hij)j≥0, there exist states xk, x′
k, such that d(T(xk, ¯u+∞[hijk]), x) < ε and
d(T(x′
k, ¯u+∞[hijk]), x′) < ε, where ¯u+∞[hijk] is the preﬁx of length hijk of
¯u+∞. Since d(x, x′) > 2ε, this contradicts the state contracting property.
“echo state ⇒state contracting”: Assume that the network is not state
contracting. This implies that there exist an input sequence ¯u+∞, a strictly
growing index sequence (hi)i≥0, states xi, x′
i, and some ε > 0, such that
∀i : d(T(xi, ¯u+∞[hi]), T(x′
i, ¯u+∞[hi])) > ε. By compactness of A × A × [0, 1],
this implies that there exists some (x, x′) ∈D+ with d(x, x′) > ε. Since
D+ ⊆D, this implies that the network has not the echo state property.
“ state contracting ⇒state forgetting”: Assume the network is not state
forgetting. This implies that there exists a left-inﬁnite input sequence ¯u−∞,
a strictly growing index sequence (hi)i≥0, states xi, x′
i, and some ε > 0, such
that ∀i : d(T(xi, ¯u+∞[−hi]), T(x′
i, ¯u+∞[−hi])) > ε, where ¯u+∞[−hi] denotes
the suﬃx of lenght hi of ¯u+∞. Complete every ¯u+∞[−hi] on the right with
an arbitrary right-inﬁnite input sequence and repeat the argument from the
case “state contracting ⇒echo state” to derive a contradiction to the state
forgetting property.
“state contracting ⇒input forgetting”: trivial.
“input forgetting ⇒echo state”: Assume that the network does not have
the echo state property. Then there exists an input sequence ¯u−∞, states
x, x′ end-compatible with ¯u−∞, d(x, x′) > 0. This leads straightforwardly to
a contradiction to input forgetting.
40
B
Proof of proposition 2
Let ¯u−∞be an input sequence, let ε > 0. Using the echo state property, we
have to provide δ and h according to the proposition. We use the “input
forgetting” characterization of echo states. Let (δh)h≥0 be the null sequence
connected with ¯u−∞according to Def. 4(3.). Then there exists some h such
that for all ¯w−∞¯uh, ¯v−∞¯uh, for all x, x′ end-compatible with ¯w−∞¯uh, ¯v−∞¯uh,
it holds that d(x, x′) < ε/2. Call ¯uh δ-close to ¯u′
h if d(u(k), u′(k)) < δ for
all −h ≤k ≤0. By continuity of T, there exists some δ such that for all
¯w−∞¯uh, ¯v−∞¯u′
h, where ¯uh is δ-close to ¯u′
h, for all x, x′ end-compatible with
¯w−∞¯uh, ¯v−∞¯u′
h, it holds that d(x, x′) < ε.
C
Proof of proposition 3
“(a)”:
d(T(x, u), T(x′, u) =
=
d(f(Winu + Wx), f(Winu + Wx′))
≤
d(Winu + Wx, Winu + Wx′)
=
d(Wx, Wx′)
=
∥W(x −x)∥
≤
Λd(x, x′),
i.e., the distance between two states x, x′ shrinks by a factor Λ < 1 at every
step, regardless of the input. This Lipschitz condition obviously results in
echo states.
“(b)”: Consider the left-inﬁnite null input sequence ¯0−∞∈U −N. The
null state sequence ¯0−∞∈A−N is compatible with the null input sequence.
But if | λmax | > 1, the null state 0 ∈RN is not asymptotically stable (see
e.g. [8], Exercise 3.52), which implies the existence of another state sequence
¯x−∞̸= ¯0−∞compatible with the null input sequence. This violates the echo
state property.
D
Proof of proposition 4
“(a)”: Observe the general fact that if Fi are functions with global Lipschitz
constants Li, then P
i Fi is globally Lipschitz with Lipschitz constant P
i Li.
Observing that the rhs of (21) is a sum of two functions. The ﬁrst has a global
Lipschitz constant of L1 = 1 −δCa. Argue like in the proof of Prop. 3(a)
41
that the second has a global Lipschitz constant L2 = δCσmax, where σmax is
the largest eigenvalue of W. Observe that L1 + L2 = 1 −δC(a −σmax).
“(b)”: With null input u(n) = 0, the null state (and output) sequence is
a solution of (21). To check its asymptotic stability, consider the system (21)
linearized in the origin. It is governed by x(n+1) = (δCW + (1 −δCa)id) x(n).
If the matrix δCW+(1−δCa)id has spectral radius greater than 1, the linear
system is not asymptotically stable. This implies the existence of another
solution of (21) besides the null solution, which in turn means that the net-
work has no echo states for any input set containing 0 and admissible state
set [−1, 1]N.
E
Erratum
E.1
Corrected version of Deﬁnition 3
The version of Deﬁnition 3 in the original techreport provided three proper-
ties that were claimed in Proposition 1 to be all equivalent with the echo state
property. However, the ﬁrst property was too weak. Here a corrected version
of this deﬁnition is given. The only change is in the statement of property
1. It was called the state contracting property in the original techreport.
Tobias Strauss in [17] calls the corrected version the uniformly state con-
tracting property, a terminology that I would want to adopt (and dismiss
the old name altogether with its defunct deﬁnition).
Deﬁnition 4 Assume standard compactness conditions and a network with-
out output feedback.
1. [Corrected] The network is called uniformly state contracting if there
exists a null sequence (δh)h≥0 such that for all right-inﬁnite input se-
quences ¯u+∞, and for all states x, x′ ∈A, for all h ≥0, for all input se-
quence preﬁxes ¯uh = u(n), . . . , u(n+h) it holds that d(T(x, ¯uh), T(x′, ¯uh))
< δh, where d is the Euclidean distance on RN.
2. The network is called state forgetting if for all left-inﬁnite input se-
quences . . . , u(n −1), u(n) ∈U −N there exists a null sequence (δh)h≥0
such that for all states x, x′ ∈A, for all h ≥0, for all input sequence
suﬃxes ¯uh = u(n −h), . . . , u(n) it holds that d(T(x, ¯uh), T(x′, ¯uh)) <
δh.
3. The network is called input forgetting if for all left-inﬁnite input se-
quences ¯u−∞there exists a null sequence (δh)h≥0 such that for all h ≥0,
42
for all input sequence suﬃxes ¯uh = u(n −h), . . . , u(n), for all left-
inﬁnite input sequences of the form ¯w−∞¯uh, ¯v−∞¯uh, for all states x
end-compatible with ¯w−∞¯uh and states x′ end-compatible with ¯v−∞¯uh
it holds that d(x, x′) < δh.
The following identically re-states Proposition 1 from the original version
of the techreport, except that state contracting has been changed to uniformly
state contracting.
Proposition 5 Assume standard compactness conditions and a network with-
out output feedback. Assume that T is continuous in state and input. Then
the properties of being uniformly state contracting, state forgetting, and input
forgetting are all equivalent to the network having echo states.
The following proof of Prop. 5 by and large replicates the original proof
from the techreport, except a re-arrangement, some completions and adding
a proof for the implication uniformly state contracting ⇒echo states.
Proof.
Part 1: echo states ⇒uniformly state contracting.
Let
D
=
{(x, x′) ∈A2 |∃¯u∞∈U Z, ∃¯x∞, ¯x′∞∈AZ, ∃n ∈Z :
¯x∞, ¯x′∞compatible with ¯u∞and x = ¯x(n) and x′ = ¯x′(n)}
denote the set of all state pairs that are compatible with some input sequence.
It is easy to see that the echo state property is equivalent to the condition
that D contain only identical pairs of the form (x, x).
Like in the original techreport, we ﬁrst derive an alternative characteri-
zation of D. Consider the set
P +
=
{(x, x′, 1/h) ∈A × A × [0, 1] |
h ∈N, ∃¯uh ∈U h, x and x′ are end-compatible with ¯uh}.
Let D+ be the set of all points (x, x′) such that (x, x′, 0) is an accumu-
lation point of P + in the product topology of A × A × [0, 1]. Note that this
topology is compact and has a countable basis. We show that D+ = D.
D ⊆D+: If (x, x′) ∈D, then ∀h : (x, x′, 1/h) ∈P + due to input shift
invariance, hence (x, x′, 0) is an accumulation point of P + .
43
D+ ⊆D: (a) From continuity of T and compactness of A, a straight-
forward argument shows that D+ is closed under network update T, i.e., if
(x, x′) ∈D+, u ∈U, then (T(x, u), T(x′, u)) ∈D+. (b) Furthermore, it
holds that for every (x, x′) ∈D+, there exist u ∈U, (z, z′) ∈D+ such
that (T(z, u), T(z′, u)) = (x, x′).
To see this, let limi→∞(xi, x′
i, 1/hi) =
(x, x′, 0).
For each of the (xi, x′
i) there exist ui, (zi, z′
i) ∈A × A such
that (T(zi, ui), T(z′
i, ui)) = (xi, x′
i). Select from the sequence (zi, z′
i, ui) a
convergent subsequence (zj, z′
j, uj) (such a convergent subsequence must ex-
ist because A × A × U is compact and has a countable topological base).
Let (z, z′, u) be the limit of this subsequence. It holds that (z, z′) ∈D+
(compactness argument) and that (T(z, u), T(z′, u)) = (x, x′) (continuity ar-
gument about T). (c) Use (a) and (b) to conclude that for every (x, x′) ∈D+
there exists an input sequence ¯u∞, state sequences ¯x(n)∞, ¯x′(n)∞compatible
with ¯u∞, and n ∈Z such that x = x(n) and x′ = x′(n).
With this preparation we proceed to the proof of echo states ⇒uniformly
state contracting, repeating (and translating to English) the argument given
by Tobias Strauss.
Assume the network is not uniformly state contracting. This implies that
for every null sequence (δi)i≥0 there exists a h ≥0, an input sequence ¯uh of
length h, and states x, x′ ∈A, such that
d(T(x, ¯uh), T(x′, ¯uh)) ≥δh.
Since A is compact, it is bounded. Therefore, the sequence (µi)i≥0 deﬁned
by
µi := sup{d(T(x, ¯ui), T(x′, ¯ui)) | x, x′ ∈A, ¯ui ∈U i}
is bounded, say by M. Because we assumed that the network is not uniformly
state contracting, (µi)i≥0 is not a null sequence. Therefore there exists a
subsequence (µij)j≥0 of (µi)i≥0, which converges to some ε > 0. Since for
all i, the space U i × A is compact and T : U i × A →A is continuous, the
supremum µi is realized by suitable x, x′ ∈A. Let (xij, x′
ij) ∈A2 be such
that
(xij, x′
ij) ∈{(T(x, ¯uij), T(x′, ¯uij)) |
¯uij ∈U ij, x, x′ ∈A, d(T(x, ¯uij), T(x′, ¯uij)) = µij}.
Since A2 is compact, there exist a subsequence (xijk, x′
ijk)k≥0 of (xij, x′
ij)j≥0
which converges to some (y, y′) ∈A2. Obviously it holds that (xij, x′
ij, 1
ij ) ∈
44
P +. Thus (y, y′, 0) is an accumulation point of P +, i.e., (y, y′) ∈D+. On
the other hand,
0 < ε = lim
k→∞µijk = lim
k→∞d(xij −x′
ij) = d(y, y′).
This contradicts the echo state property, because D+ does not contain
pairs (y, y′) with y ̸= y′.
Part 2: uniformly state contracting ⇒state forgetting.
Assume the network is not state forgetting. This implies that there exists
a left-inﬁnite input sequence ¯u−∞, a strictly growing index sequence (hi)i≥0,
states xi, x′
i, and some ε > 0, such that
∀i : d(T(xi, ¯u−∞[hi]), T(x′
i, ¯u−∞[hi])) > ε,
where ¯u−∞[hi] denotes the suﬃx of lenght hi of ¯u−∞. Complete every ¯u−∞[hi]
on the right with an arbitrary right-inﬁnite input sequence, to get a series of
right-inﬁnite input sequences (¯vi)i=1,2,.... For the i-th series ¯vi it holds that
d(T(xi, ¯vi[hi], T(x′
i, ¯vi[hi])) > ε, where ¯vi[hi] is the preﬁx of length hi of ¯vi,
which contradicts the uniform state contraction property.
Part 3: state forgetting ⇒input forgetting.
Let ¯u−∞be a left-inﬁnite input sequence, and (δh)h≥0 be an associated
null sequence according to the state forgetting property. For the suﬃx ¯uh
of length h of ¯u−∞, consider any pair y, y′ of states from A. By the state
forgetting property it holds that d(T(y, ¯uh), T(y′, ¯uh)) < δh. Now consider
any left-inﬁnite ¯w−∞and ¯v−∞.
If, speciﬁcally, y, y′ are end-compatible
with ¯w−∞and ¯v−∞, respectively, it still holds that d(T(y, ¯uh), T(y′, ¯uh)) <
δh. This implies that for all states x and x′ which are end-compatible with
¯w−∞¯uh and ¯v−∞¯uh, respectively, it holds that d(x, x′) < δh.
Part 4: input forgetting ⇒echo states.
Assume that the network does not have the echo state property. Then
there exists a left-inﬁnite input sequence ¯u−∞, states x, x′ end-compatible
with ¯u−∞, d(x, such that x′) > 0. This leads immediately to a contradiction
to input forgetting, by setting ¯w−∞¯uh = ¯v−∞¯uh = ¯u−∞.
45
References
[1] L.F. Abbott and S.B. Nelson. Synaptic plasticity: taming the beast.
Nature Neuroscience, 3(supplement):1178–1183, 2000.
[2] A.F. Atiya and A.G. Parlos. New results on recurrent network training:
Unifying the algorithms and accelerating convergence.
IEEE Trans.
Neural Networks, 11(3):697–709, 2000.
[3] K. Doya.
Recurrent neural networks: Supervised learning.
In M.A.
Arbib, editor, The Handbook of Brain Theory and Neural Networks,
pages 796–800. MIT Press / Bradford Books, 1995.
[4] B. Farhang-Boroujeny. Adaptive Filters: Theory and Applications. Wi-
ley, 1998.
[5] L.A. Feldkamp, D.V. Prokhorov, C.F. Eagen, and F. Yuan. Enhanced
multi-stream Kalman ﬁlter training for recurrent neural networks. In
J.A.K. Suykens and J. Vandewalle, editors, Nonlinear Modeling: Ad-
vanced Black-Box Techniques, pages 29–54. Kluwer, 1998.
[6] F.
Gers,
D.
Eck,
and
J.
Schmidhuber.
Applying
LSTM
to
time series predictable through time-window approaches.
Tech-
nical report IDSIA-IDSIA-22-00,
IDSI/USI-SUPSI, Instituto Dalle
Molle di studi sull’ intelligenza artiﬁciale, Manno, Switzerland, 2000.
http://www.idsia.ch/∼felix/Publications.html.
[7] F.A. Gers, J. Schmidhuber, and F. Cummins. Learning to forget: con-
tinual prediction with LSTM. Neural Computation, 12(10):2451–2471,
2000.
[8] H. K. Khalil. Nonlinear Systems (second edition). Prentice Hall, 1996.
[9] M. Kimura and R. Nakano. Learning dynamical systems by recurrent
neural networks from orbits. Neural Networks, 11(9):1589–1600, 1998.
[10] S.C. Kremer. Spatiotemporal connectionist neural networks: a taxon-
omy and review. Neural Computation, 13:249–306, 2001.
[11] W. Maass, T. Natschl¨ager, and H. Markram.
Real-time comput-
ing without stable states: A new framework for neural computation
based on perturbations. Neural Computation, 14(11):2531–2560, 2002.
http://www.cis.tugraz.at/igi/maass/psﬁles/LSM-v106.pdf.
46
[12] J McNames. Innovations in local modeling for time series prediction.
Phd thesis, Dept. of Electrical Engineering, Stanford University, 1999.
www.ee.pdx.edu/∼mcnames/Publications/Dissertation.pdf.
[13] J. McNames. Local modeling optimization for time series prediction.
In Proc. 8th European Symposium on Artiﬁcial Neural Networks, 2000.
http://www.ee.pdx.edu/∼mcnames/Publications/ESANN2000.pdf.
[14] K.S. Narendra and S. Mukhopadhyay. Adaptive control using neural
networks and approximate models. IEEE Transactions on Neural Net-
works, 8(3):475–485, 1997.
[15] B.A. Pearlmutter.
Gradient calculation for dynamic recurrent neu-
ral networks: a survey. IEEE Trans. on Neural Networks, 6(5):1212–
1228,
1995.
http://www.bcl.hamilton.ie/∼bap/papers/ieee-dynnn-
draft.ps.gz.
[16] J. Stark, D.S. Broomhead, M.E. Davies, and J. Huke. Takens embedding
theorems for forced and stochastic systems. Nonlinear Analysis, Theory,
Methods & Applications, 30(8):5303–5314, 1997.
[17] T.
Strauss.
Alternative
Konvergenzmaße
f¨ur
die
Beschreibung
des
Verhaltens
von
Echo-State-Netzen.
Diplomarbeit,
Math.-
Naturwissenschaftliche Fakult¨at, Institut f¨ur Mathematik, Universit¨at
Rostock, 2009.
[18] J.
Vesanto.
Using
the
SOM
and
local
models
in
time-
series
prediction.
In
Proceedings
of
WSOM’97,
Workshop
on
Self-Organizing
Maps,
Espoo,
Finland,
June
4–6,
1997.
http://www.cis.hut.ﬁ/projects/monitor/publications/papers/wsom97.ps.zip.
Original techreport received and put to printing Dec. 6, 2001. Erratum
note added Jan 25, 2010.
47
