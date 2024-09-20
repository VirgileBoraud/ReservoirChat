Computational Models for Generic Cortical
Microcircuits
Wolfgang Maass, Thomas Natschl¨ager
Institute for Theoretical Computer Science
Technische Universitaet Graz
A-8010 Graz, Austria
{maass, tnatschl}@igi.tu-graz.ac.at
Henry Markram
Brain Mind Institute
EPFL, Lausanne
Switzerland
henry.markram@epﬂ.ch
June 10, 2003
1
Introduction
A key challenge for neural modeling is to explain how a continuous stream
of multi-modal input from a rapidly changing environment can be processed
by neural microcircuits (columns, minicolumns, etc.)
in the cerebral cor-
tex whose anatomical and physiological structure is quite similar in many
brain areas and species. However, a model that could explain the potentially
universal computational capabilities of such microcircuits has been missing.
We propose a computational model that does not require a task-dependent
construction of neural circuits. Instead it is based on principles of high di-
mensional dynamical systems in combination with statistical learning theory,
and can be implemented on generic evolved or found recurrent circuitry. This
new approach towards understanding neural computation on the micro-level
also suggests new ways of modeling cognitive processing in larger neural sys-
tems. In particular it questions traditional ways of thinking about neural
coding.
Common models for the organization of computations, such as for exam-
ple Turing machines or attractor neural networks, are less suitable for model-
ing computations in cortical microcircuits, since these microcircuits carry out
computations on continuous streams of inputs. Often there is no time to wait
until a computation has converged, the results are needed instantly (“any-
time computing”) or within a short time window (“real-time computing”).
Furthermore biological data suggest that cortical microcircuits can support
several real-time computational tasks in parallel, a hypothesis that is incon-
sistent with most modeling approaches. In addition the components of bio-
logical neural microcircuits, neurons and synapses, are highly diverse [5] and
exhibit complex dynamical responses on several temporal scales. This makes
them completely unsuitable as building blocks of computational models that
1
require simple uniform components, such as virtually all models inspired by
computer science, statistical physics, or artiﬁcial neural nets. Furthermore,
neurons are connected by highly recurrent circuitry (“loops within loops”),
which makes it particularly diﬃcult to use such circuits for robust implemen-
tations of speciﬁc computational tasks. Finally, computations in most com-
putational models are partitioned into discrete steps, each of which requires
convergence to some stable internal state, whereas the dynamics of cortical
microcircuits appears to be continuously changing. Hence, one needs a model
for using continuous perturbations in inhomogeneous dynamical systems in
order to carry out real-time computations on continuous input streams.
In this chapter we present a conceptual framework for the organization
of computations in cortical microcircuits that is not only compatible with
all these constraints, but actually requires these biologically realistic features
of neural computation. Furthermore, like Turing machines, this conceptual
approach is supported by theoretical results that prove the universality of the
computational model, but for the biologically more relevant case of real-time
computing on continuous input streams.
2
A Conceptual Framework for Real-Time
Neural Computation
A computation is a process that assigns to inputs from some domain D cer-
tain outputs from some range R, thereby computing a function from D into
R. Obviously any systematic discussion of computations requires a mathe-
matical or conceptual framework, i.e., a computational model [22]. Perhaps
the most well-known computational model is the Turing machine. In this
case the domain D and range R are sets of ﬁnite character strings. This
computational model is universal (for deterministic oﬄine digital computa-
tion) in the sense that every deterministic digital function that is computable
at all (according to a well-established mathematical deﬁnition, see [24]) can
be computed by some Turing machine. Before a Turing machine gives its
output, it goes through a series of internal computation steps, the number of
which depends on the speciﬁc input and the diﬃculty of the computational
task (therefore it is called an “oﬄine computation”). This may not be in-
adequate for modeling human reasoning about chess end games, but most
cognitive tasks are closer related to real-time computations on continuous
input streams, where online responses are needed within speciﬁc (typically
very short) time windows, regardless of the complexity of the input. In this
case the domain D and range R consist of time-varying functions u(·), y(·)
(with analog inputs and outputs), rather than of static character strings. We
propose here an alternative computational model that is more adequate for
analyzing parallel real-time computations on analog input streams, such as
those occurring in generic cognitive information processing tasks. Further-
more, we present a theoretical result which implies that within this frame-
work the computational units of a powerful computational system can be
quite arbitrary, provided that suﬃciently diverse units are available (see the
separation property and approximation property discussed in section 4). It
also is not necessary to construct circuits to achieve substantial computa-
tional power. Instead suﬃciently large and complex “found” circuits tend to
have already large computational power for real-time computing, provided
that the reservoir from which their units are chosen is suﬃciently diverse.
2
PSfrag replacements
a
u(·)
y(t)
f M
xM(t)
LM
0
0.1
0.2
0.3
0.4
0.5
0
0.5
1
1.5
2
2.5
d(u,v)=0
d(u,v)=0.1
d(u,v)=0.2
d(u,v)=0.4
state distance
time [sec]
PSfrag replacements
b
Figure 1: a) Structure of a Liquid State Machine (LSM). b) Separation property
of a generic neural microcircuit.
Plotted on the y-axis is the average value of
∥xM
u (t)−xM
v (t)∥, where ∥·∥denotes the Euclidean norm, and xM
u (t), xM
v (t) denote
the liquid states at time t for Poisson spike trains u and v as inputs. d(u, v) is deﬁned
as distance (L2-norm) between low-pass ﬁltered versions of u and v, see section 4
for details.
Our approach is based on the following observations. If one excites a
suﬃciently complex recurrent circuit (or other medium) with a continuous
input stream u(s), and looks at a later time t > s at the current internal
state x(t) of the circuit, then x(t) is likely to hold a substantial amount of
information about recent inputs u(s) (for the case of neural circuit models
this was ﬁrst demonstrated by [4]). We as human observers may not be able
to understand the “code” by which this information about u(s) is encoded in
the current circuit state x(t), but that is obviously not essential. Essential is
whether a readout neuron that has to extract such information at time t for
a speciﬁc task can accomplish this. But this amounts to a classical pattern
recognition problem, since the temporal dynamics of the input stream u(s)
has been transformed by the recurrent circuit into a high dimensional spatial
pattern x(t). This pattern classiﬁcation problem tends to be relatively easy
to learn, even by a memoryless readout, provided the desired information is
present in the circuit state x(t). Furthermore, if the recurrent neural circuit
is suﬃciently large, it may support this learning task by acting like a ker-
nel for support vector machines (see [25]), which presents a large number of
nonlinear combinations of components of the preceding input stream to the
readout. Such nonlinear projection of the original input stream u(·) into a
high dimensional space tends to facilitate the extraction of information about
this input stream at later times t, since it boosts the power of linear readouts
for classiﬁcation and regression tasks. Linear readouts are not only better
models for the readout capabilities of a biological neuron than for example
multi-layer-perceptrons, but their training is much easier and robust because
it cannot get stuck in local minima of the error function (see [25] and [7]).
These considerations suggest new hypotheses regarding the computational
function of generic recurrent neural circuits: to serve as general-purpose tem-
poral integrators, and simultaneously as kernels (i.e., nonlinear projections
into a higher dimensional space) to facilitate subsequent linear readout of
information whenever it is needed. Note that in all experiments described in
this article only the readouts were trained for speciﬁc tasks, whereas always
a ﬁxed recurrent circuit can be used for generating x(t).
In order to analyze the potential capabilities of this approach, we intro-
duce the abstract model of a Liquid State Machine (LSM), see Figure 1a. As
the name indicates, this model has some weak resemblance to a ﬁnite state
3
machine. But whereas the ﬁnite state set and the transition function of a
ﬁnite state machine have to be custom designed for each particular computa-
tional task (since they contain its “program”), a liquid state machine might
be viewed as a universal ﬁnite state machine whose “liquid” high dimensional
analog state x(t) changes continuously over time. Furthermore if this analog
state x(t) is suﬃciently high dimensional and its dynamics is suﬃciently com-
plex, then the states and transition functions of many concrete ﬁnite state
machines F are virtually contained in it. But fortunately it is in general not
necessary to reconstruct F from the dynamics of an LSM, since the readout
can be trained to recover from x(t) directly the information contained in the
corresponding state of a ﬁnite state machine F, even if the liquid state x(t)
is corrupted by some – not too large – amount of noise.
Formally, an LSM M consists of a ﬁlter LM (i.e., a function that maps
input streams u(·) onto streams x(·), where x(t) may depend not just on
u(t), but in a quite arbitrary nonlinear fashion also on previous inputs u(s);
formally: x(t) = (LMu)(t)), and a memoryless readout function f M that
maps at any time t the ﬁlter output x(t) (i.e., the “liquid state”) into some
target output y(t) (only these readout functions are trained for speciﬁc tasks
in the following). Altogether an LSM computes a ﬁlter that maps u(·) onto
y(·).1
A recurrently connected microcircuit could be viewed in a ﬁrst approxi-
mation as an implementation of such general purpose ﬁlter LM (for example
some unbiased analog memory), from which diﬀerent readout neurons extract
and recombine diverse components of the information which was contained in
the preceding input u(·). If a target output y(t) assumes analog values, one
can use instead of a single readout neuron a pool of readout neurons whose
ﬁring activity at time t represents the value y(t) in space-rate-coding. In
reality these readout neurons are not memoryless, but their membrane time
constant is substantially shorter than the time range over which integration
of information is required for most cognitive tasks. An example where the cir-
cuit input u(·) consists of 4 spike trains is indicated in Figure 2. The generic
microcircuit model consisting of 270 neurons was drawn from the distribution
discussed in section 3. In this case 7 diﬀerent linear readout neurons were
trained to extract completely diﬀerent types of information from the input
stream u(·), which require diﬀerent integration times stretching from 30 to
150ms. The computations shown are for a novel input that did not occur
during training, showing that each readout module has learned to execute
its task for quite general circuit inputs. Since the readouts were modeled by
linear neurons with a biologically realistic short time constant of just 30 ms
for the integration of spikes, additional temporally integrated information
had to be contained at any instance t in the current ﬁring state x(t) of the
recurrent circuit (its “liquid state”), see section 3 for details. Whereas the
information extracted by some of the readouts can be described in terms of
commonly discussed schemes for “neural codes”, it appears to be hopeless to
capture the dynamics or the information content of the primary engine of the
neural computation, the circuit state x(t), in terms of such coding schemes.
This view suggests that salient information may be encoded in the very high
dimensional transient states of neural circuits in a fashion that looks like
“noise” to the untrained observer, and that traditionally discussed “neural
codes” might capture only speciﬁc aspects of the actually encoded informa-
1A closely related computational model was studied in [11].
4
0.2
0.4
0
0.6
0
0.8
0.2
0.4
0
3
0
0.15
0
0.2
0.4
0.6
0.8
1
0.1
0.3
time [sec]
PSfrag replacements
input spike trains
f1(t): sum of rates of inputs 1&2 in the interval [t-30 ms, t]
f2(t): sum of rates of inputs 3&4 in the interval [t-30 ms, t]
f3(t): sum of rates of inputs 1-4 in the interval [t-60 ms, t-30 ms]
f4(t): sum of rates of inputs 1-4 in the interval [t-150 ms, t]
f5(t): spike coincidences of inputs 1&3 in the interval [t-20 ms, t]
f6(t): nonlinear combination f6(t) = f1(t) · f2(t)
f7(t): nonlinear combination f7(t) = 2f1(t) −4f 2
1 (t) + 3
2 (f2(t) −0.3)2
Figure 2: Multi-tasking in real-time. Input spike trains were randomly generated
in such a way that at any time t the input contained no information about preceding
input more than 30 ms ago. Firing rates r(t) were randomly drawn from the uniform
distribution over [0 Hz, 80 Hz] every 30 ms, and input spike trains 1 and 2 were
generated for the present 30 ms time segment as independent Poisson spike trains
with this ﬁring rate r(t). This process was repeated (with independent drawings
of r(t) and Poisson spike trains) for each 30 ms time segment. Spike trains 3 and 4
were generated in the same way, but with independent drawings of another ﬁring
rate ˜r(t) every 30 ms.
The results shown in this ﬁgure are for test data, that
were never before shown to the circuit. Below the 4 input spike trains the target
(dashed curves) and actual outputs (solid curves) of 7 linear readout neurons are
shown in real-time (on the same time axis). Targets were to output every 30 ms the
actual ﬁring rate (rates are normalized to a maximum rate of 80 Hz) of spike trains
1&2 during the preceding 30 ms (f1), the ﬁring rate of spike trains 3&4 (f2), the
sum of f1 and f2 in an earlier time interval [t-60 ms,t-30 ms] (f3) and during the
interval [t-150 ms,t] (f4), spike coincidences between inputs 1&3 (f5(t) is deﬁned
as the number of spikes which are accompanied by a spike in the other spike train
within 5 ms during the interval [t-20 ms,t]), a simple nonlinear combinations f6 and
a randomly chosen complex nonlinear combination f7 of earlier described values.
Since that all readouts were linear units, these nonlinear combinations are computed
implicitly within the generic microcircuit model. Average correlation coeﬃcients
between targets and outputs for 200 test inputs of length 1 s for f1 to f7 were 0.91,
0.92, 0.79, 0.75, 0.68, 0.87, and 0.65.
tion. Furthermore, the concept of “neural coding” suggests an agreement
between “encoder” (the neural circuit) and “decoder” (a neural readout)
which is not really needed, as long as the information is encoded in a way so
that a generic neural readout can be trained to recover it.
5
a
b
c
Figure 3: Construction of a generic neural microcircuit model, as used for all
computer simulations discussed in this chapter (only the number of neurons varied).
a) A given number of neurons is arranged on the nodes of a 3D grid. 20% of the
neurons, marked in black, are randomly selected to be inhibitory. b) Randomly
chosen postsynaptic targets are shown for two of the neurons.
The underlying
distribution favors local connections (see footnote 2 for details). c) Connectivity
graph of a generic neural microcircuit model (for λ = 2, see footnote 2). This ﬁgure
was prepared by Christian Naeger.
3
The Generic Neural Microcircuit Model
We used a randomly connected circuit consisting of leaky integrate-and-ﬁre
(I&F) neurons, 20% of which were randomly chosen to be inhibitory, as
generic neural microcircuit model.
Best performance was achieved if the
connection probability was higher for neurons with a shorter distance between
their somata (see Figure 3). Parameters of neurons and synapses were chosen
to ﬁt data from microcircuits in rat somatosensory cortex (based on [5],
[19] and unpublished data from the Markram Lab).2 Random circuits were
2Neuron parameters: membrane time constant 30 ms, absolute refractory period 3 ms
(excitatory neurons), 2 ms (inhibitory neurons), threshold 15 mV (for a resting membrane
potential assumed to be 0), reset voltage 13.5 mV, constant nonspeciﬁc background current
Ib = 13.5 nA, input resistance 1 MΩ. Connectivity structure: The probability of a synaptic
connection from neuron a to neuron b (as well as that of a synaptic connection from neuron
b to neuron a) was deﬁned as C ·exp(−D2(a, b)/λ2), where λ is a parameter which controls
both the average number of connections and the average distance between neurons that are
synaptically connected (we set λ = 2, see [16] for details). We assumed that the neurons
were located on the integer points of a 3 dimensional grid in space, where D(a, b) is the
Euclidean distance between neurons a and b. Depending on whether a and b were excitatory
(E) or inhibitory (I), the value of C was 0.3 (EE), 0.2 (EI), 0.4 (IE), 0.1 (II). In the
case of a synaptic connection from a to b we modeled the synaptic dynamics according to
the model proposed in [19], with the synaptic parameters U (use), D (time constant for
depression), F (time constant for facilitation) randomly chosen from Gaussian distributions
that were based on empirically found data for such connections. Depending on whether
a and b were excitatory (E) or inhibitory (I), the mean values of these three parameters
(with D,F expressed in seconds, s) were chosen to be .5, 1.1, .05 (EE), .05, .125, 1.2 (EI),
.25, .7, .02 (IE), .32, .144, .06 (II). The SD of each parameter was chosen to be 50% of its
mean. The mean of the scaling parameter A (in nA) was chosen to be 30 (EE), 60 (EI),
-19 (IE), -19 (II). In the case of input synapses the parameter A had a value of 18 nA if
6
constructed with sparse, primarily local connectivity (see Figure 3), both to
ﬁt anatomical data and to avoid chaotic eﬀects.
The “liquid state” x(t) of the recurrent circuit consisting of n neurons was
modeled by an n-dimensional vector consisting of the current ﬁring activity
of these n neurons. To reﬂect the membrane time constant of the readout
neurons a low pass ﬁlter with a time constant of 30 ms was applied to the
spike trains generated by the neurons in the recurrent microcircuit.
The
output of this low pass ﬁlter applied separately to each of the n neurons,
deﬁnes the liquid state x(t). Such low pass ﬁltering of the n spike trains is
necessary for the relatively small circuits that we simulate, since at many time
points t no or just very few neurons in the circuit ﬁre (see top of Figure 5).
As readout units we used simply linear neurons, trained by linear regression
(unless stated otherwise).
4
Towards a non-Turing Theory for Real-Time
Neural Computation
Whereas the famous results of Turing have shown that one can construct
Turing machines that are universal for digital sequential oﬄine computing,
we propose here an alternative computational theory that is more adequate
for parallel real-time computing on analog input streams. Furthermore we
present a theoretical result which implies that within this framework the com-
putational units of a powerful computational system can be quite arbitrary,
provided that suﬃciently diverse units are available (see the separation prop-
erty and approximation property discussed below). It also is not necessary to
construct circuits to achieve substantial computational power. Instead suﬃ-
ciently large and complex “found” circuits (such as the generic circuit used
as the main building block for Figure 2) tend to have already large compu-
tational power, provided that the reservoir from which their units are chosen
is suﬃciently diverse.
Consider a class B of basis ﬁlters B (that may for example consist of
the components that are available for building ﬁlters LM of LSMs). We say
that this class B has the point-wise separation property if for any two input
functions u(·), v(·) with u(s) ̸= v(s) for some s ≤t there exists some B ∈B
with (Bu)(t) ̸= (Bv)(t).3 There exist completely diﬀerent classes B of ﬁlters
that satisfy this point-wise separation property: B = {all delay lines}, B =
{all linear ﬁlters}, and perhaps biologically more relevant B = {models for
dynamic synapses} (see [17]).
The complementary requirement that is demanded from the class F of
functions from which the readout maps f M are to be picked is the well-
known universal approximation property: for any continuous function h and
projecting onto a excitatory neuron and 9 nA if projecting onto an inhibitory neuron. The
SD of the A parameter was chosen to be 100% of its mean and was drawn from a gamma
distribution. The postsynaptic current was modeled as an exponential decay exp(−t/τs)
with τs = 3 ms (τs = 6 ms) for excitatory (inhibitory) synapses. The transmission delays
between liquid neurons were chosen uniformly to be 1.5 ms (EE), and 0.8 ms for the other
connections. We have shown in [16] that without synaptic dynamics the computational
power of these microcircuit models decays signiﬁcantly. For each simulation, the initial
conditions of each I&F neuron, i.e., the membrane voltage at time t = 0, were drawn
randomly (uniform distribution) from the interval [13.5 mV, 15.0 mV].
3Note that it is not required that there exists a single B ∈B which achieves this
separation for any two diﬀerent input histories u(·), v(·).
7
any closed and bounded domain one can approximate h on this domain with
any desired degree of precision by some f ∈F. Examples for such classes
are F = {feedforward sigmoidal neural nets}, and according to [3] also F =
{pools of spiking neurons with analog output in space rate coding}.
A rigorous mathematical theorem [16], states that for any class B of ﬁl-
ters that satisﬁes the point-wise separation property and for any class F of
functions that satisﬁes the universal approximation property one can approx-
imate any given real-time computation on time-varying inputs with fading
memory (and hence any biologically relevant real-time computation) by an
LSM M whose ﬁlter LM is composed of ﬁnitely many ﬁlters in B, and whose
readout map f M is chosen from the class F. This theoretical result supports
the following pragmatic procedure: In order to implement a given real-time
computation with fading memory it suﬃces to take a ﬁlter L whose dynamics
is “suﬃciently complex”, and train a “suﬃciently ﬂexible” readout to trans-
form at any time t the current state x(t) = (Lu)(t) into the target output
y(t). In principle a memoryless readout can do this, without knowledge of
the current time t, provided that states x(t) and x(t′) that require diﬀerent
outputs y(t) and y(t′) are suﬃciently distinct. We refer to [16] for details.
For physical implementations of LSMs it makes more sense to analyze
instead of the theoretically relevant point-wise separation property the fol-
lowing quantitative separation property as a test for the computational ca-
pability of a ﬁlter L: How diﬀerent are the liquid states xu(t) = (Lu)(t) and
xv(t) = (Lv)(t) for two diﬀerent input histories u(·), v(·)? This is evaluated
in Figure 1b for the case where u(·), v(·) are Poisson spike trains and L is a
generic neural microcircuit model. It turns out that the diﬀerence between
the liquid states scales roughly proportionally to the diﬀerence between the
two input histories (thereby showing that the circuit dynamic is not chaotic).
This appears to be desirable from the practical point of view since it implies
that saliently diﬀerent input histories can be distinguished more easily and
in a more noise robust fashion by the readout. We propose to use such eval-
uation of the separation capability of neural microcircuits as a new standard
test for their computational capabilities.
5
A Generic Neural Microcircuit on the Com-
putational Test Stand
The theoretical results sketched in the preceding section implies that there
are no strong a priori limitations for the power of neural microcircuits for
real-time computing with fading memory, provided they are suﬃciently large
and their components are suﬃciently heterogeneous. In order to evaluate this
somewhat surprising theoretical prediction, we tested it on several benchmark
tasks.
5.1
Speech Recognition
One well-studied computational benchmark task for which data had been
made publicly available [8] is the speech recognition task considered in [9]
and [10]. The dataset consists of 500 input ﬁles: the words “zero”, “one”, ...,
“nine” are spoken by 5 diﬀerent (female) speakers, 10 times by each speaker.
The task was to construct a network of I&F neurons that could recognize each
of the 10 spoken words w. Each of the 500 input ﬁles had been encoded in
8
the form of 40 spike trains, with at most one spike per spike train4 signaling
onset, peak, or oﬀset of activity in a particular frequency band (see top of
Figure 4). A network was presented in [10] that could solve this task with an
error5 of 0.15 for recognizing the pattern “one”. No better result had been
achieved by any competing networks constructed during a widely publicized
internet competition [9].6 A particular achievement of this network (resulting
from the smoothly and linearly decaying ﬁring activity of the 800 pools of
neurons) is that it is robust with regard to linear time-warping of the input
spike pattern.
0
45
90
135
0
0.2
0.4
time [s]
   
0
20
40
"one", speaker 5
PSfrag replacements
input
microcircuit
readout
fone
0
0.2
0.4
time [s]
   
"one", speaker 3
PSfrag replacements
0
0.2
time [s]
   
"five", speaker 1
PSfrag replacements
0
0.2
time [s]
   
"eight", speaker 4
PSfrag replacements
Figure 4: Application of our generic neural microcircuit model to the speech recog-
nition from [10]. Top row: input spike patterns. Second row: spiking response of
the 135 I&F neurons in the neural microcircuit model. Third row: output of an
I&F neuron that was trained to ﬁre as soon as possible when the word “one” was
spoken, and as little as possible else. Although the “liquid state” presented to this
readout neuron changes continuously, the readout neuron has learnt to view most
of them as equivalent if they arise while the word “one” is spoken (see [16] for more
material on such equivalence classes deﬁned by readout neurons).
We tested our generic neural microcircuit model on the same task (in
fact on exactly the same 500 input ﬁles). A randomly chosen subset of 300
4The network constructed in [10] required that each spike train contained at most one
spike.
5The error (or “recognition score”) S for a particular word w was deﬁned in [10] by
S =
Nfp
Ncp +
Nfn
Ncn , where Nfp (Ncp) is the number of false (correct) positives and Nfn and
Ncn are the numbers of false and correct negatives. We use the same deﬁnition of error to
facilitate comparison of results. The recognition scores of the network constructed in [10]
and of competing networks of other researchers can be found at [8]. For the competition
the networks were allowed to be constructed especially for their task, but only one single
pattern for each word could be used for setting the synaptic weights.
6The network constructed in [10] transformed the 40 input spike trains into linearly
decaying input currents from 800 pools, each consisting of a “large set of closely similar
unsynchronized neurons” [10]. Each of the 800 currents was delivered to a separate pair of
neurons consisting of an excitatory “α-neuron” and an inhibitory “β-neuron”. To accom-
plish the particular recognition task some of the synapses between α-neurons (β-neurons)
are set to have equal weights, the others are set to zero.
9
input ﬁles was used for training, the other 200 for testing. The generic neural
microcircuit model was drawn from the distribution described in section 3,
hence from the same distribution as the circuit drawn for the completely
diﬀerent tasks discussed in Figure 2, with randomly connected I&F neurons
located on the integer points of a 15 × 3 × 3 column. The synaptic weights of
10 readout neurons fw which received inputs from the 135 I&F neurons in the
circuit were optimized (like for SVMs with linear kernels) to ﬁre whenever
the input encoded the spoken word w. Hence the whole circuit consisted of
145 I&F neurons, less than 1/30th of the size of the network constructed in
[10] for the same task7. Nevertheless the average error achieved after training
by these randomly generated generic microcircuit models was 0.14 (measured
in the same way, for the same word “one”), hence slightly better than that of
the 30 times larger network custom designed for this task. The score given is
the average for 50 randomly drawn generic microcircuit models. It is about
the same as the error achieved by any of the networks constructed in [10] and
the associated international competition.
The comparison of the two diﬀerent approaches also provides a nice illus-
tration of the diﬀerence between oﬄine computing and real-time computing.
Whereas the network of [10] implements an algorithm that needs a few hun-
dred ms of processing time between the end of the input pattern and the
answer to the classiﬁcation task (450 ms in the example of Figure 2 in [10]),
the readout neurons from the generic neural microcircuit were trained to pro-
vide their answer (through ﬁring or non-ﬁring) immediately when the input
pattern ended.
We also compared the noise robustness of the generic neural microcircuit
models with that of [10], which had been constructed to facilitate robustness
with regard to linear time warping of the input pattern. Since no benchmark
input data were available to calculate this noise robustness we constructed
such data by creating as templates 10 patterns consisting each of 40 ran-
domly drawn Poisson spike trains at 4 Hz over 0.5 s. Noisy variations of these
templates were created by ﬁrst multiplying their time scale with a randomly
drawn factor from [1/3, 3]) (thereby allowing for a 9 fold time warp), and
subsequently dislocating each spike by an amount drawn independently from
a Gaussian distribution with mean 0 and SD 32 ms. These spike patterns
were given as inputs to the same generic neural microcircuit models con-
sisting of 135 I&F neurons as discussed before. Ten readout neurons were
trained (with 1000 randomly drawn training examples) to recognize which of
the 10 templates had been used to generate a particular input (analogously
as for the word recognition task). On 500 novel test examples (drawn from
same distributions) they achieved an error of 0.09 (average performance of
30 randomly generated microcircuit models). The best one of 30 randomly
generated circuits achieved an error of just 0.005. Furthermore it turned out
that the generic microcircuit can just as well be trained to be robust with
regard to nonlinear time warp of a spatio-temporal pattern (it is not known
whether this could also be achieved by a constructed circuit). For the case
of nonlinear (sinusoidal) time warp8 an average (50 microcircuits) error of
0.2 is achieved (error of the best circuit: 0.02). This demonstrates that it is
7If one assumes that each of the 800 “large” pools of neurons in that network would
consist of just 5 neurons, it contains together with the α and β-neurons 5600 neurons.
8A spike at time t was transformed into a spike at time t′ = g(t) := B +K ·(t+1/(2πf)·
sin(2πft + ϕ)) with f = 2 Hz, K randomly drawn from [0.5,2], ϕ randomly drawn from
[0, 2π] and B chosen such that g(0) = 0.
10
not really necessary to build noise robustness explicitly into the circuit. A
randomly generated microcircuit model can easily be trained to have at least
the same noise robustness as a circuit especially constructed to achieve that.
In fact, it can also be trained to be robust with regard to types of noise that
are very hard to handle with constructed circuits.
This test had implicitly demonstrated another point. Whereas the net-
work of [10] was only able to classify spike patterns consisting of at most
one spike per spike train, a generic neural microcircuit model can classify
spike patterns without that restriction. It can for example also classify the
original version of the speech data encoded into onsets, peaks, and oﬀsets in
various frequency bands, before all except the ﬁrst events of each kind were
artiﬁcially removed to ﬁt the requirements of the network from [10].
We have also tested the generic neural microcircuit model on a much
harder speech recognition task: to recognize the spoken word not only in real-
time right after the word has been spoken, but even earlier when the word
is still spoken.9 More precisely, each of the 10 readout neurons is trained to
recognize the spoken word at any multiple of 20 ms during the 500 ms interval
while the word is still spoken (“anytime speech recognition”). Obviously the
network from [10] is not capable to do this.
But also the trivial generic
microcircuit model where the input spike trains are injected directly into the
readout neurons perform poorly on this anytime speech classiﬁcation task:
it has an error score of 3.4 (computed as described in footnote 5, but every
20 ms). In contrast a generic neural microcircuit model consisting of 135
neurons it achieves a score of 1.4 for this anytime speech classiﬁcation task
(see Figure 4 for a sample result).
One is easily led to believe that readout neurons from a neural microcir-
cuit can give a stable output only if the ﬁring activity (or more abstractly:
the state of the dynamical system deﬁned by this microcircuit) has reached
an attractor. But this line of reasoning underestimates the capabilities of a
neural readout from high dimensional dynamical systems: even if the neural
readout is just modeled by a perceptron, it can easily be trained to recognize
completely diﬀerent states of the dynamical system as being equivalent, and
to give the same response. Indeed, Figure 4 showed already that the ﬁring
activity of readout neuron can become quite independent from the dynam-
ics of the microcircuit, even though the microcircuit neurons are their only
source of input. To examine the underlying mechanism for the possibility of
relatively independent readout response, we re-examined the readout from
Figure 4. Whereas the ﬁring activity within the circuit was highly dynamic,
the ﬁring activity of the readout neurons was quite stable after training. The
stability of the readout response does not simply come about because the
spiking activity in the circuit becomes rather stable, thereby causing quite
similar liquid states (see Figure 5).
It also does not come about because
the readout only samples a few “unusual” liquid neurons as shown by the
distribution of synaptic weights onto a sample readout neuron (bottom of
Figure 5). Since the synaptic weights do not change after learning, this indi-
cates that the readout neurons have learned to deﬁne a notion of equivalence
9It turns out that the speech classiﬁcation task from [10] is in a sense too easy for a
generic neural microcircuit. If one injects the input spike trains that encode the spoken
word directly into the 10 readout neurons (each of which is trained to recognize one of the
10 spoken words) one also gets a classiﬁcation score that is almost as good as that of the
network from [10]. Therefore we consider in the following the much harder task of anytime
speech recognition.
11
a) ﬁring activity in circuit and readout
"one", speaker 3
4
5
0
0.2
0.4
time [s]
"five", speaker 1
6 7
8
0
0.1
0.2
0.3
time [s]
"eight", speaker 4
9
10
0
0.1
0.2
time [s]
0
45
90
135
microcircuit
"one", speaker 5
1
2
3
0
0.2
0.4
readout
time [s]
b) 10 selected liquid states
−0.6
−0.37 −0.01 −0.16
−0.21
class "other"
6
7
8
9
10
1
45
90
135
neuron number
state number
0.43
0.098
0.68
0.3
0.11
class "one"
1
2
3
4
5
1
45
90
135
neuron number
state number
c) weight vector
1
45
90
135
−1
0
1
weight number
Figure 5: Readout deﬁned equivalence classes of liquid states. a) The ﬁring
activity of the microcircuit for the speech recognition task from Figure 4
is reexamined. b) The liquid state x(t) is plotted for 10 randomly chosen
time points t (see arrowheads in panel a). The target output of the readout
neuron is 1 for the ﬁrst 5 liquid states, and 0 for the other 5 liquid states.
Nevertheless the 5 liquid states in each of the 2 equivalence classes are highly
diverse. But by multiplying these liquid state vectors with the weight vector
of the linear readout (see panel c), the weighted sums yields the values shown
above the liquid state vectors, which are separated by the threshold 0 of the
readout (and by the ﬁring threshold of the corresponding leaky integrate-
and-ﬁre neuron whose output spike trains are shown in panel a). c) The
weight vector of the linear readout.
12
for dynamic states of the microcircuit. Indeed, equivalence classes are an in-
evitable consequence of collapsing the high dimensional space of microcircuit
states into a single dimension, but what is surprising is that the equivalence
classes are meaningful in terms of the task, allowing invariant and appropri-
ately scaled readout responses and therefore real-time computation on novel
inputs. Furthermore, while the input may contain salient information that is
constant for a particular readout element, it may not be for another (see for
example Figure 2), indicating that equivalence classes and dynamic stability
exist purely from the perspective of the readout elements.
5.2
Predicting Movements and Solving the Aperture
Problem
This section reports results of joint work with Robert Legenstein [13], [15].
The general setup of this simulated vision task is illustrated in Figure 6. Mov-
ing objects, a ball or a bar, are presented to an 8 x 8 array of sensors (panel
a). The time course of activations of 8 randomly selected sensors, resulting
from a typical movement of the ball, is shown in panel b. Corresponding
functions of time, but for all 64 sensors, are projected as 64 dimensional in-
put by a topographic map into a generic recurrent circuit of spiking neurons.
This circuit with randomly chosen sparse connections had been chosen in the
same way as the circuits for the preceding tasks, except that it was somewhat
larger (768 neurons) to accommodate the 64 input channels.10 The resulting
ﬁring activity of all 768 integrate-and-ﬁre neurons in the recurrent circuit is
shown in panel c. Panel d of Figure 6 shows the target output for 8 of the
102 readout pools. These 8 readout pools have the task to predict the output
that the 8 sensors shown in panel b will produce 50 ms later. Hence their
target output (dashed line) is formally the same function as shown in panel
b, but shifted by 50 ms to the left. The solid lines in panel d show the actual
output of the corresponding readout pools after unsupervised learning. Thus
in each row of panel d the diﬀerence between the dashed and predicted line
is the prediction error of the corresponding readout pool.
The diversity of object movements that are presented to the 64 sensors is
indicated in Figure 7. Any straight line that crosses the marked horizontal or
vertical line segments of length 4 in the middle of the 8 x 8 ﬁeld may occur as
trajectory for the center of an object. Training and test examples are drawn
randomly from this - in principle inﬁnite - set of trajectories, each with a
movement speed that was drawn independently from a uniform distribution
over the interval from 30 to 50 units per second (unit = side length of a unit
square). Shown in Figure 7 are 20 trajectories that were randomly drawn
from this distribution. Any such movement is carried out by an independently
drawn object type (ball or bar), where bars were assumed to be oriented
vertically to their direction of movement.
Besides movements on straight
lines one could train the same circuit just as well for predicting nonlinear
movements, since nothing in the circuit was specialized for predicting linear
movements.
36 readout pools were trained to predict for any such object movement
the sensor activations of the 6 x 6 sensors in the interior of the 8 x 8 array
25 ms into the future. Further 36 readout pools were independently trained
10A 16 x 16 x 3 neuronal sheet was divided into 64 2 x 2 x 3 input regions. Each sensor
injected input into 60 % randomly chosen neurons in the associated input region. Together
they formed a topographic map for the 8 x 8 array of sensors.
13
A
1
B
2
C
3
D
4
E
5
F
6
G
7
H
8
A
1
B
2
C
3
D
4
E
5
F
6
G
7
H
8
a
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
D7
C6
F6
B5
G4
C3
F3
E2
time (in s)
sensor #
b
sensor input
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
0
150
300
450
600
750
neuron#
resulting firing activity in recurrent circuit
c
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
D7
C6
F6
B5
G4
C3
F3
E2
time
sensor #
sensor prediction by neural readouts
d
Figure 6: The prediction task. a) Typical movements of objects over a 8 x 8 sensor
ﬁeld. b) Time course of activation of 8 randomly selected sensors caused by the
movement of the ball indicated on the l.h.s. of panel a. c) Resulting ﬁring times of
768 integrate-and-ﬁre neurons in the recurrent circuit of integrate-and-ﬁre neurons
(ﬁring of inhibitory neurons marked by +). The neurons in the 16 x 16 x 3 array
were numbered layer by layer. Hence the 3 clusters in the spike raster result from
concurrent activity in the 3 layers of the circuit. d) Prediction targets (dashed lines)
and actual predictions (solid lines) for the 8 sensors from panel b. (Predictions were
sampled every 25 ms, solid curves result from linear interpolation.)
to predict their activation 50 ms into the future, showing that the prediction
span can basically be chosen arbitrarily. At any time t (sampled every 25
ms from 0 to 400 ms) one uses for each of the 72 readout pools that predict
sensory input ∆T into the future the actual activation of the corresponding
sensor at time t + ∆T as target value (“correction”) for the learning rule.
The 72 readout pools for short-term movement prediction were trained by
14
Figure 7: 20 typical trajectories of movements of the center of an object (ball or
bar).
1500 randomly drawn examples of object movements. More precisely, they
were trained to predict future sensor activation at any time (sampled every
25 ms) during the 400 ms time interval while the object (ball or bar) moved
over the sensory ﬁeld, each with another trajectory and speed.
Among the predictions of the 72 diﬀerent readout pools on 300 novel
test inputs there were for the 25 ms prediction 8.5 % false alarms (sensory
activity erroneously predicted) and 4.8 % missed predictions of subsequent
sensor activity. For those cases where a readout pool correctly predicted that
a sensor will become active, the mean of the time period of its activation was
predicted with an average error of 10.1 ms. For the 50 ms prediction there
were for 300 novel test inputs 16.5 % false alarms, 4.6 % missed predictions
of sensory activations, and an average 14.5 ms error in the prediction of the
mean of the time interval of sensory activity.
One should keep in mind that movement prediction is actually a compu-
tationally quite diﬃcult task, especially for a moving ball, since it requires
context-dependent integration of information from past inputs over time and
space.
This computational problem is often referred to as the “aperture
problem”: from the perspective of a single sensor that is currently partially
activated because the moving ball is covering part of its associated unit square
(i.e., its “receptive ﬁeld”) it is impossible to predict whether this sensor will
become more or less activated at the next movement (see [18]). In order to
decide that question, one has to know whether the center of the ball is mov-
ing towards its receptive ﬁeld, or is just passing it tangentially. To predict
whether a sensor that is currently not activated will be activated 25 or 50 ms
later, poses an even more diﬃcult problem that requires not only informa-
tion about the direction of the moving object, but also about its speed and
15
shape. Since there exists in this experiment no preprocessor that extracts
these features, which are vital for a successful prediction, each readout pool
that carries out prediction for a particular sensor has to extract on its own
these relevant pieces of information from the raw and unﬁltered information
about the recent history of sensor activities, which are still “reverberating”
in the recurrent circuit.
28 further readout pools were trained in a similar unsupervised manner
(with 1000 training examples) to predict where the moving object is going to
leave the sensor ﬁeld. More precisely, they predict which of the 28 sensors
on the perimeter are going to be activated by more than 50 % when the
moving object leaves the 8 x 8 sensor ﬁeld. This requires a prediction for
a context-dependent time span into the future that varies by 66 % between
instances of the task, due to the varying speeds of moving objects.
We
arranged that this prediction had to be made while the object crossed the
central region of the 8 x 8 ﬁeld, hence at a time when the current position of
the moving object provided hardly any information about the location where
it will leave the ﬁeld, since all movements go through the mid area of the ﬁeld.
Therefore the tasks of these 28 readout neurons require the computation of
the direction of movement of the object, and hence a computationally diﬃcult
disambiguation of the current sensory input. We refer to the discussion of
the disambiguation problem of sequence prediction in [14] and [1]. The latter
article discusses diﬃculties of disambiguation of movement prediction that
arise already if one has just pointwise objects moving at a ﬁxed speed, and just
2 of their possible trajectories cross. Obviously the disambiguation problem
is substantially more severe in our case, since a virtually unlimited number of
trajectories (see Figure 7) of diﬀerent extended objects, moving at diﬀerent
speeds, crosses in the mid area of the sensor ﬁeld. The disambiguation is
provided in our case simply through the “context” established inside the
recurrent circuit through the traces (or “reverberations”) left by preceding
sensor activations. Figure 6 shows in panel a a typical current position of
the moving ball, as well as the sensors on the perimeter that are going to
be active by ≥50 % when the object will ﬁnally leave the sensory ﬁeld. In
panel b the predictions of the corresponding 28 readout neurons (at the time
when the object crosses the mid-area of the sensory ﬁeld) is also indicated
(striped squares). The prediction performance of these 28 readout neurons
was evaluated as follows. We considered for each movement the line from
that point on the opposite part of the perimeter, where the center of the ball
had entered the sensory ﬁeld, to the midpoint of the group of those sensors on
the perimeter that were activated when the ball left the sensory ﬁeld (dashed
line). We compared this line with the line that started at the same point,
but went to the midpoint of those sensor positions which were predicted by
the 28 readout neurons to be activated when the ball left the sensory ﬁeld
(solid line). The angle between these two lines had an average value of 4.9
degrees for 100 randomly drawn novel test movements of the ball (each with
an independently drawn trajectory and speed). Another readout pool was
independently trained in a supervised manner to classify the moving object
(ball or bar). It had an error of 0 % on 300 test examples of moving objects.
The other readout pool that was trained in a supervised manner to estimate
the speed of the moving bars and balls, which ranged from 30 to 50 units per
second, made an average error of 1.48 units per second on 300 test examples.
This shows that the same recurrent circuit that provides the input for the
movement prediction can be used simultaneously by a basically unlimited
16
= actually activated
sensors on the
perimeter
= prediction for that
a
b
Figure 8: Computation of movement direction. Dashed line is the trajectory of a
moving ball. Sensors on the perimeter that will be activated by ≥50 % when the
moving ball leaves the sensor ﬁeld are marked in panel a. Sensors marked by stripes
in panel b indicate a typical prediction of sensors on the perimeter that are going to
be activated by ≥50 %, when the ball will leave the sensor ﬁeld (time span into the
future varies for this prediction between 100 and 150 ms, depending on the speed
and angle of the object movement). Solid line in panel b represents the estimated
direction of ball movement resulting from this prediction (its right end point is the
average of sensors positions on the perimeter that are predicted to become ≥50 %
activated). The angle between the dashed and solid line (average value 4.9 for test
movements) is the error of this particular computation of movement direction by
the simulated neural circuit.
number of other readouts, that are trained to extract completely diﬀerent
information about the visual input.
We refer to [13] and [15] for details.
Currently similar methods are applied to real-time processing of input from
infra-red sensors of a mobile robot.
6
Temporal Integration and Kernel Function
of Neural Microcircuit Models
In section 2 we have proposed that the computational role of generic cortical
microcircuits can be understood in terms of two complementary computa-
tional perspectives:
1. temporal integration of continuously entering information (“analog fad-
ing memory”)
2. creation of diverse nonlinear combinations of components of such in-
formation to enhance the capabilities of linear readouts to extract non-
linear combinations of pieces of information for diverse tasks (“kernel
function”).
17
The results reported in the preceding section demonstrate implicitly that
both of these computational functions are supported by generic cortical mi-
crocircuit models, since all of the benchmark problems that we discussed
require temporal integration of information. Furthermore, for all of these
computational tasks it suﬃced to train linear readouts to transform liquid
states into target outputs (although the target function to be computed was
highly nonlinear in the inputs). In this section we provide a more quantitative
analysis of these two complementary computational functions.
6.1
Temporal Integration in Neural Microcircuit Mod-
els
In order to evaluate the temporal integration capability we considered two
input distributions. These input distributions were chosen so that the mutual
information (and hence also the correlation) between diﬀerent segments of
the input stream have value 0. Hence all temporal integration of information
from earlier input segments has to be carried out by the microcircuit circuit
model, since the input itself does not provide any clues about its past. We
ﬁrst consider a distribution of input spike trains where every 30 ms a new
ﬁring rate r(t) is chosen from the uniform distribution over the interval from
0 to 80Hz (ﬁrst row in Figure 9). Then the spikes in each of the concurrent
input spike trains are generated during each 30 ms segment by a Poisson
distribution with this current rate r(t) (second row in Figure 9). Due to
random ﬂuctuation the actual sum of ﬁring rates rmeasured(t) (plotted as
dashed line in the ﬁrst row) represented by these 4 input spike trains varies
around the intended ﬁring rate r(t). rmeasured(t) is calculated as the average
ﬁring frequency in the interval [t −30 ms, t]. Third row of Figure 9 shows
that the autocorrelation of both r(t) and rmeasured(t) vanishes after 30 ms.
Various readout neurons, that all received the same input from the mi-
crocircuit model, had been trained by linear regression to output at various
times t (more precisely: at all multiples of 30 ms) the value of rmeasured(t),
rmeasured(t−30ms), rmeasured(t−60ms), rmeasured(t−90ms), etc. Figure 10a
shows (on test data not used for training) the correlation coeﬃcients achieved
between the target value and actual output value for 8 such readouts, for the
case of two generic microcircuit models consisting of 135 and 900 neurons
(both with the same distance-dependent connection probability with λ = 2
discussed in section 3). Figure 10b shows that dynamic synapses are essential
for this analog memory capability of the circuit, since the “memory curve”
drops signiﬁcantly faster if one uses instead static (“linear”) synapses for
connections within the microcircuit model. Figure 10c shows that the inter-
mediate “hidden” neurons in the microcircuit model are also essential for this
task, since without them the memory performance also drops signiﬁcantly.
It should be noted that these memory curves not only depend on the
microcircuit model, but also on the diversity of input spike patterns that
may have occurred in the input before, at, and after that time segment in
the past from which one recalls information. Hence the recall of ﬁring rates
is particularly diﬃcult, since there exists a huge number of diverse spike
patterns that all represent the same ﬁring rate. If one restricts the diversity
of input patterns that may occur, substantially longer memory recall becomes
possible, even with a fairly small circuit. In order to demonstrate this point
8 randomly generated Poisson spike trains over 250ms, or equivalently 2
Poisson spike trains over 1000ms partitioned into 4 segments each (see top of
18
0
0.2
0.4
0.6
0.8
1
1.2
1.4
1.6
1.8
2
0
50
100
time [s]
rate per spike train [Hz]
rates
r(t)
rmeasured (∆=30ms)
0
0.2
0.4
0.6
0.8
1
1.2
1.4
1.6
1.8
2
1
2
3
4
time [s]
spike train #
spike trains
−0.5
−0.4
−0.3
−0.2
−0.1
0
0.1
0.2
0.3
0.4
0.5
0
0.5
1
lag [s]
correlation coeff
auto−correlation
r(t)
rmeasured (∆=30ms)
Figure 9: Input distribution used to determine the “memory curves” for ﬁring
rates. Input spike trains (second row) are generated as Poisson spike trains with
a randomly drawn rate r(t). The rate r(t) is chosen every 30 ms from the uniform
distribution over the interval from 0 to 80 Hz (ﬁrst row, sold line). Due to random
ﬂuctuation the actual sum of ﬁring rates rmeasured(t) (ﬁrst row, dashed line) rep-
resented by these 4 input spike trains varies around the intended ﬁring rate r(t).
rmeasured(t) is calculated as the average ﬁring frequency in the interval [t−30 ms, t].
The third row shows that the autocorrelation of both r(t) and rmeasured(t) vanishes
after 30 ms.
Figure 11), were chosen as template patterns. Then spike trains over 1000ms
were generated by choosing for each 250ms segment one of the two templates
for this segment, and by jittering each spike in the templates (more precisely:
each spike was moved by an amount drawn from a Gaussian distribution with
mean 0 and a SD that we refer to as “jitter”, see bottom of Figure 11). A
typical spike train generated in this way is shown in the middle of Figure 11.
Because of the noisy dislocation of spikes it was impossible to recognize a
speciﬁc template from a single interspike interval (and there were no spatial
cues contained in this single channel input). Instead, a pattern formed by
several interspike intervals had to be recognized and classiﬁed retrospectively.
The performance of 4 readout neurons trained by linear regression to recall
the number of the template from which the corresponding input segment had
been generated is plotted in Figure 12 (thin line).
For comparison the memory curve for the recall of ﬁring rates for the same
19
0
0.1
0.2
0.3
0
0.5
1
delay [sec]
correlation
135
900
AC
frag replacements
a
0
0.1
0.2
0.3
0
0.5
1
delay [sec]
dynamic
static
AC
PSfrag replacements
b
0
0.1
0.2
0.3
0
0.5
1
delay [sec]
λ=2
λ=0
AC
PSfrag replacements
c
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0
0.2
0.4
0.6
0.8
1
delay [sec]
correlation coefficient
135 neurons, dynamic synapses (λ=2)
900 neurons, dynamic synapses (λ=2)
135 neurons, static synapses (λ=2)
135 neurons, no connectivity (λ=0)
auto−correlation (AC)
rag replacements
d
Figure 10: Memory curves for ﬁring rates in a generic neural microcircuit model.
a) Performance improves with circuit size. b) Dynamic synapses are essential for
longer recall. c) Hidden neurons in a recurrent circuit improve recall performance
(in the control case λ = 0 the readout receives synaptic input only from those
neurons in the circuit into which one of the input spike trains is injected, hence no
“hidden” neurons are involved). d) All curves from panels a to c in one diagram
for better comparison.
In each panel the bold solid line is for a generic neural
microcircuit model (discussed in section 3) consisting of 135 neurons with sparse
local connectivity (λ = 2) employing dynamic synapses. All readouts were linear,
trained by linear regression with 500 combinations of input spike trains (1000 in
the case of the liquid with 900 neurons) of length 2 s to produce every 30 ms the
desired output.
temporal segments (i.e., for inputs generated as for Figure 10, but with each
randomly chosen target ﬁring rate r(t) held constant for 250 instead of 30 ms)
is plotted as thin line in Figure 12, both for the same generic microcircuit
model consisting of 135 neurons. Figure 12 shows that information about
spike patterns of past inputs decays in a generic neural microcircuit model
slower than information about ﬁring rates of past inputs, even if just two
possible ﬁring rates may occur. One possible explanation is that the ensemble
of liquid states reﬂecting preceding input spike trains that all represented
the same ﬁring rate forms a much more complicated equivalence class than
20
1
2
1. segment
2. segment
3. segment
4. segment
template #
possible spike train segments
1  
2  
template #
0
0.2
0.4
0.6
0.8
1
1
templ. 2
templ. 1
templ. 1
templ. 2
time [s]
spike train #
a typical resulting input spike train
0
0.05
0.1
0.15
0.2
0.25
1
2
time [s]
templ. 2 for 1. seg. (top) and a jittered version (bottom)
Figure 11:
Evaluating the fading memory of a generic neural microcircuit for
spike patterns.
In this classiﬁcation task all spike trains are of length 1000 ms
and consist of 4 segments of length 250 ms each. For each segment 2 templates
were generated randomly (Poisson spike train with a frequency of 20 Hz); see upper
traces. The actual input spike trains of length 1000 ms used for training and testing
were generated by choosing for each segment one of the two associated templates,
and then generating a noisy version by moving each spike by an amount drawn
from a Gaussian distribution with mean 0 and a SD that we refer to as “jitter”
(see lower trace for a visualization of the jitter with an SD of 4 ms). The task is
to output with 4 diﬀerent readouts at time t = 1000 ms for each of the preceding 4
input segments the number of the template from which the corresponding segment
of the input was generated.
liquid states resulting from jittered versions of a single spike pattern. This
problem is ampliﬁed by the fact that information about earlier ﬁring rates is
“overwritten” with a much more diverse set of input patterns in subsequent
input segments in the case of arbitrary Poisson inputs with randomly chosen
rates. (The number of concurrent input spike trains that represent a given
ﬁring rate is less relevant for these memory curves; not shown.)
A theoretical analysis of memory retention in somewhat similar recurrent
networks of sigmoidal neurons has been given in [12].
6.2
Kernel Function of Neural Microcircuit Models
It is well-known (see [21],[25], [23]) that the power of linear readouts can be
boosted by two types of preprocessing:
- computation of a large number of nonlinear combinations of input com-
ponents and features
- projection of the input into a very high dimensional space
21
0
0.25
0.5
0.75
0
0.2
0.4
0.6
0.8
1
1.2
delay [s]
correlation
arbitrary rates
two rates
two patterns
Figure 12: Memory curves for spike patterns and ﬁring rates. Dashed line: correla-
tion of trained linear readouts with the number of the templates used for generating
the last input segment, and the segments that had ended 250 ms, 500 ms, and 750
ms ago (for the inputs discussed in Figure 11). Solid lines: correlation of trained
linear readouts with the ﬁring rates for the same time segments of length 250 ms
that were used for the spike pattern classiﬁcation task. Thick solid line is for the
case where the ideal input ﬁring rates can assume just 2 values (30 or 60 Hz),
whereas the thin solid line is for the case where arbitrary ﬁring rates between 0
and 80 Hz are randomly chosen. In either case the actual average input rates for
the 4 time segments, which had to be recalled by the readouts, assumed of course
a wider range.
In machine learning both preprocessing steps are carried out simultane-
ously by a so-called kernel, that uses a mathematical trick to avoid explicit
computations in high-dimensional spaces. In contrast, in our model for com-
putation in neural microcircuits both operations of a kernel are physically
implemented (by the microcircuit). The high-dimensional space into which
the input is projected is the state space of the neural microcircuit (a typical
column consists of roughly 100,000 neurons). This implementation makes
use of the fact that the precise mathematical formulas by which these non-
linear combinations and high-dimensional projections are computed are less
relevant. Hence these operations can be carried out by “found” neural cir-
cuits that have not been constructed for a particular task. The fact that the
generic neural microcircuit models in our simulations automatically compute
an abundance of nonlinear combinations of input fragments can be seen from
the fact that the target output values for the tasks considered in Figures 2,
4, 6, 8 are nonlinear in the input, but are nevertheless approximated quite
well by linear readouts from the current state of the neural microcircuit.
The capability of neural microcircuits to boost the power of linear read-
outs by projecting the input into higher dimensional spaces is further un-
derlined by joint work with Stefan H¨ausler [6]. There the task to recover
the number of the template spike pattern used to generate the second-to-last
segment of the input spike train11 was carried out by generic neural microcir-
cuit models of diﬀerent sizes, ranging from 12 to 784 neurons. In each case a
11This is exactly the task of the second readout in the spike pattern classiﬁcation task
discussed in Figures 11 and 12.
22
0
200
400
600
800
0
20
40
Delta rule
Size of the recurrent circuit (neuron #)
                                        
Error [%]
Figure 13: The performance of a trained readout (perceptron trained by the ∆-
rule) for microcircuit models of diﬀerent sizes, but each time for the same input
injected into the microcircuit and the same classiﬁcation task for the readout. The
error decreases with growing circuit size, both on the training data (dashed line)
and on new test data (solid line) generated by the same distribution.
perceptron was trained by the ∆-rule to classify at time 0 the template that
had been used to generate the input in the time segment [-500, -250 ms]. The
results of the computer simulations reported in Figure 13 show that the per-
formance of such (thresholded) linear readout improves drastically with the
size of the microcircuit into which the spike train is injected, and therefore
with the dimension of the “liquid state” that is presented to the readout.
7
Software for Evaluating the Computational
Capabilities of Neural Microcircuit Models
New software for the creation, fast simulation and computational evalua-
tion of neural microcircuit models has recently been written by Thomas
Natschl¨ager (with contributions by Christian Naeger), see [20]. This software,
which has been made available for free use on WWW.LSM.TUGRAZ.AT,
uses an eﬃcient C++ kernel for the simulation of neural microcircuits.12 But
the construction and evaluation of these microcircuit models can be carried
out conveniently in MATLAB. In particular the website contains MATLAB
scripts that can be used for validating the results reported in this chapter.
The object oriented style of the software makes it easy to change the micro-
circuit model or the computational tasks used for these tests.
8
Discussion
We have presented a conceptual framework for analyzing computations in
generic neural microcircuit models that satisﬁes the biological constraints
12For example a neural microcircuit model consisting of a few hundred leaky integrate-
and-ﬁre neurons with up to 1000 dynamic synapses can be simulated in real-time on a
current generation PC.
23
listed in section 1. Thus one can now take computer models of neural micro-
circuits, that can be as realistic as one wants to, and use them not just for
demonstrating dynamic eﬀects such as synchronization or oscillations, but to
really carry out demanding computations with these models. The somewhat
surprising result is that the inherent dynamics of cortical microcircuit mod-
els, which appears to be virtually impossible to understand in detail for a
human observer, nevertheless presents information about the recent past of
its input stream in such a way that a single perceptron (or linear readout in
the case where an analog output is needed) can immediately extract from it
the “right answer”. Traditional approaches towards producing the outputs
of such complex computations in a computer usually rely on a sequential al-
gorithm consisting of a sequence of computation steps involving elementary
operations such as feature extraction, addition and multiplication of num-
bers, and “binding” of related pieces of information. The simulation results
discussed in this chapter demonstrate that a completely diﬀerent organiza-
tion of such computations is possible, which does not require to implement
these seemingly unavoidable elementary operations. Furthermore, this al-
ternative computation style is supported by theoretical results (see section
4), which suggest that it is in principle as powerful as von Neumann style
computational models such as Turing machines, but more adequate for the
type of real-time computing on analog input streams that is carried out by
the nervous system.
Obviously this alternative conceptual framework relativizes some basic
concepts of computational neuroscience such as receptive ﬁelds, neural cod-
ing and binding, or rather places them into a new context of computational
organization. Furthermore it suggests new experimental paradigms for inves-
tigating the computational role of cortical microcircuits. Instead of exper-
iments on highly trained animals that aim at isolating neural correlates of
conjectured elementary computational operations, the approach discussed in
this chapter suggests experiments on naturally behaving animals that focus
on the role of cortical microcircuits as general purpose temporal integrators
(analog fading memory) and simultaneously as high dimensional nonlinear
kernels to facilitate linear readout.
The underlying computational theory
(and related experiments in machine learning) support the intuitively rather
surprising ﬁnding that the precise details how these two tasks are carried
out (e.g. how memories from diﬀerent time windows are superimposed, or
which nonlinear combinations are produced in the kernel) are less relevant
for the performance of the computational model, since a linear readout from
a high dimensional dynamical system can in general be trained to adjust to
any particular way in which these two tasks are executed. Some evidence
for temporal integration in cortical microcircuits has already been provided
through experiments that demonstrate the dependence of the current dynam-
ics of cortical areas on their initial state at the beginning of a trial, see e.g.
[2]. Apparently this initial state contains information about preceding input
to that cortical area. Our theoretical approach suggests further experiments
that quantify the information about earlier inputs in the current state of
neural microcircuits in vivo. It also suggests to explore in detail which of
this information is read out by diverse readouts and projected to other brain
areas.
The computational theory outlined in this chapter diﬀers also in another
aspect from previous theoretical work in computational neuroscience: instead
of constructing hypothetical neural circuits for speciﬁc (typically simpliﬁed)
24
computational tasks, this theory proposes to take the existing cortical cir-
cuitry “oﬀthe shelf” and examine which adaptive principles may enable them
to carry out those diverse and demanding real-time computations on contin-
uous input streams that are characteristic for the astounding computational
capabilities of the cortex.
The generic microcircuit models discussed in this chapter were relatively
simple insofar as they did not yet take into account more speciﬁc anatomical
and neurophysiological data regarding the distribution of speciﬁc types of
neurons in speciﬁc cortical layers, and known details regarding their speciﬁc
connection patterns and regularization mechanisms to improve their perfor-
mance (work in progress). But obviously these more detailed models can
be analyzed in the same way, and it will be quite interesting to compare
their computational power with that of the simpler models discussed in this
chapter.
Acknowledgement: The work was partially supported by the Austrian
Science Fond FWF, project # P15386.
References
[1] L. F. Abbott and K. I. Blum. Functional signiﬁcance of long-term potentiation
for sequence learning and prediction. Cerebral Cortex, 6:406–416, 1996.
[2] A. Arieli, A. Sterkin, A. Grinvald, and A. Aertsen. Dynamics of ongoing activ-
ity: explanation of the large variability in evoked cortical responses. Science,
273:1868–1871, 1996.
[3] P. Auer, H. Burgsteiner, and W. Maass.
Reducing communication for
distributed learning in neural networks.
In Jos´e R. Dorronsoro,
edi-
tor, Proc. of the International Conference on Artiﬁcial Neural Networks
– ICANN 2002,
volume 2415
of Lecture Notes in Computer Science,
pages 123–128. Springer-Verlag,
2002.
Online available as #127 from
http://www.igi.tugraz.at/maass/publications.html.
[4] D. V. Buonomano and M. M. Merzenich. Temporal information transformed
into a spatial code by a neural network with realistic properties.
Science,
267:1028–1030, Feb. 1995.
[5] A. Gupta, Y. Wang, and H. Markram. Organizing principles for a diversity of
GABAergic interneurons and synapses in the neocortex. Science, 287:273–278,
2000.
[6] S. H¨ausler,
H. Markram,
and W. Maass.
Perspectives of the high
dimensional
dynamics of
neural
microcircuits
from
the
point of
view
of
low
dimensional
readouts.
Complexity
(Special
Issue
on
Complex
Adaptive
Systems,
2003.
in
press.
Online
available
as
#
137
from
http://www.igi.tugraz.at/maass/publications.html.
[7] S. Haykin. Neural Networks: A Comprehensive Foundation. Prentice Hall,
2nd edition, 1999.
[8] J. Hopﬁeld and C. Brody. The mus silicium (sonoran desert sand mouse) web
page. Base: http://moment.princeton.edu/~mus/Organism, Dataset: Base
+ /Competition/digits data.html, Scores: Base + /Docs/winners.html.
[9] J. J. Hopﬁeld and C. D. Brody. What is a moment? “cortical” sensory inte-
gration over a brief interval. Proc. Natl. Acad. Sci. USA, 97(25):13919–13924,
2000.
[10] J. J. Hopﬁeld and C. D. Brody. What is a moment? transient synchrony as
a collective mechanism for spatiotemporal integration. Proc. Natl. Acad. Sci.
USA, 98(3):1282–1287, 2001.
25
[11] H. J¨ager.
The ”echo state” approach to analyzing and training recurrent
neural networks.
GMD Report 148, German National Research Center for
Information Technology, 2001.
[12] H. J¨ager.
Short term memory in echo state networks.
GMD Report 152,
German National Research Center for Information Technology, 2002.
[13] R.
A.
Legenstein,
H.
Markram,
and
W.
Maass.
Input
prediction
and autonomous movement analysis in recurrent circuits of spiking neu-
rons.
Reviews in the Neurosciences (Special Issue on Neural and Ar-
tiﬁcial Computation,
2003.
in press. Online available as #140 from
http://www.igi.tugraz.at/maass/publications.html.
[14] W. B. Levy. A sequence predicting CA3 is a ﬂexible associator that learns and
uses context to solve hippocampal-like tasks. Hippocampus, 6:579–590, 1996.
[15] W. Maass, R. A. Legenstein, and H. Markram.
A new approach to-
wards vision suggested by biologically realistic neural microcircuit mod-
els.
In H. H. Buelthoﬀ, S. W. Lee, T. A. Poggio, and C. Wallraven,
editors, Biologically Motivated Computer Vision, Proc. of the Second In-
ternational Workshop, BMCV 2002, T¨ubingen, Germany, November 22–
24, 2002, volume 2525 of Lecture Notes in Computer Science, pages 282–
293. Springer (Berlin), 2002.
in press. Online available as #146 from
http://www.igi.tugraz.at/maass/publications.html.
[16] W. Maass, T. Natschl¨ager, and H. Markram. Real-time computing without
stable states: A new framework for neural computation based on perturba-
tions. Neural Computation, 14(11):2531–2560, 2002. Online available as #130
from http://www.igi.tugraz.at/maass/publications.html.
[17] W. Maass and E. D. Sontag.
Neural systems as nonlinear ﬁlters.
Neu-
ral Computation, 12(8):1743–1772, 2000.
Online available as #107 from
http://www.igi.tugraz.at/maass/publications.html.
[18] H. A. Mallot. Computational Vision. MIT Press, Cambridge, MA, 2000.
[19] H. Markram, Y. Wang, and M. Tsodyks. Diﬀerential signaling via the same
axon of neocortical pyramidal neurons. Proc. Natl. Acad. Sci., 95:5323–5328,
1998.
[20] T. Natschl¨ager,
H. Markram,
and W. Maass.
Computer models and
analysis tools for neural microcircuits.
In R. K¨otter,
editor,
Neuro-
science Databases. A Practical Guide, chapter 9, pages 123–138. Kluwer
Academic Publishers (Boston),
2003.
Online available as #144 from
http://www.igi.tugraz.at/maass/publications.html.
[21] J. F. Rosenblatt. Principles of Neurodynamics. Spartan Books (New York),
1962.
[22] J. E. Savage.
Models of Computation: Exploring the Power of Computing.
Addison-Wesley (Reading, MA, USA), 1998.
[23] B. Sch¨olkopf and A. J. Smola. Learning with Kernels. MIT Press (Cambridge),
2002.
[24] R. I. Soare. Recursively Enumerable Sets and Degrees: A Study of Computable
Functions and Computably Generated Sets. Springer Verlag (Berlin), 1987.
[25] V. N. Vapnik. Statistical Learning Theory. John Wiley (New York), 1998.
26
