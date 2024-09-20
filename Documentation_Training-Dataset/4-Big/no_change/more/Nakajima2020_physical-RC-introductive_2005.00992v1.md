Physical reservoir computing—An introductory perspective
Kohei Nakajima1, ∗
1Graduate School of Information Science and Technology,
The University of Tokyo, Bunkyo-ku, 113-8656 Tokyo, Japan
Understanding the fundamental relationships between physics and its information-processing ca-
pability has been an active research topic for many years. Physical reservoir computing is a re-
cently introduced framework that allows one to exploit the complex dynamics of physical systems as
information-processing devices. This framework is particularly suited for edge computing devices,
in which information processing is incorporated at the edge (e.g., into sensors) in a decentralized
manner to reduce the adaptation delay caused by data transmission overhead. This paper aims to
illustrate the potentials of the framework using examples from soft robotics and to provide a concise
overview focusing on the basic motivations for introducing it, which stem from a number of ﬁelds,
including machine learning, nonlinear dynamical systems, biological science, materials science, and
physics.
I.
INTRODUCTION
Recently, a novel information-processing scheme that
exploits physical dynamics as a computational resource
has been proposed. This scheme is called physical reser-
voir computing (PRC). The current paper aims to intro-
duce this framework concisely, focusing on its motiva-
tion and potential by using a number of examples. Un-
derstanding the original concept of reservoir computing
(RC) is important to comprehend the concept of PRC.
RC is a framework for recurrent neural network (RNN)
training and was proposed in the early 2000s as a broad
concept that allows to deal with a number of diﬀerent
models of RNN, including the echo-state network (ESN)
[1–3] and the liquid state machine (LSM) [4], under the
same umbrella [5–8].
Conventionally, to train an RNN, a backpropagation-
through-time (BPTT) method [9] is frequently used. In
this method, all the weights of the network are basically
tuned toward the target function. In the RC framework,
by preparing an RNN equipped with a massive amount
of nonlinear elements coupled with one another, called a
reservoir, only the readout part is usually trained toward
the target function. In the simplest case, this readout
part consists of linear and static weights that directly
connect the reservoir nodes and output node (Fig. 1A).
Because of this unique system construction, RC has many
advantages. Some typical examples are given below.
The ﬁrst advantage comes from the ease in the training
procedure, which makes the learning quick and stable. As
noted above, in the conventional BPTT approach, all the
weights in the network are tuned, which takes a signiﬁ-
cant amount of time in obtaining the optimal parameter
set according to the type of the given target function.
Furthermore, it is known to be unstable, in general, sug-
gesting that it cannot always obtain the optimal set of
weights after learning [10].
In the RC framework, the
weights in the network are not always targeted for train-
∗k nakajima@mech.t.u-tokyo.ac.jp
ing. Instead, the training is mainly for the readout part,
so the number of parameters that need to be tuned is gen-
erally small, making the training signiﬁcantly faster (Fig.
1A). In particular, if the readout part is set as linear and
static weights, the training can be executed with a simple
linear regression or ridge regression, and the optimal set
of weights can be induced at once through a batch learn-
ing procedure, making the entire learning process simple
and stable. Accordingly, there are many real-world appli-
cation scenarios proposed in the literature. Starting from
conventional signal processing for robust communication
against noise [3], learning of the grammatical structure of
natural language [11], robust speech recognitions [12], or
handwritten digit recognitions [13], many attempts can
be found for complex time series prediction tasks, includ-
ing the time series of stock markets [14, 15] or for the
prediction of high-dimensional spatiotemporal dynamics
found in nature [16], including weather forecasting or the
prediction of forest ﬁre spreading. In robotics, for exam-
ple, many cognitive tasks, which were previously diﬃ-
cult to implement because of the complicated procedure
of RNN training, have been revived using RC for cog-
nitive agents [17], and behavioral generations of robots,
such as the emulation of motor controller [18–21], inverse
kinematics [23], timing control [22], and central pattern
generator (CPG) [24], are successfully performed. In ad-
dition, researchers are now interested in applying the RC
framework to sensory devices, in which the raw data are
collected, and for executing processing natively on the
sensory devices in real time, which is called edge com-
puting [25]. Fonollosa et al. applied an RC framework to
a chemical gas sensory system and showed that it is suit-
able for real-time and continuous monitoring applications
and improves the time response of the chemical sensory
system [26]. Recently, the emulation of the functionality
of a sensory device in a soft robotic platform was pro-
posed using ESN, where the laser displacement sensor is
emulated in a signiﬁcantly high accuracy [27]. This ap-
proach is expected to replace the functionality of rigid
components, that is, sensory devices, freeing soft robotic
platforms from mechanical constraints to maintain their
softness and ﬂexibility. We should note that although the
arXiv:2005.00992v1  [nlin.AO]  3 May 2020
2
A
B
C
FIG. 1. Typical settings and advantages in RC. A. A typical ESN setting, a representative model in the RC framework.
The reservoir is an RNN often equipped with a nonlinear activation function, such as y = tanh(x). Only the readout part is
usually trained to the target function. B. In the RC framework, multitasking can be safely implemented in principle, because no
interference occurs among the tasks during the learning procedures. See the text for details. C. Physical reservoir computing,
which exploits the physical dynamics as a reservoir.
learning procedure of RC is simple, this does not imply
that RC is less powerful than conventional machine learn-
ing techniques [28]. For example, it has been shown that
ESN, which is a representative model system of RC, has
a universal approximation property, and many studies
are now proving its expressive power in diﬀerent settings
[29, 30]. This implies that it is largely up to the exper-
imenters using the framework and how they will utilize
it to induce its potential. In a machine learning context,
many improvements have been proposed to overcome the
instability of RNN learning based on BPTT algorithms,
which can be represented in the model of long-short term
memory [31], gated recurrent unit [32], or unitary RNNs
[33, 34]. Among these approaches, a recent systematic
comparison analysis with RC has shown that each of
these approaches has its merits and demerits (see Ref.[35]
for more details), which suggests that the best approach
depends on the experimental conditions and is largely up
to what the experimenters wish to achieve.
The second advantage is its ease in multitasking or in
sequential learning.
Consider that the network is now
implementing a task TA to the output A according to
the input u, which is expressed as yA = TA(u). Now,
we want to train the same network to additionally learn
the task TB to the output B according to the same input
u, which is expressed as yB = TB(u).
In the conven-
tional approach of backpropagation, the entire network
is optimized for the task TA ﬁrst, and then the network
is additionally trained for the task TB using the back-
propagation method, so these two tasks interfere dur-
ing the update of weights within the same network. In
this situation, there is danger that the network forgets
the previously learned tasks. The extreme case for this
phenomenon is called catastrophic interference or catas-
trophic forgetting [36, 37], and addressing this deﬁcit re-
mains a controversial topic for many researchers (see,
e.g., Ref. [38–40]). In the RC framework, because the
training is basically limited at the readout part, no in-
terference occurs among the tasks, so multitasking can
safely be implemented in principle (Fig. 1B).
The third advantage is the arbitrariness and diversity
in the choice of a reservoir. The basic concept of RC is
exploiting the intrinsic dynamics of the reservoir by out-
sourcing learning, which requires some parameter tuning,
to the readout part. According to this unique setting,
reservoirs do not have to be an RNN anymore but can
be any dynamical system. This idea naturally leads us
to exploit the physical dynamics as a reservoir instead of
using the simulated dynamics inside the PC (Fig. 1C).
This framework is called PRC and is a main theme of the
current paper. This seemingly natural step makes the
framework radically diﬀerent from other machine learn-
ing methods. That is, PRC provides a novel insight not
only into the machine learning community, but also into
the dynamical systems ﬁeld, physics, materials science,
and biological science. This point will be elaborated on
in detail later.
II.
PREREQUISITE FOR A SUCCESSFUL
RESERVOIR
As we veriﬁed in the previous section, there is a di-
versity in the choice of reservoir, and there is a freedom
to use any kind of dynamical system if you wish. How-
ever, whether that reservoir works successfully is a dif-
ferent story. There exists a prerequisite to be used as
a successful reservoir. The prerequisite is about the re-
producibility of the input–output relation, which is an in-
evitable condition for any computational device. Namely,
the reservoir should respond the same whenever the same
input sequence is injected.
Otherwise, every time you
used it, the reservoir would respond diﬀerently, meaning
that it would be operationally troublesome and unreli-
able.
Considering that the reservoir is basically a dy-
namical system, this requirement is a somewhat severe
condition because the behavior of dynamical systems is
in general determined by the initial condition.
If you
can precisely select the initial condition of the reservoir
3
and can control the timing to inject the input sequence
into the system, then for the identical input sequence,
you can always obtain the same response from the sys-
tem. However, this constraint restricts the usability of
the computational device, and it is particularly annoying
if you wish to exploit the natural and physical dynamics
as a reservoir because it is generally diﬃcult to infer or
control the initial condition of the physical dynamics. It
is preferable to guarantee the reproducibility of the re-
sponse whenever you inject the same input sequence and,
furthermore, to do so without controlling the initial con-
dition of the reservoir. The property that realizes these
conditions of the reservoir is called the echo state prop-
erty (ESP) [1]. Simply put, ESP requires the reservoir
states to be expressed as a function of the previous input
sequence only. A similar concept has been studied in the
nonlinear dynamical systems ﬁeld from a diﬀerent angle
as a synchronization phenomenon between two identical
systems induced by a common signal (or noise) or a gen-
eralized synchronization between an input sequence and
the corresponding response of the system (see, e.g., Ref.
[41]). This property suggests that even if the system is
driven by a diﬀerent initial condition, by injecting an in-
put sequence, the corresponding response of the system
becomes the same. Mathematical investigations of the
concept of ESP (e.g., Ref. [42–44]) and understanding
its relation to the nonlinear dynamical systems ﬁeld are
still ongoing research topics (e.g., Ref. [45]).
Here, we would like to summarize the situation brieﬂy
(Fig.
2).
Consider that we have the input u(t) and
the reservoir state x(t) at timestep t, and the reser-
voir dynamics is expressed as x(t + 1) = f(x(t), u(t)).
In general, a task T targeted by RNN is a function of
the previous input sequence, which is sometimes called
a temporal machine learning task; then, it is expressed
as y(t + 1) = T(u(t), u(t −1), ...). In the RC scheme,
by tuning the readout ψ (note that this readout func-
tion does not have to be linear in general), we aim
to approximate the target y(t), which is expressed as
y(t) ≈ψ(x(t)).
Now, if the reservoir fulﬁls the ESP,
then x(t) = φ(u(t −1), u(t −2), ..), where φ is called
the input echo function in Ref.
[1] and where it is a
function intrinsic to the reservoir. This implies that the
internal state of the reservoir is completely described
by the driven input sequence and is related to the ﬁl-
ter concept, which will be discussed in more detail later.
Note that when the ESP holds, then the reservoir states
from diﬀerent initial conditions, which are expressed as
x′(t) and x(t) and driven by identical input sequence,
will respond the same or become synchronized, such as
|f(x′(t), u(t)) −f(x(t), u(t))| ≈0 for a suﬃciently large
t. In summary, the RC scheme can be expressed as ex-
ploiting the function intrinsic to the reservoir φ and ad-
justing the readout function ψ to approximate the target
function T, which is expressed as T(u(t), u(t −1), ...) ≈
ψ(φ(u(t), u(t −1), ...)).
From this viewpoint, evaluating the information pro-
cessing capability or expressive power of a given reser-
voir is nothing but evaluating the property of the func-
tion φ. Currently, several approaches exist. The typical
case is evaluating how well the given reservoir can output
the previous input sequence, and this measure is called
memory capacity [46]. Focusing on ESN, the behaviors of
memory capacity and their related measures are studied
in detail with a linear activation function [46–51] and, re-
cently, with a nonlinear activation function [52–54]. This
measure is further generalized and extended to be able
to evaluate the nonlinear memory capacities by decom-
posing the function φ into the combinations of multiple
orthogonal polynomials [55], and the trade-oﬀbetween
the expressiveness of φ for linear and nonlinear functions
is investigated [55, 56]. Investigations of the relationships
between the dynamical property of the reservoir and its
information processing capability are now ongoing hot
topics in the ﬁeld [57]. Discussions that include how the
bifurcation structure or the order-chaos transition (the
critical point is often referred to as edge of chaos) aﬀects
the computational power of the reservoir are one such
example [58–61].
As we conﬁrmed in this section, although, on the one
hand, the learning procedure seems simple in RC, which
is outsourced to the readout part, on the other hand, the
reservoir part can be taken as a huge hyper parameter
that is diﬃcult to harness without knowledge of nonlinear
dynamical systems.
III.
DIVERSE VARIATIONS OF RESERVOIR:
TOWARD EXPLOITING PHYSICAL DYNAMICS
As we discussed in the previous sections, many types of
reservoirs are now proposed. Among these, a framework
that exploits the physical dynamics as the reservoir is
called PRC. Because the natural physical dynamics is di-
rectly used as a computational resource, even if the same
computation is implemented, according to the diﬀerent
physical property, there will be diverse application sce-
narios. Increasingly, many physical reservoirs have been
reported worldwide (Fig. 3), such as the case using pho-
tonics [62–72], spintronics [73–76, 78–85], quantum dy-
namics [86–91], nanomaterials [92–100], analog circuits
and ﬁeld programmable gate arrays [101–108], mechanics
[109–124], ﬂuids [125–128], and biological materials [129–
131]. Readers interested in which types of reservoirs are
currently proposed can refer to, e.g., Ref. [132, 133].
Before going into PRC, which is the main theme of
the current paper, we would like to overview the typical
misapprehensions that we frequently face when it comes
to the RC framework in this section.
The ﬁrst one is
the belief that the weights of the reservoir should be set
randomly. Of course, there exists a reservoir that imple-
ments a random weight matrix, such as ESN, but this
is not an essential requisite. RC was originally inspired
by the type of information processing that occures in the
brain, and the connections between neurons are usually
not random but have speciﬁc structures.
Accordingly,
4
FIG. 2. Schematics showing how the echo state property works in RC. As can be seen in the diagram, input echo
function φ is a part intrinsic to the reservoir, and experimenters can adjust the output using readout function ψ. See the text
for details.
several reservoir settings implement brain-inspired con-
nections [134] or simply implement the neighboring con-
nections [4], introducing a spatial dimension that is not
random at all. More coherent network structures, such
as cyclic reservoirs, are also investigated [50]. One inter-
esting aspect of RC is that it is capable of exploring the
computational account of the structure of the reservoir,
and as we will see later, this point is important for PRC.
The second misconception is that the reservoir weights
should remain unchanged, and experimenters cannot
tune them in any sense. This is untrue. This mistake is
thought to be raised from the expression of the RC learn-
ing scheme that the training is performed in the readout
part. This expression of the learning scheme is true, but
this does not always mean experimenters cannot tune the
weights of the reservoir. An obvious counterexample is
that when setting the ESN, it is common to tune the spec-
tral radius of the reservoir weights [1–3, 5, 135]. This is
nothing but the tuning, or preconditioning, of the inter-
nal weights before training the readout to some speciﬁc
task. Another example can be found in cases that imple-
ment pretraining in the reservoir part before training the
entire system for some speciﬁc target task. The use of re-
current infomax [136], which maximizes the mutual infor-
mation between the past and future within the internal
dynamics, or the implementation of the plasticity rule,
such as Hebbian learning [137] or spike-timing-dependent
plasticity (STDP) [138, 139], into the input-driven RNN
have been reported in pretraining the reservoir.
From
this viewpoint, the recently introduced RNN called AL-
BERT [140] for language processing can be included as
a pretrained reservoir whose internal networks are pre-
trained based on predicting the ordering of two consecu-
tive segments of text in the language data set; here, the
readout part is trained for speciﬁc language-processing
tasks.
The third misunderstanding is that if the reservoir is
exhibiting chaos, which is a frequently observed behav-
ior of nonlinear dynamical systems, then this means it
cannot be used successfully.
Chaos can be character-
ized by sensitivity to initial conditions, where a slight
initial diﬀerence in the state expands exponentially, and
in this sense, the current state of the system is certainly
aﬀected by the initial condition. Accordingly, although
the chaotic dynamics show a rich diversity of patterns for
function emulation, it seems that chaos does not show
ESP and is not suitable for RC. However, this is not the
case. Even if the dynamical system exhibits chaos, when
it is driven by the input sequence (or noise), chaos is
sometimes suppressed, and generalized synchronization
occurs between the input sequence and the response of
the dynamics [41], which is an outcome of ESP. In par-
ticular, chaos in a large ESN equipped with a sigmoidal
function [141] can be suppressed with noise [142]. There
exists a learning scheme that exploits this property of
chaos suppression eﬀectively, and it is found in the study
of a ﬁrst-order-reduced and controlled-error (FORCE)
learning approach [143]. In the study of FORCE learn-
ing, it was found that a chaotic reservoir is capable of
implementing coherent patterns by adjusting the read-
out weights with the output fed back to the reservoir,
or interestingly, the learning performance was even bet-
ter than a non-chaotic reservoir in this condition. Fur-
thermore, because there is no fundamental diﬀerence be-
tween the output node fed back to the reservoir and the
reservoir nodes interacting with each other, both through
linear connection weights (although there is a slight dif-
ference concerning whether the output is injected into
the nonlinear activation function before fed back to the
reservoir), the FORCE learning scheme has been applied
not only to the readout weights, but also to the inter-
nal weights of the reservoir [143–145].
Chaos is more
5
A
B
C
D
E
FIG. 3.
Variations of physical reservoirs.
A. The physical liquid state machine proposed in Ref.
[127].
It exploits
the Faraday wave as a computational resource. B. Quantum reservoir computing proposed in Ref. [86]. It allows to exploit
disordered ensemble quantum dynamics as a computational resource. Figure reprinted with permission from Ref. [86], Copyright
(2017) by the American Physical Society. C. Variations of the spintronics reservoir. The upper and lower diagrams show
reservoirs, which exploit vortex-type spintronics [78] and spatially multiplexed magnetic tunnel junctions [74], respectively.
The upper diagram is a ﬁgure reprinted with permission from Ref. [78] by the author. The lower diagram is a ﬁgure reprinted
with permission from Ref. [74], Copyright (2018) by the American Physical Society. D. Complex Turing B-type atomic switch
networks proposed in Ref. [92]. The complex nanowire network extends throughout the device and is probed via macroscopic
electrodes. Figure reprinted with permission from Ref. [92], Copyright (2012) by John Wiley and Sons. E. A skyrmion network
embedded in frustrated magnetic ﬁlms proposed in Ref. [77]. The current path is visualized after the voltage is applied to the
frustrated magnetic texture including Bloch skyrmions. Figure reprinted with permission from Ref. [77], Copyright (2018) by
the American Physical Society.
apparently exploited in the learning scheme, which is
called innate training [146]. Chaos has rich dynamics but
does not guarantee reproducible input–output relations.
Then, why not keep the richness of the dynamics and
make it reproducible? In the innate training approach,
preparing the chaotic reservoir at ﬁrst and collecting its
own chaotic dynamics as training data, the internal con-
nection weights are trained using FORCE learning to out-
put their own chaotic dynamics reproducibly. This ap-
proach can be also viewed as pretraining of the reservoir
and has been applied for several machine learning and
robot control tasks (see, e.g., Ref. [147–149]). Recently,
many neuromorphic devices have been shown to exhibit
chaos (e.g., Ref. [150–152]), and it is expected that these
chaotic dynamics can be harnessed and exploited as a
computational resource based on an RC framework.
Because the tuning of the internal weights were in-
troduced in the above approaches, it may be helpful to
6
clarify the diﬀerence between the conventional training
scheme, such as BPTT, and the above introduced ap-
proaches. The main diﬀerence comes from the design of
the cost function. In BPTT, there usually exists a global
target function, and the gradient is obtained based on it;
in addition, the error is backpropagated to each internal
node to be used to update the concerning weights. In the
above approaches, however, the internal weights are not
usually tuned for the global target function but can be
tuned for any global or local target function that the ex-
perimenter designs. In this sense, the above approaches
contain more freedom in the setting of cost functions, or
it may be more appropriate to say that these approaches
even include the conventional setting of cost function.
This RC property, which can be composed of multiple
cost functions, is also an important aspect to be kept in
mind when trying to step toward the PRC.
In this paper, we discuss what becomes interesting
when we proceed from conventional RC driven inside a
PC (this is also physical dynamics, though) to PRC that
exploits physical dynamics as a reservoir. The story be-
gins from the genesis of LSM, which is one of the original
RC model systems.
IV.
LIQUID STATE MACHINE
A.
“Wetware” and its implication
When Wolfgang Maass, Thomas Natschl¨ager, and
Henry Markram proposed the seminal model of the LSM,
at around the same time, Wolfgang Maass presented
some interesting insights in his paper entitled “Wetware”
about the modality of information processing in the
brain [153]. This paper starts as follows:
“If you pour water over your PC, the PC will stop
working.
This is because very late in the history of
computing
which started about 500 million years ago
the PC and other devices for information processing
were developed that require a dry environment.
But
these new devices, consisting of hardware and software,
have a disadvantage: they do not work as well as the
older and more common computational devices that are
called nervous systems, or brains, and which consist of
wetware.
These superior computational devices were
made to function in a somewhat salty aqueous solution,
apparently because many of the ﬁrst creatures with a
nervous system were coming from the sea. We still carry
an echo of this history of computing in our heads: the
neurons in our brain are embedded into an artiﬁcial
sea-environment, the salty aqueous extracellular ﬂuid
which surrounds the neurons in our brain. ...”
Maass’s paper [153] subsequently discusses how to
capture the information processing function of the hu-
man brain. The idea expressed in the above introductory
paragraph already penetrates the fundamental aspect of
PRC.
The important point that we should conﬁrm here is
that once computation, which is an abstract inputout-
put operation in principle, was implemented in the real-
world through a physical entity or substrate, then the
physical property of the substrate and the inﬂuence
of its execution environment came to aﬀect the imple-
mented computation and inevitably added a novel prop-
erty/functionality to the system.
The above example
clearly suggests that even if the same computation is
implemented, according to the choice of physics for the
substrate (in the above case, the conventional PC and
brain), the robustness against water is diﬀerent.
The conventional PC consists of hardware and soft-
ware; the hardware is the “physical” part of the PC, and
the software is a set of commands used to run it. These
two components function complementarily. That is, the
hardware is specialized and designed to execute the com-
mand sent from the software. In contrast, the nervous
system can function in a somewhat salty aqueous solu-
tion, but this physical condition is not fully designed for
information processing. Rather, the nervous system ex-
ploits its given environmental constraints and physical
conditionswhich are shaped by its original context (i.e.,
many of the ﬁrst creatures with a nervous system came
from the sea)to enable information processing.
When we look at the background of the seminal model
of the LSM, it is evident that Wolfgang Maass and his
colleagues were not adopting a conventional view of the
brain as a network consisting of interacting elements (i.e.,
neurons) as many researchers do; instead, they char-
acterized its behavior based on the surrounding liquid
physical substrate. Furthermore, the idea is not merely
a metaphor; the researchers even proposed a concrete
sketch of their proposed model. This system is called the
“liquid computer.” [125]
B.
Liquid computer and the liquid brain
Thomas Natschl¨ager, Wolfgang Maass, and Henry
Markram suggested that the brain is constantly exposed
to a massive ﬂow of sensory information, including both
audio and visual inputs, and that it does not exist in
the stable state frequently expressed as an attractor but
rather in a transient state (except when it is in the “dead”
state) [125].
According to this view, they proposed a
scheme to exploit the surface of a liquid (such as a cup
of coﬀee) for computation.
We focus on a transformation over a time series. We
consider the issue by mapping from input sequences u(·),
which are a function of time, to output sequences v(·),
which are also a function of time; this transformation is
usually called a ﬁlter (or operator).
See Fig. 4A. The schematics show the conceptual de-
sign of a “liquid computer.” [125] To illustrate this con-
cept, one could imagine a situation where he or she pre-
pares a cup of coﬀee and perturbs the coﬀee surface by
7
A
B
C
FIG. 4. A Natschlager-Maass-Markram-type liquid computer and its analogy to neural information processing.
A. Schematics of a “Liquid computer.” Figure reprinted with permission from Ref. [125] by the author. B. The system takes a
video image of a liquid surface as a state of the system. The liquid surface shows diﬀerent spatiotemporal patterns according to
how it is perturbed (e.g., manual perturbations using a spoon or dropping a cube of sugar, and their temporal orderings make
the patterns of the liquid surface diﬀerent, such as “state 1” and “state 2”). Figure reprinted with permission from Ref. [125]
by the author. C. Understanding neural circuits as an LSM. Figure reprinted with permission from Ref. [125] by the author.
using a spoon or dropping a cube of sugar in, thereby in-
jecting an “input.” Consider that we have a video camera
that can monitor the coﬀee’s surface in real time and de-
ﬁne this camera image at time t as the liquid state x(t)
(Fig. 4B). The liquid (in this case, coﬀee) transforms the
input time series u(·) into a liquid state x(t), expressed
as x(t) = (Lu)(t), where L is called a liquid ﬁlter. The
image is sent to the PC, and by using the state of the
surface, the PC processes the state and outputs the re-
sult. The interesting point of this system is that one can
design various ﬁlters without using the memory storage
inside the PC; in other words, this process can be car-
ried out with the memory-less readout f, expressed as
v(t) = f(x(t)).
Let us consider an example of information processing
using this system. Assume that we want the system to
output the number of cubes of sugar injected over the
last two seconds. Because the readout part in the PC
is memory-less, to perform this task, the current liquid
state should be able to express the number of cubes of
sugar droped inside over the last two seconds in a distin-
guishable form. Let us call this ability to distinguish the
previous input state as a diﬀerence in the current liquid
state the “separation property” of the liquid. Then, to
perform the task, it is necessary to map the separated
states into the required output (e.g., the liquid states
perturbed in the order of “spoon →cube →cube” and
“cube →spoon →cube” should be mapped to output
“2”).
This property of the readout function is called
the “approximation property.” Interestingly, it has been
shown that any time invariant ﬁlter with fading memory
can be approximated in an arbitrary precision composing
these two properties (with a ﬁlter bank containing point-
wise separation property and readout function having
FIG. 5. A Fernando-Sojakka-type liquid brain. Figure
reprinted with permission from Ref. [126], Copyright (2003)
by Springer Nature.
universal approximation property) (see, Ref. [4, 154, 155]
for detailed discussions). In Ref. [125], it is stated that
the formalization of this liquid computer is an LSM and
is proposed to understand the information processing of
a neural circuit (Fig. 4C).
As soon as the concept of the liquid computer and
its formalization under an LSM were proposed, two
computer scientists, Chrisantha Fernando and Sampsa
Sojakka from the University of Sussex, integrated the
idea into a physical system; they called this model
the “liquid brain” (Fig. 5). In their paper [126], they
described it as follows:
“ ...
Here we have taken the metaphor seriously
and demonstrated that real water can be used as an LSM
8
for solving the XOR problem and ... ”
Using
this
system,
they
showed
that
water
in
a
bucket is capable of implementing an XOR task and a
speech recognition task [126]. We can conﬁrm that water
in a bucket is not made or designed for computation but
can be exploited for it. (Note that, recently in complex
systems study,
coginitive networks that lack stable
connections and static elements are also called liquid
brains [156].
These networks include such as ant and
termite colonies, immune systems, and slime moulds.)
V.
SOFT ROBOTICS
A computer is, in simple terms, a machine that is
made to compute. Accordingly, the hardware structure
of a computer is specialized to implement computation
in general.
Here, the concrete form of computation is
determined beforehand in a top-down manner, and to
realize it, the component arrangement is designed and
decided in detail. On this point, the liquid computer and
liquid brain are composed in somewhat opposite direc-
tions compared with the conventional computer. They
both started from the physical property of liquid, and
by considering how to exploit this property for computa-
tion, they came to invent a novel scheme to implement
it, which is a bottom-up approach.
A.
Embodiment and morphological computation
In robotics, a concept that accounts for these unex-
pected and intrinsic properties associated with the phys-
ical body when implementing computations, abstract op-
erations, or behavior control has been around for a long
time. This concept is called “embodiment.” [157–159]
For example, a seminal platform called a “passive dy-
namic walker” can walk naturally like a human with-
out having an external controller [160].
Just by using
a well-designed body (a compass-like shape) and a well-
designed environment (a slope), the natural walking be-
havior can be realized, where the behavior control is par-
tially outsourced to the physical body. In bio-inspired
robotics, this property of embodiment is studied in var-
ious platforms, including not only bipedal walkers, but
also quadruped robots (e.g., Ref.
[161–164]).
Similar
properties can be found in animals. There is a famous ex-
periment that used the dead body of ﬁsh (a trout, specif-
ically) where the body was able to generate a vivid and
natural swimming motion by exploiting the vortex in a
water tank [165]. In this experiment, because the ﬁsh
was dead, we can guarantee that the central nervous sys-
tem of the ﬁsh was not functioning at all, so we can also
conﬁrm that the speciﬁc morphology and material prop-
erty of the body and its interaction with the vortex in
the surrounding water environment were capable of real-
izing the natural swimming motion of a ﬁsh [165]. In the
ﬁeld of self-assembling systems, there are many studies
that investigate how the shape of each element induces or
aﬀects the global behavior of the system (e.g., Ref. [166–
169]). Here, the research ﬁeld that aims at investigating
and pursuing the nature of how the shape or morphology
of the system aﬀects the behavior of the entire system is
called morphological computation [170].
Are there any quantitative ways to characterize the
intrinsic information processing capability of the phys-
ical body?
Helmut Hauser et al.
tried to propose a
framework to theoretically investigate the morphological
computation of compliant bodies [109]. In their study,
they considered a mass-damper system, which is often
used to model the body of robots, and explained that
by using a linear mass-damper system, it is possible to
compose a ﬁlter bank, which we discussed earlier (Fig.
6A). This implies that if you design the readout func-
tion nicely, it is possible to approximate time-invariant
ﬁlters with a fading memory property using a linear mass-
damper system, which is consistent with the arguments
corresponding to the LSM model. Furthermore, the au-
thors numerically demonstrated that by using a complex
nonlinear mass-damper system, even the nonlinearity re-
quired in the readout function can be outsourced to the
mass-damper system, and the system would be capable
of emulating nonlinear ﬁlters with fading memory only
by composing the linear readouts. That is, this approach
suggests that the physical body of robots can be, in some
conditions, used to emulate nonlinear ﬁlters with a fad-
ing memory, which implies that the physical body can
be used as a successful reservoir. Subsequently, Helmut
Hauser et al. have investigated the role of feedback on the
mass-damper system implementing nonlinear limit cy-
cles based on this framework [110]. Tensegrity structures
serve as an appropriate testbed to implement this frame-
work, where it enables the structures to embed closed-
loop control and realize locomotion by exploiting its in-
trinsic body dynamics as a computational resource, here
being a controller [111, 116, 120] (Fig. 6B). (Note that,
although we do not go into details in this paper, infor-
mation theoretic approachs to characterize morphological
computation have also been investigated [172, 173].)
B.
Physical reservoir computing using a soft
robotic arm
Soft robotics is a recently developed ﬁeld that actively
investigates the account of soft and compliant bodies
to functionality and behavioral control [174–176]. Com-
pared with conventional rigid-bodied robots, soft robots
introduce a number of novel challenges into the ﬁeld re-
garding the material properties and deformable morphol-
ogy of the body as well as the complexity and diversity
of body dynamics. Soft robots hold many advantages,
which are linked to the mechanical softness of the body
[174–177]. For example, they are considered to be use-
ful in the situation of humanrobot interaction, rescue,
9
A
D
B
C
E
FIG. 6. Physical reservoir computing using compliant and soft bodies. A. A generic mass-spring network used as a
reservoir in Ref. [109]. Figure reprinted from Ref. [109] under the Creative Commons CC-BY-NC license. B. A tensegrity
robot called SUPERball proposed in Ref. [171]. Figure reprinted with permission from Ref. [171], Copyright (2015) by IEEE.
C. A quadruped robot called Kitty proposed in Ref. [114], which exploits soft spine dynamics as a reservoir. Figure reprinted
with permission from Ref. [114], Copyright (2013) by IEEE. D. A picture of a physical soft robotic arm inspired by the octopus
used in the experiment in Ref. [117]. It is made of silicone and embeds ten bending sensors, monitoring the soft body dynamics
every 0.03 [s]. Figure reprinted from Ref. [117] under the Creative Commons license. E. Schematics explaining how to exploit
the soft robotic arm as a reservoir. Figure reprinted from Ref. [117] under the Creative Commons license.
and biomedical applications because they do not dam-
age people in the same way that rigid robots do; in
other words, they are generally considered a safer op-
tion. These robots, however, include challenges in terms
of control [177]. Soft robots are often classiﬁed into the
category of an underactuated system, where the num-
ber of the actuation points are less than the degrees of
freedom. Furthermore, they usually generate diverse and
complex body dynamics when actuated, which are high-
dimensional, nonlinear, and contain short-term memory
[178–180]. These properties make soft robots diﬃcult to
control using the conventional control scheme.
On the other hand, these seemingly undesirable prop-
erties of soft robot control can be viewed as a positive
from PRC perspectives. That is, we can exploit the di-
verse, rich dynamics of a soft body as a computational
resourcemore speciﬁcally, as a reservoir (Fig. 6C, D, and
E). In previous studies, we have shown that a silicone-
based soft robotic arm inspired by an octopus can be used
as a successful reservoir by taking the actuation sequence
as the input and sensory reading as the reservoir state;
indeed, this method exhibits high information-processing
capability in some conditions [115, 117, 122, 123] (Fig.
6D). Interestingly, octopus arms have characteristic mus-
cle organizations termed muscular-hydrostats [181]. In
these structures, the volume of the organ remains con-
stant during their motion, enabling diverse and complex
behaviors. We showed that using biologically plausible
parameter settings, the dynamic model of the muscular-
hydrostat system has the computational capacity to
achieve a complex nonlinear computation [112, 113, 118].
Furthermore, by incorporating the feedback-loop from
the output to the next input (i.e., the next actuation
pattern), we have demonstrated that the robot’s behav-
ioral control for the next time step can be implemented
by using its current state of its body as a computational
10
resource, suggesting that the “controller” and “to be con-
trolled” is the same in this scheme [115]. This concept
has been also applied to the study of a quadruped robot,
where the robot exploits its spine dynamics as a physi-
cal reservoir to control its actuation patterns and loco-
motions [114] (Fig. 6C). In short, the drawbacks of soft
robot control became assets for control from a PRC view-
point.
VI.
EXPLOITING PHYSICAL DYNAMICS FOR
COMPUTATIONAL PURPOSES
We began by reviewing the concept of wetware by
Wolfgang Maass, and from there, we illustrated the de-
velopment of physical platforms, such as the liquid com-
puter, liquid brain, mass-damper systems, and silicone-
based soft robotic arms inspired by octopuses. In this
section, we would like to review three signiﬁcant phases
that we can ﬁnd in this evolution.
Phase 0: Inferring the computational power of
physical systems.
PRC provides a method to exploit natural physical dy-
namics as a computational device. It implies that this
method is also useful for investigating which physical sys-
tems are suitable to implement which types of computa-
tion and for analyzing the information-processing capa-
bility of the physical dynamics. In particular, if we use
linear and static readouts to generate outputs for spe-
ciﬁc tasks requiring a certain amount of nonlinearity and
memory, because we are not adding any nonlinear terms
and memory externally, by evaluating the task perfor-
mance, we can infer back which amount of nonlinearity
and memory has been positively contributed or exploited
from the physical reservoir to perform the task (Fig. 7A).
That is, in this way, if we use a previously introduced
symbol, we can pursue the nature of the function φ in the
physical systems. Systematic investigations are needed to
reveal the response characteristics of the physical system
against the type, intensity, and timescale of the input,
and these properties are intrinsic to each physical system.
Accordingly, we can expect the diversity of the type of
information processing according to the type of physics,
where each physical system has a preference in terms of
the type of functions it can express.
In neuroscience, there are several studies that have in-
ferred the computational capability of the neural circuits
[129] or the cultured neural systems [130, 131].
Obvi-
ously, their motivation is not to make a high-performance
computer but rather to reveal the functional characteris-
tics of the natural systems from information-processing
perspectives. This approach can also be applied to infer
the functionality of the body of living systems quantita-
tively. As we discussed in the concept of embodiment, a
functionality that is thought to be handled by the brain
is often partially outsourced to the physical body. Un-
like the randomly coupled ESN, the biological body has
a speciﬁc structure or morphology that is intrinsic to re-
spective living organisms.
This speciﬁc morphology is
evolved through the respective ecological niche of living
things, which is a driving force of the diversity of mor-
phology. It is expected that the PRC framework has the
potential to reveal the property of the body’s morphol-
ogy from information-processing perspectives. (Related
to this issue, there exists a research project that aims
to characterize RC from evolutionary perspectives [182].)
The above directions of research can be summarized and
stated as the study of φ within the physical system. This
penetration is the basics and is fundamentally important
in the PRC framework and can thus be taken as a ground
basis, which we call phase 0.
We should note, however, that once the function φ
of the physical system is revealed, then because it is a
mathematical description in principle, there is no mean-
ing to use the actual physical system as an information-
processing device anymore, but we can implement the
same functionality of the physical system using a con-
ventional PC. If we only stick to this perspective, then
PRC does not diﬀer so much from the original RC any-
more. Shortly, the diversity we can ﬁnd here is in fact the
diversity of function φ. Now, much like as we overviewed
from the examples starting from wetwares, PRC has the
potential to go beyond this perspective. That is, the PRC
framework can deal with a property that is not described
in φ. This point is elaborated subsequently in phase 1
and phase 2.
Phase 1: Physical properties of a computer.
A computer is a machine that is designed for com-
putation. As long as it is made of a physical entity, it
inevitably and, sometimes unexpectedly, adds physical
and material properties to the system that are not always
directly connected to the computational purpose (these
properties can present as both advantages and disadvan-
tages for the user). We have clearly conﬁrmed this point
using the example of wetware, and this constraint is also
true for PRC. Even if you implement the same computa-
tion, depending on the type of physics you exploit, you
may gain additional or unexpected properties beyond the
computation itself (Fig. 7B). For example, if you use a
laser as a computational resource, you can implement
an extremely fast computation, or if you use water as a
substrate, the system will be tolerant of water (Fig. 7B).
Many currently discussed assets of physical reservoirs can
be understood from this perspective. In particular, spin-
tronics devices have been gaining attention as an appro-
priate substrate for PRC because of their compactness,
high-speed processing, and energy eﬃciency while being
able to function at normal temperatures [73–76, 78–85].
These assets are somewhat common in the computer sci-
ence ﬁeld, but spintronics devices also contain an inter-
11
A
B
C
FIG. 7. Three phases in PRC. A. Phase 0. PRC can be used as a method to infer the information processing capability of
natural physical dynamics. B. Phase 1. Physical properties, which are potentially diﬀerent according to the type of physics,
are added to the reservoir in PRC. C. Phase 2. PRC enables to exploit physical dynamics as a computational resource that is
already functioning for diﬀerent purposes. See the text for details.
esting additional property: they show high durability in
radioactive environments [183] (Fig. 7B). This property
opens up the potential for spintronics reservoirs to be
used as a computational substrate in extreme environ-
ments where conventional electronic devices break down
or do not function at all. Another example can be found
in quantum reservoir computing.
Since the ﬁrst con-
ception to exploit quantum dynamics as a reservoir in
Ref. [86], there have been many variants and extensions
proposed in the literature [87–91]. In quantum reservoir
computing, by using the property of quantum computa-
tional supremacy, a huge amount of computational nodes
can be equipped, which then provide a direct inﬂuence
to the information-processing capability of the system
[86, 87].
Another important property is that because
quantum reservoir computing exploits quantum dynam-
ics, it is capable of implementing a quantum task (a task
deﬁned in the quantum scale) (Fig. 7B). In Ref. [89], the
preparation of desired quantum states, such as single-
photon states, Schr¨odinger’s cat states, and two-mode
entangled states, is introduced as an eﬀective application
domain.
To induce these assets, which originate from the phys-
ical properties of a reservoir, current technologies still re-
quire conventional electronics and external devices, such
as for the readout part, to maintain the temperature
during the reservoir executions and to make the phys-
ical reservoir work in the real environment.
This is a
weakness of currently available technologies; these points
should be improved, and a novel scheme should be pro-
posed in the future.
Phase 2: Exploiting a physical substrate that is not
made for computation for computation.
If we think of the body of a robot, it is, of course,
not made for computation.
The body is an essential
constituent of a robot and is inevitably associated when
generating behaviors. That is, the robot’s intended func-
tionality is to realize behaviors in the real world. As we
have seen in the example of soft robots, if the body itself
exerts certain dynamic conditions, then according to the
PRC framework, the body can also be used as a com-
putational resource (Fig. 7C). This implies that the two
functionalities“behavioral generation” and “information
processing”are associated with the same physical body.
Then, when the robot generates behavior, we can simul-
taneously use its dynamics for information processing.
Considering this property, as we conﬁrmed in the above
examples of soft robotic arms, together with an incorpo-
ration of the feedback-loop, the resulting body dynamics
of the target behavior can be exploited to calculate the
target motor command that controls its own behavior. It
shows that this approach is more eﬃcient than any other
controller attached externally to realize target robotic
12
FIG. 8. Schematics of the closed-loop control in phase
2 of PRC. In conventional control, information processing
is prepared outside a system that is to be controlled or acted
on (left diagram). In phase 2, information processing is ac-
companied by the behavior of the system (right diagram).
The system behaves in a certain manner and performs in-
formation processing simultaneously. Note that the required
information processing to generate behavioral control can be
bypassed from the digital processor and embedded in the sys-
tem itself.
behavior (Fig. 8).
Phase 2 may be classiﬁed as a derivative of Phase 1,
but the major turn from Phase 1 to Phase 2 is that in
the latter, the physical substrate is not prepared for com-
putational purposes whatsoever in the ﬁrst place. The
most interesting point of PRC among other computa-
tional frameworks can be found here.
PRC can easily
generate the transition from Phase 1 to Phase 2. This
is because the RC framework allows one to exploit the
natural dynamics of physical systems for information pro-
cessing. Accordingly, in PRC, we do not need to precisely
design the physical substrate speciﬁc to target computa-
tion in many cases; rather, the implemented information
processing depends on the input-driven dynamics of the
physical substrate, which results in a diversity of infor-
mation processing.
Then, which kind of physical reservoirs is classiﬁed in
Phase 2 other than a soft robotic arm inspired by an octo-
pus? This question is a fundamental theme that should
be further explored in the ﬁeld of PRC. One direction
would be to exploit real living things, such as animals
(e.g., rats, ﬁsh, etc.) or the human brain, as a physi-
cal reservoir. In principle, living things are free from the
intended purposes introduced by users. Needless to say,
they are not made for computational purposes. Recently,
there have been several studies suggesting that the brain
wave of animals and humans exhibit consistent responses
against external inputs, and it is expected that the PRC
approach can be directly applied to brain waves (e.g.,
Ref.
[184]).
This direction of research has long been
studied in the ﬁeld of brain-machine-interface. Together
with the recent advancement of sensing technology that
allows us to monitor massive amounts of data from liv-
ing things (e.g., Ref. [185]), the PRC approach presents
a high potential for further study of the issue, and it can
be actively applied not only to our daily devices, such
as smart phones, but also to wearables and biomedical
devices.
VII.
ACKNOWLEDGEMENTS
K. N. would like to acknowledge Taichi Haruna, Kat-
sushi Kagaya, Megumi Akai-Kasaya, Atsushi Uchida,
Kazutaka Kanno, Sumito Tsunegi, Quoc Hoan Tran, and
Yongping Pan for their fruitful discussions and thought-
ful suggestions.
This work was based on results ob-
tained from a project commissioned by the New En-
ergy and Industrial Technology Development Organiza-
tion (NEDO). K. N. was supported by JSPS KAKENHI
Grant Numbers JP18H05472 and by MEXT Quantum
Leap Flagship Program (MEXT Q-LEAP) Grant Num-
ber JPMXS0118067394.
[1] H. Jaeger, The “echo state” approach to analysing
and training recurrent neural networks-with an erratum
note. German National Research Center for Information
Technology GMD Technical Report 148 (34), 13 (2001).
[2] H. Jaeger, Tutorial on training recurrent neural net-
works, covering BPPT, RTRL, EKF and the “echo
state network” approach (Vol. 5, p. 01). Bonn: GMD-
Forschungszentrum Informationstechnik (2002).
[3] H. Jaeger, H. Haas, Harnessing nonlinearity: Predicting
chaotic systems and saving energy in wireless commu-
nication. Science, 304 (5667), p.78-80 (2004).
[4] W. Maass, T. Natschl¨ager, H. Markram, Real-time com-
puting without stable states: A new framework for neu-
ral computation based on perturbations. Neural compu-
tation 14 (11), p.2531-2560 (2002).
[5] D.
Verstraeten,
B.
Schrauwen,
M.
d’Haene,
D.
Stroobandt, An experimental uniﬁcation of reservoir
computing methods. Neural Networks 20 (3), p.391-403
(2007).
[6] B. Schrauwen, D. Verstraeten, J. Van Campenhout, An
overview of reservoir computing: theory, applications
and implementations. Proc. of the 15th european sym-
posium on artiﬁcial neural networks. p. 471-482 (2007).
[7] M. Lukoˇseviˇcius, H. Jaeger, Reservoir computing ap-
proaches to recurrent neural network training. Com-
puter Science Review 3 (3), p.127-149 (2009).
[8] M. Lukoˇseviˇcius, H. Jaeger, B. Schrauwen, Reservoir
computing trends. KI-K¨unstliche Intelligenz, 26(4), 365-
371 (2012).
13
[9] P. J. Werbos, Backpropagation through time: what it
does and how to do it, Proceedings of the IEEE 78, 1550
(1990).
[10] Y. Bengio, P. Simard, P. Frasconi, Learning long-term
dependencies with gradient descent is diﬃcult, IEEE
transactions on neural networks 5, 157 (1994).
[11] M. H. Tong, A. D. Bickett, E. M. Christiansen, G.
W. Cottrell, Learning grammatical structure with echo
state networks. Neural networks 20(3), 424-432 (2007).
[12] M. D. Skowronski, J. G. Harris, Automatic speech
recognition using a predictive echo state network classi-
ﬁer. Neural networks 20(3), 414-423 (2007).
[13] N. Schaetti, M. Salomon, R. Couturier, Echo state
networks-based reservoir computing for mnist handwrit-
ten digits recognition. Proc. of 2016 IEEE Intl Confer-
ence on Computational Science and Engineering (CSE)
and IEEE Intl Conference on Embedded and Ubiqui-
tous Computing (EUC) and 15th Intl Symposium on
Distributed Computing and Applications for Business
Engineering (DCABES), pp. 484-491 (2016).
[14] I. Ilies, H. Jaeger, O. Kosuchinas, M. Rincon, V. Sake-
nas, N. Vaskevicius, Stepping forward through echoes of
the past: forecasting with echo state networks. Proc. of
2006/07 Forecasting Competition for Neural Networks
& Computational Intelligence (NN3), pp.1-4 (2007).
[15] X. Lin, Z. Yang, Y. Song, Short-term stock price pre-
diction based on echo state networks. Expert systems
with applications 36(3), 7313-7317 (2009).
[16] J. Pathak, B. Hunt, M. Girvan, Z. Lu, E. Ott, Model-
free prediction of large spatiotemporally chaotic systems
from data: A reservoir computing approach. Phys. Rev.
Lett. 120, 024102 (2018).
[17] M. Inada, Y. Tanaka, H. Tamukoh, K. Tateno, T. Morie,
Y. Katori, Prediction of Sensory Information and Gen-
eration of Motor Commands for Autonomous Mobile
Robots Using Reservoir Computing, Proc. of the 2019
Int. Symp. on Nonlinear Theory and its Applications
(NOLTA2019), pp. 333-336 (2019).
[18] M. Salmen, P. G. Ploger, Echo state networks used for
motor control. Proc. of the 2005 IEEE international con-
ference on robotics and automation (ICRA), pp. 1953-
1958 (2005).
[19] T. Li, K. Nakajima, M. Calisti, C. Laschi, R. Pfeifer,
Octopus-Inspired Sensorimotor Control of a Multi-Arm
Soft Robot. Proc. Int. Conf. on Mechatronics and Au-
tomation (ICMA), p.948-955 (2012).
[20] T. Li, K. Nakajima, M. Cianchetti, C. Laschi, R. Pfeifer,
Behavior Switching by Using Reservoir Computing for
a Soft Robotic Arm. Proc. IEEE Int. Conf. on Robotics
and Automation (ICRA), p.4918-4924 (2012).
[21] T. Li, K. Nakajima, R. Pfeifer, Online Learning Tech-
nique for Behavior Switching in a Soft Robotic Arm.
Proc. IEEE Int. Conf. on Robotics and Automation
(ICRA), p.1288-1294 (2013).
[22] J. Kuwabara, K. Nakajima, R. Kang, D. T. Branson, E.
Guglielmino, D. G. Caldwell, R. Pfeifer, Timing-Based
Control via Echo State Network for Soft Robotic Arm.
Proc. Int. Joint Conf. on Neural Networks (IJCNN),
p.1-8 (2012).
[23] C. Hartmann, J. Boedecker, O. Obst, S. Ikemoto, M.
Asada, Real-Time Inverse Dynamics Learning for Mus-
culoskeletal Robots based on Echo State Gaussian Pro-
cess Regression, Proc. of Robotics: Science and Systems
VIII, p15, 2012 [10.15607/RSS.2012.VIII.015].
[24] F. Wyﬀels, B. Schrauwen, Design of a Central Pattern
Generator Using Reservoir Computing for Learning Hu-
man Motion. Proc. of the 2009 Advanced Technologies
for Enhanced Quality of Life, pp. 118-122 (2009).
[25] W. Shi, J. Cao, Q. Zhang, Y. Li, L. Xu, Edge com-
puting: Vision and challenges. IEEE Internet of Things
Journal 3, 637 (2016).
[26] J. Fonollosa, S. Sheik, R. Huerta, S. Marco, Reservoir
computing compensates slow response of chemosensor
arrays exposed to fast varying gas concentrations in con-
tinuous monitoring. Sensors and Actuators B: Chemical,
215, 618-629 (2015).
[27] R. Sakurai, M. Nishida, H. Sakurai, Y. Wakao, N.
Akashi, Y. Kuniyoshi, Y. Minami, K. Nakajima, Em-
ulating a sensor using soft material dynamics: A reser-
voir computing approach to pneumatic artiﬁcial muscle,
Proc. IEEE Int. Conf. on Soft Robotics (RoboSoft) 2020
(in press).
[28] J. C. Principe, B. Chen, Universal approximation with
convex optimization: gimmick or reality? IEEE Com-
putational Intelligence Magazine 10 (2), p.68-77 (2015).
[29] L. Grigoryeva, J.-P. Ortega, Echo state networks are
universal, Neural Networks 108, p. 495508 (2018).
[30] L. Gonon, J.-P. Ortega, Reservoir Computing Univer-
sality With Stochastic Inputs, IEEE Transactions on
Neural Networks and Learning Systems 31(1), p.100-
112 (2020).
[31] S. Hochreiter, J. Schmidhuber, Long short-term mem-
ory. Neural Comput. 9, 17351780 (1997).
[32] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau,
F. Bougares, H. Schwenk, Y. Bengio, Learning phrase
representations using rnn encoderdecoder for statisti-
cal machine translation, Proc. of the 2014 Conference
on Empirical Methods in Natural Language Processing
(EMNLP), Association for Computational Linguistics.
pp. 17241734 (2014).
[33] M. Arjovsky, A. Shah, Y. Bengio, Unitary evolution
recurrent neural networks, Proc. of the 33rd Interna-
tional Conference on International Conference on Ma-
chine Learning - Volume 48, JMLR.org. pp. 11201128
(2016).
[34] L. Jing, Y. Shen, T. Dubcek, J. Peurifoy, S. A. Skirlo,
Y. LeCun, M. Tegmark, M. Soljacic, Tunable eﬃcient
unitary neural networks (EUNN) and their application
to rnns, ICML, PMLR. pp. 17331741 (2017).
[35] P. R. Vlachas, J. Pathak, B. R. Hunt, T. P. Sapsis,
M. Girvan, E. Ott, P. Koumoutsakos, Forecasting of
spatio-temporal chaotic dynamics with recurrent neu-
ral networks: A comparative study of reservoir com-
puting and backpropagation algorithms. arXiv preprint
arXiv:1910.05266.
[36] M. McCloskey, N. J Cohen. Catastrophic interference in
connectionist networks: The sequential learning prob-
lem. Psychology of Learning and Motivation, 24:109165,
1989.
[37] R. Ratcliﬀ. Connectionist models of recognition mem-
ory:
Constraints imposed by learning and forgetting
functions. Psychological Review, 97(2):285308, 1990.
[38] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness,
G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ra-
malho, A. Grabska-Barwinska, D. Hassabis, C. Clopath,
D. Kumaran, R. Hadsell, Overcoming catastrophic for-
getting in neural networks, PNAS 114 (13), p.3521-3526
(2017).
14
[39] S. W. Lee, J. H. Kim, J. Jun, J. W. Ha, B. T. Zhang,
Overcoming catastrophic forgetting by incremental mo-
ment matching. In Advances in neural information pro-
cessing systems, pp. 4652-4662 (2017).
[40] X. He, H. Jaeger, Overcoming Catastrophic Interference
by Conceptors. Jacobs University Technical Report Nr
35 (2017).
[41] R. Toral, C. R. Mirasso, E. Hernandez-Garcia, O. Piro,
Analytical and numerical studies of noise-induced syn-
chronization of chaotic systems, CHAOS 11, 665 (2001).
[42] I. B. Yildiz, H. Jaeger, S. J. Kiebel, Re-visiting the echo
state property. Neural networks 35, 1-9 (2012).
[43] G. Manjunath, H. Jaeger, Echo state property linked
to an input: Exploring a fundamental characteristic of
recurrent neural networks. Neural computation 25(3),
671-696 (2013).
[44] M.
Komatsu,
T.
Yaguchi,
K.
Nakajima,
Alge-
braic
approach
towards
the
exploitation
of
“soft-
ness”:
the
input-output
equation
for
morpho-
logical
computation.
The
International
Journal
of
Robotics
Research,
0278364920912298
(2020)
[doi.org/10.1177/0278364920912298].
[45] Z. Lu, B. R. Hunt, E. Ott, Attractor reconstruction by
machine learning. CHAOS 28, 061104 (2018).
[46] H. Jaeger, Short term memory in echo state networks.
GMD Report 152, German National Research Center
for Information Technology (2001).
[47] O. L. White, D. D. Lee, H. Sompolinsky, Short-term
memory in orthogonal neural networks. Phys. Rev. Lett.
92, 148102 (2004).
[48] S. Ganguli, D. Huh, H. Sompolinsky, Memory traces
in dynamical systems. Proc. Natl. Acad. Sci. USA 105,
1897018975 (2008).
[49] M. Hermans, B. Schrauwen, Memory in linear recurrent
neural networks in continuous time. Neural Netw. 23,
341355 (2010).
[50] A. Rodan, P. Tino, Minimum complexity echo state net-
work. IEEE transactions on neural networks 22(1), 131-
144 (2010).
[51] A. Goudarzi, D. Stefanovic, Towards a calculus of echo
state networks, Procedia Computer Science 41, 176181
(2014).
[52] T. Toyoizumi, L. F. Abbott, Beyond the edge of chaos:
Ampliﬁcation and temporal integration by recurrent
networks in the chaotic regime. Phys. Rev. E 84, 051908
(2010).
[53] J. Schuecker, S. Goedeke, M. Helias, Optimal sequence
memory in driven random networks. Phys. Rev. X 8,
041029 (2018).
[54] T. Haruna and K. Nakajima, Optimal short-term mem-
ory before the edge of chaos in driven random recurrent
networks, Phys. Rev. E 100 (6), 062312 (2019).
[55] J. Dambre, D. Verstraeten, B. Schrauwen, S. Massar,
Information processing capacity of dynamical systems.
Sci. Rep. 2, 514 (2012).
[56] M. Inubushi, K. Yoshimura, Reservoir computing be-
yond memory-nonlinearity trade-oﬀ. Sci. Rep. 7(1), 1-10
(2017).
[57] M. Dale, J. F. Miller, S. Stepney, M. A. Trefzer, A
substrate-independent framework to characterize reser-
voir computers. Proceedings of the Royal Society A
475(2226), 20180723 (2019).
[58] N. Bertschinger, T. Natschl¨ager, Real-time computation
at the edge of chaos in recurrent neural networks. Neural
computation 16(7), 1413-1436 (2004).
[59] R. Legenstein, W. Maass, Edge of chaos and prediction
of computational performance for neural circuit models.
Neural networks 20(3), 323-334 (2007).
[60] J. Boedecker, O. Obst, J. T. Lizier, N. M. Mayer, M.
Asada, Information processing in echo state networks at
the edge of chaos. Theory in Biosciences 131(3), 205-213
(2012).
[61] G. Wainrib, M. N. Galtier, A local echo state property
through the largest Lyapunov exponent. Neural Net-
works 76, 39-45 (2016).
[62] K. Vandoorne, W. Dierckx, B. Schrauwen, D. Ver-
straeten, R. Baets, P. Bienstman, J. Van Campenhout,
Toward optical signal processing using photonic reser-
voir computing. Optics express, 16(15), 11182-11192
(2008).
[63] L. Larger, M. C. Soriano, D. Brunner, L. Appeltant,
J. M. Gutierrez, L. Pesquera, C. R. Mirasso, I. Fis-
cher, Photonic information processing beyond Turing:
an optoelectronic implementation of reservoir comput-
ing.. Optics Express 20, 3241 (2012).
[64] Y. Paquot, F. Duport, A. Smerieri, J. Dambre, B.
Schrauwen, M. Haelterman, S. Massar, Optoelectronic
reservoir computing. Sci. Rep. 2, 287 (2012).
[65] F. Duport, B. Schneider, A. Smerieri, A. M. Haelter-
man, S. Masser, All-optical reservoir computing, Opt.
Express 20 (20), pp.22783-22795 (2012).
[66] D. Brunner, M. C. Soriano, C. R. Mirasso, I. Fischer,
Parallel photonic information processing at gigabyte per
second data rates using transient states. Nature commu-
nications 4(1), 1-7 (2013).
[67] K. Vandoorne, P. Mechet, T. Van Vaerenbergh, M.
Fiers, G. Morthier, D. Verstraeten, B. Schrauwen, J.
Dambre, P. Bienstman, Experimental demonstration of
reservoir computing on a silicon photonics chip, Nature
Communications 5, 3541 (2014).
[68] J. Nakayama, K. Kanno, and A. Uchida, Laser dynam-
ical reservoir computing with consistency: an approach
of a chaos mask signal, Optics Express, Vol. 24, No. 8,
pp. 8679-8692 (2016).
[69] L. Larger, A. Baylon-Fuentes, R. Martinenghi, V. S.
Udaltsov, Y. K. Chembo, M. Jacquot, High-speed
photonic reservoir computing using a time-delay-based
architecture:
Million words per second classiﬁcation.
Phys. Rev. X 7, 011015 (2017).
[70] G. Van der Sande, D. Brunner, M. C. Soriano, Advances
in photonic reservoir computing. Nanophotonics, 6(3),
561-576 (2017).
[71] J. Bueno, S. Maktoobi, L. Froehly, I. Fischer, M.
Jacquot, L. Larger, D. Brunner, Reinforcement learn-
ing in a large-scale photonic recurrent neural network,
Optica 5(6), pp. 756-760 (2018).
[72] K. Takano, C. Sugano, M. Inubushi, K. Yoshimura, S.
Sunada, K. Kanno, A. Uchida, Compact reservoir com-
puting with photonic integrated circuit, Opt. Express
26(22), pp. 29424-29439 (2018).
[73] J. Torrejon,
M. Riou,
F. A. Araujo,
S. Tsunegi,
G. Khalsa, D. Querlioz, P. Bortolotti, V. Cros, K.
Yakushiji, A. Fukushima, H. Kubota, S. Yuasa, M. D.
Stiles, and J. Grollier, Neuromorphic computing with
nanoscale spintronic oscillators. Nature 547, p.428-431
(2017).
[74] T. Furuta, K. Fujii, K. Nakajima, S. Tsunegi, H. Kub-
ota, Y. Suzuki, S. Miwa, Macromagnetic simulation for
15
reservoir computing utilizing spin dynamics in magnetic
tunnel junctions. Phys. Rev. Appl. 10, 034063 (2018).
[75] S. Tsunegi, T. Taniguchi, S. Miwa, K. Nakajima, K.
Yakushiji, A. Fukushima, S. Yuasa, H. Kubota, Evalua-
tion of memory capacity of spin torque oscillator for re-
current neural networks. Jpn. J. Appl. Phys. 57, 120307
(2018).
[76] R. Nakane, G. Tanaka, A. Hirose, Reservoir Comput-
ing With Spin Waves Excited in a Garnet Film, IEEE
ACCESS, vol. 6, pp. 4462-4469 (2018).
[77] D. Prychynenko, M. Sitte, K. Litzius, B. Kr¨uger, G.
Bourianoﬀ, M. Kl¨aui, J. Sinova, K. Everschor-Sitte,
Magnetic skyrmion as a nonlinear resistive element: a
potential building block for reservoir computing. Phys-
ical Review Applied 9(1), 014034 (2018).
[78] S. Tsunegi, T. Taniguchi, K. Nakajima, S. Miwa, K.
Yakushiji, A. Fukushima, S. Yuasa, H. Kubota, Physical
reservoir computing based on spin torque oscillator with
forced synchronization. Appl. Phys. Lett. 114, 164101
(2019).
[79] D. Markovi, N. Leroux, M. Riou, F. Abreu Araujo,
J. Torrejon, D. Querlioz, A. Fukushima, S. Yuasa, J.
Trastoy, P. Bortolotti, J. Grollier, Reservoir computing
with the frequency, phase, and amplitude of spin-torque
nano-oscillators, Appl. Phys. Lett. 114, 012409 (2019).
[80] H. Nomura, T. Furuta, K. Tsujimoto, Y. Kuwabiraki, F.
Peper, E. Tamura, S. Miwa, M. Goto, R. Nakatani, Y.
Suzuki, Reservoir computing with dipole-coupled nano-
magnets, Jpn. J. Appl. Phys. 58, 070901 (2019).
[81] T. Kanao,
H. Suto,
K. Mizushima,
H. Goto,
T.
Tanamoto, T. Nagasawa, Reservoir Computing on Spin-
Torque Oscillator Array, Phys. Rev. Applied 12, 024052
(2019)
[82] M. Riou, J. Torrejon, B. Garitaine, F. Abreu Araujo,
P. Bortolotti, V. Cros, S. Tsunegi, K. Yakushiji, A.
Fukushima, H. Kubota, S. Yuasa, D. Querlioz, M. D.
Stiles, J. Grollier, Temporal Pattern Recognition with
Delayed-Feedback Spin-Torque Nano-Oscillators, Phys.
Rev. Applied 12, 024049 (2019).
[83] W. Jiang, L. Chen, K. Zhou, L. Li, Q. Fu, Y. Du, R.
H. Liu, Physical reservoir computing using magnetic
skyrmion memristor and spin torque nano-oscillator,
Appl. Phys. Lett. 115, 192403 (2019).
[84] H. Nomura, K. Tsujimoto, M. Goto, N. Samura, R.
Nakatani, Y. Suzuki, Reservoir computing with two-bit
input task using dipole-coupled nanomagnet array, Jpn.
J. Appl. Phys. 59, SEEG02 (2019)
[85] F. A. Araujo, M. Riou, J. Torrejon, S. Tsunegi, D. Quer-
lioz, K. Yakushiji, A. Fukushima, H. Kubota, S. Yuasa,
M. D. Stiles, J. Grollier, Role of non-linear data pro-
cessing on speech recognition task in the framework of
reservoir computing, Sci. Rep. 10, 328 (2020).
[86] K. Fujii, K. Nakajima, Harnessing disordered-ensemble
quantum dynamics for machine learning. Phys. Rev.
Appl. 8, 024030 (2017).
[87] K. Nakajima, K. Fujii, M. Negoro, K. Mitarai, and M.
Kitagawa, Boosting computational power through spa-
tial multiplexing in quantum reservoir computing. Phys.
Rev. Appl. 11, 034021 (2019).
[88] S. Ghosh, A. Opala, M. Matuszewski, T. Paterek, T.
C. Liew, Quantum reservoir processing. npj Quantum
Information, 5(1), 1-6 (2019).
[89] S. Ghosh, T. Paterek, T. C. Liew, Quantum Neuromor-
phic Platform for Quantum State Preparation. Phys.
Rev. Lett., 123(26), 260404 (2019).
[90] S. Ghosh, T. Krisnanda, T. Paterek, T. C. Liew, Uni-
versal quantum reservoir computing. arXiv preprint
arXiv:2003.09569.
[91] J. Chen, H. I. Nurdin, N. Yamamoto, Temporal infor-
mation processing on noisy quantum computers. arXiv
preprint arXiv:2001.09498.
[92] A. Z. Stieg, A. V. Avizienis, H. O. Sillin, C. MartinOl-
mos, M. Aono, J. K. Gimzewski, Emergent criticality
in complex turing Btype atomic switch networks. Adv.
Mater. 24, 286 (2012).
[93] H. O. Sillin, R. Aguilera, H. H. Shieh, A. V. Avizie-
nis, M. Aono, A. Z. Stieg, J. K. Gimzewski, A theo-
retical and experimental study of neuromorphic atomic
switch networks for reservoir computing. Nanotechnol-
ogy 24(38), 384004 (2013).
[94] M. Dale, J. F. Miller, S. Stepney, M. A. Trefzer, Evolv-
ing carbon nanotube reservoir computers. Proc. of In-
ternational Conference on Unconventional Computation
and Natural Computation, pp. 49-61 (2016).
[95] M. Dale, J. F. Miller, S. Stepney, Reservoir computing
as a model for in-materio computing. In Advances in
Unconventional Computing, pp. 533-571 (2017).
[96] C. Du, F. Cai, M. A. Zidan, W. Ma, S. H. Lee, W.
D. Lu, Reservoir computing using dynamic memristors
for temporal information processing. Nature communi-
cations 8(1), 2204 (2017).
[97] K. S. Scharnhorst, J. P. Carbajal, R. C. Aguilera, E.
J. Sandouk, M. Aono, A. Z. Stieg, J. K. Gimzewski,
Atomic switch networks as complex adaptive systems.
Japanese Journal of Applied Physics, 57(3S2), 03ED02
(2018).
[98] H. Tanaka, M. Akai-Kasaya, A. TermehYouseﬁ, L.
Hong, L. Fu, H. Tamukoh, D. Tanaka, T. Asai, T.
Ogawa, A molecular neuromorphic network device con-
sisting of single-walled carbon nanotubes complexed
with polyoxometalate, Nature Communications 9, 2693
(2018).
[99] J. Moon, W. Ma, J. H. Shin, F. Cai, C. Du, S. H. Lee,
W. D. Lu, Temporal data classiﬁcation and forecasting
using a memristor-based reservoir computing system.
Nature Electronics 2(10), 480-487 (2019).
[100] R. Midya, Z. Wang, S. Asapu, X. Zhang, M. Rao, W.
Song, Y. Zhuo, N. Upadhyay, Q. Xia, J. Joshua Yang,
Reservoir computing using diﬀusive memristors. Ad-
vanced Intelligent Systems 1(7), 1900084 (2019).
[101] L. Appeltant, M. C. Soriano, G. Van der Sande, J.
Danckaert, S. Massar, J. Dambre, B. Schrauwen, C. R.
Mirasso, I. Fischer, Information processing using a sin-
gle dynamical node as complex system. Nature Commu-
nications 2 , 468 (2011).
[102] L. Appeltant, G. Van der Sande, J. Danckaert, I. Fis-
cher, Constructing optimized binary masks for reservoir
computing with delay systems. Sci. Rep. 4 , 3629 (2014).
[103] M. L. Alomar, V. Canals, V. Mart´ınez-Moll, J. L.
Rossell´o, Low-cost hardware implementation of reser-
voir computers. Proc. of 2014 24th International Work-
shop on Power and Timing Modeling, Optimization and
Simulation (PATMOS), pp. 1-5 (2014).
[104] M. L. Alomar, M. C. Soriano, M. Escalona-Mor´an, V.
Canals, I. Fischer, C. R. Mirasso, J. L. Rossell´o, Digi-
tal implementation of a single dynamical node reservoir
computer. IEEE Transactions on Circuits and Systems
II: Express Briefs 62(10), 977-981 (2015).
16
[105] P. Antonik, A. Smerieri, F. Duport, M. Haelterman, S.
Massar, FPGA implementation of reservoir computing
with online learning. Proc. of the 24th Belgian-Dutch
Conference on Machine Learning, pp.1-8 (2015).
[106] M. L. Alomar, V. Canals, N. Perez-Mora, V. Mart´ınez-
Moll, J. L. Rossell´o, FPGA-based stochastic echo state
networks for time-series forecasting. Computational in-
telligence and neuroscience 2016, 3917892 (2016).
[107] M. L. Alomar, V. Canals, A. Morro, A. Oliver, J. L.
Rossello, Stochastic hardware implementation of liquid
state machines. Proc. of 2016 International Joint Con-
ference on Neural Networks (IJCNN), pp. 1128-1133
(2016).
[108] P. Antonik, Application of FPGA to real-time machine
learning:
hardware reservoir computers and software
image processing. Springer (2018).
[109] H. Hauser, A. J. Ijspeert, R. M. F¨uchslin, R. Pfeifer, W.
Maass, Towards a theoretical foundation for morpho-
logical computation with compliant bodies. Biological
cybernetics 105(5-6), 355-370 (2011).
[110] H. Hauser, A. J. Ijspeert, R. M. F¨uchslin, R. Pfeifer,
W. Maass, The role of feedback in morphological com-
putation with compliant bodies. Biological cybernetics
106(10), 595-613 (2012).
[111] K.
Caluwaerts,
M.
D’Haene,
D.
Verstraeten,
B.
Schrauwen, Locomotion without a brain: physical reser-
voir computing in tensegrity structures. Artiﬁcial life
19(1), 35-66 (2013).
[112] K. Nakajima, H. Hauser, R. Kang, E. Guglielmino, D.
G. Caldwell, R. Pfeifer, Computing with a Muscular-
Hydrostat System. Proc. IEEE Int. Conf. on Robotics
and Automation (ICRA), p.1496-1503 (2013).
[113] K. Nakajima, H. Hauser, R. Kang, E. Guglielmino, D.
G. Caldwell, R. Pfeifer, A Soft Body as a Reservoir:
Case Studies in a Dynamic Model of Octopus-Inspired
Soft Robotic Arm. Frontiers in Computational Neuro-
science 7, 91 (2013).
[114] Q. Zhao, K. Nakajima, H. Sumioka, H. Hauser, R.
Pfeifer, Spine dynamics as a computational resource in
spine-driven quadruped locomotion. Proc. IEEE/RSJ
Int. Conf. on Intelligent Robots and Systems (IROS),
p.1445-1451 (2013).
[115] K. Nakajima, T. Li, H. Hauser, R. Pfeifer, Exploiting
short-term memory in soft body dynamics as a compu-
tational resource. J. Royal Society Interface 11 (100),
20140437 (2014).
[116] K. Caluwaerts, J. Despraz, A. I¸s¸cen, A. P. Sabelhaus, J.
Bruce, B. Schrauwen, V. SunSpiral, Design and control
of compliant tensegrity robots through simulation and
hardware validation. J. Royal Society Interface 11 (98),
20140520 (2014).
[117] K. Nakajima, H. Hauser, T. Li, R. Pfeifer, Information
processing via physical soft body. Scientiﬁc Reports 5,
10487 (2015).
[118] K. Nakajima, Muscular-Hydrostat Computers: Physical
Reservoir Computing for Octopus-Inspired Soft Robots.
Brain Evolution by Design (Springer Japan, 2017),
p.403-414.
[119] J. C. Coulombe, M. C. York, J. Sylvestre, Computing
with networks of nonlinear mechanical oscillators. PloS
one 12(6), e0178663 (2017).
[120] G. Urbain, J. Degrave, B. Carette, J. Dambre, F. Wyf-
fels, Morphological properties of massspring networks
for optimal locomotion learning. Frontiers in neuro-
robotics 11, 16 (2017).
[121] Y. Yamanaka, T. Yaguchi, K. Nakajima, H. Hauser,
Mass-Spring Damper Array as a Mechanical Medium
for Computation. Lecture Notes in Computer Science
11141:
International Conference on Artiﬁcial Neural
Networks (ICANN2018), Springer, Cham., p.781-794
(2018).
[122] K. Nakajima, T. Li, N. Akashi, Soft timer: dynamic
clock embedded in soft body. Robotic Systems and Au-
tonomous Platforms: Advances in Materials and Man-
ufacturing (Woodhead Publishing in Materials, 2018),
p.181-196.
[123] K. Nakajima, H. Hauser, T. Li, R. Pfeifer, Exploiting
the Dynamics of Soft Materials for Machine Learning.
Soft Robotics 5 (3), p.339-347 (2018).
[124] E. A. Torres, K. Nakajima, I. S. Godage, Information
Processing Capability of Soft Continuum Arms. Proc.
IEEE Int. Conf. on Soft Robotics (RoboSoft), p.441-447
(2019).
[125] T. Natschl¨ager, W. Maass, H. Markram, The “liquid
computer”: A novel strategy for real-time computing on
time series. Special Issue on Foundations of Information
Processing of TELEMATIK 8 (1), p.39-43 (2002).
[126] C. Fernando, S. Sojakka, Pattern recognition in a
bucket. Lecture Notes in Computer Science 2801, p.588,
Springer (2003).
[127] K. Nakajima, T. Aoyagi, The Memory Capacity of a
Physical Liquid State Machine, IEICE Technical Report
vol.115. No.300, pp.109-112 (2015).
[128] K. Goto, K. Nakajima, H. Notsu, Computing with
vortices: Bridging ﬂuid dynamics and its information-
processing capability. arXiv preprint arXiv:2001.08502,
(2020).
[129] D. Nikoli´c, S. H¨ausler, W. Singer, W. Maass. Dis-
tributed fading memory for stimulus properties in the
primary visual cortex. PLoS biology, 7(12):e1000260,
2009.
[130] M. R Dranias, H. Ju, E. Rajaram, A. M J VanDongen.
Short-term memory in networks of dissociated cortical
neurons. Journal of Neuroscience, 33(5):19401953, 2013.
[131] T. Kubota, K. Nakajima, H. Takahashi, Echo State
Property of Neuronal Cell Cultures. Lecture Notes in
Computer Science 11731: Int. Conf. on Artiﬁcial Neural
Networks (ICANN2019), Springer, Cham., p.137-148,
2019.
[132] G. Tanaka, T. Yamane, J. B. H´eroux, R. Nakane, N.
Kanazawa, S. Takeda, H. Numata, D. Nakano, A. Hi-
rose, Recent advances in physical reservoir computing:
A review. Neural Networks 115, p.100-123 (2019).
[133] K. Nakajima, I. Fischer, ed.:
Reservoir Computing:
Theory, Physical Implementations, and Applications
(Springer, in preparation).
[134] Y. Kawai, J. Park, M. Asada, A small-world topology
enhances the echo state property and signal propagation
in reservoir computing, Neural Networks, Vol. 112, pp.
15-23 (2019).
[135] M. C. Ozturk, D. Xu, J. C. Pr´ıncipe, Analysis and de-
sign of echo state networks. Neural computation 19(1),
111-138 (2007).
[136] T.
Tanaka,
K.
Nakajima,
T.
Aoyagi,
Eﬀect
of
recurrent
infomax
on
the
information
process-
ing
capability
of
input-driven
recurrent
neu-
ral
networks.
Neuroscience
Research
(in
press)
[doi.org/10.1016/j.neures.2020.02.001].
17
[137] D. Norton, D. Ventura, Preparing more eﬀective liq-
uid state machines using hebbian learning. Proc. of The
2006 IEEE International Joint Conference on Neural
Network, pp. 4243-4248 (2006).
[138] J. Yin, Y. Meng, Y. Jin, A developmental approach
to structural self-organization in reservoir computing.
IEEE transactions on autonomous mental development
4(4), 273-289 (2012).
[139] F. Xue, Z. Hou, X. Li, Computational capability of liq-
uid state machines with spike-timing-dependent plastic-
ity. Neurocomputing 122, 324-329 (2013).
[140] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma,
R. Soricut, Albert:
A lite bert for self-supervised
learning of language representations. arXiv:1909.11942
(2019).
[141] H. Sompolinsky, A. Crisanti, H. J. Sommers, Chaos
in random neural networks. Phys. Rev. Lett. 61, 259
(1988).
[142] L. Molgedey, J. Schuchhardt, H. G. Schuster, Suppress-
ing chaos in neural networks by noise. Phys. Rev. Lett.
69, 3717 (1992).
[143] D. Sussillo, L. F. Abbott, Generating coherent patterns
of activity from chaotic neural networks. Neuron 63, 544
(2009).
[144] D. Sussillo, L. Abbott, Transferring learning from ex-
ternal to internal weights in echo-state networks with
sparse connectivity. PloS one 7, e37372 (2012).
[145] B. DePasquale, C. J. Cueva, K. Rajan, G. S. Es-
cola, L. F. Abbott, full-FORCE: A target-based method
for training recurrent networks. PloS one 13, e0191527
(2018).
[146] R. Laje, D. V. Buonomano, Robust timing and motor
patterns by taming chaos in recurrent neural networks.
Nature neuroscience 16, 925 (2013).
[147] V. Goudar, D. V. Buonomano, Encoding sensory and
motor patterns as time-invariant trajectories in recur-
rent neural networks, eLife 7: e31134 (2018).
[148] K. Inoue, K. Nakajima, Y. Kuniyoshi, Soft bodies as
input reservoir:
role of softness from the viewpoint
of reservoir computing. Proc. Int. Symp. on Micro-
NanoMechatronics and Human Science (MHS), p.60-65,
2019.
[149] K. Inoue, K. Nakajima, Y. Kuniyoshi, Designing sponta-
neous behavioral switching via chaotic itinerancy, arXiv
preprint arXiv:2002.08332
[150] M. Yamaguchi, Y. Katori, D. Kamimura, H. Tamukoh,
T. Morie, A Chaotic Boltzmann Machine Working as a
Reservoir and Its Analog VLSI Implementation, Proc.
of Int. Joint Conf. on Neural Networks (IJCNN 2019),
N-20163 (2019).
[151] T. Taniguchi, N. Akashi, H. Notsu, M. Kimura, H.
Tsukahara, K. Nakajima, Chaos in nanomagnet via
feedback current. Phys. Rev. B 100 (17), 174425 (2019).
[152] T. Yamaguchi, N. Akashi, K. Nakajima, S. Tsunegi, H.
Kubota, T. Taniguchi, Synchronization and chaos in a
spin torque oscillator with a perpendicularly magnetized
free layer. Phys. Rev. B 100 (22), 224422 (2019).
[153] W. Maass, wetware, TAKEOVER: Who is Doing the
Art of Tomorrow (Ars Electronica 2001, Springer),
p.148-152.
[154] S. Boyd, L. Chua, Fading memory and the problem
of approximating nonlinear operators with Volterra se-
ries. IEEE Transactions on circuits and systems 32 (11),
p.1150-1161 (1985).
[155] W. Maass, H. Markram, On the computational power
of circuits of spiking neurons. Journal of computer and
system sciences 69 (4), p.593-616 (2004).
[156] R. Sol´e, M. Moses, S. Forrest, Liquid brains, solid
brains. Philosophical Transactions of the Royal Society
B: Biological Sciences 374(1774), 20190040 (2019).
[157] R. Pfeifer, C. Scheier, Understanding Intelligence (MIT
Press, 2001).
[158] R. Pfeifer, J. Bongard, How the Body Shapes the Way
We Think: A New View of Intelligence (MIT Press,
2006).
[159] R. Pfeifer, M. Lungarella, F. Iida, Self-organization,
embodiment, and biologically inspired robotics. Science
318 (5853), p.1088-1093 (2007).
[160] T. McGeer, Passive dynamic walking. I. J. Robotic Res.
9 (2), p.62-82 (1990).
[161] Q. Zhao, H. Sumioka, X. Yu, K. Nakajima, Z. Wang, R.
Pfeifer, The Function of the Spine and its Morphological
Eﬀect in Quadruped Robot Locomotion. Proc. IEEE
Int. Conf. on Robotics and Biomimetics (ROBIO), p.66-
71 (2012).
[162] Q. Zhao, K. Nakajima, H. Sumioka, X. Yu, R. Pfeifer,
Embodiment Enables the Spinal Engine in Quadruped
Robot Locomotion. Proc. IEEE/RSJ Int. Conf. on Intel-
ligent Robots and Systems (IROS), p.2449-2456 (2012).
[163] N. Schmidt, M. Hoﬀmann, K. Nakajima, R. Pfeifer,
Bootstrapping Perception Using Information Theory:
Case Studies in a Quadruped Robot Running on Diﬀer-
ent Grounds. Advances in Complex Systems 16 (1&2),
1250078 (2013).
[164] Q. Zhao, H. Sumioka, K. Nakajima, X. Yu, R. Pfeifer,
Spine as an Engine:
Eﬀect of Spine Morphology
on Spine-Driven Quadruped Locomotion. Advanced
Robotics 28 (6), p.367-378 (2014).
[165] J. C. Liao, Neuromuscular control of trout swimming in
a vortex street: implications for energy economy during
the Karman gait. Journal of Experimental Biology 207
(20), p.3495-3506 (2004).
[166] A. M. T. Ngouabeu, S. Miyashita, R. M. F¨uchslin, K.
Nakajima, M. Goldi, R. Pfeifer, Self-organized Segre-
gation Eﬀect on Water Based Self-Assembling Robots.
Proc. Int. Conf. on the Simulation and Synthesis of Liv-
ing System (Artiﬁcial Life XII), MIT Press, p.232-238
(2010).
[167] S. Miyashita, A. M. T. Ngouabeu, R. M. F¨uchslin, K.
Nakajima, C. Audretsch, R. Pfeifer, Basic Problems in
Self-Assembling Robots and a Case Study of Segrega-
tion on Tribolon Platform. Studies in Computational
Intelligence 355, p.173-191 (2011).
[168] K. Nakajima, A. M. T. Ngouabeu, S. Miyashita, M.
Goldi, R. M. F¨uchslin, R. Pfeifer, Morphology-Induced
Collective Behaviors: Dynamic Pattern Formation in
Water-Floating Elements. PLoS ONE 7 (6), e37805
(2012).
[169] S. Miyashita, K. Nakajima, Z. Nagy, R. Pfeifer, Self-
Organized Translational Wheeling Behavior in Stochas-
tic Self-Assembling Modules. Artiﬁcial Life 19 (1), p.79-
95 (2013).
[170] R. Pfeifer, G. G´omez, Creating brain-like intelligence
(Springer, Berlin, Heidelberg, 2009), p.66-83.
[171] A. P. Sabelhaus, J. Bruce, K. Caluwaerts, P. Manovi,
R. F. Firoozi, S. Dobi, A. M. Agogino, V. SunSpiral,
System design and locomotion of SUPERball, an un-
tethered tensegrity robot. Proc. of 2015 IEEE interna-
18
tional conference on robotics and automation (ICRA),
pp. 2867-2873 (2015).
[172] K. Ghazi-Zahedi, N. Ay, Quantifying morphological
computation. Entropy 15(5), 1887-1915 (2013).
[173] K. Ghazi-Zahedi, Morphological Intelligence: Measuring
the Body’s Contribution to Intelligence (Springer Na-
ture, 2019).
[174] S. Kim, C. Laschi, B. Trimmer, Soft robotics: a bioin-
spired evolution in robotics. Trends in Biotechnology 31
(5), p.287-294 (2013).
[175] C. Laschi, B. Mazzolai, M. Cianchetti, Soft robotics:
Technologies and systems pushing the boundaries of
robot abilities. Science Robotics 1 (1), eaah3690 (2016).
[176] D. Rus, M. T. Tolley, Design, fabrication and control of
soft robots. Nature 521 (7553), p.467 (2015).
[177] T. Li, K. Nakajima, M. J. Kuba, T. Gutnick, B.
Hochner, R. Pfeifer, From the octopus to soft robots
control: an octopus inspired behavior control architec-
ture for soft robots. Vie et Milieu 61, p.211-217 (2012).
[178] K. Nakajima,
T. Li,
H. Sumioka,
M. Cianchetti,
R. Pfeifer, Information Theoretic Analysis on a Soft
Robotic Arm Inspired by the Octopus. Proc. IEEE Int.
Conf. on Robotics and Biomimetics (ROBIO), p.110-
117 (2011).
[179] K. Nakajima, T. Li, R. Kang, E. Guglielmino, D. G.
Caldwell, R. Pfeifer, Local Information Transfer in Soft
Robotic Arm. Proc. IEEE Int. Conf. on Robotics and
Biomimetics (ROBIO), p.1273-1280 (2012).
[180] K. Nakajima, N. Schmidt, R. Pfeifer, Measuring Infor-
mation Transfer in a Soft Robotic Arm. Bioinspiration
& Biomimetics 10 (3), 035007 (2015).
[181] D. Trivedi, C. D. Rahn, W. M. Kier, and I. D. Walker,
Soft robotics: Biological inspiration, state of the art,
and future research. Applied Bionics and Biomechanics
5 (3), pp. 99117 (2008).
[182] L. F. Seoane, Evolutionary aspects of reservoir comput-
ing. Philosophical Transactions of the Royal Society B
374(1774), 20180377 (2019).
[183] D. Kobayashi, Y. Kakehashi, K. Hirose, S. Onoda, T.
Makino, T. Ohshima, S. Ikeda, M. Yamanouchi, H.
Sato, E. C. Enobio, T. Endoh, H. Ohno, Inﬂuence
of heavy ion irradiation on perpendicular-anisotropy
CoFeB-MgO magnetic tunnel junctions. IEEE Trans.
Nuclear Science 61, 1710 (2014).
[184] K. Kitajo, T. Sase, Y. Mizuno, H. Suetani, Consistency
in macroscopic human brain responses to noisy time-
varying visual inputs. bioRxiv, 645499 (2019).
[185] C. Stringer, M. Pachitariu, N. Steinmetz, C. B. Reddy,
M. Carandini, K. D. Harris. Spontaneous behaviors
drive multidimensional,
brainwide activity. Science,
364(6437):255255, 2019.
