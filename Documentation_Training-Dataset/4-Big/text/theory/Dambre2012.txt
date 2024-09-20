Information Processing Capacity of
Dynamical Systems
Joni Dambre1, David Verstraeten1, Benjamin Schrauwen1 & Serge Massar2
1Department of Electronics and Information Systems (ELIS), Ghent University, Sint-Pietersnieuwstraat 41, 9000 Ghent, Belgium,
2Laboratoire d’Information Quantique (LIQ), Universite´ libre de Bruxelles (U.L.B.), 50 Avenue F. D. Roosevelt, CP 225, B-1050
Bruxelles, Belgium.
Many dynamical systems, both natural and artificial, are stimulated by time dependent external signals,
somehow processing the information contained therein. We demonstrate how to quantify the different
modes in which information can be processed by such systems and combine them to define the
computational capacity of a dynamical system. This is bounded by the number of linearly independent state
variables of the dynamical system, equaling it if the system obeys the fading memory condition. It can be
interpreted as the total number of linearly independent functions of its stimuli the system can compute. Our
theory combines concepts from machine learning (reservoir computing), system modeling, stochastic
processes, and functional analysis. We illustrate our theory by numerical simulations for the logistic map, a
recurrent neural network, and a two-dimensional reaction diffusion system, uncovering universal trade-offs
between the non-linearity of the computation and the system’s short-term memory.
M
any dynamical systems, both natural and artificial, are driven by external input signals and can be seen as
performing non-trivial, real-time computations of these signals. The aim of this work is to study the
online information processing that takes place within these dynamical systems. The importance of this
question is underlined by the multitude of examples from nature and engineering which fall into this category.
These include the activation patterns of biological neural circuits to sensory input streams1, the dynamics of many
classes of artificial neural network models used in artificial intelligence (including the whole field of ‘‘reservoir
computing’’2–5 as well as its recent implementation using time delay systems6–8), and systems of interacting
chemicals, such as those found within a cell, that display intra-cellular control mechanisms in response to external
stimuli9. Recent insight in robotics has demonstrated that the control of animal and robot locomotion can greatly
be simplified by exploiting the physical body’s response to its environment10,11. Additional systems to which our
theory could apply are population dynamics responding to, e.g., changes in food supply, ecosystems responding
to external signals such as global climate changes, and the spread of information in social network graphs. All of
these exemplary systems display completely different physical implementations.
We present a general framework that allows to compare the computational properties of a broad class of
dynamical systems. We are able to characterize the information processing that takes place within these systems
in a way which is independent of their physical realization, using a quantitive measure. It is normalized appro-
priately such that completely different systems can be compared and it allows us to characterize different
computational regimes (linear vs. non-linear; long vs. short memory).
Some initial steps in this direction are provided by the linear memory capacity introduced in12 (later theoret-
ically extended to linear systems in discrete time13 and continuous time systems14, and using Fischer informa-
tion15). Although it has been argued that short-term memory of a neural system is crucial for the brain to perform
useful computation on sensory input streams15,16, it is our belief that the general paradigm underlying the brain’s
computation cannot solely rely on maximizing linear memory. The neural circuits have to compute complex non-
linear and spatio-temporal functions of the inputs. Prior models focusing on linear memory capacity in essence
assumed the dynamic system to only implement memory, while the complex non-linear mapping is off-loaded to
an unspecified read-out mechanism. We propose to endow the dynamic system with all the required computa-
tional capacity, and use only simple linear read-out functions. The capacity measures we introduce are therefore
of great interest since they quantify all the information processing that takes place within the dynamical system,
and don’t introduce an artificial separation between linear and non-linear information processing.
One of the startling results of this work with potentially far reaching consequences is that all dynamical
systems, provided they obey the condition of having a fading memory17 and have linearly independent internal
variables, have in principle the same total normalized capacity to process information. The ability to carry out
SUBJECT AREAS:
THEORY
INFORMATION THEORY AND
COMPUTATION
MATHEMATICS AND
COMPUTING
STATISTICAL PHYSICS,
THERMODYNAMICS AND
NONLINEAR DYNAMICS
Received
4 April 2012
Accepted
18 June 2012
Published
19 July 2012
Correspondence and
requests for materials
should be addressed to
J.D. (Joni.Dambre@
ugent.be)
SCIENTIFIC REPORTS | 2 : 514 | DOI: 10.1038/srep00514
1
useful information processing is therefore an almost universal char-
acteristic of dynamical systems. This result provides theoretical jus-
tification for the widely used paradigm of reservoir computing with a
linear readout layer2–4; indeed it confirms that such systems can be
universal for computation of time invariant functions with fading
memory.
We illustrate the versatility of our approach by using numerical
simulations on three very different dynamical systems: the logistic
map, a recurrent neural circuit, and a two-dimensional reaction
diffusion system. These examples allow us to uncover the (potentially
universal) trade-off between memory depth and non-linear compu-
tations performed by the dynamic system itself. We also discuss how
the influence of noise decreases the computation capacity of a
dynamical system.
Results
Dynamical systems. The dynamical systems we consider are
observed in discrete time t [ Z. They are described by internal
variables xj(t), j g J, being the set of indices of all internal
variables. The set J may be finite or (enumerably or continuously)
infinite.
The internal variables are driven by external signals u(t) g UK
which can have arbitrary dimensionality. Typical examples for the
set U include a discrete set, or subsets of RK such as the unit interval
or the whole real line R. When studying a dynamical system, we do
not necessarily have access to all the dynamical variables x(t). This
corresponds to experimental situations where many variables may
not be measurable, and is also the case if the number of dynamical
variables jJj is infinite. We therefore assume that we can only access
the instantaneous state of the system through a finite number N of
the internal variables: xi(t), i 5 1, …, N. Note that we use the term
internal variables throughout the paper for easy of notation. The
theory however also holds if we cannot directly observe the internal
states, but only have access to them via N observation functions. An
important special case is when the dynamical system comprises a
finite number jJj 5 N of dynamical variables and we have access to all
of them.
The system evolves according to an evolution equation of the
form:
xj tð Þ~Tj x t{1
ð
Þ,u tð Þ
ð
Þ,
ð1Þ
with Tj the mapping from input and previous state to the j’th current
state. We denote a dynamical system with the evolution law eq. (1) by
the capital letter X.
In order to evaluate the performance of the system in a standar-
dized way and to make analytical derivations tractable, we will take
the inputs u(t) to be independent and identically drawn from some
probability distribution p(u). It should be noted that in many, if not
all, real world situations the inputs are not i.i.d. Most of the theory
should also hold for the non-i.i.d. case, but this significantly compli-
cates the construction of a basis for the Hilbert space defined in
Definition 5. However our aim is to characterize the dynamical sys-
tem itself. By taking the inputs to be i.i.d. we eliminate as much as
possible any observed structure originating from the inputs, so that
any measured structure will be due to the dynamical system itself.
The distribution p(u) can be chosen according to what is most rel-
evant for the dynamical system under study.
In order to study this dynamical system experimentally, we pro-
ceed as follows. First the system is initialized in an arbitrary initial
state at a time -T9, and is driven for T9 time steps with a sequence of
i.i.d. inputs drawn from the distribution p(u). This to ensure that
potential initial transients have died out. Next, we drive the system
for T timesteps with inputs from the same distribution. During this
later phase we record both the values of the inputs u(t) and of the
state variables x(t) which will be used to get insight into how informa-
tion is processed by the dynamical system.
Capacity for reconstructing a function of the inputs. We will
denote by u–h(t) 5 (u(t – h 1 1), …, u(t)) the sequence of h
previous inputs up to time t. It follows that u–‘(t) 5 (…, u(t – 2),
u(t – 1), u(t)) is the left infinite sequence of all previous inputs up to
time t. We will often consider functions of a finite number h, or an
infinite number of time-steps (the latter corresponding to the limit
T, T9 R ‘). We often wish to refer to a sequence of inputs without
referring to the time at which they act, in which case we denote by
u–h 5 (u–h11, …, u0) g Uh a sequence of h inputs and by u–‘ 5 (…,
u–2, u–1, u0) g U‘ a left infinite sequence of inputs.
Consider a function on sequences of h inputs
z : Uh?R : u{h?z u{h


:
ð2Þ
Given the recorded inputs u(t), it induces a time dependent function
z(t) 5 z(u–h(t)).
A central quantity in our analysis will be the capacity to recon-
struct the function z from the state of a dynamical system using a
linear estimator. We introduce this notion as follows. Consider a
linear estimator constructed from the observed data
^z tð Þ~
X
N
i~1
Wixi tð Þ:
ð3Þ
We measure the quality of the estimator by the Mean Square Error
(MSE):
MSET ^z½ ~ 1
T
X
T
t~1
^z tð Þ{z tð Þ
ð
Þ2:
ð4Þ
where here and throughout we denote by a subscript T quantities that
are generated from the measured data. We are interested in the
estimator which minimizes the MSE.
Definition 1. The Capacity for the dynamical system X to reconstruct
the function z is
CT X,z
½
~1{ minWi MSET ^z½ 
z2
h
iT
ð5Þ
where z2
h
iT~ 1
T
XT
t~1 z tð Þ2.
The notion of capacity can also be interpreted from the point
of view of computation. In this approach, one wishes to use the
dynamical system to compute the function z(u–h), and to this end
one uses, following the ideas introduced in reservoir computing2–4,
the simple linear readout eq. (3). The capacity C[X, z] measures how
successful the dynamical system is at computing z. Note that this
capacity is essentially the same measure that was introduced in12
when z(t) 5 u(t – l) is a linear function of one of the past inputs.
The main difference with the notation of12 is the treatment of a
possible bias term in eq. (3). This corresponds to writing
^z tð Þ~ PN
i~1 Wixi tð ÞzWNz1 and is equivalent to introducing an
additional observation function xN11(t) 5 1 which is time independ-
ent. In what follows we do not introduce such a bias term, as this
makes the statement of the results simpler. We discuss the relation
between the two approaches in the supplementary material.
We note several properties of the capacity:
Proposition 2. For any function z(t), and any set of recorded data x(t),
t 5 1, …, T the capacity can be expressed as
CT X,z
½
~
P
ij zxi
h
iT xixj

{1
T
xjz


T
z2
h
iT
,
ð6Þ
where v
h iT~ 1
T
XT
t~1 v tð Þ denotes the time average of a set of data,
and xixj

{1
T
denotes the inverse of the matrix ÆxixjæT. If ÆxixjæT does
not have maximum rank, then xixj

{1
T
is zero on the kernel of ÆxixjæT
and equal to its inverse on the complementary subspace and
www.nature.com/scientificreports
SCIENTIFIC REPORTS | 2 : 514 | DOI: 10.1038/srep00514
2
Proposition 3. For any function z, and any set of recorded data, the
capacity is normalized according to:
0ƒCT X,z
½
ƒ1,
ð7Þ
where CT [X, z] 5 0 if the dynamical system is unable to even partially
reconstruct the function z, and CT 5 1 if perfect reconstruction of z is
possible (that is if there is a linear combination of the observation
functions xi(t) that equals z(t)).
Note that CT [X, z] depends on a single set of recorded data. The
computed value of the capacity is therefore affected by statistical
fluctuations: if one repeats the experiment one will get a slightly
different value. This must be taken into account, particularly when
interpreting very small values of the capacity CT. We discuss this
issue in the supplementary material.
The capacity is based on simple linear estimators ^z tð Þ~
PN
i~1 Wixi tð Þ. For this reason the capacity characterizes properties
of the dynamical system itself, rather than the properties of the
estimator. Consider for instance the case where z[u–h] 5 u–3 is the
input 3 time steps in the past. Because the estimator is a linear
combination of the states xi(t) of the dynamical system xi at time t,
a nonzero capacity for this function indicates that the dynamical
system can store information in a linear subspace of x(t) about the
previous inputs for at least 3 time steps. Consider now the case where
z[u–h] 5 u–2u–3 is a nonlinear function of the previous inputs (in this
case the product of the inputs 2 and 3 time steps in the past). Then a
nonzero capacity for this function indicates that the dynamical sys-
tem both has memory, and is capable of nonlinear transformations
on the inputs, since the dynamical system is the only place where a
nonlinear transformation can occur. We therefore say that if CT (X, z)
. 0, the dynamical system X can, with capacity C(X, z), approximate
the function z.
In summary, by studying the capacity for different functions we
can learn about the information processing capability of the dynam-
ical system itself. Because the capacity is normalized to 1, we can
compare the capacity for reconstructing the function z for different
dynamical systems.
Maximum capacity of a dynamical system. Consider two functions
z and z9, and the associated capacities CT [X, z] and CT [X, z9]. If the
functions z and z9 are close to each other, then we expect the
capacities to also be close to each other. On the other hand if z and
z9 are sufficiently distinct, then the two capacities will be giving
us independent information about the information processing
capability of the dynamical system. To further exploit the notion of
capacity for reconstructing functions z, we therefore need to
introduce a distance on the space of functions z : Uh?R. This
distance should tell us when different capacities are giving us
independent information about the dynamical system.
To introduce this distance, we note that the probability distri-
bution p(u) on the space of inputs, and associated notion of integr-
ability, provide us with an inner product, distance, and norm on the space
of functions z : Uh?R : z,z0
h
iUh~EUh zz0
½
, d z,z0
ð
Þ~EUh
z{z0
ð
Þ2


,
z
k k2
Uh~ z,z
h
iUh, where EUh :½ , is the expectation taken with respect to the
measure p(u) on Uh. We define the Hilbert space HUh as the
space of functions z : Uh?R with bounded norm
z,z
h
iUhv?
(this is technically known as the weighted L2 space). In the fol-
lowing we suppose that HU, and hence HUh, is separable, i.e., that
it has a finite or denumerable basis.
The following theorem is one of our main results. It shows that if
functions z and z9 in HUh are orthogonal, then the corresponding
capacities CT [X, z] and CT [X, z9] measure independent properties of
the dynamical system, and furthermore the sum of the capacities for
orthogonal state variables cannot exceed the number N of output
functions of the dynamical system.
Theorem 4. Consider a dynamical system as described above with N
output functions xi(t) and choose a positive integer h [ N. Consider
any finite set YL 5 {y1, …, yL} of size jYLj 5 L of orthogonal functions
in HUh, yl [ HUh, yl,yl0
h
iUh~ yl
j j2dll0, l, l9 5 1, …, L. We furthermore
require that the fourth moment of the y1 is finite y4
l


Uhv?. Then,
in the limit of an infinite data set T R ‘, the sum of the capacities for
these functions is bounded by the number N of output functions
(independently of h, of the set YL, or of its size jYLj 5 L):
lim
T??
X
L
l~1
CT x,yl
½
ƒN:
ð8Þ
The fact that the right hand side in eq. (8) depends only on the
number of output functions N implies that we can compare sums of
capacities, for the same sets of functions YL, but for completely dif-
ferent dynamical systems.
Fading Memory. In order to characterize when equality is attained in
Theorem 4 we introduce the notion of fading memory Hilbert space
and fading memory dynamical system:
Definition 5. Hilbert space of fading memory functions. The Hilbert
space HU? of fading memory functions is defined as the limit of
Cauchy sequences in HUh, as h increases, as follows:
1. If x [ HUh, then x [ HU?, and x,x
h
iU?~ x,x
h
iUh.
2. Consider a sequence xh [ HUh, h [ N, then the limit limhR‘ xh
exists and is in HU? if for all . 0, there exits h0 [ N, such that for
all h, h9 . h0, xh{xh0
k
k2
Umax h,h0
ð
Þv .
3. Conversely, all x [ HU? are the limit of a sequence of the type given
in 2.
4. The scalar product of x, x0 [ HU? is given by
x0,x
h
iU?~
limh,h0?? x0
h0,xh


, where xh R x and x0
h?x0 are any two Cauchy
sequences that converge to x and x9 according to 2) above.
In the supplementary information we prove that HU? indeed con-
stitutes a Hilbert space, and that it has a denumerable basis if HU is
separable.
Definition 6. Fading Memory Dynamical System17. Consider a
dynamical system given by the recurrence x(t) 5 T(x(t – 1), u(t))
with some initial conditions x(–T9), and suppose that one has access
to a finite number of internal variables xi(t), i 5 1, …,N. This dynam-
ical system has fading memory if, for all . 0 , there exists a positive
integer h0 [ N, such that for all h . h0, for all initial conditions x0,
and for all sufficiently long initialization time T9 . h, the variables
xi(t) at any time t $ 0 are well approximated by the functions xh
i :
EUtzT0 xi tð Þ{xh
i u{h tð Þ



2v ,
ð9Þ
where the expectation is taken over the t 1 T9 previous inputs.
When the dynamical system has fading memory, its output functions
xi(t) can be identified with vectors in the Hilbert space of fading mem-
ory functions. Indeed the convergence condition eq. (9) in the definition
of fading memory dynamical systems is precisely the condition that the
functions xh
i
have a limit in HU?. We can thus identify limT0??
xi tð Þ~xi u{? tð Þ
½
 [ HU?, where xi u{?
½
~ limh?? xh
i u{h


.
The following result states that if the dynamical system has fading
memory, and if the dynamical variables xi, i 5 1, …,N, are linearly
independent (their correlation matrix has full rank), and if the func-
tion yl constitute a complete orthonormal basis of the fading memory
Hilbert space HU?, then one has equality in Theorem 4:
Theorem 7. Consider a dynamical system with fading memory as
described above with N accessible variables xi(t). Because the dynam-
ical system has fading memory, we can identify the output functions
with functions xi(u–‘) in HU?. Consider an increasing family of
orthogonal functions in HU? : YL~ y1, . . . ,yL
f
g with YL(YL0 if
L9 $ L and yl [ HU?, yl,yl0
h
iU?~ yl
j j2dll0, l, l9 5 1, …,L , such that
in the limit L R ‘, the sets YL tends towards a complete orthogonal
set of functions in HU?. Suppose that the readout functions xi(u–‘)
and the basis functions y1(u–‘) have finite fourth order moment:
www.nature.com/scientificreports
SCIENTIFIC REPORTS | 2 : 514 | DOI: 10.1038/srep00514
3
x4
i


U?v?,
y4
l


U?v?. Consider the limit of an infinite data set
T R ‘ and infinite initialization time T9 R ‘. Suppose the correla-
tion matrix Rii0~ limT,T0??
1
T
XT
t~1 xi tð Þxi0 tð Þ has rank N. Then
the sum of the capacities for the sets YL tends towards the number
N of output functions:
YL?
lim
complete set
lim
T,T0??
X
L
l~1
CT x,yl
½

"
#
~N:
ð10Þ
This result tells us that under the non-strict conditions of Theorem
7 (fading memory and linearly independent internal variables), all
dynamical systems driven by an external input and comprising N
linearly independent internal variables have the same ability to carry
out computation on the inputs, at least as measured using the capa-
cities. Some systems may carry out more interesting computations
than others, but at the fundamental level the amount of computation
carried out by the different systems is equivalent.
Connections with other areas. There is a close connection between
the sum of capacities for orthogonal functions P
l CT [X, yl] and
system modeling, and in particular Wiener series representation of
dynamical systems18,19. Indeed suppose that one wishes to model the
dynamical system with variables xi (which for simplicity of the
argument we suppose to belong to HU?) by linear combinations of
the finite set of orthonormal functions yl [ HU?, l 5 1, …,L. The best
such approximation is PL
l~1 yl yl,xi
h
iU?, and we can write:
xi~ PL
l~1 yl yl,xi
h
iU?zDi where the errors Di are orthogonal to
the yl. We can then write the sum of the capacities as
lim
T??
X
L
l~1
CT x,yl
½
~N
1{O
D
k k2
U?
x
k k2
U?
 
!
 
!
:
That is, the extent to which the sum of the capacities for a finite set of
orthogonal functions saturates the bound given in Theorem 7 is
proportional to how well the dynamical system can be modeled by
the linear combinations of these functions. This connection between
the degree to which dynamical systems can be used for computation
and system modeling has to our knowledge not been noted before.
Reservoir computing was one of the main inspirations for the
present work, and is closely connected to it. We shed new light on
it, and in particular on the question raised in2 of characterizing the
combinations of dynamical systems and output layer that have uni-
versal computing power on time invariant functions with fading
memory. In2 it was shown that the system has universal computing
power if the dynamical system has the point-wise separation property
(a weak constraint), and if the readout layer has the approximation
property (a strong constraint). Here we consider simple readout
layers which are linear functions of the internal states of the dynam-
ical system, see eq. (3). Our work shows that the combination of a
dynamical system and a linear output layer has universal computing
power on time invariant functions with fading memory if, in the limit
N R ‘, the set of internal states xi, i 5 1, …,N, spans all of the Hilbert
space HU? of fading memory functions. This provides a new class of
systems which have universal computing power on time dependent
inputs. In particular it provides theoretical justification for the widely
used paradigm of reservoir computing with linear readout layer20,3,21,
by proving that such systems can be universal.
Applications. To illustrate our theory, we have chosen driven
versions of three (simulated) dynamical systems. The first is the
well-known logistic map22, governed by the equation
x tz1
ð
Þ~r v tð Þ 1{v tð Þ
ð
Þ,
with u(t) the input, v tð Þ~S x tð Þziu tð Þ
ð
Þ, and S y
ð Þ a piecewise
linear saturation function, i.e.,
S y
ð Þ~
0, yv0
y, 0ƒyƒ1
1, yw1
8
>
<
>
:
to ensure that 0 # v (t) # 1. The parameter i can be used to scale the
strength with which the system is driven. When not driven, this
system becomes unstable for r . 3.
The second system is an echo-state network (ESN)20, which can be
thought of as a simplified model of a random recurrent neural net-
work. ESNs are very successful at processing time dependent
information3,21. The N internal variables xi(t), i 5 1 … N, evolve
according to the following equation
xi tz1
ð
Þ~tanh r
X
N
j~1
wijxj tð Þziviu tð Þ
 
!
,
ð11Þ
where the coefficients wij and vi are independently chosen at random
from the uniform distribution on the interval [–1, 11], and the
matrix wij is then rescaled such that its maximal singular value equals
1. The parameters r and i, called the feedback gain and input gain
respectively, can be varied. It was shown in20,23 that, if the maximal
singular value of the weight matrix W 5 [wij] equals one, ESN are
guaranteed to have fading memory whenever r , 1. In the absence of
input, they are unstable for r . 1. Because of the saturating nature of
the tanh nonlinearity, driving an ESN effectively stabilizes the system
dynamics24. In this work we took an ESN with N 5 50 variables. A
single instantiation of the coefficients wij and vi was used for all
calculations presented below.
The last system is a driven version of the continuous-time con-
tinuous space two-dimensional Gray-Scott reaction diffusion (RD)
system25,26, a one-dimensional version of which was used in9 in a
machine learning context. This system consists of two reagents, A
and B, which have concentrations axy(t) and bxy(t) at spatial coordi-
nates (x, y) and at (continuous-valued) time t. The reaction kinetics
of the undriven system are described by the following differential
equations:
Laxy
Lt ~da+2axy{axyb2
xyzf
1:0{axy


Lbxy
Lt ~db+2bxyzaxyb2
xy{ f zk
ð
Þ
bxy:
8
>
>
<
>
>
:
Reagents A and B diffuse through the system with diffusion constants
da and db, respectively. A is converted into B at rate axyb2
xy. B cannot
be converted back into A. A is supplied to the system through a
semipermeable membrane, one side of which is kept at a fixed (nor-
malised) concentration of 1.0. The rate with which A is fed to the
system is proportional to the concentration difference accross the
membrane with a rate constant f (the feed rate). B is eliminated from
the system through a semipermeable membrane at a rate propor-
tional to its concentration (the outside concentration is set to 0.0),
with rate constant f 1 k. To drive the system with input signal u (t),
we first convert it to continuous time as follows: u (t) 5 u (t) , tTS # t
, (t 1 1) TS, for a given value the sampling period TS. This signal is
then used to modulate the concentration of A outside the membrane:
Laxy
Lt ~da+2axy{axyb2
xyzf 1:0ziwxyu t
ð Þ{axy


with iwxyu t
ð Þ

v1:0:
In order to break the spatial symmetry of the system, the modulation
strength wxy is randomized accross the system. The concentration of
A is observed at N discrete positions (25 in our experiments) by
taking one sample per sampling period TS. In our numerical experi-
ments, we keep all parameters da, db, f, k, i and wxy constant, and
explore the behavior of the systems as TS is varied. Further details
about the RD system are given in the supplementary material.
www.nature.com/scientificreports
SCIENTIFIC REPORTS | 2 : 514 | DOI: 10.1038/srep00514
4
Total capacity. We now apply our theoretical results to the study of
these three dynamical systems. We first choose the probability
distribution over the inputs p(u) and the complete orthonormal set
of functions yl. We take p(u) to be the uniform distribution over the
interval [–1, 11]. As corresponding basis we use finite products of
normalized Legendre polynomials for each time step:
y di
f g~ P
i Pdi u t{i
ð
Þ
ð
Þ
ð12Þ
where Pdi :ð Þ, di $ 0, is the normalized Legendre polynomial of
degree di. The normalized Legendre polynomial of degree 0 is a
constant, P0~1. The index l ; {di} therefore corresponds to the
set of di, only a finite number of which are non zero. We do not
evaluate the capacity when {di} is the all zero string, i.e., at least one of
the di must be non zero (see supplementary material for details of the
expressions used). Other natural choices for the probability distri-
butions p(u) and the set of orthonormal functions (which will not be
studied here) are for instance inputs drawn from a standard normal
distribution and a basis of products of Hermite polynomials, or
uniformly distributed inputs with a basis of products of trigo-
nometric functions.
Each of the dynamical systems in our experiments were simulated.
After an initialization period the inputs and outputs were recorded for
a duration T 5 106 time steps. When using finite data, one must take
care not to overestimate the capacities. According to the procedure
outlined in the supplementary material, we set to zero all capacities
for which CT (X, {di}) , , and therefore use in our analysis the trun-
cated capacities CT X, di
f g
ð
Þ~h CT X, di
f g
ð
Þ{
ð
ÞCT X, di
f g
ð
Þ where h
is the Heaviside function. The value of
is taken to be
5 1.7 10–4 for
ESN,
5 1.1 10–4 for RD system, and
5 2.2 10–5 for the logistic map
(see supplementary material for the rationale behind these choices of ).
We have carefully checked that the values we obtain for CT X, di
f g
ð
Þ
are reliable.
According to the above theorems, for fading memory systems with
linearly independent state variables, the sum of all capacities
C~ P
di
f gC X, di
f g
ð
Þ should equal the number of independent
observed variables N. To test this, in Figure 1 we plot the sum of
the measured capacities
CTOT C
½ ~
X
di
f g
CT X, di
f g
ð
Þ,
as a function of the parameter r for logistic map and ESN, and for
three values of the input scaling parameter i. For very small input
scaling, the capacity CTOT remains close to N up to the bifurcation
point of the undriven system and then drops off rapidly. As the input
scaling increases, the loss of capacity becomes less abrupt, but its
onset shifts to lower values. We attribute this to two effects which
act in opposite ways. On the one hand, when the input strength
increases, the response of the system to the input becomes more
and more nonlinear, and is not well approximated by a polynomial
with coefficients , . As a consequence CTOT underestimates the
total capacity. On the other hand driving these systems on average
stabilizes their dynamics24, and this effect increases with the input
amplitude i. Therefore the transition to instability becomes less
abrupt for larger values of r.
The measured capacities CT X, di
f g
ð
Þ contain much additional
information about how the dynamical systems process their inputs.
To illustrate this, we have selected a parameter that changes the degree
of nonlinearity for each system: the input scaling parameter i for the
ESN and logistic map, and the sampling period TS for the RD system,
and studied how the capacities change as a function of these para-
meters. In Figure 2 we show the breakdown of the total capacity
according to the degree P?
i~1 di of the individual basis functions.
The figure illustrates how, by increasing i or TS, the information pro-
cessing that takes place in these systems becomes increasingly non-
linear. It also shows that these systems process information in inequi-
valent ways. For instance from the figure it appears that, contrary to the
other systems, the ESN is essentially incapable of computing basis
functions of even degree when the input is unbiased (we attribute this
to the fact that the tanh appearing in the ESN is an odd function).
 
 
 
 
Figure 1 | Total measured capacity CTOT for the logistic map (left) and an
ESN with 50 nodes (right) as a function of the gain parameter r and for
three different values of the input scaling parameter i. The edge of
stability of the undriven system is indicated in both figures by the dotted
line.
 
 
Figure 2 | Breakdown of total measured capacity according to the degree
X?
i~1 di of the basis function as a function of the parameter i for ESN and logistic
map, and Ts for RD system. The values of r (0.95 for ESN and 2.5 for logistic map) were chosen close to the edge of stability, a region in which the most useful
processing often occurs 27. The scale bar indicates the degree of the polynomial. Capacities for polynomials up to degree 9 were measured, but the higher degree
contributions are too small to appear in the plots. Note how, when the parameters i and TS increase, the systems become increasingly nonlinear. Due to the fact
that the hyperbolic tangent is an odd function and the input is unbiased, the capacities for the ESN essentially vanish for even degrees.
www.nature.com/scientificreports
SCIENTIFIC REPORTS | 2 : 514 | DOI: 10.1038/srep00514
5
Memory vs. non-linearity trade-off. Further insight into the
different ways dynamical systems can process information is
obtained by noting that the measured capacities rescaled by the
number of observed signals, CT X, di
f g
ð
Þ=N, constitute a (possibly
sub-normalised because of underestimation or failure of the fading
memory property) probability distribution over the basis functions.
This enables one to compute quantities which summarize how
information is processed in the dynamical systems, and compare
between systems. We first (following12) consider the linear memory
capacity (the fraction of the total capacity associated to linear
functions):
L C
½ ~
X
di
f g
d
X
i
di{1
 
!
CT X, di
f g
ð
Þ
N
,
ð13Þ
where the delta function is equal to 1 iff. its argument is zero, and
otherwise is equal to 0. Conversely, the non-linearity of the
dynamical system is defined by associating to each Legendre
polynomial Pdi in eq. (12) the corresponding degree di, and to
products of Legendre polynomials the sums of the corresponding
degrees, i.e., the total degree of the basis function, to obtain
N L C
½ ~
X
di
f g
X
?
i~1
di
 
!
CT X, di
f g
ð
Þ
N
:
ð14Þ
In Figure 3 we plot how these quantities evolve for the three
dynamical systems as a function of the chosen parameters i and TS.
Unsurprisingly the linearity L and non-linearity NL are com-
plementary, when the first decreases the second increases. We inter-
pret this as a fundamental tradeoff in the way dynamical systems
process information: on one end of the scale they store the informa-
tion but then are linear, while on the other end they immediately
process the input in a highly non-linear way, but then have small
linear memory.
Influence of noise on the computation in dynamical systems. Real
life systems are often affected by noise. The capacities enable one to
quantify how the presence of noise decreases the computational
power of a dynamical system. Here we model noise by assuming
that there are two i.i.d. inputs u and v, representing the signal and
the noise respectively. The aim is to carry out computation on u. The
presence of the unwanted noise v will degrade the performances. The
origin of this degradation can be understood as follows. We can
define the Hilbert space HU? of fading memory functions of u2‘,
as well as the Hilbert space HV? of fading memory functions of v–‘.
The dynamical system belongs to the tensor product of these two
spaces xi [ HU?6HV?, and a basis of the full space is given by all
possible products of basis functions for each space. It is only over the
full space that Theorem 7 holds and that the capacities will sum to N.
However since v is unknown, the capacities which are measurable are
those which depend only on u. The corresponding basis functions
project onto a subspace, and therefore these capacities will not sum to
N. Furthermore, we expect that the more non-linear the system, the
larger the fraction of the total capacity which will lie outside the
subspace HU?. This is illustrated in Figure 4 (left panel). We also
consider in Figure 4 (right panel) the case where there are K input
signals of equal strength, and the aim is to carry out computation on
only one of them.
Discussion
In the present work we introduce a framework for quantifying the
information processing capacity of a broad class of input driven
dynamical systems, independently of their implementation. Aside
from the fundamental result that under general conditions dynam-
ical systems of the same dimensionality have globally the same com-
putational capacity, our framework also allows testing of hypotheses
about these systems.
We have already outlined several insights about the influence of
system parameters on the kind of computation carried out by the
dynamical system, but the potential implications go much further.
While previous work on memory capacities12–15 seemed to indicate
that non-linear networks tend to have bad (linear) memory prop-
erties and are thus unsuitable for long-term memory tasks, we now
conclude that information is not lost inside these networks but
merely transformed in various non-linear ways. This implies that
non-linear mappings with memory in dynamical systems are not
only realisable with linear systems providing the memory followed
by a non-linear observation function, but that it is possible to find a
system that inherently implements the desired mappings. Moreover,
Figure 3 | Trade-off between linear memory L[C] (red, dashed, left axis) and nonlinearity NL[C] (blue, full line, right axis), for ESN (r 5 0.95), logistic
map (r 5 2.5) and Reaction Diffusion systems as the parameters i and Ts are changed.
 
 
 
 
Figure 4 | Decrease of total capacity CTOT due to noise, for an ESN with r
5 0.95, for different values of i, corresponding to varying degree of
nonlinearity. In the left panel there are two i.i.d. inputs, the signal u and
the noise v. The horizontal axis is the input signal to noise ratio in dB
(SNR~10 log10
var u
ð Þ
var v
ð Þ
	

). The fraction of the total capacity which is
usable increases when the SNR increases, and decreases when the system
becomes more non-linear (increasing i). In the right panel there are a
varying number K of i.i.d. inputs with equal power, the total power being
kept constant as K varies. The capacity for computing functions of a single
input in the presence of multiple inputs is plotted. The black line indicates
the situation for a strictly linear system, where the capacity for a single
input should equal N/K.
www.nature.com/scientificreports
SCIENTIFIC REPORTS | 2 : 514 | DOI: 10.1038/srep00514
6
our framework allows an a-priori evaluation of the suitability of a
given system to provide a desired functional mapping, e.g., required
for a class of applications.
When considering sequences that are not i.i.d. (as is the case for
most real-world data), the construction of an orthonormal function
basis, although it exists in theory, is often not feasible. From this
perspective, the upper bound on total capacity is largely a theoretical
insight. However, by using either an over-complete set of functions
or an orthonormal but incomplete set, a lot of useful information
about the way a system processes a given type of input data or about
the type of processing required for a given task can still be gained.
An avenue for further research with high potential would be to
extend the current framework such that the systems can be adapted
or even constructed so as to provide certain functional mappings.
Also, a study of the effects of coupling several of these systems with
known computational properties on the global capacity of the whole
would be very interesting in the context of larger real-world dynam-
ical systems.
The framework is applicable under rather loose conditions and
thus to a variety of dynamical systems representative of realistic
physical processes. It is for instance now possible to give a complete
description of the functional mappings realised by neuronal net-
works, as well as the influence of network structure and even homeo-
static and synaptic plasticity on the realised functional mappings.
The application of the proposed measures to models for different
neural substrates such as the hippocampus or prefrontal cortex could
further elucidate their roles in the brain. Additionally, it allows for
instance to evaluate the suitability of the physical embodiment of
dynamical systems such as robots to inherently perform computa-
tion, thus creating an opportunity for simplifying their controller
systems. Finally, it provides a novel way to approach systems which
are not generally considered to explicitly perform computation, such
as intra-cellular chemical reactions or social networks, potentially
yielding entirely new insights into information processing and pro-
pagation in these systems.
1. Arbib, M. (Ed.) The handbook of brain theory and neural networks, second
edition. The MIT Press, Cambridge MA (2003).
2. Maass, W., Natschla¨ger, T. & Markram H. Real-time computing without stable
states: a new framework for neural computation based on perturbations. Neural
Comp. 14, 2531–2560 (2002).
3. Jaeger, H. & Haas H. Harnessing nonlinearity: predicting chaotic systems and
saving energy in wireless communication. Science 304, 78–80 (2004).
4. Verstraeten, D., Schrauwen, B., D’Haene, M. & Stroobandt, D. An experimental
unification of reservoir computing methods. Neural Networks 20, 391–403
(2007).
5. Vandoorne, K. et al. Toward optical signal processing using Photonic Reservoir
Computing. Optics Express 16(15), 11182–11192 (2008).
6. Appeltant, L. et al. Information processing using a single dynamical node as
complex system. Nat. Commun. 2, 468–472 (2011).
7. Paquot, Y. et al. Optoelectronic Reservoir Computing. Nat. Sci. Rep. In press
(2011).
8. Larger, L. et al. Photonic information processing beyond Turing: an
optoelectronic implementation of reservoir computing. Optics Express 20(3),
3241–3249 (2012).
9. Dale, K. & Husbands, P. The evolution of Reaction-Diffusion Controllers for
Minimally Cognitive Agents. Artificial Life 16, 1–19 (2010).
10. Pfeifer, R., Iida, F. & Bongard, J. C. New Robotics: Design Principles for Intelligent
Systems. Artificial Life 11, 1–2 (2005).
11. Hauser, H., Ijspeert, A. J., Fu¨chslin, R. M., Pfeifer, R. & Maass, W. Towards a
theoretical foundation for morphological computation with compliant bodies.
Biol. Cyber. 105, 355–370 (2011).
12. Jaeger, H. Short Term Memory in Echo State Networks. Fraunhofer Institute for
Autonomous Intelligent Systems, Tech. rep. 152 (2002).
13. White, O., Lee, D. & Sompolinsky, H. Short-term memory in orthogonal neural
networks. Phys. Rev. Lett. 92(14), 148102 (2002).
14. Hermans, M. & Schrauwen B. Memory in linear recurrent neural networks in
continuous time. Neural Networks 23(3), 341–355 (2010).
15. Gangulia, S., Huhc, D. & Sompolinsky, H. Memory traces in dynamical systems
Proc. Natl. Acad. Sci. USA 105, 18970–18975 (2008).
16. Buonomano, D. & Maass, W. State-dependent computations: Spatiotemporal
processing in cortical networks. Nat. Rev. Neurosci. 10(2), 113–125 (2009).
17. Boyd, S. & Chua, L. Fading memory and the problem of approximating nonlinear
operators with Volterra series. IEEE Trans. Circuits Syst. 32, 1150–1161 (1985).
18. Wiener, N. Nonlinear Problems in Random Theory. John Wiley, New York (1958).
19. Lee, Y. W. & Schetzen, M. Measurement of the Wiener kernels of a nonlinear
system by cross-correlation. Int. J. Control 2, 237–254 (1965).
20. Jaeger, H. The ‘‘echo state’’ approach to analysing and training recurrent neural
networks. Fraunhofer Institute for Autonomous Intelligent Systems, Tech. rep. 148
(2001).
21. Lukosevicius, M. & Jaeger, H. Reservoir computing approaches to recurrent
neural network training. Comp. Sci. Rev. 3, 127–149 (2009).
22. May, R. M. Simple mathematical models with very complicated dynamics. Nature
261, 459–467 (1976).
23. Buehner, M. & Young, P. A tighter bound for the echo state property. IEEE Trans.
Neural Netw. 17, 820–824 (2006).
24. Ozturk, M. C., Xu, D. & Principe, J. C. Analysis and Design of Echo State
Networks. Neural Comp. 19, 111–138 (2006).
25. Gray, P. & Scott, S. K. Sustained oscillations and other exotic patterns of behavior
in isothermal reactions. J. Phys. Chem. 89, 22–32 (1985).
26. Pearson, J. Complex patterns in a simple system. Science 261, 189–192 (1993).
27. Langton, C. G. Computation at the edge of chaos. Physica D 42, 12–37 (1990).
Acknowledgements
The authors acknowledge financial support by the IAP project Photonics@be (Belgian
Science Policy Office) and the FP7 funded AMARSi EU project under grant agreement
FP7-248311.
Author contributions
J.D., B.S. and S.M. designed research, J.D., D.V. and S.M. performed research, all authors
analyzed data and wrote the paper.
Additional information
Supplementary information accompanies this paper at http://www.nature.com/
scientificreports
Competing financial interests: The authors declare no competing financial interests.
License: This work is licensed under a Creative Commons
Attribution-NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/
How to cite this article: Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S.
Information Processing Capacity of Dynamical Systems. Sci. Rep. 2, 514; DOI:10.1038/
srep00514 (2012).
www.nature.com/scientificreports
SCIENTIFIC REPORTS | 2 : 514 | DOI: 10.1038/srep00514
7
