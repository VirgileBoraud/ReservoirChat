Edge of Stability Echo State Network
Andrea Cenia,∗, Claudio Gallicchioa
aDepartment of Computer Science, University of Pisa, Largo Bruno Pontecorvo, 3 - 56127, IT
Abstract
Echo State Networks (ESNs) are time-series processing models working under the Echo State Property (ESP)
principle. The ESP is a notion of stability that imposes an asymptotic fading of the memory of the input. On
the other hand, the resulting inherent architectural bias of ESNs may lead to an excessive loss of information,
which in turn harms the performance in certain tasks with long short-term memory requirements. With the
goal of bringing together the fading memory property and the ability to retain as much memory as possible,
in this paper we introduce a new ESN architecture, called the Edge of Stability Echo State Network (ES2N).
The introduced ES2N model is based on defining the reservoir layer as a convex combination of a nonlinear
reservoir (as in the standard ESN), and a linear reservoir that implements an orthogonal transformation. We
provide a thorough mathematical analysis of the introduced model, proving that the whole eigenspectrum
of the Jacobian of the ES2N map can be contained in an annular neighbourhood of a complex circle of
controllable radius, and exploit this property to demonstrate that the ES2N’s forward dynamics evolves
close to the edge-of-chaos regime by design. Remarkably, our experimental analysis shows that the newly
introduced reservoir model is able to reach the theoretical maximum short-term memory capacity. At the
same time, in comparison to standard ESN, ES2N is shown to offer an excellent trade-off between memory
and nonlinearity, as well as a significant improvement of performance in autoregressive nonlinear modeling.
Keywords:
Echo state networks, Reservoir computing, Memory capacity, Recurrent neural networks,
Input-driven systems, Edge of chaos.
1. Introduction
Recurrent neural networks (RNNs) [21] are computational models designed to extract features from data
with temporal structures. Applications ranges from speech recognition to classification of time series. The
most common way of training RNNs is via stochastic gradient descent methods, usually via the backprop-
agation through-time algorithm [46]. Unfortunately, these methods come with a significant computational
effort. Modern hardware unleashed the power of parallel computing techniques, allowing to reduce the com-
putational time of training deep learning models as RNNs. However, the price to pay is a massive energy
consumption. Moreover, a fundamental limitation of theoretical nature prevents RNNs to be fully exploited,
namely the vanishing/exploding (V/E) gradient issue [2]. In this regard, an appealing alternative is rep-
resented by Reservoir Computing (RC) [32, 29], a different paradigm of training RNNs dodging the V/E
while being computationally fast, and energy efficient. The flexibility of the RC paradigm offers a suitable
theoretical framework for computing with physical substrates [26, 41, 47, 45], for fast and scalable graph
neural networks models [13, 7], and for implementing digital twins of real-world nonlinear dynamical systems
[22]. The key idea of RC is to inject the input signal into a large random untrained recurrent layer, the
reservoir, from which a readout layer is optimised to fit the desired target signal. The Echo State Network
(ESN) [17, 19] provides a popular discrete-time class of RC machines. The name ESN recalls the imagery of
∗Corresponding author
Email addresses: andrea.ceni@di.unipi.it (Andrea Ceni), claudio.gallicchio@unipi.it (Claudio Gallicchio)
Preprint submitted to Elsevier
September 6, 2023
arXiv:2308.02902v2  [cs.LG]  3 Sep 2023
the input signal echoing and reverberating within the pool of neuronal activations in the reservoir, which in
turn serves as an high-dimensional representation of the past history of the input. Although gradients are
not backpropagated in the RC paradigm, the ESN’s forward dynamics are ruled by the very same equation
of conventional RNNs. Thus, ESNs inherit a problem which closely relates to the V/E gradient problem of
plain RNNs, namely the degradation of memory [16]. As revealed by previous studies [9, 39], the degradation
of memory is linked to the nonlinearity of the system in an inherent trade-off. Nonlinear computation and
short-term memory are two fundamental aspects of neural systems. Therefore, the existence of a trade-off
between them compels to design nonlinear RC systems able to retain as much memory as possible. In fact,
the memory capacity is a key feature to reach desirable results in certain learning tasks [5].
In this paper, we propose and analyse a novel RC model, called Edge of Stability Echo State Networks
(ES2N), that copes with the degradation of memory of nonlinear reservoir systems. We tackle the degradation
of memory of RNNs from a dynamical system perspective, framing the problem within the edge of chaos
hypothesis. The proposal of this hypothesis can be traced back at least to the late eighties [24, 34], where
it has been observed that extensive computational capabilities are achieved by adaptive systems whose
dynamics are neither chaotic nor ordered but somewhere in between order and chaos. In this paper we link
these ideas rooted in the study of adaptive systems to the case of partially randomised RNNs, as ESNs are.
ESNs work “properly” provided with the so called Echo State Property (ESP) [17, 49]. In coarse terms,
the ESP guarantees the ESN to possess a unique input-driven solution such that all the trajectories originat-
ing from different initial conditions (in the infinite past) synchronise with it (in present time). Following the
imagery, such a unique input-driven solution would represent the echo of the input signal from the infinite
past. The simplest known criterion to ensure the ESP is to impose the maximum singular value of the reser-
voir’s connections matrix to be less than one. This condition implies straight contraction in the phase space
at each time step. Therefore, any two different internal states of the RNN, when driven by the same input
sequence, will get closer and closer to each other as time flows ahead. Although stable, such a dynamical
system would have truly little margin to exploit the transient dynamics for computational purposes, due to
the straightforward contraction. An ideal situation would be for the RNN to stay in a regime of balance
between stable contractive dynamics and unstable chaotic dynamics, i.e. along the edge of chaos [3, 25].
This led the RC community to adopt the rule of thumb of setting the reservoir matrix to have spectral radius
approximately one. However, this rule of thumb inevitably overweights the contractive dynamics, especially
when considering the action of the input driving neurons towards the saturation regime. This overweighting
of the contractive dynamics reflects in the degradation of memory in the forward dynamics of ESNs. A
stronger condition would be to have the entire eigenspectrum of the reservoir matrix to be “around” the
complex unit circle. We study the eigenspectrum of ES2N, and prove in Theorem 3.3 that all the eigenvalues
of an ES2N lie within an annular neighbourhood of the complex unit circle. The radius of such neighbour-
hood can be tuned via a specific hyperparameter of the ES2N that we called the proximity hyperparameter.
Moreover, in the limit of small values of the proximity hyperparameter, we prove in Theorem 3.5 that the
ES2N dynamics narrowly hover over the edge of chaos while being on average over time in the stable regime,
hence the name Edge of Stability Echo State Network. Through experiments, we show the empirical advan-
tages of the proposed approach in terms of short-term memory capacity, memory-nonlinearity tradeoff, and
autoregressive time-series modeling.
The rest of this paper is organized as follows. In Section 2 we introduce the reader to the RC fundamentals
and the classical ESN model. In Section 3 we propose our ES2N model and provide a theoretical analysis
of its dynamics. Section 4 is dedicated to the experimental results, focusing on the short-term memory
capacity, the trade-off between nonlinearity and memory, and autoregressive time series modeling. Finally,
in Section 5, we discuss our findings, draw the conclusions, and point out interesting research directions to
explore further.
2. The ESN model
RC neural networks [8, 42] identify a class of fastly trainable RNNs, in which a non-linear untrained
dynamical layer is followed by a linear trainable readout component. In this contribution, we focus on the
2
Echo State Networks (ESNs) [19, 17] approach within RC, and we recall the well-established [20, 29, 28]
formulation of the leaky ESN model, which consists of a nonlinear reservoir layer made up of leaky recurrent
neurons followed by a linear readout. The equations read as follows:
x[t] =α tanh(ρWrx[t −1] + ωWinu[t]) + (1 −α)x[t −1],
(1)
z[t] =Wox[t].
(2)
The internal state x[t], input u[t], and output z[t] are, respectively, Nr-dimensional, Ni-dimensional and
No-dimensional vectors of real values. The nonlinearity is expressed by the element-wise applied hyperbolic
tangent function tanh, and the system is typically initialised in the origin, i.e., x[0] = 0. Matrices Wr, Win
are, respectively, the recurrent reservoir weight matrix and the input weight matrix, both randomly instan-
tiated and left untouched. In this paper, we initialise Win with i.i.d. random uniformly distributed entries
in (−1, 1), and Wr with i.i.d. normally distributed entries with zero mean and standard deviation
1
√Nr
.
This initialisation scheme for Wr ensures that, for large Nr, the spectral radius of Wr is approximately 1,
thanks to the circular law from random matrix theory [31]. This allows us to interpret the hyperparameter ρ
as the spectral radius (i.e., the largest eigenvalue in modulus) of the effective recurrent matrix, i.e. of ρWr.
While, the input scaling ω is an hyperparameter entitled to rescale the weight of the current input into the
reservoir dynamics. The leaky ESN owes its name to the presence of the hyperparameter α ∈(0, 1], the leak
rate, that is designated to modify the time scale of the ESN dynamics according to the input at hand. In
this paper we always consider the recurrent reservoir matrix Wr a fully connected weight matrix.
2.1. Training ESNs via ridge regression
Given a training set of input-target samples, {u[t], y[t]}t=1,...,T , we train a leaky ESN by means of
optimising the readout matrix Wo in order to solve the linear regression problem y[t] = Wox[t]. Usually,
this is achieved via ridge regression (or Tichonov regularisation) [29] by means of the following formula:
Wo = YXT (XXT + µI)−1,
(3)
where X is the matrix of dimension Nr × T, containing all the internal states x[t] of the ESN driven by the
input u[t] for k = 1, . . . , T, Y the matrix of dimension No × T, containing all the target values y[t], I is
the identity matrix of dimension Nr × Nr, and µ is the regularisation parameter. Throughout the paper we
denote the transposed of a matrix X with XT . Usually, when dealing with regression tasks, a number of
initial time steps are discarded for the computation of eq. (3) to allow the reservoir to “warm up”. This is
to ensure that the ESN transient dynamics are washed out, so that the internal dynamics of the ESN get
linked with the driving input.
2.2. Echo state property
ESNs work under the fundamental assumption of the ESP, a condition ensuring a unique stable input-
driven response [17]. Roughly speaking, the ESP guarantees that the internal state x[t] is uniquely determined
by the entire past history of the input signal. The easiest condition to ensure the ESP is to set a reservoir
such that
∥ρWr∥< 1,
(4)
where, for a given matrix M ∈RN×N, ∥M∥denotes the matrix norm induced by the Euclidean norm in
RN, or equivalently the maximum singular value of M. For the rest of the paper, we will always use ∥·∥to
denote either the Euclidean norm on RN, or the matrix norm induced by that, depending on whether the
argument of the norm is respectively a vector or a matrix.
The condition expressed by eq. (4) implies contraction of the internal states at each time step for whatever
input; that is a strong stability condition implying a Markovian state space organisation [12]. However,
stability is not the only property we crave from an ideal recurrent neural system. We strive for a stable
dynamical system that is able to recall the information conveyed from the external input for as long as
possible.
In other words, we want to keep the ESP while staying close to the border of its domain of
3
existence. This led the RC community to adopt the less restrictive rule of thumb of setting the spectral
radius of ρWr approximately (or often slightly less than) 1, e.g. initialising Wr to have spectral radius 1 and
then setting setting the hyperparameter ρ ≈1. Intuitively, ρ controls the amount of nonlinearity into the
reservoir and the contribution of the past activations. A small ρ promotes stable dynamics at the expenses
of forgetting faster the past internal activations, thus amplifying the fading memory property. However, as
previous works demonstrated [49], the only constraint of ρ < 1 is not generally sufficient to guarantee the
ESP in an input-driven ESN; and, due to the presence of the external input driving the dynamics, it is not
even necessary. In fact, the ESP is not a property of the reservoir alone, but rather of the reservoir plus the
forcing input. Some efforts to comprehend the input-dependence in the analysis of the ESP in RC have been
exerted in [30, 44, 11, 6]. As demonstrated in the literature, the ESP might hold even with large values of
ρ, as long as the amplitude of the inputs (or the hyperparameter ω) are large enough to counterbalance the
effect. As a consequence, ρ should be optimised in synergy with the input scaling ω.
Although ESNs present many advantages, it remains unclear how to tune a priori an ESN close to the
edge of chaos keeping the benefits of a stable nonlinear computational system. Driven by the need to reconcile
the properties of stability, nonlinearity, long-term memory, and ease of computation, we introduce in the
next section a new RC architecture.
3. The ES2N model
We propose a variant of ESN, called Edge of Stability ESN (ES2N), whose equations with linear readout
read as follows:
x[t] =β ϕ(ρWrx[t −1] + ωWinu[t]) + (1 −β) Ox[t −1],
(5)
z[t] =Wox[t],
(6)
Where x[t], u[t], z[t], Wr, Win, and Wo are as in eq. (1) and eq. (2). Here, in addition, the matrix O is a
randomly generated orthogonal matrix, and β ∈(0, 1] a real valued hyperparameter that we call proximity.
All the random orthogonal matrices in the experiments of this paper have been obtained by generating first
a random matrix D of the desired dimension Nr × Nr with i.i.d. uniformly random entries in (−1, 1), hence
performing a QR decomposition of D = QR, and taking the resulting Nr × Nr orthogonal matrix Q as
the random orthogonal matrix O in eq. (5). Although apparently the hyperparameters α of a leaky ESN
of eq. (1) and β of an ES2N of eq. (5) share the same position in the equation, they play quite a different
role in the dynamics of the RNNs, as it will be evident after our theoretical and experimental analysis. In
fact, while the value of α in eq. (1) is intended to slow down the speed of the reservoir dynamics relative
to those of the input signal [20], as we will see later in this section, the role of β in eq. (5) is to determine
the proximity of the reservoir dynamics to the edge of chaos. With the setting in eq. (5), the reservoir of
the ES2N results into a convex combination of a standard nonlinear input-driven reservoir (first term on the
right hand side) and a linear orthogonal input-free reservoir (second term on the right hand side). Note that
for β →0, and α →0, the ES2N model of eq. (5) and the leaky ESN model of eq. (1) both coincide to the
standard (non-leaky) ESN model.
3.1. Edge of chaos in ES2N
In this section we present a detailed mathematical analysis of the structure of the eigenspectrum for
the newly introduced ES2N model, and its maximum local Lyapunov exponent. We start introducing our
notation. Denoting the ES2N reservoir map as G(u, x) = βϕ(ρWrx + ωWinu) + (1 −β)Ox, where we use
ϕ to generally indicate the nonlinear activation function, then the corresponding Jacobian reads as follows:
∂G
∂x (u, x) = β D(u,x) ρWr + (1 −β) O,
(7)
where we defined the following diagonal matrix
D(u,x) := diag(ϕ′(ρWrx + ωWinu)),
(8)
4
whose supremum of its norm we denote as follows
γ := sup
u,x ∥D(u,x)∥.
(9)
Note that whenever |ϕ′| ≤1, then it holds that γ ∈[0, 1]. Considering a specific input-driven trajectory
{(u[t + 1], x[t])}t=0,...,T −1, we denote as
J[t] := ∂G
∂x (u[t + 1], x[t])
(10)
the Jacobian map of eq. (7) evaluated along the input-driven trajectory. Finally, we will denote the maximum
singular value of the matrix ρWr as follows
σ := ∥ρWr∥.
(11)
We start our analysis providing a sufficient condition for the ESP to hold for the ES2N model, as stated
in the proposition below.
Proposition 3.1. Let us assume that |ϕ′| ≤1, e.g., ϕ = tanh. If σ < 1 then the ES2N has the ESP for all
inputs.
Proof.
Following the same proof of [17, Proposition 3], for an ES2N to have the ESP for all input, it is sufficient to
demonstrate that ∥∂G
∂x (u, x)∥< 1 for all x, u. The following holds
∥∂G
∂x (u, x)∥≤β∥D(u,x)∥∥ρWr∥+ (1 −β)∥O∥≤βσ + (1 −β) = 1 −β(1 −σ).
The first inequality is the triangle inequality, the second holds since ∥D(u,x)∥≤1 for all activation functions
such that |ϕ′| ≤1, and because of the isometric property of orthogonal matrices which implies ∥O∥= 1.
Now the thesis holds by hypothesis since σ < 1 =⇒1 −β(1 −σ) < 1.
□
Remark 3.1. Proposition 3.1 informs us that, regardless of the particular value of β ∈(0, 1], the ES2N
model owns the ESP under the same contractivity condition of a standard ESN.
In an ES2N with a relatively low value of β, the dependence from the hyperparameters ρ and ω is
attenuated. In fact, the forward dynamics of such an ES2N evolve by design close to the edge of chaos.
Since the Jacobian of the ES2N map in eq. (7) is itself a convex combination, we can weight more the
orthogonal part of the ES2N approaching the hyperparameter β to 0, which by a continuity argument tune
the eigenvalues of the Jacobian to stay closer to the unitary circle, pretty much regardless of the spectral
radius of the reservoir and the scaling of the input matrix (provided that the spectral radius is bounded). We
formalise this intuition exploiting the Bauer-Fike theorem [1], that we report here for ease of comprehension.
Theorem 3.2 (Bauer-Fike). Let A be a diagonalisable matrix, and let V be the eigenvector matrix such
that A = VΛV−1 where Λ is the diagonal matrix of the eigenvalues of A. Let E be an arbitrary matrix of
the same dimension of A. Then, for all µ eigenvalues of A + E, there exists an eigenvalue λ of A such that
|λ −µ| ≤∥V∥∥V−1∥∥E∥.
(12)
The Bauer-Fike theorem allows us to derive the following characterisation of the eigenspectrum of the
Jacobian of the ES2N map.
5
Theorem 3.3. Let us consider an ES2N model whose state update equation is given by eq. (5), and recall
the definitions of σ and γ of eq. (11) and eq. (9). Then, the eigenspectrum of the Jacobian of the ES2N map
is confined in the annular neighbourhood of radius βγσ of the circle centered in the origin of radius 1 −β.
In formulas, for each eigenvalue µ of the Jacobian matrix of eq. (7) there exists a θ ∈[0, 2π) such that
|(1 −β)eiθ −µ| ≤βγσ.
(13)
Proof.
Define A = (1 −β)O, and E = βD(u,x)ρWr, so that the Jacobian of the ES2N model is ∂G
∂x (u, x) = A + E.
The matrix O is orthogonal, hence there exists an unitary matrix V such that O = VΛV−1, where Λ is
the diagonal matrix of the eigenvalues of O. In particular, each eigenvalue of O is of the kind eiθ, for some
argument θ ∈[0, 2π), due to the orthogonality of O. Therefore, A = (1 −β)VΛV−1, and all eigenvalues
of A have the form (1 −β)eiθ, for some argument θ ∈[0, 2π). In other words, all the eigenvalues of A lie
on the complex circle centered in the origin with radius (1 −β). Now, since V is unitary, we have that
∥V∥= ∥V−1∥= 1. Therefore, eq. (12) tells us that for each µ eigenvalue of ∂G
∂x (u, x) there exists a complex
number λ = (1 −β)eiθ such that
|(1 −β)eiθ −µ| ≤∥βD(u,x)ρWr∥≤βγ∥ρWr∥= βγσ.
(14)
This implies that each eigenvalue of ∂G
∂x (u, x) must be inside the circle of radius 1 −β + βγσ, and outside
the circle of radius 1 −β −βγσ, both centered in zero, which is the thesis.
□
A similar argument leads to the following characterisation of a leaky ESN’s eigenspectrum.
Corollary 3.4. Let us consider a leaky ESN model whose state update equation is given by eq. (1). Then,
the eigenspectrum of the Jacobian of the leaky ESN map is confined in a neighbourhood of radius αγσ of the
complex number (1 −α, 0). In formulas, for each eigenvalue µ of the Jacobian matrix of a leaky ESN’s map
it holds
|(1 −α) −µ| ≤αγσ.
(15)
Proof. The proof follows the same steps of the proof of Theorem (3.3), replacing β with α, and O with the
identity matrix I. Now, since A = (1 −α)I has a unique eigenvalue 1 −α with multiplicity the dimension of
A, then eq. (13) reads |(1 −α) −µ| ≤αγσ, which is the thesis.
□
Remark 3.2. The key feature that differentiates an ES2N model from a leaky ESN model is that the latter
has an eigenspectrum that shrinks towards 1 (for small values of α), while the former has an eigenspectrum
that tends to dispose along the unitary circle (for small values of β). This spread of the eigenspectrum can add
richness and diversity to the resulting reservoir dynamics, while the “collapse” of the eigenspectrum towards
1 might harm the expressiveness of the recurrent neural dynamics in some tasks like retrieving memory. In
Fig. 1 the eigenspectrum of the Jacobian of the ES2N model and leaky ESN model are plotted for various
combinations of the hyperparameters ρ, ω, β and α.
In the context of autonomous dynamical systems, the notion of Maximum Lyapunov Exponent (MLE)
is widely adopted to detect whether a system is sensitive to initial conditions [43]. The idea is to consider
two infinitesimally close initial conditions and measure the average (over time) maximum expansion rate of
the distance between those two initial conditions. Although there exists a spectrum of Lyapunov Exponents,
exactly one for each dimension of the system, the maximum among them is the most important. If the MLE
is less than zero, it means that any perturbation of an initial condition gets damped on average over time.
On the contrary if the MLE is greater than zero, then there exists at least one direction in tangent space
along which the perturbation gets magnified on average over time; this characteristic expansion behaviour
of the linearised system is often one of the basic ingredients for the definition of chaotic dynamics [33, 40].
6
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
immaginary axis
=10, 
=2
=0.1
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
=10, 
=0
=0.01
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
=1, 
=0
=0.1
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
=1, 
=0
=0.5
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
=1, 
=2
=0.5
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
=1, 
=0
=0.9
1
0
1
real axis
1.0
0.5
0.0
0.5
1.0
immaginary axis
=0.1
1.0
0.5
0.0
0.5
1.0
real axis
1.0
0.5
0.0
0.5
1.0
=0.01
1.0
0.5
0.0
0.5
1.0
real axis
1.0
0.5
0.0
0.5
1.0
=0.1
1.0
0.5
0.0
0.5
1.0
real axis
1.0
0.5
0.0
0.5
1.0
=0.5
1.0
0.5
0.0
0.5
1.0
real axis
1.0
0.5
0.0
0.5
1.0
=0.5
1.0
0.5
0.0
0.5
1.0
real axis
1.0
0.5
0.0
0.5
1.0
=0.9
Figure 1: Eigenspectrum of the Jacobian of the proposed ES2N model of eq. (5) (in red) and the leaky ESN model of eq. (1)
(in green) for various combinations of ρ, ω, β, and α. In black the unitary complex circle. The input driving the reservoirs is
set to the constant 1, which is then scaled by the value of the hyperparmameter ω.
Therefore, in the literature the edge of chaos is often defined as the locus of parameters where the MLE
is exactly zero [4]. However, when we allow an external input to drive the dynamics, as usual in RNNs,
the definition of the Lyapunov exponents becomes input-dependent. In the literature [42, 27], one way to
assess the degree of regularity of the dynamics of an input-driven system is to compute its Maximum Local
Lyapunov Exponent (MLLE) on a given input-driven trajectory, whose definition is provided below.
Definition 3.1. Let us consider an input-driven system of equation x[t] = G(u[t], x[t −1]), e.g. the one
defined by the ES2N’s map whose state update equation is given by eq. (5). Let be given an initial internal
state condition x[0] at time t = 0, and a sequence of T inputs u[1], . . . , u[T]. Then the input-driven trajectory
{(u[t + 1], x[t])}t=0,...,T −1 is well defined via eq. (5). Then, the MLLE of the ES2N on such input-driven
trajectory is defined as follows
Λ :=
max
n=1,...,Nr
1
T
T −1
X
t=0
log(rn[t])
(16)
where Nr is the dimension of the reservoir matrix Wr, and rn[t] is the square root of the modulus of the nth
eigenvalue of the symmetric real matrix J[t]J[t]T , with J[t] defined as in eq. (10).
Roughly speaking, eq. (16) gives us an estimation of the maximum expansion rate, averaged over the
window of time [0, T], locally to a given input-driven trajectory. On the same vein of autonomous dynamical
systems, a Λ < 0 denotes local contractive stable dynamics, Λ > 0 is a blueprint of chaotic dynamics since
it implies local exponential divergence of trajectories, while Λ = 0 characterises the edge of chaos.
Below we provide an estimation of the MLLE for the ES2N model.
Theorem 3.5. Let us consider an ES2N model whose state update equation is given by eq. (5). Then, for all
time lenghts T, initial internal state x[0], and inputs u[1], . . . , u[T], the MLLE defined by eq. (16) is bounded
as follows
log
 1 −β(γσ + 1)

≤Λ ≤log
 1 + β(γσ −1)

,
(17)
In particular, in the first order approximation of small values of β it holds that
Λ ≈−β.
(18)
Proof. Thanks to Theorem 3.3 we know that the modulus of each eigenvalue µ of the Jacobian J[t] is
bounded as 1 −β −βγσ ≤|µ| ≤1 −β + βγσ, regardless of t. Each eigenvalue ν of J[t]J[t]T is bounded as
(1 −β −βγσ)2 ≤|ν| ≤(1 −β + βγσ)2, regardless of t. In particular, the square root rn[t] of the modulus
of the nth eigenvalue of the matrix J[t]J[t]T is bounded as 1 −β −βγσ ≤rn[t] ≤1 −β + βγσ, regardless of
n ∈{1, . . . , Nr}, and t. Therefore, it follows (17) from Definition 3.1. In particular, for small values of β ≈0
7
we have a tight squeeze that justifies the estimation of Λ as the arithmetic mean of the first order approxima-
tion of the left bound, log
 1−β(γσ +1)

≈−β(γσ +1), and the right bound, log
 1+β(γσ −1)

≈β(γσ −1),
which results in the approximation Λ ≈−β.
□
Remark 3.3. Theorem 3.5 implies that, in an ES2N, we can tune the recurrent neural dynamics towards
the edge of chaos via tuning the proximity hyperparameter β, regardless of the input. More precisely, for
decreasing values of β the bounds of eq. (17) become tighter, and in the approximation of small values of β
the ES2N model gets close to the edge of chaos (Λ ≈0) while being on average over time in the stable regime
characterised by Λ < 0.
4. Experiments
In this section we present our experimental analysis on the ES2N model, in comparison with well-
established approaches from the ESN literature. Specifically, the short-term memory capacity is analysed in
Section 4.1, while in Section 4.2 we investigate the trade-off between nonlinearity and memory. Finally, in
Section 4.3 we focus on the autoregressive time series modeling, using the multiple superimposed oscillators
case as a reference task.
4.1. Memory Capacity
The Memory Capacity (MC) task was introduced by Jaeger in [18] to measure the short-term memory
ability of an ESN to retrieve from the pool of internal neuronal activations the past input history. The task
later became very popular for analysing the computational properties of RC-based models [35]. A systematic
analysis of MC varying various hyperparameters of ESNs can be found in [10], in which the authors also
propose gradient descent based orthogonalization procedures to increase the MC. In the following, we set an
MC experiment similarly to [14].
We consider reservoirs of Nr = 100 neurons, with a linear readout trained by ridge regression. The input
u[t] is an i.i.d. signal uniform in [−0.8, 0.8] of discrete-time length T = 6000. The first 5000 time steps are
exploited for training (excluding the very first 100 time steps to warm up the reservoir), and the last 1000
time steps are left for test. The task is to reproduce in output a signal zk[t] that is a delayed version of k
time steps of the input signal, i.e., to have zk[t] as close as possible to u[t −k]. The MC score is defined as
follows:
MC =
∞
X
k=1
MCk,
(19)
where MCk is the squared correlation coefficient between the output zk[t] and the target u[t −k], defined as
follows:
MCk =

<
 zk[t]−< zk[t] >t
 u[t −k]−< u[t −k] >t

>t
2
<

zk[t]−< zk[t] >t
2
>t <

u[t −k]−< u[t −k] >t
2
>t
.
(20)
Angular bracket in eq. (20) denotes average over time, and are calculated with regard to the test session, i.e.
for t = 5001, . . . , 6000. Moreover, the calculation of the sum in eq. (19) has been truncated to k = 200, i.e.
twice the reservoir size. This choice of truncating the sum in eq. (19), widely used in the RC community,
makes sense considering that the maximum MC achievable by an Nr-dimensional reservoir is Nr; for a proof
of this fact see [18, Proposition 2].
For the calculations, both ESNs and ES2N have been set with a spectral radius of 0.9, and an input
scaling of 0.1. This setting has been tested as good for ESNs in previous works [14, 38]. Keeping fixed those
hyperparameters, the leak rate α for the ESN model, and the proximity hyperparameter β for the ES2N
model, have been varied in (10−3, 1). Precisely, we used the same grid of 50 values for α and β, generated via
the formula a10−s, with a random uniform in (0.1, 1), and s random uniform in {0, 1, 2}. This methodology,
8
0.00
0.25
0.50
0.75
1.00
 (green),  (red)
0
20
40
60
80
100
Memory Capacity
ES2N
leaky ESN
0
5
10
15
20
25
30
35
0
1
Delay=30
leaky ESN
ES2N
ground truth
0
5
10
15
20
25
30
35
Time steps
0.5
0.0
0.5
Delay=98
Figure 2: Left: MC values of eq. (19) averaged over 10 trials for various values of α (for the ESN), and β (for the ES2N).
Right: Output signals of ESN with α = 1 (green), and ES2N with β = 0.05 (red), over the ground truth signal (dashed), for
the two cases of delay k = 30 and k = 98.
while covering a large range of values for the leak rate of ESNs, also ensures to properly explore values close
to zero where the ES2N model presents an interesting behaviour.
Both ESNs and ES2Ns have been run for 10 different initialisations for each delay k, and the computed
MC has been averaged over these trials in order to have statistical significance. In the left picture of Figure 2
the computed MC values for both ESN and ES2N are plotted. The MC of ES2N exhibits a peculiar nonlinear
dependence on the proximity hyperparameter β, peaking around β = 0.05. On the contrary, the MC of ESN
appears monotonic in α, reaching the highest value at α = 1. As expected, for β values approaching to
1, the ES2N’s MC and the ESN’s MC curves overlap on the right part of the graph. In the right plots of
Figure 2, two examples with delay k = 30 and k = 98 are reported for leaky ESN and ES2N (each with its
best hyperparameter setting found) highlighting the supremacy of the ES2N model over a standard ESN.
Note that recalling the input signal up to 98 time steps in the past is challenging for a reservoir of 100 units.
The computed mean MC values for the optimal α and β are reported in Table 1 along with their empirical
standard deviations.
Additionally, we computed the MC of linear ESN, i.e. the model of eq. (1) with α = 1 and the identity
function as ϕ (called linearESN in Table 1); the MC of an ESN with an orthogonal reservoir, i.e.
the
model of eq. (1) with α = 1 and a randomly generated orthogonal matrix Wr (called orthoESN in Table
1); and the MC of a linear ESN with a specific orthogonal structure that realises a circular shift (called
linearSCR in Table 1), i.e. the model of eq. (1) with α = 1 and Wr with nonzero elements in the lower
subdiagonal and the upper-right corner, all filled with 1.1
All the other hyperparameters of linearESN,
orthoESN, and linearSCR have been set identically to those specified previously. Linear ESNs are known to
perform better than nonlinear ESNs in the MC task. However, as evident from the large standard deviation
of linearESN in Table 1, linear models occasionally give very poor performance. One way to stabilise linear
ESN’s performance is to employ an orthogonal matrix as reservoir. We used linearSCR as benchmark because
theoretical results (see [18, Proposition 4] and [35, Theorem 1]) guarantee for it optimal performance on the
MC task. Remarkably, the ES2N model can get very close to the MC optimal value (of 100, for a reservoir
of 100 neurons) with a noticeably narrow standard deviation, almost matching the MC of linearSCR, see
Table 1.
Finally, we plot in the left picture of Figure 3 the squared correlation coefficient MCk for all k = 1, . . . , 200,
for all of the five considered models: leaky ESN (with α = 1, the best found), linearESN, orthoESN,
linearSCR, and ES2N (with β = 0.05, the best found). These MCk values are not averaged over more trials.
1The linearSCR model implements the same reservoir topology used in the Simple Cycle Reservoir in [35], from which the
name linearSCR.
9
Model
MC
leaky ESN
30.40 ± 3.76
linearESN
49.35 ± 17.13
orthoESN
89.42 ± 1.50
linearSCR
99.09 ± 0.01
ES2N
98.43 ± 0.11
Table 1: Mean and standard deviation
of the MC computed over 10 different
initialisations of reservoir models of 100
neurons.
Leaky ESN is with α = 1,
ES2N is with β = 0.05
0
25
50
75
100
125
150
175
200
k (delay)
0.0
0.2
0.4
0.6
0.8
1.0
MCk
ES2N (
= 0.05)
orthoESN
leaky ESN (
= 1)
linearESN
linearSCR
0
5
10
15
20
25
30
35
(Test) time steps
0.8
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
Delay=1.    Output vs Target signal
orthoESN
ES2N
ground truth
Figure 3: Left: MCk values for delays k = 1, . . . , 200, for all
the 5 models.
Right: orthoESN versus ES2N in the simplest case of k = 1.
However, apart from linearESN (which sometimes fails), the MCk values of all the other four models are
relatively insensitive to the random initialisation. As evident from the left plot of Figure 3, ES2N presents
an MC curve (red) particularly close to the optimal one of linearSCR (black), while we note that despite the
large MC of the orthoESN model, the orthoESN never excels in reconstructing the delayed version of the
input, even for the simplest case of k = 1. In the right plot of Figure 3 a comparison between orthoESN and
ES2N outputs is plotted, for the case of delay k = 1. The ES2N (red) is able to perfectly reconstruct the
target (black dashed line), while the orthoESN (blue) seems to struggle. Interestingly, the orthoESN trades
its poor reconstruction of the delayed signal with the ability to mildly correlate its output with the target
for very large delays, and given the definition of MC = P
k MCk, this results in an overall large value of
memory capacity.
4.2. Memory-nonlinearity trade-off
From previous RC literature, it is well known the existence of a trade-off between the strength of nonlin-
earity of a dynamical reservoir system and its short-term memory abilities [9]. In [16] the authors propose
a task with the aim of measuring this trade-off. The task consists of extracting from an i.i.d. uniform input
signal u[t] in [−1, 1] a target signal of the form y[t] = sin(ν ∗u[t −τ]); here ν quantifies the nonlinearity
strength, while τ measures the memory depth. We use this task to benchmark the memory-nonlinearity
trade-off for various combinations of τ and ν comparing ES2N against leaky ESN, and linearSCR. For this
experiment, an input signal of length 6000 has been generated, of which the first 5000 for training (excluding
the very first 100 steps), and the remaining 1000 for testing. The metric used to evaluate the task is the
NRMSE between target y(t) and output z(t) in the test session, defined as follows:
NRMSE(y, z) =
s
< ∥y[t] −z[t]∥2 >t
< ∥y[t]−< y[t] >t∥2 >t
,
(21)
where ∥v∥represents the Euclidean norm of a vector v, and angular brackets means the average over time.
According to this metric, the lower the better.
In the following experiment, we considered a grid of values of (log(ν), τ), with τ ∈[1, 20], with a step of
1, and log(ν) ∈[−1.6, 1.6] with a step of 0.1. For each pair (log(ν), τ), we ran 100 instantiations of leaky
ESN, linearSCR, and ES2N. For each run we generated randomly the following hyperparameters
• uniformly random input scaling in [0.2, 6],
• uniformly random spectral radii in [0.1, 3],
• α (for leaky ESN) and β (for ES2N) generated via the formula a10−s, with a uniformly random in
(0.1, 1), and s uniformly random in {0, 1}, so that they vary in (10−2, 1).
10
In Figure 4 are plotted the best NRMSE found on test on a coloured scale from black (NRMSE = 0) to
yellow (NRMSE = 1 or greater), the lower the better. Results in Figure 4 show that ES2N significantly
outperforms both leaky ESN and linearSCR. Note that, linearSCR starts to increasingly underperform as
soon as log(ν) > 0, i.e.
in the region where nonlinearity is needed, while ES2N is able to retrieve the
information for much stronger nonlinearly transformed input signals. This indicates that the ES2N model
can truly exploit nonlinearity in the computation. On the other hand, leaky ESN is able to reconstruct the
input signal in the strong nonlinear regime, but only for very small delays. In particular, for the challenging
case of log(ν) > 1 (strong nonlinearity), leaky ESN’s performance significantly degrades already at τ = 4,
with NRMSE values always above 0.5. On the contrary, the ES2N model obtains NRMSE values always
below of 0.5 up to delays of τ = 16 in the strong nonlinearity regime of log(ν) > 1.
In Figure 5 are plotted the output signals in the test session for the best hyperparameters found on the three
models ES2N, linearSCR, and leaky ESN, in comparison with the ground truth target signal for the case of
strong nonlinearity with log(ν) = 1.3 and τ = 10. These results demonstrate how ES2N can conciliate the two
contrasting properties of having a large memory capacity and the ability to perform nonlinear computations.
1.5
1.0
0.5
0.0
0.5
1.0
1.5
Nonlinearity strength log( )
2
4
6
8
10
12
14
16
18
20
Delay 
NRMSE optimised leaky ESN
0.0
0.2
0.4
0.6
0.8
1.0
1.5
1.0
0.5
0.0
0.5
1.0
1.5
Nonlinearity strength log( )
2
4
6
8
10
12
14
16
18
20
Delay 
NRMSE optimised linearSCR
0.0
0.2
0.4
0.6
0.8
1.0
1.5
1.0
0.5
0.0
0.5
1.0
1.5
Nonlinearity strength log( )
2
4
6
8
10
12
14
16
18
20
Delay 
NRMSE optimised ES2N
0.0
0.2
0.4
0.6
0.8
1.0
Figure 4: Results of the memory-nonlinearity trade-off task explained in Section 4.2.
NRMSE values ranging from black
(NRMSE= 0) to yellow (NRMSE= 1 or greater) are plotted for various combinations of delay τ and nonlinearity strength ν.
Left: best NRMSE values after a random search of 100 trials on a leaky ESN. Centre: best NRMSE values after a random
search of 100 trials on a linearSCR. Right: best NRMSE values after a random search of 100 trials on our proposed ES2N
model.
0
5
10
15
20
25
30
Test time steps
1
0
1
Test predictions for best hyperparameters with log( )=1.3 and =10.
optimised leaky ESN (NRMSE
0.81)
optimised linearSCR (NRMSE
0.82)
optimised ES2N (NRMSE
0.12)
ground truth
Figure 5: Input u[t] is i.i.d. uniformly sampled in [−1, 1]. Target is a strong nonlinear transformation of a 10-delayed version of
the input given by the function sin(νu[t −10]) with ν ≈3.67 represented in dashed line. The best hyperparameters found for
the setting of log(ν) = 1.3 and τ = 10 are the following in the form of (input scaling, spectral radius, α or β): (0.43, 1.03, 0.98)
for leaky ESN, (0.35, 0.49, 1.0) for linearSCR, and (1.44, 0.10, 0.18) for ES2N.
4.3. Multiple superimposed oscillators in auto-regressive mode
The Multiple Superimposed Oscillators pattern generation (MSO) is a popular benchmark task where an
RNN is trained to generate autonomously (i.e. without the input driving the dynamics) a one-dimensional
signal made from the superpositions of a few incommensurate sines. This is achieved providing in input the
target signal during the training phase, then, once the readout matrix is trained, closing the loop in the
testing phase via u[t] = z[t] in eq. (5), i.e. self-driving the ESN dynamics with its own generated output. In
these auto-regressive tasks, the ridge regression training of the readout matrix can be regarded as a teacher
forcing training strategy. In the literature [48, 37, 15, 36, 23] the MSO task with different numbers of sine
11
waves has been inspected. In all of the mentioned studies the frequencies of the sine waves were taken from
the same set: ν1 = 0.2, ν2 = 0.311, ν3 = 0.42, ν4 = 0.51, ν5 = 0.63, ν6 = 0.74, ν7 = 0.85, and ν8 = 0.97. In
this section, we only consider the case of 8 frequencies, the most challenging one among those mentioned
above, which we will denote concisely with MSO8. Thus, the target signal takes the following form:
y[t] =
8
X
i=1
sin(νit).
(22)
The MSO8 signal of eq. (22) is then rescaled in order to have zero mean and be bounded in (−1, 1). From
now on, we will refer to y[t] as the normalised signal. The function of eq. (22) has an almost-period of
τP = 6283 time steps.2 We use 6383 time steps for training, i.e. a whole “period” of 6283 steps excluding
the first 100 time steps to wash out transient dynamics. In this phase, the target signal (the teacher) is
injected into the reservoir as the input signal, i.e. u[t] = y[t] for t = 1, . . . , 6383. During the training phase,
a small Gaussian noise of zero mean and standard deviation of 10−4 has been added to the argument of the
tanh in order stabilise the dynamics. Therefore, the ES2N’ state-update equation during training was
x[t] = βϕ(Wrx[t −1] + Winu[t] + η[t]) + (1 −β)Ox[t −1],
(23)
with η ∈N(0, 10−4) the source of Gaussian noise. We set zero regularisation, i.e. µ = 0 in eq. (3). Thus,
we train the linear readout to reproduce the target signal. Then, we close the loop feeding back the output
into the reservoir in place of the external input, i.e. u[t] = z[t] for t ≥6384. From this moment on, the RNN
runs autonomously (without noise).
4.3.1. Random hyperparameter search
First of all, we accomplished a random search to optimise the hyperparameters of both leaky ESN and
ES2N. Varying the reservoir size revealed that, as expected, larger models leads to better results, for both
leaky ESNs and ES2Ns. We considered leaky ESNs of 600 reservoir neurons, and ES2Ns of 100 reservoir
neurons. We considered leaky ESNs with reservoir size 6 times larger of the ES2N’s, in order for the leaky
ESN to get competitive performance on the challenging MSO8 task. We ran 10000 different initialisations
for both leaky ESN and ES2N with uniformly random generated hyperparameters as follows:
• uniformly random spectral radius ρ ∈[0.8, 1.2]
• uniformly random input scaling ωin ∈[0, 0.4]
• (for leaky ESN) uniformly random α ∈(0.1, 1)
• (for ES2N) uniformly random β ∈(0.01, 0.1).
We also tried to vary the reservoir connectivity, but it did not influence the performance. Thus, we used fully
connected reservoirs. In the training phase, a small Gaussian noise with zero mean and standard deviation
of 10−4 has been introduced in the state-update equation as in eq. (23), for both leaky ESN and ES2N. To
evaluate the performance we compute the NRMSE as defined in eq. (21) for 300 time steps in the testing
phase, i.e. for t = 6384, . . . , 6684. We refer the reader to Appendix A for the visualisation of the data
obtained from this random search. In summary, from this search emerged that leaky ESNs need spectral
radii strictly around 1 for good results on the MSO8 task. On the contrary, ES2Ns are quite insensitive
on the choice of the spectral radius. However, by design ES2N reflects the insensitivity on the spectral
radius on a dependence on the proximity hyperparameter β, which for this task reaches its optimum around
β = 0.03. Apart from that, the most influential hyperparameter turned out to be the input scaling, in line
with previous works. In particular, we found the optimal combination of ρ = 0.99, ωin = 0.05, and α = 0.9
for leaky ESNs. While for ES2N ωin = 0.11, and β = 0.03 (regardless of ρ). All the NRMSEs computed
2More precisely, for the normalised signal, it turns out that <
y[t + τP ] −y[t]
>t=0,...,τP ≈0.024 ± 0.02.
12
in this random search are reported in the histograms in the left plots of Figure 6. The mean and standard
deviation of NRMSE for leaky ESN is 1.44 ± 13.34, while for the ES2N is 0.05 ± 0.11. The difference of
mean NRMSE values highlights the supremacy of ES2N over the leaky ESN on the MSO8 task, despite a 6
times smaller reservoir size. As a side result, this random search revealed how ES2Ns are characterised by a
wider “good” hyperparameter region compared to leaky ESNs, testified via the more than 120 times larger
standard deviation of leaky ESN over ES2N.3
4.3.2. Stability in the long run
In this section, we compare the quality of the learned signal in the long run. For the purpose, we fixed
the hyperparameters in their respective optimal setting, precisely we set (ρ = 1, ωin = 0.11, β = 0.03) for
ES2Ns, and (ρ = 0.99, ωin = 0.05, α = 0.9) for leaky ESNs.4 Therefore, we trained a large leaky ESN of
3000 neurons, and a relatively small ES2N of 300 neurons, on 6383 training steps as explained above (i.e.
100 for transient, and 6283 for actual training). The resulting output signals in the test session of these
trained models are reported in the centre plots of Figure 6. In the beginning both models follow tightly
the target (dashed line). After 15000 time steps the leaky ESN output (green) already deviates significantly
from the target, while the ES2N output (red) still performs very well. After 20000 time steps the leaky
ESN output is completely decorrelated with the target. On the contrary, the ES2N manages to generate a
meaningful output signal even after 50000 time steps of autonomous run. Remarkably, the ES2N model is
able to substantially outperform the leaky ESN model while having a number of neurons which is an order
of magnitude smaller.
0.0
0.5
1.0
1.5
2.0
0
200
400
600
800
mean = 1.44
std = 13.34
0.0
0.5
1.0
1.5
2.0
0
1000
2000
3000
4000
5000
mean = 0.05
std = 0.11
0.00
0.02
0.04
0.06
0.08
0.10
NRMSE (test)
0
50
100
150
200
250
300
350
400
ES2N
leaky ESN
0
50
100
150
200
250
0.5
0.0
0.5
leaky ESN vs ES2N
15000
15050
15100
15150
15200
15250
0.5
0.0
0.5
20000
20050
20100
20150
20200
20250
0.5
0.0
0.5
ES2N
leaky ESN
50000
50050
50100
50150
50200
50250
(Test) time steps
1
0
1
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
immaginary axis
Jacobian spectrum
1.0
0.5
0.0
0.5
1.0
real axis
1.0
0.5
0.0
0.5
1.0
immaginary axis
Figure 6: MSO task with 8 frequencies. In all plots the colour green corresponds to the leaky ESN model, while red to the ES2N
model. Left: hystograms of NRMSE of 10000 initialisations of leaky ESN (600 reservoir neurons) and ES2N (100 reservoir
neurons) with hyperparameters uniformly generated as explained in Section 4.3.1. In the upper plot, there are almost 300
cases with NRMSE over 2 which have been cut off from the picture (the maximum reaching a NRMSE of 942). In the bottom
plot both leaky ESN and ES2N’s histograms are plotted together and magnified around low NRMSE values of 0.05. Centre
and right: A large fine tuned leaky ESN of 3000 neurons and a relatively small ES2N of 300 neurons have been trained
to reproduce with the feedback of the output the MSO8 signal (dashed line). The output signals generated by leaky ESN
(green) and ES2N (red) after various time intervals ∆t of running in auto-generation mode are plotted, from top to bottom,
∆t = 0, 15000, 20000, 50000. On the right plots, the eigenvalues of the Jacobians of the corresponding trained leaky ESN model,
and ES2N model, on a randomly selected time step in the testing session.
As explained in [20], the MSO is a relatively easy task for a linear ESN, as long as one has at least two
reservoir neurons per frequency. However, the apparent perfection of trained linear ESNs hides an intrinsic
3There are few hundreds NRMSE values of leaky ESN greater than 2 which do not appear in the histogram in Figure 6,
some of them exceeding NRMSE of 100.
4The choice of ρ = 1 for ES2N was arbitrary since it does not influence the outcome of this experiment, see also Appendix
A.
13
unstable phase-coupling dynamics, which manifests itself as soon as little perturbations are applied to the
system. On the other hand, nonlinear ESNs are more resilient to perturbations, but they struggle to learn
functions composed of even just two superimposed oscillators, especially if one wishes to maintain the learned
oscillations for long time. The ES2N is a nonlinear model lying in between these two extremes, since it can
be trained to self-sustain complex oscillatory dynamics for very long time, as shown in the centre plots of
Figure 6. This can be attributed to the peculiar form of the ES2N’s Jacobian spectrum which tends to
dispose its eigenvalues along the unitary circle promoting the dynamics to take place at the edge of stability
as evident from the right plots in Figure 6.
5. Conclusions
In this paper, we have developed a new RC architecture known as the Edge of Stability Echo State
Network (ES2N), which has the unique feature of being controllably tunable to the edge of chaos dynamical
behavior.
Our mathematical analysis first showed that the proposed model has dynamic behavior that
is able to exhibit contracting dynamics and the Echo State Property in a manner similar to a standard
ESN. Furthermore, and relevantly, we provided precise analytical bounds for the entire eigenspectrum of the
Jacobian of the forward map that hold on each input-driven trajectory and for all time steps. As a result,
the reservoir exhibits a behavior whose quality is determined by a specific proximity hyper-parameter. By
architectural construction, smaller values of this hyper-parameter result in dynamics that are progressively
closer to the edge of chaos. Overall, the introduced model takes advantage of both the benefits of having a
linear orthogonal reservoir and a nonlinear contracting dynamics.
We empirically showed that ES2N can reach the maximum memory capacity obtainable within a given
reservoir size, showing significant advantages over standard nonlinear and linear ESN alternatives. Further-
more, we tested the trade-off between nonlinear computation and long short-term memory, and found that
ES2N can reconstruct strongly nonlinear transformations of relatively large delayed input signals, where
both standard (nonlinear) ESN and linear ESN fail. Finally, we empirically demonstrated the superiority of
ES2N in the generation of complex oscillatory patterns. Remarkably, the recurrent network driven with its
own output signal can produce meaningful oscillations for 50 thousands time steps of autonomous run on
the MSO8 task.
The analytical and experimental results presented in this paper have shown, already in this form, the
advantages of combining nonlinear dynamics and orthogonal transformations in the state space of a recurrent
network. Building upon these findings, our future research aims to investigate alternative forms of reservoir
construction in a ES2N. For instance, we plan to explore reservoir construction methods based on permutation
or circular shift matrices, which on the one hand allow to further improve the computational efficiency of the
approach while maintaining its computational properties, and on the other hand are prone to implementation
in edge or neuromorphic devices. Moreover, in forthcoming studies, we will explore the performance of ES2N
in various applications, such as time series forecasting, attractor reconstruction, and classification tasks.
Acknowledgements
This work is partially supported by the EC H2020 programme under project TEACHING (grant n.
871385), and by the EU Horizon research and innovation programme under project EMERGE (grant n.
101070918).
References
[1] F. L. Bauer and C. T. Fike. Norms and exclusion theorems. Numerische Mathematik, 2(1):137–141,
1960.
[2] Y. Bengio, P. Simard, and P. Frasconi.
Learning long-term dependencies with gradient descent is
difficult. IEEE transactions on neural networks, 5(2):157–166, 1994.
14
[3] N. Bertschinger and T. Natschl¨ager. Real-time computation at the edge of chaos in recurrent neural
networks. Neural computation, 16(7):1413–1436, 2004.
[4] J. Boedecker, O. Obst, J. T. Lizier, N. M. Mayer, and M. Asada. Information processing in echo state
networks at the edge of chaos. Theory in Biosciences, 131:205–213, 2012.
[5] T. L. Carroll. Optimizing memory in reservoir computers. Chaos: An Interdisciplinary Journal of
Nonlinear Science, 32(2), 2022.
[6] A. Ceni, P. Ashwin, L. Livi, and C. Postlethwaite. The echo index and multistability in input-driven
recurrent neural networks. Physica D: Nonlinear Phenomena, 412:132609, 2020.
[7] A. Cini, I. Marisca, F. M. Bianchi, and C. Alippi. Scalable spatiotemporal graph neural networks. In
Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 7218–7226, 2023.
[8] M. Cucchi, S. Abreu, G. Ciccone, D. Brunner, and H. Kleemann. Hands-on reservoir computing: a
tutorial for practical implementation. Neuromorphic Computing and Engineering, 2(3):032002, 2022.
[9] J. Dambre, D. Verstraeten, B. Schrauwen, and S. Massar. Information processing capacity of dynamical
systems. Scientific reports, 2(1):1–7, 2012.
[10] I. Farkaˇs, R. Bos´ak, and P. Gergel’. Computational analysis of memory capacity in echo state networks.
Neural Networks, 83:109–120, 2016.
[11] C. Gallicchio. Chasing the echo state property. arXiv preprint arXiv:1811.10892, 2018.
[12] C. Gallicchio and A. Micheli.
Architectural and markovian factors of echo state networks.
Neural
Networks, 24(5):440–456, 2011.
[13] C. Gallicchio and A. Micheli.
Fast and deep graph neural networks.
In Proceedings of the AAAI
conference on artificial intelligence, volume 34, pages 3898–3905, 2020.
[14] C. Gallicchio, A. Micheli, and L. Pedrelli. Deep reservoir computing: A critical experimental analysis.
Neurocomputing, 268:87–99, 2017.
[15] G. Holzmann and H. Hauser. Echo state networks with filter neurons and a delay&sum readout. Neural
Networks, 23(2):244–256, 2010.
[16] M. Inubushi and K. Yoshimura. Reservoir computing beyond memory-nonlinearity trade-off. Scientific
reports, 7(1):1–10, 2017.
[17] H. Jaeger. The “echo state” approach to analysing and training recurrent neural networks-with an
erratum note. Bonn, Germany: German National Research Center for Information Technology GMD
Technical Report, 148(34):13, 2001.
[18] H. Jaeger.
Short term memory in echo state networks. gmd-report 152.
In GMD-German
National Research Institute for Computer Science (2002), http://www. faculty. jacobs-university.
de/hjaeger/pubs/STMEchoStatesTechRep. pdf. Citeseer, 2002.
[19] H. Jaeger and H. Haas.
Harnessing nonlinearity: Predicting chaotic systems and saving energy in
wireless communication. science, 304(5667):78–80, 2004.
[20] H. Jaeger, M. Lukoˇseviˇcius, D. Popovici, and U. Siewert. Optimization and applications of echo state
networks with leaky-integrator neurons. Neural networks, 20(3):335–352, 2007.
[21] J. F. Kolen and S. C. Kremer. A field guide to dynamical recurrent networks. John Wiley & Sons, 2001.
[22] L.-W. Kong, Y. Weng, B. Glaz, M. Haile, and Y.-C. Lai. Reservoir computing as digital twins for
nonlinear dynamical systems. Chaos: An Interdisciplinary Journal of Nonlinear Science, 33(3), 2023.
15
[23] D. Koryakin, J. Lohmann, and M. V. Butz. Balanced echo state networks. Neural Networks, 36:35–45,
2012.
[24] C. G. Langton. Computation at the edge of chaos: Phase transitions and emergent computation. Physica
D: nonlinear phenomena, 42(1-3):12–37, 1990.
[25] R. Legenstein and W. Maass. What makes a dynamical system computationally powerful. New directions
in statistical signal processing: From systems to brain, pages 127–154, 2007.
[26] X. Liang, Y. Zhong, J. Tang, Z. Liu, P. Yao, K. Sun, Q. Zhang, B. Gao, H. Heidari, H. Qian, et al.
Rotating neurons for all-analog implementation of cyclic reservoir computing. Nature communications,
13(1):1549, 2022.
[27] L. Livi, F. M. Bianchi, and C. Alippi. Determination of the edge of criticality in echo state networks
through fisher information maximization. IEEE transactions on neural networks and learning systems,
29(3):706–717, 2017.
[28] M. Lukoˇseviˇcius. A practical guide to applying echo state networks. In Neural networks: Tricks of the
trade, pages 659–686. Springer, 2012.
[29] M. Lukoˇseviˇcius and H. Jaeger. Reservoir computing approaches to recurrent neural network training.
Computer Science Review, 3(3):127–149, 2009.
[30] G. Manjunath and H. Jaeger. Echo state property linked to an input: Exploring a fundamental char-
acteristic of recurrent neural networks. Neural computation, 25(3):671–696, 2013.
[31] E. Meckes. The eigenvalues of random matrices. arXiv preprint arXiv:2101.02928, 2021.
[32] K. Nakajima and I. Fischer. Reservoir Computing. Springer, 2021.
[33] E. Ott. Chaos in dynamical systems. Cambridge university press, 2002.
[34] N. H. Packard.
Adaptation toward the edge of chaos.
Dynamic patterns in complex systems, 212:
293–301, 1988.
[35] A. Rodan and P. Tino. Minimum complexity echo state network. IEEE transactions on neural networks,
22(1):131–144, 2010.
[36] B. Roeschies and C. Igel. Structure optimization of reservoir networks. Logic Journal of IGPL, 18(5):
635–669, 2010.
[37] J. Schmidhuber, D. Wierstra, M. Gagliolo, and F. Gomez. Training recurrent networks by evolino.
Neural computation, 19(3):757–779, 2007.
[38] B. Schrauwen, M. Wardermann, D. Verstraeten, J. J. Steil, and D. Stroobandt. Improving reservoirs
using intrinsic plasticity. Neurocomputing, 71(7-9):1159–1171, 2008.
[39] T. Schulte to Brinke, M. Dick, R. Duarte, and A. Morrison. A refined information processing capacity
metric allows an in-depth analysis of memory and nonlinearity trade-offs in neurocomputational systems.
Scientific Reports, 13(1):10517, 2023.
[40] S. H. Strogatz. Nonlinear dynamics and chaos with student solutions manual: With applications to
physics, biology, chemistry, and engineering. CRC press, 2018.
[41] H. Tan and S. van Dijken. Dynamic machine vision with retinomorphic photomemristor-reservoir com-
puting. Nature Communications, 14(1):2169, 2023.
[42] D. Verstraeten, B. Schrauwen, M. d’Haene, and D. Stroobandt. An experimental unification of reservoir
computing methods. Neural networks, 20(3):391–403, 2007.
16
[43] A. Vulpiani, F. Cecconi, and M. Cencini. Chaos: from simple models to complex systems, volume 17.
World Scientific, 2009.
[44] G. Wainrib and M. N. Galtier. A local echo state property through the largest lyapunov exponent.
Neural Networks, 76:39–45, 2016.
[45] S. Wang, Y. Li, D. Wang, W. Zhang, X. Chen, D. Dong, S. Wang, X. Zhang, P. Lin, C. Gallicchio,
et al. Echo state graph neural networks with analogue random resistive memory arrays. Nature Machine
Intelligence, 5(2):104–113, 2023.
[46] P. J. Werbos. Backpropagation through time: what it does and how to do it. Proceedings of the IEEE,
78(10):1550–1560, 1990.
[47] X. Wu, S. Wang, W. Huang, Y. Dong, Z. Wang, and W. Huang. Wearable in-sensor reservoir com-
puting using optoelectronic polymers with through-space charge-transport characteristics for multi-task
learning. Nature Communications, 14(1):468, 2023.
[48] Y. Xue, L. Yang, and S. Haykin. Decoupled echo state networks with lateral inhibition. Neural Networks,
20(3):365–376, 2007.
[49] I. B. Yildiz, H. Jaeger, and S. J. Kiebel. Re-visiting the echo state property. Neural networks, 35:1–9,
2012.
17
Appendix A
Data from the random search of the MSO8 task
Here we report the NRMSE values of the random search of the 10000 runs for the MSO8 task for both
the ES2N model (left plots), and the leaky ESN model (right plots). Note that a great portion of the 10000
runs of the ES2N model has NRMSE value less than 2e-2, while the vast majority of runs of the leaky ESN
model has NRMSE value greater than 1e-1.
0.02
0.04
0.06
0.08
0.10
0
1
NRMSE
0.2
0.4
0.6
0.8
1.0
0
500
NRMSE
0.01
0.02
0.03
0.04
0.06
0.08
0.10
0.00
0.01
0.02
NRMSE
0.2
0.4
0.6
0.8
0.9
1.0
0.0
0.1
0.2
NRMSE
0.80
0.85
0.90
0.95
1.00
1.05
1.10
1.15
1.20
0
1
NRMSE
0.80
0.85
0.90
0.95
1.00
1.05
1.10
1.15
1.20
0
500
NRMSE
0.8
0.9
1.0
1.1
1.2
0.00
0.01
0.02
NRMSE
0.80
0.85
0.90
0.95
0.99
1.05
1.10
1.15
1.20
0.0
0.1
0.2
NRMSE
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
in
0
1
NRMSE
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
in
0
100
NRMSE
0.00
0.05
0.11
0.15
0.20
0.25
0.30
0.35
0.40
in
0.00
0.01
0.02
NRMSE
0.00
0.05
0.10
0.20
0.30
0.40
in
0.0
0.1
0.2
NRMSE
Figure 7: NRMSEs resulted from the random search of section 4.3.1 plotted versus each one of the hyperparameter.
The
horizontal orange lines correspond to the minimum NRMSE values reached, while the corresponding minimum points are
highlighted in red on the abscissa. Left: data of ES2N. Right: data of leaky ESN.
18
