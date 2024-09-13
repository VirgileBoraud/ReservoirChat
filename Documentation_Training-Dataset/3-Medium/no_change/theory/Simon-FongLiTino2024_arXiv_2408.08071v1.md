UNIVERSALITY OF REAL MINIMAL COMPLEXITY
RESERVOIR
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
Abstract. Reservoir Computing (RC) models, a subclass of recurrent neural
networks, are distinguished by their fixed, non-trainable input layer and dynam-
ically coupled reservoir, with only the static readout layer being trained. This
design circumvents the issues associated with backpropagating error signals
through time, thereby enhancing both stability and training efficiency. RC
models have been successfully applied across a broad range of application do-
mains. Crucially, they have been demonstrated to be universal approximators
of time-invariant dynamic filters with fading memory, under various settings of
approximation norms and input driving sources.
Simple Cycle Reservoirs (SCR) represent a specialized class of RC models
with a highly constrained reservoir architecture, characterized by uniform
ring connectivity and binary input-to-reservoir weights with an aperiodic sign
pattern. For linear reservoirs, given the reservoir size, the reservoir construction
has only one degree of freedom – the reservoir cycle weight. Such architectures
are particularly amenable to hardware implementations without significant
performance degradation in many practical tasks. In this study we endow
these observations with solid theoretical foundations by proving that SCRs
operating in real domain are universal approximators of time-invariant dynamic
filters with fading memory. Our results supplement recent research showing
that SCRs in the complex domain can approximate, to arbitrary precision,
any unrestricted linear reservoir with a non-linear readout. We furthermore
introduce a novel method to drastically reduce the number of SCR units, making
such highly constrained architectures natural candidates for low-complexity
hardware implementations. Our findings are supported by empirical studies on
real-world time series datasets.
1. Introduction
Reservoir Computing (RC) is a subclass of Recurrent Neural Network defined by
a fixed parametric state space representation (the reservoir) and a static trained
readout map. This distinctive approach not only simplifies the training process by
focusing adjustments solely on the static readout layer but also enhances computa-
tional efficiency.
The simplest recurrent neural network realization of RC [Jae01, MNM02, TD01,
LJ09] are known as Echo State Networks (ESN) [Jae01, Jae02a, Jae02b, JH04].
The representation capacity of ESNs have been demonstrated in a series of papers,
showing existentially that ESNs can approximate any time-invariant dynamic filters
with fading memory in a variety of settings [GO18b, GO18a, GO18a, GO19].
Simple Cycle Reservoirs (SCRs) [RT10] are RC models characterized by a highly
restricted architecture: a uniform weight ring connectivity among reservoir neurons
and binary uniformly weighted input-to-reservoir connections. This simplicity is
particularly advantageous for hardware implementations, reducing implementation
and computational costs, as well as enhancing real-time processing capabilities
1
arXiv:2408.08071v1  [cs.LG]  15 Aug 2024
2
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
without degrading performance. However, while cyclic reservoir topology with
a single connection weight has been adopted in a variety of hardware implemen-
tations [ASdS+11, NTH21, CSK+18, ANH+24], its theoretical foundations have
been missing. To rectify this situation we rigorously prove that SCRs operating
in real domain are universal approximators of time-invariant dynamic filters with
fading memory. Our results supplement recent research [LFT24] showing that SCRs
in the complex domain can approximate, to arbitrary precision, any unrestricted
linear reservoir with a non-linear readout, opening the door to a broad range of
practical scenarios involving real-valued computations. Furthermore, based on our
constructive proof arguments, we formulate a novel method to drastically reduce
the number of SCR units, making such highly constrained architectures natural
candidates for low-complexity hardware implementations.
We emphasize that proving that SCR architecture retains universality when
moving from the complex to the real domain is far from straightforward. Indeed, as
shown in [LFT24], attempts to even partially restrict SCR in the complex domain
to the real one result in more complex multi-reservoir structures.1
We conclude the paper with numerical experiments that validate the structural
approximation properties demonstrated in our theoretical analysis. In particular, we
begin with an arbitrary linear reservoir system with linear readout and demonstrate,
on real-life datasets, that the approximation SCR counterparts will gradually ap-
proach the original system as the number of reservoir neurons increase, reinforcing
the theoretical insights provided.
2. The setup
We first introduce the basic building blocks needed for the developments in this
study.
Definition 1. A linear reservoir system over R is formally defined as the triplet
R := (W, V, h) where the dynamic coupling W is an n × n real-valued matrix, the
input-to-state coupling V is an n×m real-valued matrix, and the state-to-output
mapping (readout) h : Rn →Rd is a (trainable) continuous function.
The corresponding linear dynamical system is given by:
(1)
(
xt
= Wxt−1 + V ut
yt
= h(xt)
where {ut}t∈Z−⊂Rm, {xt}t∈Z−⊂Rn, and {yt}t∈Z−⊂Rd are the external inputs,
states and outputs, respectively. We abbreviate the dimensions of R by (n, m, d).
We make the same assumption as in [LFT24, Definition 1], that
(1) The input stream {ut} is uniformly bounded by a constant M.
(2) The dynamic coupling matrix W has its operator norm ∥W∥< 1.
The only difference in this present work is the requirement that all the matrices
and vectors are over R. Under the assumptions, for each left infinite time series
1As we will show, to retain advantages of the single simple SCR structure in the real domain
and retain universality one needs to consider orthogonal similarity throughout the approximation
pipeline, as well as completion of the set of roots of unity in the canonical form of orthogonal
matrices for cyclic dilation of orthogonally dilated coupling matrices.
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
3
u = {ut}t∈Z−, the system (1) has a unique solution given by
xt(u) =
X
n≥0
W nV ut−n,
yt(u) = h(xt(u)).
We refer to the solution simply as {(xt, yt)}t.
Definition 2. For two reservoir systems R = (W, V, h) (with dimensions (n, m, d))
and R′ = (W ′, V ′, h′) (with dimensions (n′, m, d)):
(1) We say the two systems are equivalent if for any input stream u = {ut}t∈Z−,
the solutions {(xt, yt)}t and {(x′
t, y′
t)}t for systems R and R′ satisfy yt = y′
t
for all t.
(2) For ϵ > 0, we say the two systems are ϵ-close if for any input stream
u = {ut}t∈Z−, the solutions of the two systems (under the notation above)
satisfy ∥yt −y′
t∥2 < ϵ for all t.
We now define the main object of interest in this paper. We begin by the following
definition.
Definition 3. Let P = [pij] be an n × n matrix. We say P is a permutation
matrix if there exists a permutation σ in the symmetric group Sn such that:
pij =
(
1,
if σ(i) = j,
0,
if otherwise.
We say a permutation matrix P is a full-cycle permutation2 if its corresponding
permutation σ ∈Sn is a cycle permutation of length n. Finally, a matrix W = cP
is called a contractive full-cycle permutation if c ∈(0, 1) and P is a full-cycle
permutation.
Simple cycle reservoir systems originate from the minimum complexity reservoir
systems introduced in [RT10]:
Definition 4. A linear reservoir system R = (W, V, h) with dimensions (n, m, d) is
called a Simple Cycle Reservoir (SCR) over R 3 if
(1) W is a contractive full-cycle permutation, and
(2) V ∈Mn×m ({−1, 1}).
Our goal is to show that every linear reservoir system is ϵ-close to a SCR over R.
This is done in a way that does not increase the complexity of the readout function
h. To be precise:
Definition 5. We say that a function g is h with linearly transformed domain
if g(x) = h(Ax) for some matrix A.
2Also called left circular shift or cyclic permutation in the literature
3We note that the assumption on the aperiodicity of the sign pattern in V is not required for
this study
4
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
3. Universality of Orthogonal Dynamical Coupling
In [Hal50], Halmos raised the question of what kind of operators can be embedded
as a corner of a normal operator. He observed that one can embed a contractive
operator W inside a unitary operator:
˜U =
 W
DW ⊤
DW
−W ⊤

where DW = (I −W ⊤W)1/2 and DW ⊤= (I −WW ⊤)1/2. This motivated the rich
study of the dilation theory of linear operators. One may refer to [Pau02] for more
comprehensive background on dilation theory. In the recent study [LFT24], the
authors used a dilation theory technique to obtain an ϵ-close reservoir system with
a unitary matrix as the dynamic coupling. A key idea is the theorem of Egerváry,
which states that for any n × n matrix W with ∥W∥≤1 and N > 1, there exists a
(N + 1)n × (N + 1)n unitary matrix U such that the upper left n × n corner of U k
is W k for all 1 ≤k ≤N. In fact, this matrix U can be constructed explicitly as:
U =


W
0
0
. . .
0
DW ⊤
DW
0
0
. . .
0
−W ⊤
0
I
0
. . .
0
0
...
...
...
...
0
. . .
I
0


.
We first notice that when W is a matrix over R, this dilation matrix U is over R as
well. Therefore, U is an orthogonal matrix. This technique allows us to obtain an
ϵ-close reservoir system with an orthogonal dynamic coupling matrix.
Theorem 6. Let R = (W, V, h) be a reservoir system defined by contraction W with
∥W∥=: λ ∈(0, 1). Given ϵ > 0, there exists a reservoir system R′ = (W ′, V ′, h′)
that is ϵ-close to R, with dynamic coupling W ′ = λU, where U is orthogonal.
Moreover, h′ is h with linearly transformed domain.
Proof. The proof follows that of an analogous statement in the complex domain
in [LFT24, Theorem 11].
The arguments follow through by replacing unitary
matrices by orthogonal matrices and conjugate transpose by regular transpose. For
completeness we present the proof in Appendix 8.1.
□
4. Universality of Cyclic Permutation Dynamical Coupling
Having established universality of orthogonal dynamic coupling in the reservoir
domain, we now continue to show that for any reservoir system with orthogonal
state coupling, we can construct an equivalent reservoir system with cyclic coupling
of state units weighted by a single common connection weight value. In broad terms
we will employ the strategy of [LFT24], but here we need to pay special attention
to maintain all the matrices in the real domain.
We begin by invoking [LFT24, Proposition 12], which stated that matrix similarity
of dynamical coupling implies reservoir equivalence. It therefore remains to be
shown that for any given orthogonal state coupling we can always find a full-cycle
permutation that is close to it to arbitrary precision. Specifically, when given an
orthogonal matrix, the goal is to perturb it to another orthogonal matrix that
is orthogonally equivalent to a permutation matrix. Here we cannot adopt the
strategy in [LFT24, Section 5] because it would inevitably involve a unitary matrix
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
5
over C during the diagonalization process. Instead, we convert an orthogonal matrix
to its canonical form via a (real) orthogonal matrix.
The core idea of our approach is schematically illustrated in Figure 1. Given
a reservoir system with dynamic coupling W, we first find its equivalent with
orthogonal coupling U. Rotational angles in the canonical form of U are shown
as red dots (a). Roots of unity corresponding to a sufficiently large cyclic dilation
coupling can approximate the rotational angles to arbitrary precision ϵ (b).
(a)
(b)
Figure 1. Schematic illustration of the core idea enabling us to
prove universality of SCRs in the real domain. Given a reservoir
system with dynamic coupling W and its equivalent with orthogonal
coupling U, rotational angles in the canonical form of U are shown
as red dots (a). Roots of unity corresponding to cyclic dilation
approximate the rotational angles to a prescribed precision ϵ (b).
We begin by recalling some elementary results of orthogonal and permutation
matrices. For each θ ∈[0, 2π), define the following rotation matrix:
Rθ =
cos(θ)
−sin(θ)
sin(θ)
cos(θ)

.
The eigenvalues of Rθ are precisely e±iθ. Moreover, notice that:
0
1
1
0

R−θ
0
1
1
0
−1
=
0
1
1
0
  cos(θ)
sin(θ)
−sin(θ)
cos(θ)
 0
1
1
0

=
cos(θ)
−sin(θ)
sin(θ)
cos(θ)

= Rθ.
Therefore, Rθ and R−θ are orthogonally similar. Employing the real version of Schur
decomposition, for any orthogonal matrix C ∈O(n), there exists an orthogonal
matrix S such that the product S⊤CS has the following form:
6
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
S⊤CS =


Rθ1
...
Rθk
±1
...
±1


=


Rθ1
...
Rθk
Υ

,
where θi ∈(0, π), and Υ := diag{a1, a2, ..., aq}, ai ∈{−1, +1}, i = 1, 2, ..., q, is a
diagonal matrix with q entries of ±1’s. For simplicity, we will assume for the
rest of the paper an even dimension n, which inherently implies that q is also
even. The case when n is odd is analogous and follows from similar arguments.
Note that without loss of generality, Υ can always take the form where +1’s (if
any) preceded −1’s (if any). This can be achieved by permuting rows of Υ which is
an orthogonality preserving operation and invoking [LFT24, Proposition 12]. This
can be further simplified by observing:
R0 =
1
0
0
1

, Rπ =
−1
0
0
−1

.
Hence, pairs of +1’s (and −1’s) can therefore be grouped as blocks of R0 (and Rπ).
Therefore, without loss of generality, S⊤CS is a block diagonal matrix consisting
of {Rθ1, . . . , Rθm}, θi ∈[0, π], i = 1, 2, ..., m, and at most one block of the form
1
0
0
−1

. In the literature this is known as the canonical form of the orthogonal
matrix C.
Given the inherent orthogonality of permutation matrices, their corresponding
canonical forms can be computed. Given an integer ℓ≥1, the complete set of ℓ-th
roots of unity is a collection of uniformly positioned points along the complex circle,
denoted by {ei 2jπ
ℓ
: 0 ≤j ≤ℓ−1}.
It is well-known from elementary matrix analysis that the eigenvalues of a full-
cycle permutation matrix form a complete set of roots of unity.
Therefore, given an ℓ× ℓfull-cycle permutation P (ℓeven), we can find an
orthogonal matrix Q such that Q⊤PQ is a block diagonal matrix of {1, −1, R 2πj
ℓ
:
1 ≤j < ℓ
2}. Here, note that for each 1 ≤j < ℓ
2, R 2πj
ℓ
has two conjugate eigenvalues
ei 2πj
ℓ
and e−i 2πj
ℓ. Hence, an ℓ× ℓorthogonal matrix X is orthogonally equivalent
to a full-cycle permutation if and only if its canonical form consists of:
(1) A complete set of rotation matrices {R 2πj
ℓ
: 1 ≤j < ℓ
2}, and
(2) Two additional diagonal entries of 1 and −1.
Theorem 7. Let U be an n × n orthogonal matrix and δ > 0 be an arbitrarily small
positive number. There exists n1 ≥n, an n1 × n1 orthogonal matrix S, an n1 × n1
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
7
full-cycle permutation P and an (n1 −n) × (n1 −n) orthogonal matrix D, such that
S⊤PS −
U
0
0
D
 < δ.
Proof. We only prove the case when the canonical form of U is in a block diagonal
form consisting of Rθ1, Rθ2, . . . , Rθk. The case when the canonical form contains
additional entries of ±1 is analogous. Let S1 be the orthogonal matrix such that
S⊤
1 US1 is in the canonical form. For fixed δ > 0, pick an integer ℓ0 > 0 such that:
|1 −e
πi
ℓ0 | < δ.
For each j = 1, . . . k, the interval Ij :=

θjℓ0(k+1)
π
−(k + 1),
θjℓ0(k+1)
π
+ (k + 1)

has length 2(k + 1), and therefore contain 2(k + 1) distinctive integers. Moreover,
since θj ∈[0, π], the interval Ij contains at least k distinct integers strictly within
(0, ℓ0 · (k + 1)). From each interval I1, . . . , Ik we can thus choose distinct integers
a1, · · · , ak, with aj ∈Ij and 0 < aj < ℓ0 · (k + 1) such that for each j = 1, . . . , k:
0 <

θjℓ0(k + 1)
π
−aj
 < (k + 1),
or equivalently,
0 <
θj −
aj
ℓ0(k + 1) · π
 < π
ℓ0
.
This implies:
eiθj −eπi
aj
ℓ0·(k+1)
 =
1 −e
i

−θj+
aj
ℓ0·(k+1) ·π

=
1 −e
i
θj−
aj
ℓ0·(k+1) ·π

 ≤
1 −ei π
ℓ0
 < δ.
Now for each j = 1, . . . , k, set βj :=
πaj
ℓ0(k+1) ∈(0, π). Let A0 be the block diagonal
matrix consisting of {Rβ1, . . . , Rβk}. Since the eigenvalues of Rβj is within δ to Rθj,
we obtain:
A0 −S⊤
1 US1
 =



Rβ1 −Rθ1
...
Rβk −Rθk



= max{
Rβj −Rθj
} < δ,
where ∥·∥denotes the operator norm.
Let n1 := 2ℓ0(k + 1). We have each βj = aj
n1 · (2π), and therefore the rotations
Rβ1, . . . , Rβk are all rotations of distinct n1-roots of unity. Whilst this is not a
complete set of rotations of the roots of unity, we can complete the set of rotations
by filling in the missing ones. In particular the missing set of rotations is given
explicitly by:
R1 :=
n
R 2πa
n1 : a ∈Z, 0 < a < n1
2 , a ̸= aj
o
Let D denote the (n1 −n) × (n1 −n) block diagonal matrix consisting of:
(1) All the missing blocks of rotations described in R1, and
(2) Two additional diagonal entries of 1 and −1.
8
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
By construction D is orthogonal since it contains block diagonal matrices of ±1 and
Rθ. Consider the n1 × n1 matrix A := (S1A0S⊤
1 ) ⊕D. Then A is orthogonal by
construction and the canonical form of A consists of:
(1) A complete set of rotations R πa
n1 , a ∈Z, 0 < a < n1
2 , and
(2) An additional diagonal entry of 1 when n1 is odd and two additional diagonal
entries of 1 and −1 when n1 is even.
This is precisely the canonical form of a n1 × n1 full-cycle permutation. Let P
be a full-cycle permutation of dimension n1, then there exists an orthogonal matrix
S such that A = S⊤PS. We have,
S⊤PS −
U
0
0
D
 =


S1A0S⊤
1 −U
0
0
0

=
A0 −S⊤
1 US1
 < δ,
as desired.
□
Remark 8. In practice, the dimension n1 is usually much smaller than the theoret-
ical upper bound of 2ℓ0(k + 1). Here, the integer ℓ0 is chosen to satisfy |1 −e
πi
ℓ0 | < δ,
which equivalently means:
π
ℓ0
< arccos

1 −δ2
2

.
In practice, a much lower dimension can be achieved. Given a set of angles {θi} from
an n × n orthogonal matrix U. For a fixed n′ > n, we can use maximum matching
program in a bipartite graph to check whether: for each θi there exists a distinct ki
such that the root-of-unity 2kiπ
n′
approximates θi. For fixed n′, define a bipartite
graph G with vertex set A ∪B with A = {θi} and B = { 2aπ
n′ : 0 < a < n′
2 }. An
edge e joins θi ∈A with 2aπ
n′ ∈B if
eθii −e
2aπi
n′
 < δ. One can easily see that we
can find distinct ki to approximate θi by 2kiπ
n′
if and only if there exists a matching
for this bipartite graph with exactly |A| edges. We let nC denote the smallest n′
such that a desired maximum matching is achieved. We shall demonstrate later, in
the numerical experiment section, that nC is significantly lower than the theoretical
upper bound given by Theorem 7.
Theorem 9. Let U be an n × n orthogonal matrix and W = λU with λ ∈(0, 1).
Let R = (W, V, h) be a reservoir system with state coupling W. For any ϵ > 0, there
exists a reservoir system Rc = (Wc, Vc, hc) that is ϵ-close to R such that:
(1) Wc is a contractive full-cycle permutation with ∥Wc∥= ∥W∥= λ ∈(0, 1),
and
(2) hc is h with linearly transformed domain.
Proof. The proof follows that of an analogous statement in the complex domain in
[LFT24, Theorem 14]. The arguments follow through by replacing unitary matrices
by orthogonal matrices and conjugate transpose by regular transpose.
For completeness we present the proof in Appendix 8.2
□
5. From Cyclic Permutation to SCR
We have now ready to prove the main result (Theorem 12). So far, we have
proved that any linear reservoir system R = (W, V, h) is ϵ-close to another reservoir
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
9
system R′ = (W ′, V ′, h′) where W ′ is a permutation or a full-cycle permutation. It
remains to show that one can make the entries in the input-to-state coupling matrix
V to be all ±1.
We first recall the following useful Lemmas.
Lemma 10 ([LFT24, Lemma 16]). Let n, k be two natural numbers such that
gcd(n, k) = 1. Let P be an n × n full-cycle permutation. Consider the nk × nk
matrix:
P1 =


0
0
0
. . .
0
P
P
0
0
. . .
0
0
0
P
0
. . .
0
0
...
...
...
...
0
. . .
P
0


.
Then P1 is a full-cycle permutation.
Lemma 11 ([LFT24, Lemma 17]). For any n × m real matrix V and δ > 0, there
exists k matrices {F1, · · · , Fk} ⊂Mn×m ({−1, 1}) and a constant integer N > 0
such that:

V −1
N
k
X
j=1
Fj

< δ
Moreover, k can be chosen such that gcd(k, n) = 1.
We now obtain our main theorem on the universality of SCR over R. In comparison
to [LFT24, Theorem 20], the coupling matrix V in a SCR over R contains only ±1
instead of {±1, ±i}.
Theorem 12. For any reservoir system R = (W, V, h) of dimensions (n, m, d) and
any ϵ > 0, there exists a SCR R′ = (W ′, V ′, h′) of dimension (n′, m, d) that is
ϵ-close to R. Moreover, ∥W∥= ∥W ′∥and h′ is h with linearly transformed domain.
Proof. One may refer to the proof of [LFT24, Theorem 20]. Crucially, because the
dynamic coupling matrix V in the intermediate steps are all over R instead of C,
the resulting matrix V ′ only have ±1.
□
[GO18a](Corollary 11) shows that linear reservoir systems with polynomial read-
outs are universal. They can approximate to arbitrary precision time-invariant
fading memory filters. This result, together with Theorem 12, establish universality
of SCRs in the real domain. Indeed, given a time-invariant fading memory filter,
one can find an approximating linear reservoir system with polynomial readout h
approximating the filter to the desired precision. By Theorem 12, we can in turn
constructively approximate this reservoir system with a SCR, again to arbitrary
precision. Moreover, the SCR readout is a polynomial of the same degree as h, since
it is h with linearly transformed domain.
6. Numerical Experiments
We conclude the paper with numerical experiments illustrating our contributions.
For reproducibility of the experiments, all experiments are CPU-based and are
performed on Apple M3 Max with 128GB of RAM. The source code and data of the
numerical analysis is openly available at https://github.com/Lampertos/RSCR.
10
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
6.1. Dilation of Linear Reservoirs on Time Series Forecasting. This sec-
tion illustrates the structural approximation properties of linear reservoir systems
when dilating the reservoir coupling matrix, when applied to univariate time series
forecasting. The readout function will be assumed to be linear throughout the
numerical analysis in this section, as the primary objective of this paper is to
examine the structural approximation properties of the state-space representation
of linear reservoirs as determined by the coupling matrix.
Initially, a linear reservoir system featuring a randomly generated coupling matrix
W is constructed. We then approximate this system by linear reservoir systems
under two types of dilated coupling matrices: (1) U – Orthogonal dilation of W
(Theorem 6) , and (2) C – Cyclic dilation of U (Theorem 9).
The results demonstrate that the prediction loss approximation error diminishes
progressively as the dimension of dilation increases. For demonstration purposes,
we keep dimensionality of the original reservoir system to be approximated by SCR
low (n = 5). The elements of W are independently sampled from the uniform
distribution U(0, 1). The elements of input-to-state coupling V is generated by
scaling the binary expansion of the digits of π by 0.05 [RT10]. Univariate forecasting
performance of the initial and the approximating reservoir systems are compared on
two popular datasets used in recent time series forecasting studies (e.g. [ZZP+20]
(adopting their training/validation/test data splits)):
ETT. The Electricity Transformer Temperature dataset4consists of measurements
of oil temperature and six external power-load features from transformers in two
regions of China. The data was recorded for two years, and measurements are
provided either hourly (indicated by ’h’) or every 15 minutes (indicated by ’m’). In
this paper we used oil temperature of the ETTm2 dataset for univariate prediction
with train/validation/test split being 12/4/4 months.
ECL. The Electricity Consuming Load5 consists of hourly measurements of electricity
consumption in kWh for 321 Portuguese clients during two years. In this paper
we used client MT 320 for univariate prediction. The train/validation/test split is
15/3/4 months.
The readout h of the original reservoir system is trained using ridge regression
with a ridge coefficient of 10−9. Note that modified versions of the input-to-state
map V and the readout map h will be employed in all subsequent dilated systems.
Specifically, the readout map will not be subject to further training in these systems.
In all simulations, we maintain a spectral radius λ = 0.9 and prediction horizon is
set to be 300.6
The initial system R = (W, V, h) is dilated over a set of pre-defined dilation
dimensions D := {2, 6, 10, 15, 19, 24, 28, 33, 37, 42}7. For each N ∈D, by Theorem 6,
we construct a linear reservoir system RU with an orthogonal dynamic coupling
WU of dimension nU = (N + 1)n. Then by Theorem 9, we dilate RU into an ϵ-close
linear reservoir system RC with contractive cyclic-permutation dynamic coupling.
4https://github.com/zhouhaoyi/ETDataset
5https://archive.ics.uci.edu/dataset/321/
electricityloaddiagrams20112014
6These two parameters are primarily chosen not for accuracy of the forecasting capacities but
to demonstrate the structural approximation properties proven in this paper.
7Recall that the corresponding orthogonal dilation will have dimensions (N + 1) · 5 × (N + 1) · 5
for each N ∈D.
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
11
For the orthogonal dilation, the linear reservoir system RU := (WU, VU, hU) is
defined by:
WU := λ · U, where
U :=


W
DW ⊤
DW
−W ⊤
I
...
I
0


∈M5·(N+1)×5·(N+1)(R)
VU :=
V
0

,
hU(x) = h (Pn(x)) ,
where Pn : R5(N+1) ,→R5 denote the projection onto the first n = 5 coordinates.
Since U is orthogonal, it’s canonical form TU can be obtained via the real version
of Schur’s decomposition:
U = JUTUJ⊤
U ,
where we note that both U and TU have unit spectral radius.
By Remark 8, given ϵ > 0 and the canonical form TU of U, the maximum matching
program in bipartite graphs allows us to find the canonical form of the nC × nC-
dimensional root-completed-matrix A (along with the corresponding dimension nC),
given by T := A0 ⊕D, described in the proof of Theorem 7. By construction T is
ϵ-close to TU ⊕D in terms of operator norm.
Let C be the nC × nC-dimensional full cycle permutation matrix of unit spectral
radius. Since C is orthogonal, we can once again apply Schur’s decomposition to
obtain it’s canonical form:
C = JCTCJ⊤
C .
By construction, T contains rotational matrices with angles at the roots of unity
that are rearrangements of the eigenvalues of cyclic permutation of the same size.
Therefore there exists permutation matrix ˜P such that:
˜PT ˜P ⊤= TC.
Therefore by the proof of Theorem 9 (following that of [LFT24, Theorem 14]),
the linear reservoir system RC := (WC, VC, hC) for the cyclic dilation is defined by:
WC := λ · C,
VC := P
VU
0

,
hC(x) = h
 Pn(P ⊤x)

,
P := JC ˜PJ
⊤
U,
(2)
where C ∈Mnc×nc(R) is a cyclic permutation of dimension nc > 5 · (N + 1), and
JU is a nc −5 · (N + 1) × nc −5 · (N + 1) matrix with an JU on the upper left hand
corner and zero everywhere else.
By the uniform continuity of the readout h, it suffices to evaluate the closeness of
the state trajectories of the original and the approximating cyclic dilation systems,
as they are driven by the same input time series ETTm2 and ECL. The two state
activation sequences are not directly comparable, but they become comparable if
the states of the approximating system are transformed by the orthogonal matrix P
(Equation (2)) and projected into the first n coordinates. Figure 2 shows the mean
square differences between the states of the two systems as a function of the dilation
12
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
dimension N. As expected, MSE between the states decays to zero exponentially as
the dilation dimension increases.
0
5
10
15
20
25
30
35
40
Dilation N
10
6
10
5
10
4
10
3
10
2
Mean difference in Euclidean norm (log)
(a) ECL
0
5
10
15
20
25
30
35
40
Dilation N
10
6
10
5
10
4
10
3
10
2
Mean difference in Euclidean norm (log)
(b) ETTm2
Figure 2. Mean and 95% confidence intervals of the mean square
differences of the states of the original reservoir and the approxi-
mating cyclic dilation systems over 15 randomized generations of
the original system. The data used is labelled in the sub-caption.
6.2. Reduction of dilation dimension with maximum matching in bipartite
graph. In this section we illustrate how the dimension nC of the cyclic dilation
obtained from the maximum matching program in bipartite graphs discussed in
in Remark 8 can yield reservoir sizes drastically lower than the theoretical upper
bound given by the approximating full-cycle permutation P:
n1 = 2 · ℓ0 · (k + 1) >
&
2 ·
π
arccos
 1 −δ2
2
 · (k + 1)
'
.
We generate 10 orthogonal matrices U uniformly randomly for each initial di-
mension n ∈{20, 40, . . . , 140, 160}. We perform cyclic dilation as described in the
previous section and compare it against the theoretical upper bound. Notice that
the y-axis is in log scale. The dimension nC is significantly lower than the theoretical
upper bound, reaching ≈300 −400 units for initial reservoirs of size 80 −160, which
is well within possibilities of hardware implementations of such reservoirs.
7. Conclusion
In this paper, we rigorously demonstrated the universality of Simple Cycle
Reservoir (SCR) in the real domain, adopting the strategy from [LFT24]. Specifically,
we proved that SCRs are universal approximations for any real-valued unrestricted
linear reservoir system and any real-valued time-invariant fading memory filter over
uniformly bounded input streams.
To achieve this, we constrained our approach to the real domain throughout
the approximation pipeline. We performed cyclic dilation of orthogonally dilated
coupling matrices by completing the set of roots of unity in the canonical form of
orthogonal matrices, rather than using the eigendecomposition method involving
unitary matrices. This ensured that all approximant systems remained in the real
domain under orthogonal similarity.
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
13
20
40
60
80
100
120
140
160
Initial Dimension of U
103
104
105
Dilation dimension (log scale)
Comparing theoretical upperbound and dimension obtained maximum matching
Theoretical upperbound n1
Mean of nC
95% CI of nC
Figure 3. Theoretical upperbound v.s. dimension from maximum
matching of bipartite graph.
We facilitated the completion of roots of unity by utilizing a maximum matching
program in bipartite graphs, enabling a tighter dimension expansion of the approxi-
mation system. This method ensured efficient and effective expansion to achieve
the desired approximation accuracy.
The fully constructive nature of our results is a crucial step towards the physical
implementations of reservoir computing [ASdS+11, NTH21, CSK+18, ANH+24].
14
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
References
[ANH+24] Yuki Abe, Kazuki Nakada, Naruki Hagiwara, Eiji Suzuki, Keita Suda, Shin-ichiro
Mochizuki, Yukio Terasaki, Tomoyuki Sasaki, and Tetsuya Asai. Highly-integrable
analogue reservoir circuits based on a simple cycle architecture. Sci. Rep., 14(10966):1–
10, May 2024.
[ASdS+11] Lennert Appeltant, Miguel C. Soriano, Guy Van der Sande, Jan Danckaert, Serge
Massar, Joni Dambre, Benjamin Schrauwen, Claudio R. Mirasso, and Ingo Fischer.
Information processing using a single dynamical node as complex system. Nature
Communications, 2, 2011.
[CSK+18]
Florian Denis-Le Coarer, Marc Sciamanna, Andrew Katumba, Matthias Freiberger,
Joni Dambre, Peter Bienstman, and Damien Rontani. All-Optical Reservoir Computing
on a Photonic Chip Using Silicon-Based Ring Resonators. IEEE Journal of Selected
Topics in Quantum Electronics, 24(6):1 – 8, November 2018.
[GO18a]
L. Grigoryeva and J.-P. Ortega. Universal discrete-time reservoir computers with
stochastic inputs and linear readouts using non-homogeneous state-affine systems. J.
Mach. Learn. Res., 19(1):892–931, January 2018.
[GO18b]
Lyudmila Grigoryeva and Juan-Pablo Ortega. Echo state networks are universal. Neural
Networks, 108:495–508, 2018.
[GO19]
Lukas Gonon and Juan-Pablo Ortega. Reservoir computing universality with stochastic
inputs. IEEE transactions on neural networks and learning systems, 31(1):100–112,
2019.
[Hal50]
Paul R. Halmos. Normal dilations and extensions of operators. Summa Brasil. Math.,
2:125–134, 1950.
[Jae01]
H. Jaeger. The "echo state" approach to analysing and training recurrent neural
networks. Technical report gmd report 148, German National Research Center for
Information Technology, 2001.
[Jae02a]
H. Jaeger. Short term memory in echo state networks. Technical report gmd report
152, German National Research Center for Information Technology, 2002.
[Jae02b]
H. Jaeger. A tutorial on training recurrent neural networks, covering bppt, rtrl, ekf
and the "echo state network" approach. Technical report gmd report 159, German
National Research Center for Information Technology, 2002.
[JH04]
H. Jaeger and H. Haas. Harnessing nonlinearity: predicting chaotic systems and saving
energy in wireless telecommunication. Science, 304:78–80, 2004.
[LFT24]
Boyu Li, Robert Simon Fong, and Peter Tino. Simple Cycle Reservoirs are Universal.
Journal of Machine Learning Research, 25(158):1–28, 2024.
[LJ09]
M. Lukosevicius and H. Jaeger. Reservoir computing approaches to recurrent neural
network training. Computer Science Review, 3(3):127–149, 2009.
[MNM02]
W. Maass, T. Natschlager, and H. Markram. Real-time computing without stable states:
a new framework for neural computation based on perturbations. Neural Computation,
14(11):2531–2560, 2002.
[NTH21]
Mitsumasa Nakajima, Kenji Tanaka, and Toshikazu Hashimoto. Scalable reservoir
computing on coherent linear photonic processor. Communications Physics, 4(1):20,
December 2021.
[Pau02]
Vern Paulsen. Completely bounded maps and operator algebras, volume 78 of Cambridge
Studies in Advanced Mathematics. Cambridge University Press, Cambridge, 2002.
[RT10]
Ali Rodan and Peter Tiňo. Minimum complexity echo state network. IEEE transactions
on neural networks, 22(1):131–144, 2010.
[TD01]
P. Tiňo and G. Dorffner. Predicting the future of discrete sequences from fractal
representations of the past. Machine Learning, 45(2):187–218, 2001.
[ZZP+20]
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong,
and Wan Zhang. Informer: Beyond efficient transformer for long sequence time-series
forecasting. In AAAI Conference on Artificial Intelligence, 2020.
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
15
8. Appendix: proofs of results
8.1. Proof of Theorem 6.
Proof. The uniform boundedness of input stream and contractiveness of W imply
that the state space X ⊆Rn is closed and bounded, hence compact. The continuous
readout map h is therefore uniformly continuous on the state space X.
By the uniform continuity of h, for any ϵ > 0, there exists δ > 0 such that for
any x, x′ ∈X with ∥x −x′∥< δ, we have ∥h(x) −h(x′)∥< ϵ. Let λ = ∥W∥and let
M denote the uniform bound of {ut} such that ∥ut∥≤M for all t. Since λ < 1, we
can choose N, such that:
2M∥V ∥
X
t>N
∥W∥t = 2M∥V ∥λN+1
1 −λ < δ.
Let W1 = W/λ and n′ = (N +1)·n. We have ∥W1∥= 1 and therefore by Egerváry’s
dilation, there exists a orthogonal n′ × n′ matrix U such that for all 1 ≤k ≤N, we
have:
W k
1 = J⊤U kJ,
where J : Rn ,→Rn′ is the canonical embedding of Rn onto the first n-coordinates
of Rn′. Let W ′ = λU, then it follows immediately that:
W k = λkW k
1 = J⊤(λU)k J = J⊤(W ′)k J.
Define an n′ × n matrix by:
(3)
V ′ =
V
0

,
and the map h′ : Rn′ →Rd given by:
h′(x1, x2, · · · , xn, · · · , xn′)
(4)
:= h(x1, x2, · · · , xn) = h ◦J⊤(x1, x2, · · · , xn, · · · , xn′)
We now show that the reservoir system R′ = (W ′, V ′, h′) is ϵ-close to R =
(W, V, h).
For any input stream {ut}t∈Z−, consider the states under the reservoir systems
R and R′ given by:
xt =
X
k≥0
W kV ut−k
x′
t =
X
k≥0
(W ′)k V ′ut−k
(5)
For each k ≥0, we denote the upper left n × n block of (W ′)k by Ak. In other
words:
(W ′)k =

Ak
∗
∗
∗

.
This splits into two cases. For each 0 ≤k ≤N, we have Ak = W k by construction
of W ′. Otherwise, for k > N, the power k is beyond the dilation power bound and
16
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
we no longer have Ak = W k in general. Nevertheless, since Ak is a submatrix of
(W ′)k, it’s operator norm is bounded from above:
∥Ak∥≤
W k ≤
(W ′)k ≤∥W ′∥k = λk.
By Equation (3), we have V ′ut−k =
V ut−k
0

and the state x′
t of R′ from
Equation (5) thus becomes:
x′
t =
X
k≥0
(W ′)k V ′ut−k
=
X
k≥0

Ak
∗
∗
∗
 
V ut−k
0

=
N
X
k=0

W k
∗
∗
∗
 V ut−k
0

+
X
k>N
Ak
∗
∗
∗
 V ut−k
0

=
PN
k=0 W kV ut−k
∗

+
P
k>N AkV ut−k
∗

.
Let J⊤(x′
t) be the first n-coordinates of x′
t. We have
J⊤(x′
t) =
N
X
k=0
W kV ut−k +
X
k>N
AkV ut−k.
Comparing with
xt =
X
k≥0
W kV ut−k =
N
X
k=0
W kV ut−k +
X
k>N
W kV ut−k,
it follows immediately that:
J⊤(x′
t) −xt
 =
0 +
X
k>N
 Ak −W k
V ut−k

≤
X
k>N
 ∥Ak∥+
W k
∥V ∥M
Notice we have ∥W k∥≤∥W∥k = λk and we also showed ∥Ak∥≤λk, and therefore:
∥J⊤(x′
t) −xt∥≤
X
k>N
2λk∥V ∥M < δ
By Equation (4) h′(xt) = h(J⊤(xt)) and by uniform continuity of h we have:
∥yt −y′
t∥= ∥h(xt) −h(J⊤(xt))∥< ϵ
This finishes the proof.
□
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
17
8.2. Proof of Theorem 9.
Proof. Let ϵ > 0 be arbitrary. By the proof of Theorem 6, the state space X is
compact and we can choose δ such that ∥x −x′∥< δ implies ∥h(x) −h(x′)∥< ϵ.
Let M := sup ∥ut∥< ∞, since λ < 1 we can pick N > 0 such that
2M∥V ∥
X
k>N
λk < δ
2.
(6)
Once we fix such an N, pick δ0 > 0 such that
M∥V ∥
N
X
k=0
((λ + δ0)k −λk) < δ
2.
(7)
Such a δ0 exists because the left-hand side is a finite sum that is continuous in δ0
and tends to 0 as δ0 →0. According to Theorem 7, there exists a n1 × n1 full-cycle
permutation matrix P, an orthogonal matrix S, and an orthogonal matrix D such
that:
S⊤PS −
U
0
0
D
 < 1
λ min{δ, δ0}.
Let A = S⊤PS and let Qn : Rn1 ,→Rn be the canonical projection onto
the first n coordinates. Consider the reservoir systems R0 := (W0, V0, h0) and
R1 := (W1, V0, h0) defined by the following:
W0 = λ
U
0
0
D

,
V0 =
V
0

W1 = λA,
h0(x) = h(Qn(x)).
Notice that the choice of A ensures that ∥W1 −W0∥< min{δ, δ0}.
The rest of the proof is outlined as follows: We first show that R0 is equivalent
to R, and then prove that R1 is ϵ-close to R0. By Theorem 7, A is orthogonally
equivalent to a full-cycle permutation matrix, and the desired results follow from
[LFT24, Proposition 12]. We now flesh out the above outline: We first establish
that R0 is equivalent to R. For any input stream {ut}t∈Z−, the solution to R0 is
given by
y(0)
t
= h0

X
k≥0
W k
0 V0ut−k


= h

Qn

X
k≥0
(λU)k
0
0
(λD)k
 V
0

ut−k




= h

Qn
P
k≥0 W kV ut−k
0

= h

X
k≥0
W kV ut−k

.
This is precisely the solution to R.
We now show that R1 is ϵ-close to R0.
First, we observe that since Qn is a projection onto the first n-coordinates, it
has operator norm ∥Qn∥= 1 and thus whenever ∥x −x′∥< δ, x, x′ ∈Rn1, we have
18
ROBERT SIMON FONG, BOYU LI, AND PETER TIŇO
∥Qnx −Qnx′∥< δ and thus ∥h(Qnx)−h(Qnx′)∥< ϵ. Therefore it suffices to prove
that for any input {ut}, the solution to R0, given by
x(0)
t
=
X
k≥0
W k
0 V0ut−k,
is within δ to the solution to R1, given by
x(1)
t
=
X
k≥0
W k
1 V0ut−k.
By construction V0 =
V
0

has ∥V0∥= ∥V ∥, hence:
x(0)
t
−x(1)
t
 =

X
k≥0
(W k
0 −W k
1 )V0ut−k

≤
X
k≥0
(W k
0 −W ′k
1 )
 ∥V0∥M
=
N
X
k=0
 W k
0 −W k
1
 ∥V ∥M
+
X
j>N
 W k
0 −W k
1
 ∥V ∥M.
(8)
Consider ∆= W0 −W1, we then have ∥∆∥< δ0 and for each 0 ≤j ≤N,
W j
0 −W j
1 = (W1 + ∆)j −W j
1 . Expanding (W1 + ∆)j, we get a summation of 2j
terms of the form Qj
i=1 Xi, where each Xi = W1 or ∆. For s = 0, . . . , j, each of
the 2j terms has norm
Qj
i=1 Xi
 ≤∥W1∥j−s∥∆∥s if there are s copies of ∆among
Xi. Removing the term W j
1 from (W1 + ∆)j results in all the remaining terms
containing at least one copy of ∆. We thus arrive at:
W j
0 −W j
1
 =
(W1 + ∆)j −W j
1

≤
j
X
s=1
j
s

∥W1∥j−s ∥∆∥s
≤
j
X
s=1
j
s

λj−sδs
0 = (λ + δ0)j −λj.
(9)
Combining the above with Equation (7), we have:
M∥V ∥
N
X
j=0
∥(W j
0 −W j
1 )∥≤M∥V ∥
N
X
j=0
((λ + δ0)j −λj) < δ
2.
UNIVERSALITY OF REAL MINIMAL COMPLEXITY RESERVOIR
19
On the other hand by Equation (6), we obtain:
M∥V ∥
X
j>N
∥(W j
0 −W j
1 )∥
≤
M∥V ∥
X
j>N
(∥W0∥j + ∥W1∥j)
≤
2M∥V ∥
X
j>N
λj
<
δ
2.
With the two inequalities above, Equation (8) thus becomes:
∥x(0)
t
−x(1)
t ∥< δ.
Uniform continuity of h implies
h(x(0)
t ) −h(x(1)
t )
 < ϵ, proving R1 is ϵ-close to
R0. Finally, by Theorem 7, A is orthogonally equivalent to a full-cycle permutation
matrix P, i.e. there exists orthogonal matrix S such that S⊤AS = P. By [LFT24,
Proposition 12], we obtain a reservoir system Rc = (Wc, Vc, hc) with Wc = λP,
such that Rc is equivalent to R1, which is in turn ϵ-close to R0. Since the original
reservoir system R is equivalent to R0, R is therefore ϵ-close to Rc, as desired.
□
School of Computer Science, University of Birmingham, Birmingham, B15 2TT, UK
Email address: r.s.fong@bham.ac.uk
Department of Mathematical Sciences, New Mexico State University, Las Cruces,
New Mexico, 88003, USA
Email address: boyuli@nmsu.edu
School of Computer Science, University of Birmingham, Birmingham, B15 2TT, UK
Email address: p.tino@bham.ac.uk
