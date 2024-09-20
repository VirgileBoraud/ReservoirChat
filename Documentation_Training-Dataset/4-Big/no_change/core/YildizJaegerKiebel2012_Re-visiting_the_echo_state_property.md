Re-Visiting the Echo State Property
Izzet B. Yildiza,∗, Herbert Jaegerb, Stefan J. Kiebela
aMax Planck Institute for Human Cognitive and Brain Sciences, Leipzig, Germany
bJacobs University, Bremen, Germany
Abstract
An echo state network (ESN) consists of a large, randomly connected neural network, the
reservoir, which is driven by an input signal and projects to output units. During training,
only the connections from the reservoir to these output units are learned. A key requisite
for output-only training is the echo state property (ESP), which means that the eﬀect of
initial conditions should vanish as time passes. In this paper, we use analytical examples
to show that a widely used criterion for the ESP, the spectral radius of the weight matrix
being smaller than unity, is not suﬃcient to satisfy the echo state property. We obtain these
examples by investigating local bifurcation properties of the standard ESNs. Moreover, we
provide new suﬃcient conditions for the echo state property of standard sigmoid and leaky
integrator ESNs. We furthermore suggest an improved technical deﬁnition of the echo state
property, and discuss what practicians should (and should not) observe when they optimize
their reservoirs for speciﬁc tasks.
Keywords:
echo state network, spectral radius, bifurcation, diagonally Schur stable,
Lyapunov
1. Introduction
Echo state networks (ESN) [8, 12] provide an architecture and supervised learning principle
for recurrent neural networks (RNNs). The main idea is (i) to drive a random, large, ﬁxed
recurrent neural network with the input signal, thereby inducing in each neuron within this
“reservoir” network a nonlinear response signal, and (ii) combine a desired output signal by
a trainable linear combination of all of these response signals. The internal weights of the
underlying reservoir network are not changed by the learning; only the reservoir-to-output
connections are trained.
This basic functional principle is shared with Liquid State Machines (LSM), which were
developed independently from and simultaneously with ESNs by Wolfgang Maass [19]. An
∗Corresponding Author: Max Planck Institute for Human Cognitive and Brain Sciences, Stephanstrasse
1A, 04103 Leipzig,
Germany.
Phone:
+49 341 9940-2216,
Fax:
+49 341 9940-2221,
Email:
yildiz@cbs.mpg.de, Website: www.cbs.mpg.de/~yildiz
Preprint submitted to Elsevier
June 23, 2012
2
earlier precursor is a biological neural learning mechanism investigated by Peter F. Dominey
in the context of modeling sequence processing in mammalian brains [6]. Increasingly often,
LSMs, ESNs and some other related methods are subsumed under the name of reservoir
computing (introduction: [10], survey of current trends: [18]). Today, reservoir computing
has established itself as one of the standard approaches to supervised RNN training.
A crucial, enabling precondition for ESN learning algorithms to function is that the underly-
ing reservoir network possesses the echo state property (ESP). Roughly speaking, the ESP is
a condition of asymptotic state convergence of the reservoir network, under the inﬂuence of
driving input. The ESP is connected to algebraic properties of the reservoir weight matrix,
and to properties of the driving input. It is a rather subtle mathematical concept. Often the
ESP is violated if the spectral radius of the weight matrix exceeds unity. Conversely, under
rather general conditions, the ESP is obtained most of the time when the spectral radius is
smaller than unity. This combination of facts has led to a widespread misconception that
all one has to observe in order to obtain the ESP is to scale the reservoir weight matrix to
a spectral radius below unity. We witness that a signiﬁcant fraction – even a majority – of
“end-users” of reservoir computing fall prey to this misconception. In fact, neither does a
spectral radius below unity generally ensure the ESP, nor does a spectral radius above unity
generally destroy it. In numerous applications – depending on the nature of the driving
input and on the nature of the desired readout signal – a spectral radius well above unity
serves best. The widespread practice of scaling the spectral radius to below unity thus leads
to an under-exploitation of the learning and modeling capacities of reservoirs.
Here we re-visit the ESP, with the general aim to illuminate this concept from several sides for
the practical beneﬁt of reservoir computing practice. Besides this didactic goal, the technical
contribution of this article is twofold. First, after summarizing the standard formalism and
ESP deﬁnition in Section 2, we present a bifurcation analysis to show in detail how the
ESP can be lost even for spectral radii below unity (Section 3). Second, we derive a new,
convenient-to-use formulation of a suﬃcient algebraic criterion for the ESP (Section 4).
Then, in Section 5, we comment on situations where the ESP is obtained for spectral radii
exceeding unity, which are of signiﬁcant practical importance. We conclude with a short
appreciation of the entire subject in a ﬁnal discussion section.
2. Echo state networks
In this section we deﬁne the standard ESN and the echo state property.
The standard discrete-time ESN, which we denote shortly by xk+1 = F(xk, xout
k , uk+1), is
deﬁned as follows:
xk+1 = f(Wxk + W inuk+1 + W fbxout
k ),
(1)
xout
k
= g(W out[xk ; uk]),
where W ∈RN×N is the internal weight matrix or the reservoir, W in ∈RN×K is the input
matrix, W fb ∈RN×L is the feedback matrix, W out ∈RL×(N+K) is the output matrix and
3
Figure 1: The basic structure of an ESN. Solid arrows denote the ﬁxed connections and dashed arrows denote
the trainable connections.
xk ∈RN×1, uk ∈RK×1 and xout
k
∈RL×1 are the internal, input and output vectors at time
k, respectively (see Figure 1). The state activation function f = (f1, . . . , fN)T is a sigmoid
function (usually fi = tanh) applied component-wise with f(0) = 0 and the output activa-
tion function is g = (g1, . . . , gL)T where each gi is usually the identity or a sigmoid function.
[ ; ] denotes vector concatenation and xT denotes the transpose of a vector x.
Here we consider only ESNs without feedback, i.e. W fb = 0. The echo state network F with
no feedback connection becomes:
xk+1 = F(xk, uk+1) = f(Wxk + W inuk+1),
(2)
For the supervised learning algorithms which are used with ESNs (e.g. [8, 18]) it is crucial
that the current network state xk is uniquely determined by any left-inﬁnite input sequence
. . . , uk−1, uk. This is made precise by requesting the echo state property (ESP). Since this ef-
fect depends on the input sequence, the deﬁnition of the ESP is stated relative to constraining
the input range to a compact set U.
Concretely, we require the compactness condition which means F is deﬁned on X ×U where
X ⊂RN, U ⊂RK are compact sets and F(xk, uk+1) ∈X and uk ∈U, ∀k ∈Z. Note that
the compactness of the state space X is automatically warranted when the reservoir unit
nonlinearity f is bounded, like the tanh or the logistic sigmoid. Furthermore, in practical
applications the input will always be bounded, so compactness of U will typically be war-
ranted too.
Let U−∞:= {u−∞= (. . . , u−1, u0) | uk ∈U ∀k ≤0} and X−∞:= {x−∞= (. . . , x−1, x0) | xk ∈
X ∀k ≤0} denote the set of left inﬁnite input and state vector sequences, respectively. We
say x−∞is compatible with u−∞when xk = F(xk−1, uk), ∀k ≤0.
The deﬁnition of the echo state property when W fb = 0 is as follows (adopted from [8]):
4
Deﬁnition 2.1 (echo state property). A network F : X × U →X (with the compactness
condition) has the echo state property with respect to U: if for any left inﬁnite input sequence
u−∞∈U−∞and any two state vector sequences x−∞, y−∞∈X−∞compatible with u−∞, it
holds that x0 = y0.
This “backward-oriented” deﬁnition can be equivalently stated in a forward direction. We
remark that the original forward version given in [8] was too weak, and here present the
corrected version [11]. Similarly, let U+∞:= {u+∞= (u1, u2, . . .) | uk ∈U ∀k ≥1} and
X+∞:= {x+∞= (x0, x1, . . .) | xk ∈X ∀k ≥0} denote the set of right-inﬁnite input and
state sequences, respectively. Then,
Theorem 2.1 (Forward speciﬁcation of ESP). A network F : X × U →X (with the
compactness condition) satisﬁes the echo state property with respect to U if and only if it
has the uniform state contraction property, i.e. if there exists a null sequence (δk)k≥0 such
that for all u+∞∈U+∞, for all x+∞, y+∞∈X+∞compatible with u+∞, it holds that for all
k ≥0, ∥xk −yk∥≤δk.
For practical work with ESNs, it was mentioned in [8] that one usually obtains the echo
state property by taking a random W and scaling it so that its spectral radius ρ(W) is
smaller than unity, where the spectral radius is the maximum of the absolute values of the
eigenvalues of W. Although this recipe is used widely in reservoir computing practice, it is
neither necessary nor suﬃcient to ensure the echo state property. In the next section, we
investigate in more detail how and why it is not suﬃcient, and in Section 4 we provide a new,
suﬃcient condition which is more practical than the current best known condition given in
[4]. In Section 5 we discuss relevant implications of the fact that it is not necessary, and
comment on shortcomings of the current deﬁnition of the ESP.
3. Bifurcations in 2-dim echo state networks
Here we investigate ESNs with internal weight matrix W and a spectral radius ρ(W) < 1
where the network does not have the echo state property. We will constrain our analysis to
the constant zero-input case, that is, U = {0}, because this basic case supports the present
arguments already. In other words, we are interested in W matrices with ρ(W) < 1 for
which the system
xk+1 = f(Wxk)
(3)
does not have the echo state property. In particular, we investigate some bifurcation types
which yield systems with non-trivial ﬁxed points and periodic orbits. Note that the zero
state (origin) is always a ﬁxed point of the above system since f(0) = 0. For linear systems,
xk+1 = Wxk, the origin is indeed the global attractor of the system when ρ(W) < 1. The
question is: For nonlinear systems, is the origin always the global attractor of the system in
Eq. 3 when ρ(W) < 1? The answer is no and we give analytical examples below.
5
3.1. One dimensional case
When xk ∈R and f = tanh, the ﬁxed points can be found by solving x = tanh(wx) where
w ∈R. In this case, the spectral radius is w. If w > 1, the origin is unstable and therefore
we do not have the echo state property. If w < 1, then tanh(wx) is a contraction map
and therefore the origin is globally asymptotically stable. Therefore, the examples we are
interested in do not exist in one dimension.
3.2. Two dimensional case
The analysis in two dimension is non-trivial. For a 2 ×2 weight matrix W =
!
w11
w12
w21
w22
"
,
the ﬁxed points satisfy:
!
x1
x2
"
=
!
tanh(w11x1 + w12x2)
tanh(w21x1 + w22x2)
"
The ﬁrst question is, for which values of wij the matrix W has a spectral radius smaller than
1? This question was studied to investigate the region of stability for linear systems and is
the well-known stability triangle [23]:
∆= {W ∈R2×2 : |tr(W)| −1 < det(W) < 1}
where tr(W) = w11 + w22 is the trace of W and det(W) = w11w22 −w12w21 is the determi-
nant of W. One can show this by computing the eigenvalues of W directly and imposing the
condition that both eigenvalues should have norm smaller than 1. Therefore, we are looking
for W matrices in ∆for which the origin is not globally asymptotically stable.
Since the nonlinear system can be analyzed locally using the linear system and the origin
is asymptotically stable in the linear case (in ∆), we can conclude that the origin is locally
asymptotically stable in the nonlinear case. But there may be points away from the origin
that are not attracted to the origin.
To be able to investigate this two dimensional system using one dimensional techniques,
we also assume that w11 = 0. Then tr(W) = w22 and det(W) = −w12w21. We call the
corresponding stability triangle ∆0 (see Fig. 2). The equation for ﬁxed points reduces to:
!
x1
x2
"
=
!
tanh(w12x2)
tanh(w21x1 + w22x2)
"
(4)
Note that it is enough to consider (x1, x2) ∈(−1, 1) × (−1, 1) since all states are mapped by
tanh into this set after one iteration. The ﬁxed points of this system can be found by solving
x1 = tanh(w12x2) and x2 = tanh(w21x1 + w22x2).
The second equation can be written
as arctanh(x2) = w21x1 + w22x2. Plugging the ﬁrst equation into the second one, we get
6
∆0
1
1
−1
−1
−2
2
det(W) = −w12w21
tr(W) = w22
ℓ1
ℓ2
Figure 2: The triangular region ∆0 in the determinant-trace space. First, under some algebraic conditions
described in the text, the system goes through a degenerate bifurcation along the side ℓ1 of ∆0 (shown by
circles on ℓ1). Then, pitchfork bifurcations occur towards the inside of ∆0 which are shown by the horizontal
arrows. The diamond-shaped region with dashed boundary gives the parameters for which W is diagonally
Schur stable and therefore, no bifurcations can occur inside this region.
7
arctanh(x2) = w21tanh(w12x2) + w22x2. Therefore, we need to ﬁnd the ﬁxed points of the
function ϕ(x2) which for simplicity we write as ϕ(x), where ϕ(x) is:
ϕ(x) = −1
w22
(w21tanh(w12x) −arctanh(x)).
If this function, with W in the stability triangle, has a ﬁxed point other than the origin, the
echo state property does not hold.
Remark: Note that ϕ is not the system update equation for x2, i.e. it does not hold that
(x2)k+1 = ϕ((x2)k). The following bifurcation analysis of ϕ(x) will only inform us about the
creation of new ﬁxed points for the system (4) as we move through parameter space within
the stability triangle. The stability of these newborn ﬁxed points has to be checked using
local linearization (Jacobian) of the two dimensional system. Note that the origin is always
going to be locally attracting (therefore stable) when the parameters are in ∆0.
Remark: In Section 4.1, we will show that if W is diagonally Schur stable then the echo
state property is satisﬁed for all inputs. In dimension 2, these matrices form the following
set [14]: {W ∈R2×2 : |det(W)| < 1, |w11+w22| < 1+det(W) and |w11−w22| < 1−det(W)}.
This corresponds to the diamond-shaped region in ∆0 whose boundary is shown with dashed
lines in Fig. 2. Therefore, bifurcations can only exist outside of this region.
First, we would like to show that along one of the sides of ∆0, i.e. along ℓ1 = {wij | w11 =
0, w22 = −w12w21 + 1 and 0 ≤w22 ≤2}, for a given ﬁxed determinant and trace value (see
little circles on the side ℓ1 of ∆0 in Fig. 2), it is possible to change w12 and w21 appropriately
(keeping the determinant and trace constant) so that a degenerate bifurcation occurs. This
bifurcation creates two more ﬁxed points away from the origin. Moreover, following this de-
generate bifurcation, if w12 and w21 are changed so that this time the determinant is increased
and moved towards the inside of ∆0 (horizontal arrows in the upper part of Fig. 2), then
a pitchfork bifurcation occurs which creates two more ﬁxed points from the origin. There-
fore, we obtain examples which do not satisfy the echo state property even though ρ(W) < 1.
First, we look at the Taylor series expansion of ϕ(x):
ϕ(x) =
−w21
w22
!
w12x −(w12x)3
3
+ O(x5)
"
+
1
w22
!
x + x3
3 + O(x5)
"
=
−1
w22
(w12w21 −1)x +
1
3w22
(1 + w3
12w21)x3 + O(x5)
=
px + qx3 + O(x5)
(5)
where p = −1
w22(w12w21−1) = (1+det(W))/tr(W) and q =
1
3w22(1+w3
12w21). We investigate
the following bifurcations:
8
3.2.1. Degenerate Bifurcations
We start with bifurcations that occur on the line ℓ1 at some ﬁxed (det(W), tr(W)) = (c, c+1)
value where det(W) = c > 0. Note that there are many w12 and w21 values which give the
same determinant, i.e. −w12w21 = c. Also note that p equals 1 on ℓ1 and q =
1
3w22(1+w3
12w21).
We want to show that q changes sign as w12 and w21 are changed along the curve −w12w21 = c
(see Fig. 3). Note that q = 0 if and only if 1 + w3
12w21 = 0. Solving these two equations
together, i.e. −w12w21 = c and 1 + w3
12w21 = 0, one gets w12 = ±1/√c and w21 = ∓c√c. In
fact, when |w12| < 1/√c then q > 0 and when |w12| > 1/√c then q < 0. This means that
the sign of q changes at this value.
To understand why the change of sign of q creates new ﬁxed points, it is helpful to con-
sider y = ϕ(x) −x since the x-intercepts of this function gives the ﬁxed points. Note that
y = ϕ(x) −x ≈qx3 around the origin and the sign of q changes this function locally (see
Fig. 4). On the other hand, as x →1−and x →−1+, the arctanh(x) term in the deﬁni-
tion of ϕ(x) becomes dominant and therefore y = ϕ(x) −x ≈
1
w22arctanh(x). This means,
independent of the sign of q, y = ϕ(x) −x approaches to +∞and −∞as x →1−and
x →−1+, respectively. Therefore, the local change in y = ϕ(x) −x creates two new ﬁxed
points away from the origin as shown in Fig. 4. The existence of these ﬁxed points can be
shown rigorously using the Intermediate Value Theorem.
However, when det(W) = c < 0, i.e.
w12w21 > 0, we obtain q =
1
3w22(1 + w3
12w21) =
1
3w22(1 + w2
12(w12w21) > 0 (w22 = tr(W) > 0 along ℓ1). In other words, the sign of q does not
change and no bifurcation occurs.
There are further bifurcations as the parameters are moved towards the inside of ∆0.
3.2.2. Pitchfork Bifurcation:
Eq. (5) is similar to the normal form of pitchfork bifurcation which is px + qx3 where q < 0
for the supercritical case and q > 0 for the subcritical case (e.g. see [16]). In the supercritical
case (q < 0), as p changes from p < 1 to p > 1, the ﬁxed point at the origin changes its
stability from stable to unstable, and two new, stable ﬁxed points appear. In the subcritical
case (q > 0), as p changes from p > 1 to p < 1, the ﬁxed point at the origin changes its
stability and from unstable to stable, and two new, unstable ﬁxed points appear.
We have shown above that for ﬁxed values of (det(W), tr(W)) on ℓ1, q can be positive or
negative depending on w12 and w21. Now, we look at what happens when tr(W) is ﬁxed but
the determinant is increased so that we move from ℓ1 towards the inside of ∆0:
The case det(W) > 0 and q < 0:
When det(W) = −w12w21 = c > 0, as mentioned in the degenerate bifurcation case, there
exist w12 and w21 values such that q < 0. If we perturb w12 and w21 slightly, q stays negative
since it is a continuous function. Therefore, one can indeed increase det(W) = −w12w21
9
0
0
w12
w21
 
 
1 + w3
12w21 = 0
-w12w21 = c
- 1
√c
q > 0
q < 0
q > 0
1
√c
q < 0
Figure 3: The description of the degenerate bifurcation at a point (det(W), tr(W)) = (c, c + 1) where
0 < c < 1 (shown with little circles on the side ℓ1 of ∆0 in Fig. 2). The degenerate bifurcation occurs as the
sign of q =
1
3w22 (1 + w3
12w21) changes from positive to negative. The parameters where this change of sign
occurs are given by the curve 1 + w3
12w21 = 0. Since we also have the constraint det(W) = −w12w21 = c,
the bifurcations we describe in the text occur in the direction of the two black arrows.
slightly and preserve q < 0. As det(W) increases, p = −1
w22(w12w21 −1) changes from p = 1
to p > 1 (tr(W) = w22 is ﬁxed but det(W) increases towards the inside of ∆0, see the
horizontal arrows in the upper part of Fig. 2). Therefore, we observe a pitchfork bifurcation
where the origin changes to stable (since we move towards the inside of ∆0) and two new
ﬁxed points appear in addition to the existing ﬁxed points that appeared in the degenerate
bifurcation (see Fig. 5). Thus, we now have ﬁve ﬁxed points of ϕ. To further characterize
the original two dimensional system, one can numerically compute the basin of attraction
for each ﬁxed point. Note that in Fig. 6, there are two attracting ﬁxed points in addition to
the origin itself. These are the ﬁxed points that were born from the degenerate bifurcation
described previously. By computing the Jacobians, one can see that that the new ﬁxed points
born from the pitchfork bifurcation correspond to saddle points in the original dynamics (4)
(see Fig. 6). This gives an example of the case where ρ(W) < 1 and the echo state property
does not hold.
The case det(W) > 0 and q > 0:
When det(W) > 0, as described previously, there exist w12 and w21 values such that q > 0.
As p = −1
w22(w12w21 −1) changes from p > 1 to p < 1, a pitchfork bifurcation occurs but
this direction is towards the outside of ∆0 and therefore not relevant here.
Remark: One can observe numerically that under some conditions, similar bifurcations can
10
−0.5
0
0.5
−0.04
−0.03
−0.02
−0.01
0
0.01
0.02
0.03
0.04
x
y
 
 
det = 0.1 q>0
det = 0.1 q<0
Figure 4: The degenerate bifurcation occurs at the value (det(W), tr(W)) = (0.1, 1.1) as q changes sign.
Here, we draw y = ϕ(x) −x for easier visualization of the ﬁxed points which corresponds to the x-intercepts
of the graphs. Two new ﬁxed points appear which are marked by asterisks. The entries used are w11 = 0,
w12 = 2, w21 = −0.05 and w22 = 1.1 (q ≈0.18) for the dashed curve and w11 = 0, w12 = 10, w21 = −0.01
and w22 = 1.1 (q ≈−2.7) for the solid curve. In both cases, ρ(W) = 1.
11
−0.5
0
0.5
−0.01
−0.005
0
0.005
0.01
x
y
 
 
det = 0.1
det = 0.12
Figure 5: The pitchfork bifurcation occurs along the side ℓ1 of ∆0 as the parameters are moved towards
the inside of ∆0 (see the horizontal arrows in the upper part of Fig. 2). Here, we draw y = ϕ(x) −x for
easier visualization of the ﬁxed points which corresponds to the x-intercepts of the graphs. Two new saddle
ﬁxed points appear which are marked by circles. The entries used are w11 = 0, w12 = 10, w21 = −0.01 and
w22 = 1.1 for the dashed curve (which is the same as the solid curve in Fig. 4, ρ(W) = 1) and w11 = 0,
w12 = 10, w21 = −0.012 and w22 = 1.1 for the solid curve (ρ(W) = 0.9772).
12
Figure 6: The shaded region is the basin of attraction for the origin of the system with internal weight matrix
w11 = 0, w12 = 10, w21 = −0.012 and w22 = 1.1. All the points (x1, x2) above this region are attracted to
the stable ﬁxed point (0.999, 0.438) and all the points (x1, x2) below this region are attracted to the ﬁxed
point (−0.999, −0.438). Two saddle ﬁxed points born from the pitchfork bifurcation are shown with circles
and they lie on the boundary of the shaded region.
be observed along the side ℓ2 of ∆0 (Fig. 2) where the pitchfork bifurcation is replaced by
period doubling bifurcation as the parameters are moved towards the inside of ∆0. Since
this requires the investigation of the second iteration of the system, we do not explore it
analytically here.
3.2.3. Neimark-Sacker Bifurcation
The Neimark-Sacker bifurcation is the discrete-time version of the Hopf bifurcation in the
continuous case where the ﬁxed point changes stability because of a pair of complex eigen-
values on the unit circle. At the critical parameter, the origin is surrounded by a closed
invariant curve which can attract or repel nearby orbits. In dimension 2, we observed the
Neimark-Sacker bifurcation only towards the outside of the stability triangle. However, one
can ﬁnd higher dimensional examples with ρ(W) < 1 where there exists an invariant orbit
around the origin and therefore the echo state property is not satisﬁed. A numerically found,
4-dimensional example can be given with the internal weight matrix:
W =





0
1.95
3.3
1.56
0
−0.47
−1.38
1.3
0
0
0
1.95
0.23
0.67
−1.13
0




.
(6)
13
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
x
y
Figure 7: Illustration of an example with ρ(W) < 1 for which the echo state property does not hold. The
orbits of arbitrary initial points approach an invariant set where the internal weight matrix is given in Eq. 6.
The orbit of an initial point (the ﬁrst 8000 iterations are not shown) is attracted to a 4-dimensional high-
order periodic orbit for which the dots show the projection to the ﬁrst two coordinates and the asterisks
show the projection to the last two coordinates.
The eigenvalues of W are 0.2546 ± 0.9201i and −0.4896 ± 0.5668i where the spectral radius
is 0.9547. The invariant set which is the orbit of a high-order periodic point can be observed
by taking arbitrary initial values and observing their forward iterations (Fig. 7).
3.3. Higher dimensions
The direct analysis of bifurcations in higher dimensions gets complicated quickly because of
the number of parameters involved. However, one can always ﬁnd examples in any dimen-
sion by generalizing from the lower dimensional examples. For example, let W ∈RN×N be
one of the interesting examples found above where the echo state property is not satisﬁed
even though ρ(W) < 1. Then, let J1, J2 be two positive integers and consider the matrix
¯W =
!
W
Q
0
R
"
where Q ∈RN×J1 is an arbitrary matrix (no constraints) and R ∈RJ2×J1
is an arbitrary matrix with ρ(R) < 1. This ensures that ρ( ¯W) < 1 and if z ∈RN×1 is a ﬁxed
point of the system given by Eq. 3, then ¯z = [z; 0] ∈R(N+J1)×1 is a ﬁxed point of the higher
dimensional system with weight matrix ¯W.
In summary, using bifurcation analysis, we have shown that the condition ρ(W) < 1 is not
a suﬃcient condition for the echo state property. The question remains how one can adapt
this condition to establish the echo state property.
14
4. New suﬃcient conditions for the echo state property
In this section we provide suﬃcient conditions for the echo state property of the standard and
leaky integrator ESNs. These suﬃcient conditions are important because, in practice, less
restrictive conditions are typically used which do not guarantee the echo state property. In
the standard ESNs (Eq. 1), one usually samples a random internal weight matrix W with a
subsequent scaling of the connectivity matrix W to ensure that its spectral radius is less than
unity, ρ(W) < 1. In Section 3, we gave analytical examples for a simple but instructive case
where the condition ρ(W) < 1 is insuﬃcient to guarantee the echo state property. Similarly,
for the leaky integrator ESNs, the commonly used condition, the eﬀective spectral radius
(see Section 4.2) being smaller than unity, is insuﬃcient to obtain the echo state property as
shown below in Example 4.1. Here, we derive new suﬃcient conditions and simple recipes
to obtain the echo state property.
4.1. A suﬃcient condition for the standard ESN
A rather restrictive condition for the echo state property of the standard ESN was given in
[8] as ¯σ(W) < 1 where ¯σ(W) denotes the maximum singular value of W. Since this condition
is too restrictive and the input is washed out very fast, it is not commonly used in practice.
In this part, we state and prove Theorem 4.1 which provides a less restrictive condition in
terms of diagonal Schur stability. We show that this condition is equivalent to the condition
described in [4]. The advantage of the Schur stability condition is that it is well studied in
the literature. This enables us to list some important types of matrices which can be used
as internal weight matrices guaranteed to have the echo state property. Finally, we give a
simple recipe to obtain internal weight matrices which satisfy the echo state property.
We ﬁrst deﬁne the following set of matrices which are important for the present analysis:
Deﬁnition 4.1. A matrix W ∈RN×N is called Schur stable if there exists a positive deﬁnite
symmetric matrix P > 0 such that W TPW −P is negative deﬁnite. If the matrix P can be
chosen as a positive deﬁnite diagonal matrix, then W is called diagonally Schur stable. The
positive deﬁnite and negative deﬁnite matrices are denoted by P > 0 and P < 0, respectively.
The notion of diagonal Schur stability is enough to state our result:
Theorem 4.1. The network given by Eq. (2) with internal weight matrix W satisﬁes the
echo state property for any input if W is diagonally Schur stable, i.e. there exists a diagonal
P > 0 such that W TPW −P is negative deﬁnite.
The proof of this theorem is given in the Appendix.
The diagonal Schur stability was investigated in [3] and more recently, in [14], the authors
included matlab code for checking whether a given matrix is diagonally Schur stable. More-
over, the following set of matrices are proven to be diagonally Schur stable. Therefore, the
15
echo state property for all inputs is satisﬁed for internal weight matrices which fulﬁll one of
the following criteria:
• W = (wij) such that ρ(|W|) < 1 where |W| = (|wij|).
• W = (wij) such that wij ≥0, ∀i, j and ρ(W) < 1.
• W such that ρ(W) < 1 and there exists a nonsingular diagonal D such that D−1WD
is symmetric (this also includes symmetric matrices).
• W is a triangular matrix and ρ(W) < 1.
• W ∈R2×2, |det(W)| < 1, |w11 + w22| < 1 + det(W) and |w11 −w22| < 1 −det(W).
More examples such as quasidominant and checkerboard matrices are given in [14] with the
relevant deﬁnitions.
A simple recipe for the echo state property of the standard ESNs
The condition ρ(|W|) < 1 where |W| = (|wij|) gives a simple way to construct internal
weight matrices for the standard ESNs that satisfy the echo state property:
(i) Start with a random W with all non-negative entries, wij ≥0.
(ii) Scale W so that ρ(W) < 1.
(iii) Change the signs of a desired number of entries of W to get negative connection weights
as well.
Note that this recipe is more restrictive than the necessary condition ρ(W) < 1; however,
the echo state property is guaranteed for any input.
Remark: The diagonal Schur stability condition turns out to be equivalent to the con-
dition given in [4].
The authors have shown that the echo state property is satisﬁed if
inf
D∈D ¯σ(DWD−1) < 1 where D is the set of nonsingular diagonal matrices and ¯σ is the
largest singular value.
The equivalence of this condition to the diagonal Schur stability
can be obtained by noting [21]: (i) The inﬁmum does not change when we restrict D to
be the set of positive diagonal matrices since any nonsingular diagonal matrix D can be
written as D = UD+ where U is a diagonal matrix with entries ±1 (therefore unitary)
and D+ is a positive diagonal matrix. Therefore, inf
D∈D ¯σ(DWD−1) = inf
D+∈D ¯σ(D+WD−1
+ ), (ii)
The following equivalences hold: ¯σ(DWD−1) < 1 ⇔ρ(D−1W TDDWD−1) < 12 = 1 ⇔
D−1W TDDWD−1 −I < 0 ⇔W TD2W −D2 < 0 where the last inequality is equivalent to
being diagonally Schur stable.
16
Next, we use similar techniques to ﬁnd suﬃcient conditions for the echo state property in
the leaky integrator case.
4.2. Suﬃcient conditions for leaky integrator ESNs
In this section, we will ﬁrst describe the leaky integrator ESNs and state the discretized
version. Then, in Theorem 4.2, we will deﬁne a new system (Eq. 8) such that its stability
implies the echo state property for the discretized leaky integrator ESN (Eq. 7) with no
feedback. There are various ways to check the stability of this new system and we give two
closely related conditions as corollaries. One of these corollaries provides a simple recipe to
obtain internal weight matrices with the echo state property.
For dealing with slowly and continuously changing systems, a continuous version of ESN was
introduced in [8] and investigated in more detail in [13]. It is deﬁned by:
˙x = 1
c(−ax + f(W inu + Wx + W fbxout)),
xout = g(W out[x ; u]),
where the matrices W in ∈RN×K, W ∈RN×N, W out ∈RL×(K+N) and W fb ∈RN×L are the
input, internal, output and feedback connection weight matrices, respectively. u ∈RK is the
external input, x ∈RN is the internal weight activation state, xout ∈RL is the output vector,
c > 0 is a global time constant, a > 0 is the leaking rate, f is a sigmoid function (usually
tanh applied component-wise), g is the output activation function (usually the identity or a
sigmoid) and [ ; ] denotes vector concatenation.
The Euler discretization with step-size δ > 0 gives the following discrete-time version with
a discrete-time sampled input uδ
k. We use the notation ∆t = δ
c for simplicity:
xk+1 = (1 −a∆t)xk + ∆tf(W inuδ
k+1 + Wxk + W fbxout
k ),
(7)
xout
k
= g(W out[xk ; uδ
k]).
It is assumed that a∆t < 1.
A suﬃcient condition for the echo state property of this
discretized version with no feedback connection was given in [8]: If |1 −∆t(a −σmax)| < 1
where σmax is the largest singular value of W, then the echo state property is satisﬁed.
Furthermore, it was also stated that if the matrix ˜W = ∆tW + (1 −a∆t)I where I is
the identity matrix, has a spectral radius ρ( ˜W) > 1 then the echo state property (for zero
input) is not satisﬁed since the origin becomes unstable. In practice, the necessary condition
ρ( ˜W) < 1 is used to obtain stable leaky integrator ESN’s where ρ( ˜W) is called the eﬀective
spectral radius. However, one should be aware that the necessary condition ρ( ˜W) < 1 is
indeed not suﬃcient and counterexamples can be given where the echo state property is not
satisﬁed:
17
Example 4.1. Let a = 1, c = 1, δ = 0.1 (∆t = 0.1) and ˜W = ∆tW + (1 −a∆t)I =
!
0
10
−0.012
1.1
"
. Note that the eﬀective spectral radius, ρ( ˜W) = 0.9772 < 1. However,
the system given in Eq. 7 with zero-input, no feedback and internal weight matrix W, has
ﬁxed points other than the origin and therefore it does not satisfy the echo state property.
In particular, the ﬁxed points (−0.999, −0.943) and (0.999, 0.943) also attract nearby points.
The phase space is very similar to the one given in Fig. 6 which was investigated for the
standard ESNs in more detail in Section 3.
Since the commonly used eﬀective spectral radius condition, ρ( ˜W) < 1, is not suﬃcient for
the echo state property, it is still important to ﬁnd a less restrictive suﬃcient condition than
the one given by |1 −∆t(a −σmax)| < 1. In fact, using a similar idea as in the proof of
Theorem 4.1, we will deﬁne a new system (Eq. 8) for which, once stability is established, the
echo state property is implied for the corresponding leaky integrator ESN.
Theorem 4.2. The leaky integrator ESN given in Eq. 7 with W fb = 0 and f = tanh has
the echo state property for all inputs if the following system converges to z = 0 uniformly for
all input sequences u+∞and state sequences z+∞:
zk+1 = [(1 −a∆t)I + ∆tLkW]zk,
(8)
where ∀k ≥0, zk = xk −yk with x+∞, y+∞∈X+∞compatible with u+∞and Lk =
Lk(zk, uk+1) are diagonal matrices with entries in the interval (0, 1].
Proof. Note that when W fb = 0, Eq. 7 becomes xk+1 = (1−a∆t)xk +∆tf(W inuδ
k+1+Wxk).
For any right inﬁnite input sequence (uδ
k+1)k≥0 = u+∞∈U+∞and any two right inﬁnite
state vector sequences x+∞, y+∞∈X+∞compatible with u+∞, we have:
xk+1 −yk+1 = (1 −a∆t)xk + ∆tf(W inuδ
k+1 + Wxk) −(1 −a∆t)yk −∆tf(W inuδ
k+1 + Wyk).
Applying the Mean-Value Theorem component-wise:
f(Wxk + W inuδ
k+1) −f(Wyk + W inuδ
k+1) = Lk(Wxk + W inuδ
k+1 −Wyk −W inuδ
k+1)
= LkW(xk −yk),
where each Lk = Lk(xk, yk, uk+1) is given by Lk = diag(ℓ1
k, . . . , ℓN
k ) with 0 < ℓi
k ≤1 since the
derivative of tanh is bounded between 0 and 1. Then we get,
xk+1 −yk+1 = (1 −a∆t)(xk −yk) + ∆tLkW(xk −yk).
Now, letting zk = xk −yk, we obtain the system zk+1 = [(1 −a∆t)I + ∆tLkW]zk. If one can
show the uniform convergence of this system to z = 0 for all u+∞and z+∞then, by Theorem
2.1, the echo state property of the system given by Eq. 7 is satisﬁed for all inputs.
18
The system given in Eq. 8 was studied in [7] (see Eq. 11 in [7]). Several suﬃcient condi-
tions for the global asymptotic stability and global exponential stability (which means states
approach the unique equilibrium point exponentially fast) were provided. Since global ex-
ponential stability implies uniform convergence, all the conditions provided in [7] for global
exponential stability also imply the echo state property for the leaky integrator ESNs in
Eq. 7 with no feedback. We provide two of those conditions (Theorem 6 and Corollary 10
in [7]) because of their similarity to the eﬀective spectral radius:
Corollary 4.3. If the spectral radius of the matrix
ˆ
M ∈RN×N deﬁned below is smaller
than 1, i.e. ρ( ˆ
M) < 1, then the echo state property for all inputs is satisﬁed for the leaky
integrator ESN given in Eq. 7 with W fb = 0 and f = tanh:
ˆ
M =






|1 −a∆t + ˆℓ1∆tw11|
∆t|w12|
· · ·
∆t|w1N|
∆t|w21|
|1 −a∆t + ˆℓ2∆tw22|
· · ·
∆t|w2N|
...
...
...
...
∆t|wN1|
∆t|wN2|
· · ·
|1 −a∆t + ˆℓN∆twNN|






,
(9)
where |1 −a∆t + ˆℓi∆twii| = max0≤ℓi≤1 |1 −a∆t + ℓi∆twii|.
A more restrictive condition can be given as:
Corollary 4.4. If the spectral radius of the matrix ˜
M = ∆t|W| + (1 −a∆t)I where |W| =
(|wij|), is smaller than 1, i.e. ρ( ˜
M) < 1, then the leaky integrator ESN given in Eq. 7 with
W fb = 0 and f = tanh has the echo state property for all inputs.
Note that the matrix ˜
M = ∆t|W|+(1−a∆t)I diﬀers from the matrix ˜W = ∆tW +(1−a∆t)I
only by the involvement of the absolute value. Even though the eﬀective spectral radius con-
dition, i.e. ρ( ˜W) < 1, is not suﬃcient for the echo state property, the new condition, i.e.
ρ( ˜
M) < 1, is suﬃcient. This is similar to the standard ESN case where ρ(W) < 1 is not
suﬃcient for the echo state property (as described in Section 3) but ρ(|W|) < 1 is suﬃcient
as discussed in Section 4.1.
A simple recipe for the echo state property of the leaky integrator ESNs
The condition ρ( ˜
M) < 1 gives a simple way to construct internal weight matrices for the
leaky integrator ESNs that satisfy the echo state property:
(i) Start with a random W with all non-negative entries, wij ≥0 (note that W = |W| in
this case).
(ii) Scale W so that ρ( ˜
M) < 1 where ˜
M = ∆tW + (1 −a∆t)I.
(iii) Change the signs of a desired number of entries of W to get negative connection weights
as well.
19
Finally, we would like to point out that the new suﬃcient conditions are less restrictive than
the existing |1 −∆t(a −σmax)| < 1 condition:
Example 4.2. Let us take a = 1, c = 1, δ = 0.5 (∆t = 0.5) and W =
!
−0.9
−1
0
−0.9
"
for the leaky integrator ESN given in Eq. 7 with W fb = 0. The maximum singular value
of W, σmax ≈1.52, gives |1 −∆t(a −σmax)| ≈|1 −0.5(1 −1.52)| = 1.26 > 1. Therefore,
based on this condition, we cannot be sure that the system has the echo state property.
However, using Eq. 9, ˆ
M =
!
0.5
0.5
0
0.5
"
where ρ( ˆ
M) = 0.5 < 1 actually proves the echo
state property. This can also be seen from the more restrictive condition ρ( ˜
M) = 0.95 < 1
where ˜
M = 0.5|W| + 0.5I =
!
0.95
0.5
0
0.95
"
.
5. A new deﬁnition for the echo state property
So far, we have investigated how the ESP can be lost for a spectral radius below unity, in
the case of zero input. Furthermore, for zero input a spectral radius not exceeding unity
is a necessary condition for the ESP. Obviously in practical applications one will usually
have nonzero input.
For nonzero input, the ESP can be obtained even with a spectral
radius exceeding unity. A very simple example is the one-dimensional “reservoir” xk+1 =
tanh(2xk + 1) driven by constant input uk ≡1. A quick look at a function plot (not shown)
reveals that it has a global point attractor at x ≈0.995 with uniform attraction from within
X = [−1, 1], thus has the ESP, but with a spectral radius of 2.
An intuitive explanation for the presence of ESP even at spectral radii beyond unity is
that the input drives the reservoir units (which we assume here to be tanh units for the
sake of discussion) toward the positive or negative branches of the sigmoid where stabilizing
saturation eﬀects start to become eﬀective. In this context, it becomes apparent that the
classical deﬁnition of the ESP, as in Deﬁnition 2.1, is not satisfactory. Consider a case where
the input is some scalar random process which takes values in a compact interval, say uk is
uniformly distributed in U = [−2, 2]. A typical realization of the input process will have an
expected absolute value of E[|uk|] = 1, which would lead to the desired uniform convergence
of state sequences in the sense of Theorem 2.1 for almost all input sequences even for a
range of spectral radii beyond unity. However, for the input sequence uk ≡0, which is a
valid realization of the input process, the ESP would be lost for any spectral radius beyond 1.
The original deﬁnition of the ESP takes account only of just the range U of admissible inputs,
not of the distribution of the input process. In practice, it is however this distribution which
determines the admissible range of spectral radius for almost all input sequences – which are
the practically relevant ones, not the “pathological” ones, which destroy the ESP but occur
with zero probability. Therefore, a more useful deﬁnition of the ESP is the following:
20
Deﬁnition 5.1 (ESP relative to an input process). Let (Uk)k∈Z be a stochastic process,
where the random variables Uk take values in a set U. A network F : X × U →X satisﬁes
the echo state property with respect to the process (Uk) if with probability one, for any left
inﬁnite input realization u−∞∈U−∞and any two state vector sequences x−∞, y−∞∈X−∞
compatible with u−∞, it holds that x0 = y0.
However, a mathematical analysis of the implications of this deﬁnition needs a combination
of tools from ergodic theory and non-autonomous dynamical systems. This is much harder
than for the original deﬁnition of the ESP. Research in this direction is currently being
pursued in the group of the second author.
6. Discussion
In this article we discussed, from various angles, the echo state property (ESP) and the
closely related issue of the spectral radius of the reservoir weight matrix. The main technical
contribution is a detailed analysis how the ESP is lost for speciﬁc weight patterns even when
the spectral radius is below unity.
Furthermore, we provided a novel algebraic criterion
which is suﬃcient for the ESP for any input in reservoirs whose nonlinearity has a derivative
bounded in [−1, 1] (such as tanh reservoirs). We hope that the bifurcation analyses and
link to the mathematically well-known concept of Schur stability will be interesting to both
reservoir computing researchers and users.
Another motivation to write this paper was to clarify a number of problematic preconceptions
which are rather wide-spread among researchers and engineers who use reservoir computing
for their applications. The authors witness that a signiﬁcant fraction of end-users of ESNs
only consider reservoirs with a spectral radius below unity, following the (widespread but
misguided) assumption that a spectral radius beyond unity destroys the ESP. One purpose
of this article is to dissolve this misperception and encourage users to explore spectral radii
greater than unity.
Conversely, the authors also witness examples of what one might call the inverse fallacy: by
experimentation the spectral radius is scaled up to the point where the ESP is lost and a
spectral radius slightly below this critical value is used. This leads to choices of spectral
radii which are typically much greater than 1. The justiﬁcation for this scheme is that in
some research it has been found that reservoirs scaled to the “edge of chaos” give best per-
formance. This line of thought apparently originates – within the reservoir computing arena
– in [2, 1], where it was shown that, for reservoirs with binary threshold units, performance
peaked when the reservoir weights were scaled to just below a critical value after which
state convergence was lost. The authors of these papers were careful in stating the speciﬁc
conditions under which these results were obtained and these results should not lead one to
assume that scaling a reservoir to the borders of chaos is always beneﬁcial.
21
In fact, there are tasks like the multistable switching circuits described in [9] (subsequently
employed to model stable working memory mechanisms [22]) which require fast reaction
times of reservoirs and fast locking into attractor states. Such behavior is best achieved
with spectral radii much smaller than unity. Furthermore, it was shown later [5] that binary
reservoirs show a much more marked dependency of task performance on parameter scaling
than the analog ESNs which are mostly used in applications.
In another study [20], which explored the task performance of analog reservoirs with respect
to speciﬁc information-theoretic metrics of the reservoir, it was found that performance is
coupled to the average reservoir state entropy, which – on the one hand – increases when
reservoirs are scaled toward the edge of chaos, but – on the other hand – can also be maxi-
mized by algebraically conﬁguring the reservoir weight matrix such that its eigenvalues are
spread as uniformly as possible about a circle in the complex plane. It was found here that
task-optimal spectral radii varied, and were found to lie below 1 for all tasks considered, i.e.
far away from chaos.
Finally, we would like to express our scepticism about the use of the term “edge of chaos”
in the context of reservoir computing. This term was introduced in the seminal paper by
Langton [17] in the context of the emergence of universal computation in cellular automata.
This is a setting which is at best indirectly related to reservoir computing, both with respect
to the computational substrate (eminently discrete in cellular automata, typically analog in
reservoirs) and with respect to the notion of “computation” (Turing computability in cellular
automata vs. online signal processing in reservoirs). In fact, when the spectral radius of a
reservoir weight matrix is scaled toward the critical value where the ESP is lost, one obtains
a phase transition which typically does not lead into chaos but into some oscillatory mode.
In the reservoir computing context it therefore is more appropriate to speak of an “edge of
stability”, as has been discussed in [24].
To conclude, we brieﬂy summarize a number of dispersed ﬁndings which amount to the fol-
lowing characterization of the current state of insight about the ESP in reservoir computing:
• The ESP is a property that is deﬁned with respect to the reservoir and the nature of
the driving input.
• The current deﬁnition of ESP is unsatisfactory in that it is uninformed about statis-
tical properties of the input, which are however crucial in applications. An improved
deﬁnition (Deﬁnition 5.1) is suggested, which however is mathematically diﬃcult to
analyze.
• The ESP (according to the new deﬁnition 5.1) may be obtainable almost surely even
for spectral radii (much) larger than unity if the driving input is suﬃciently strong.
22
• While ensuring the ESP is mandatory for training and employing reservoirs, there is
typically a wide range of spectral radii beyond 1 under which the ESP is obtained.
• There are no generally applicable recipes for the optimal setting of spectral radius, in
particular:
– It is not required to scale the spectral radius below 1.
– There is no general beneﬁt in scaling the spectral radius toward the “edge of
chaos”.
• An appropriate setting of the spectral radius still has to be found by task-speciﬁc
experimentation.
Appendix A. Proof of Theorem 4.1
Proof. A straightforward proof can be given by noting that diagonally Schur stable condition
is shown in the text to be equivalent to the condition given in [4] for the echo state property
(for the proof of this equivalence, see the Remark at the end of section 4.1).
Another direct proof can be given using the Lyapunov theory.
Let us denote a ﬁnite input sequence (u1, . . . , uk) ∈Uk, k ≥1 by ¯uk and the iterations of
a point x0 ∈X ⊂RN under the echo state network F : X × Uk →X by xk = F(x0, ¯uk).
For the echo state property to hold, we want to show, by Theorem 2.1, that for all u+∞∈
U+∞, for all x+∞, y+∞∈X+∞compatible with u+∞, the diﬀerence ∥xk −yk∥goes to zero
uniformly. Note that since X × Uk is compact and F : X × Uk →X is continuous, the
image sets Xk := {F(x0, ¯uk) | x0 ∈X, ¯uk ∈Uk} are compact. It also holds that ∀k ≥1,
Xk+1 ⊆Xk ⊆X0 = X since any element F(x0, ¯uk+1) ∈Xk+1 can be written as F(x0, ¯uk+1) =
F(F(x0, u1), (u2, . . . , uk+1)) ∈Xk.
Similarly, we deﬁne the diﬀerence sets Zk := {x −y | x, y ∈Xk}. Note that Zk are compact
and similarly we have Zk+1 ⊆Zk ⊆Z0 = Z.
We ﬁrst deﬁne an equivalent system using the variable zk = xk −yk where zk ∈Zk. Note
that zk+1 = xk+1 −yk+1 = f(Wxk + W inuk+1) −f(Wyk + W inuk+1).
Using the Mean-Value Theorem component-wise:
f(Wxk + W inuk+1) −f(Wyk + W inuk+1) = Lk(Wxk + W inuk+1 −Wyk −W inuk+1)
= LkW(xk −yk),
where each Lk is given by Lk = diag(ℓ1
k, . . . , ℓN
k ) with 0 < ℓi
k ≤1 since the derivative of tanh
is bounded between 0 and 1. Therefore, we can deﬁne the new system as:
zk+1 = LkWzk.
Note that z = 0 is an equilibrium point of this system and Lk = Lk(xk, yk, uk+1). We need
to show that all solutions converge to z = 0 uniformly for all (compatible) z+∞∈Z+∞and
23
u+∞∈U+∞.
Let us deﬁne the quadratic function V : Z →R, V (z) = zT Pz where P > 0 is the diagonal
positive matrix mentioned in the statement of the theorem. Note that V (z) = 0 ⇔z = 0
and V (z) > 0, ∀z ∈Z \ {0}, V (z) →∞as ∥z∥→∞and
V (zk+1) −V (zk) = (LkWzk)TP(LkWzk) −zT
k Pzk
= zT
k (LkW)TP(LkW)zk −zT
k Pzk
= zT
k [(LkW)TP(LkW) −P]zk.
To prove uniform global asymptotic stability (see for e.g. [15]), one needs to ﬁnd a pos-
itive deﬁnite, time independent function N(z) such that V (zk+1) −V (zk) = ∆V (z) =
zT [(LkW)TP(LkW) −P]z ≤−N(z) for all Lk and for all z ∈Ba(0) (Ba(0) is a ball around
zero).
By a simple calculation, one can see that zT
k [(LkW)TP(LkW)−P]zk ≤zT
k [W TPW −P]zk for
all zk, i.e. the maximum is attained when Lk is the identity. Since W TPW −P is negative
deﬁnite (by the assumption of the theorem), choosing −N = W TPW −P, we prove the
uniform global asymptotic stability of the origin.
24
REFERENCES
[1] N. Bertschinger and T. Natschl¨ager. Real-time computation at the edge of chaos in
recurrent neural networks. Neural Computation, 16(7):1413–1436, 2004.
[2] Nils Bertschinger, Thomas Natschl¨ager, and Robert A. Legenstein. At the edge of chaos:
Real-time computations and self-organized criticality in recurrent neural networks. In
Advances in Neural Information Processing Systems 17 [Neural Information Processing
Systems, NIPS 2004, December 13-18, 2004, Vancouver, British Columbia, Canada],
2004.
[3] Amit Bhaya and Eugenius Kaszkurewicz. On discrete-time diagonal and d-stability.
Linear Algebra and its Applications, 187(0):87 – 104, 1993.
[4] M. Buehner and P. Young. A tighter bound for the echo state property. IEEE Trans-
actions on Neural Networks, 17(3):820 –824, 2006.
[5] L. B¨using, B. Schrauwen, and R. Legenstein. Connectivity, dynamics, and memory in
reservoir computing with binary and analog neurons. Neural Computation, 22(5):1272–
1311, 2010.
[6] P. F. Dominey.
Complex sensory-motor sequence learning based on recurrent state
representation and reinforcement learning. Biological Cybernetics, 73:265–274, 1995.
[7] S. Hu and J. Wang. Global stability of a class of discrete-time recurrent neural networks.
Circuits and Systems I: Fundamental Theory and Applications, IEEE Transactions on,
49(8):1104 – 1117, 2002.
[8] H. Jaeger. The ”echo state” approach to analysing and training recurrent neural net-
works. GMD Report 148, GMD - German National Research Institute for Computer
Science, 2001.
[9] H. Jaeger.
Short term memory in echo state networks.
GMD-Report 152, GMD -
German National Research Institute for Computer Science, 2002.
[10] H. Jaeger. Echo state network. In Scholarpedia, volume 2, page 2330. 2007.
[11] H. Jaeger. Erratum note for the techreport: The ”echo state” approach to analysing
and training recurrent neural networks. Technical report, 2010.
[12] H. Jaeger and H. Haas. Harnessing nonlinearity: Predicting chaotic systems and saving
energy in wireless communication. Science, 304:78–80, 2004.
[13] H. Jaeger, M. Lukosevicius, D. Popovici, and U. Siewert. Optimization and applications
of echo state networks with leaky- integrator neurons. Neural Networks, 20(3):335 – 352,
2007. Echo State Networks and Liquid State Machines.
25
[14] E. Kaszkurewicz and A. Bhaya. Matrix diagonal stability in systems and computation.
Matrix Diagonal Stability in Systems and Computation. Birkhauser, 2000.
[15] H.K. Khalil. Nonlinear Systems. Prentice Hall, 2002.
[16] I.U.A. Kuzne&tsov. Elements of applied bifurcation theory. Applied mathematical sci-
ences. Springer-Verlag, 2004.
[17] C. G. Langton. Computation at the edge of chaos: Phase transitions and emergent
computation. Physica D, 42(1-3):12–37, 1990.
[18] Mantas Lukoˇseviˇcius and Herbert Jaeger. Reservoir computing approaches to recurrent
neural network training. Computer Science Review, 3(3):127–149, 2009.
[19] W. Maass, T. Natschl¨ager, and H. Markram. Real-time computing without stable states:
A new framework for neural computation based on perturbations. Neural Computation,
14(11):2531–2560, 2002. http://www.cis.tugraz.at/igi/maass/psﬁles/LSM-v106.pdf.
[20] M.C. Ozturk, D. Xu, and J.C. Principe. Analysis and design of echo state networks for
function approximation. Neural Computation, 19:111–138, 2006.
[21] A. Packard and J. Doyle. The complex structured singular value. Automatica, 29(1):71–
109, 1993.
[22] R. Pascanu and H. Jaeger.
A neurodynamical model for working memory.
Neural
Networks, 24(2):199–207, 2011. DOI: 10.1016/j.neunet.2010.10.003.
[23] J.M.T. Thompson and H.B. Stewart. Nonlinear dynamics and chaos. Wiley, 2002.
[24] D. Verstraeten.
Reservoir Computing: Computation with Dynamical Systems.
Phd
thesis, Electronics and Information Systems, University of Ghent, 2009.
