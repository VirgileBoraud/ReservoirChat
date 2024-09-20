Herbert Jaeger
Controlling Recurrent Neural Networks by
Conceptors
Technical Report No. 31
March 2014
School of Engineering and Science
arXiv:1403.3369v1  [cs.NE]  13 Mar 2014
Controlling Recurrent Neural Net-
works by Conceptors
Herbert Jaeger
Jacobs University Bremen
School of Engineering and Science
Campus Ring
28759 Bremen
Germany
E-Mail: h.jaeger@jacobs-university.de
http: // minds. jacobs-university. de
Abstract
The human brain is a dynamical system whose extremely complex sensor-
driven neural processes give rise to conceptual, logical cognition. Under-
standing the interplay between nonlinear neural dynamics and concept-level
cognition remains a major scientiﬁc challenge. Here I propose a mechanism
of neurodynamical organization, called conceptors, which unites nonlinear
dynamics with basic principles of conceptual abstraction and logic. It be-
comes possible to learn, store, abstract, focus, morph, generalize, de-noise
and recognize a large number of dynamical patterns within a single neural
system; novel patterns can be added without interfering with previously
acquired ones; neural noise is automatically ﬁltered. Conceptors help ex-
plaining how conceptual-level information processing emerges naturally and
robustly in neural systems, and remove a number of roadblocks in the theory
and applications of recurrent neural networks.
Notes on the Structure of this Report.
This report introduces several novel
analytical concepts describing neural dynamics; develops the corresponding math-
ematical theory under aspects of linear algebra, dynamical systems theory, and
formal logic; introduces a number of novel learning, adaptation and control al-
gorithms for recurrent neural networks; demonstrates these in a number of case
studies; proposes biologically (not too im-)plausible realizations of the dynamical
mechanisms; and discusses relationships to other work. Said shortly, it’s long.
Not all parts will be of interest to all readers. In order to facilitate navigation
through the text and selection of relevant components, I start with an overview
section which gives an intuitive explanation of the novel concepts and informal
sketches of the main results and demonstrations (Section 1). After this overview,
the material is presented in detail, starting with an introduction (Section 2) which
relates this contribution to other research. The main part is Section 3, where I
systematically develop the theory and algorithms, interspersed with simulation
demos. A graphical dependency map for this section is given at the beginning of
Section 3. The technical documentation of the computer simulations is provided
in Section 4, and mathematical proofs are collected in Section 5. The detailed
presentation in Sections 2 – 5 is self-contained. Reading the overview in Section
1 may be helpful but is not necessary for reading these sections. For convenience
some ﬁgures from the overview section are repeated in Section 3.
Acknowledgements.
The work described in this report was partly funded through
the European FP7 project AMARSi (www.amarsi-project.eu). The author is in-
debted to Dr. Mathieu Galtier and Dr. Manjunath Ghandi for careful proofreading
(not an easy task).
3
Contents
1
Overview
6
2
Introduction
25
2.1
Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25
2.2
Mathematical Preliminaries
. . . . . . . . . . . . . . . . . . . . . .
27
3
Theory and Demonstrations
29
3.1
Networks and Signals . . . . . . . . . . . . . . . . . . . . . . . . . .
29
3.2
Driving a Reservoir with Diﬀerent Patterns . . . . . . . . . . . . . .
30
3.3
Storing Patterns in a Reservoir, and Training the Readout . . . . .
34
3.4
Conceptors: Introduction and Basic Usage in Retrieval . . . . . . .
34
3.5
A Similarity Measure for Excited Network Dynamics
. . . . . . . .
38
3.6
Online Learning of Conceptor Matrices . . . . . . . . . . . . . . . .
38
3.7
Morphing Patterns . . . . . . . . . . . . . . . . . . . . . . . . . . .
39
3.8
Understanding Aperture . . . . . . . . . . . . . . . . . . . . . . . .
43
3.8.1
The Semantics of α as “Aperture” . . . . . . . . . . . . . . .
43
3.8.2
Aperture Adaptation and Final Deﬁnition of Conceptor Ma-
trices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
44
3.8.3
Aperture Adaptation: Example . . . . . . . . . . . . . . . .
46
3.8.4
Guides for Aperture Adjustment
. . . . . . . . . . . . . . .
46
3.9
Boolean Operations on Conceptors
. . . . . . . . . . . . . . . . . .
50
3.9.1
Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . .
50
3.9.2
Preliminary Deﬁnition of Boolean Operations
. . . . . . . .
51
3.9.3
Final Deﬁnition of Boolean Operations . . . . . . . . . . . .
53
3.9.4
Facts Concerning Subspaces . . . . . . . . . . . . . . . . . .
55
3.9.5
Boolean Operators and Aperture Adaptation . . . . . . . . .
56
3.9.6
Logic Laws
. . . . . . . . . . . . . . . . . . . . . . . . . . .
56
3.10 An Abstraction Relationship between Conceptors . . . . . . . . . .
58
3.11 Example: Memory Management in RNNs . . . . . . . . . . . . . . .
59
3.12 Example: Dynamical Pattern Recognition
. . . . . . . . . . . . . .
67
3.13 Autoconceptors . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
74
3.13.1 Motivation and Overview
. . . . . . . . . . . . . . . . . . .
74
3.13.2 Basic Equations . . . . . . . . . . . . . . . . . . . . . . . . .
75
3.13.3 Example: Autoconceptive Reservoirs as Content-Addressable
Memories
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
77
3.13.4 Analysis of Autoconceptor Adaptation Dynamics
. . . . . .
89
3.14 Toward Biologically Plausible Neural Circuits: Random Feature
Conceptors
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
3.15 A Hierarchical Filtering and Classiﬁcation Architecture . . . . . . . 117
3.16 Toward a Formal Marriage of Dynamics with Logic . . . . . . . . . 133
3.17 Conceptor Logic as Institutions: Category-Theoretical Detail . . . . 143
3.18 Final Summary and Outlook . . . . . . . . . . . . . . . . . . . . . . 154
4
4
Documentation of Experiments and Methods
157
4.1
General Set-Up, Initial Demonstrations (Section 1 and Section 3.2
- 3.4) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 157
4.2
Aperture Adaptation (Sections 3.8.3 and 3.8.4)
. . . . . . . . . . . 158
4.3
Memory Management (Section 3.11)
. . . . . . . . . . . . . . . . . 160
4.4
Content-Addressable Memory (Section 3.13.3) . . . . . . . . . . . . 161
4.5
The Japanese Vowels Classiﬁcation (Section 3.12) . . . . . . . . . . 162
4.6
Conceptor Dynamics Based on RFC Conceptors (Section 3.14) . . . 163
4.7
Hierarchical Classiﬁcation and Filtering Architecture (Section 3.15) 163
5
Proofs and Algorithms
166
5.1
Proof of Proposition 1 (Section 3.4) . . . . . . . . . . . . . . . . . . 166
5.2
Proof of Proposition 6 (Section 3.9.3) . . . . . . . . . . . . . . . . . 167
5.3
Proof of Proposition 7 (Section 3.9.3) . . . . . . . . . . . . . . . . . 170
5.4
Proof of Proposition 8 (Section 3.9.3) . . . . . . . . . . . . . . . . . 170
5.5
Proof of Proposition 9 (Section 3.9.4) . . . . . . . . . . . . . . . . . 171
5.6
Proof of Proposition 10 (Section 3.9.5)
. . . . . . . . . . . . . . . . 173
5.7
Proof of Proposition 11 (Section 3.9.6)
. . . . . . . . . . . . . . . . 176
5.8
Proof of Proposition 13 (Section 3.9.6)
. . . . . . . . . . . . . . . . 176
5.9
Proof of Proposition 14 (Section 3.10) . . . . . . . . . . . . . . . . . 177
5.10 Proof of Proposition 16 (Section 3.13.4) . . . . . . . . . . . . . . . . 181
5.11 Proof of Proposition 18 (Section 3.17) . . . . . . . . . . . . . . . . . 186
References
187
5
1
Overview
Scientiﬁc context.
Research on brains and cognition unfolds in two directions.
Top-down oriented research starts from the “higher” levels of cognitive perfor-
mance, like rational reasoning, conceptual knowledge representation, command of
language. These phenomena are typically described in symbolic formalisms de-
veloped in mathematical logic, artiﬁcial intelligence (AI), computer science and
linguistics. In the bottom-up direction, one departs from “low-level” sensor data
processing and motor control, using the analytical tools oﬀered by dynamical
systems theory, signal processing and control theory, statistics and information
theory. The human brain obviously has found a way to implement high-level log-
ical reasoning on the basis of low-level neuro-dynamical processes. How this is
possible, and how the top-down and bottom-up research directions can be united,
has largely remained an open question despite long-standing eﬀorts in neural net-
works research and computational neuroscience [80, 87, 33, 2, 36, 43], machine
learning [35, 47], robotics [11, 81], artiﬁcial intelligence [83, 104, 8, 10], dynam-
ical systems modeling of cognitive processes [94, 98, 105], cognitive science and
linguistics [22, 96], or cognitive neuroscience [5, 26].
Summary of contribution.
Here I establish a fresh view on the neuro-symbolic
integration problem.
I show how dynamical neural activation patterns can be
characterized by certain neural ﬁlters which I call conceptors. Conceptors derive
naturally from the following key observation. When a recurrent neural network
(RNN) is actively generating, or is passively being driven by diﬀerent dynamical
patterns (say a, b, c, . . .), its neural states populate diﬀerent regions Ra, Rb, Rc, . . .
of neural state space. These regions are characteristic of the respective patterns.
For these regions, neural ﬁlters Ca, Cb, Cc, . . . (the conceptors) can be incremen-
tally learnt. A conceptor Cx representing a pattern x can then be invoked after
learning to constrain the neural dynamics to the state region Rx, and the network
will select and re-generate pattern x. Learnt conceptors can be blended, combined
by Boolean operations, specialized or abstracted in various ways, yielding novel
patterns on the ﬂy. Conceptors can be economically represented by single neu-
rons (addressing patterns by neurons, leading to explicit command over pattern
generation), or they may be constituted spontaneously upon the presentation of
cue patterns (content-addressing, leading to pattern imitation). The logical oper-
ations on conceptors admit a rigorous semantical interpretation; conceptors can
be arranged in conceptual hierarchies which are structured like semantic networks
known from artiﬁcial intelligence. Conceptors can be economically implemented
by single neurons (addressing patterns by neurons, leading to explicit command
over pattern generation), or they may self-organize spontaneously and quickly
upon the presentation of cue patterns (content-addressing, leading to pattern im-
itation). Conceptors can also be employed to “allocate free memory space” when
new patterns are learnt and stored in long-term memory, enabling incremental
6
life-long learning without the danger of freshly learnt patterns disrupting already
acquired ones. Conceptors are robust against neural noise and parameter varia-
tions. The basic mechanisms are generic and can be realized in any kind of dynam-
ical neural network. All taken together, conceptors oﬀer a principled, transparent,
and computationally eﬃcient account of how neural dynamics can self-organize in
conceptual structures.
Going bottom-up:
from neural dynamics to conceptors.
The neural
model system in this report are standard recurrent neural networks (RNNs, Figure
1 A) whose dynamics is mathematically described be the state update equations
x(n + 1)
=
tanh(W ∗x(n) + W in p(n)),
y(n)
=
W outx(n).
Time here progresses in unit steps n = 1, 2, . . ..
The network consists of N
neurons (typically in the order of a hundred in this report), whose activations
x1(n), . . . , xN(n) at time n are collected in an N-dimensional state vector x(n).
The neurons are linked by random synaptic connections, whose strengths are col-
lected in a weight matrix W ∗of size N × N. An input signal p(n) is fed to the
network through synaptic input connections assembled in the input weight ma-
trix W in. The “S-shaped” function tanh squashes the neuronal activation values
into a range between −1 and 1. The second equation speciﬁes that an ouput sig-
nal y(n) can be read from the network activation state x(n) by means of output
weights W out. These weights are pre-computed such that the output signal y(n)
just repeats the input signal p(n).
The output signal plays no functional role
in what follows; it merely serves as a convenient 1-dimensional observer of the
high-dimensional network dynamics.
The network-internal neuron-to-neuron connections W ∗are created at random.
This will lead to the existence of cyclic (“recurrent”) connection pathways inside
the network. Neural activation can reverberate inside the network along these
cyclic pathways. The network therefore can autonomously generate complex neu-
rodynamical patterns even when it receives no input. Following the terminology of
the reservoir computing [56, 6], I refer to such randomly connected neural networks
as reservoirs.
For the sake of introducing conceptors by way of an example, consider a reser-
voir with N = 100 neurons. I drive this system with a simple sinewave input p(n)
(ﬁrst panel in ﬁrst row in Fig. 1 B). The reservoir becomes entrained to this input,
each neuron showing individual variations thereof (Fig. 1 B second panel). The
resulting reservoir state sequence x(1), x(2), . . . can be represented as a cloud of
points in the 100-dimensional reservoir state space. The dots in the ﬁrst panel
of Fig. 1 C show a 2-dimensional projection of this point cloud. By a statistical
method known as principal component analysis, the shape of this point cloud can
be captured by an N-dimensional ellipsoid whose main axes point in the main
7
!!
"
!
!!
"
!
!!
"
!
!!
"
!
!!
"
!
!!
"
!
s1 
!"
W  in 
W  out 
W   
y!
"#$%#$!!
&'('&)"*&!
p!!
*+%#$!!
x!
#"
$"
!!
"
!
#$%&'$(
!!
"
!
)*+$&,-.+&/
!!"
"
0+1!"$!
"
!
/
!!
"
!
!!
"
!
!!"
"
"
!
!!
"
!
!!
"
!
!!"
"
"
!
"
!"
2"
!!
"
!
"
!"
2"
!!
"
!
"
3"
!""
!!"
"
"
3"
!""
"
!
!1 !2 
s2 
Figure 1: Deriving conceptors from network dynamics. A. Network layout. Ar-
rows indicate synaptic links. B. Driving the reservoir with four diﬀerent input
patterns. Left panels: 20 timesteps of input pattern p(n) (black thin line) and
conceptor-controlled output y(n) (bold light gray). Second column: 20 timesteps
of traces xi(n), xj(n) of two randomly picked reservoir neurons. Third column: the
singular values σi of the reservoir state correlation matrix R in logarithmic scale.
Last column: the singular values si of the conceptors C in linear plotting scale. C.
From pattern to conceptor. Left: plots of value pairs xi(n), xj(n) (dots) of the two
neurons shown in ﬁrst row of B and the resulting ellipse with axis lengths σ1, σ2.
Right: from R (thin light gray) to conceptor C (bold dark gray) by normalizing
axis lengths σ1, σ2 to s1, s2.
8
scattering directions of the point cloud. This ellipsoid is a geometrical representa-
tion of the correlation matrix R of the state points. The lengths σ1, . . . , σN of the
ellipsoid axes are known as the singular values of R. The directions and lengths of
these axes provide a succinct characterization of the geometry of the state point
cloud. The N = 100 lengths σi resulting in this example are log-plotted in Fig. 1
B, third column, revealing an exponential fall-oﬀin this case.
As a next step, these lengths σi are normalized to become si = σi/(σi + α−2),
where α ≥0 is a design parameter that I call aperture. This normalization ensures
that all si are not larger than 1 (last column in Fig. 1 B). A new ellipsoid is
obtained (Fig. 1 C right) which is located inside the unit sphere. The normalized
ellipsoid can be described by a N-dimensional matrix C, which I call a conceptor
matrix. C can be directly expressed in terms of R by C = R(R + α−2I)−1, where
I is the identity matrix.
When a diﬀerent driving pattern p is used, the shape of the state point cloud,
and subsequently the conceptor matrix C, will be characteristically diﬀerent. In
the example, I drove the reservoir with four patterns p1 – p4 (rows in Fig. 1B). The
ﬁrst two patterns were sines of slightly diﬀerent frequencies, the last two patterns
were minor variations of a 5-periodic random pattern. The conceptors derived
from the two sine patterns diﬀer considerably from the conceptors induced by the
two 5-periodic patterns (last column in Fig. 1B). Within each of these two pairs,
the conceptor diﬀerences are too small to become visible in the plots.
There is an instructive alternative way to deﬁne conceptors.
Given a se-
quence of reservoir states x(1), . . . , x(L), the conceptor C which characterizes
this state point cloud is the unique matrix which minimizes the cost function
P
n=1,...,L ∥x(n) −Cx(n)∥2/L + α−2∥C∥2, where ∥C∥2 is the sum of all squared
matrix entries. The ﬁrst term in this cost would become minimal if C were the
identity map, the second term would become minimal if C would be the all-zero
map. The aperture α strikes a balance between these two competing cost com-
ponents. For increasing apertures, C will tend toward the identity matrix I; for
shrinking apertures it will come out closer to the zero matrix.
In the termi-
nology of machine learning, C is hereby deﬁned as a regularized identity map.
The explicit solution to this minimization problem is again given by the formula
C = R (R + α−2I)−1.
Summing up: if a reservoir is driven by a pattern p(n), a conceptor matrix C
can be obtained from the driven reservoir states x(n) as the regularized identity
map on these states. C can be likewise seen as a normalized ellipsoid characteri-
zation of the shape of the x(n) point cloud. I write C(p, α) to denote a conceptor
derived from a pattern p using aperture α, or C(R, α) to denote that C was ob-
tained from a state correlation matrix R.
Loading a reservoir.
With the aid of conceptors a reservoir can re-generate a
number of diﬀerent patterns p1, . . . , pK that it has previously been driven with.
For this to work, these patterns have to be learnt by the reservoir in a special
9
sense, which I call loading a reservoir with patterns. The loading procedure works
as follows. First, drive the reservoir with the patterns p1, . . . , pK in turn, collect-
ing reservoir states xj(n) (where j = 1, . . . , K). Then, recompute the reservoir
connection weights W ∗into W such that W optimally balances between the fol-
lowing two goals. First, W should be such that W xj(n) ≈W ∗xj(n) + W inpj(n)
for all times n and patterns j. That is, W should allow the reservoir to “simu-
late” the driving input in the absence of the same. Second, W should be such
that the weights collected in this matrix become as small as possible. Technically
this compromise-seeking learning task amounts to computing what is known as a
regularized linear regression, a standard and simple computational task. This idea
of “internalizing” a driven dynamics into a reservoir has been independently (re-
)introduced under diﬀerent names and for a variety of purposes (self-prediction
[72], equilibration [55], reservoir regularization [90], self-sensing networks [100],
innate training [61]) and appears to be a fundamental RNN adaptation principle.
Going top-down: from conceptors to neural dynamics.
Assume that con-
ceptors Cj = C(pj, α) have been derived for patterns p1, . . . , pK, and that these
patterns have been loaded into the reservoir, replacing the original random weights
W ∗by W.
Intuitively, the loaded reservoir, when it is run using x(n + 1) =
tanh(W x(n)) (no input!) should behave exactly as when it was driven with input
earlier, because W has been trained such that W x(n) ≈W ∗x(n) + W inpj(n).
In fact, if only a single pattern had been loaded, the loaded reservoir would
readily re-generate it.
But if more than one patter had been loaded, the au-
tonomous (input-free) update x(n + 1) = tanh(W x(n)) will lead to an entirely
unpredictable dynamics: the network can’t “decide” which of the loaded patterns
it should re-generate! This is where conceptors come in. The reservoir dynam-
ics is ﬁltered through Cj. This is eﬀected by using the augmented update rule
x(n + 1) = Cj tanh(W x(n)). By virtue of inserting Cj into the feedback loop,
the reservoir states become clipped to fall within the ellipsoid associated with Cj.
As a result, the pattern pj will be re-generated: when the reservoir is observed
through the previously trained output weights, one gets y(n) = W out x(n) ≈pj(n).
The ﬁrst column of panels in Fig. 1 B shows an overlay of the four autonomously
re-generated patterns y(n) with the original drivers pj used in that example. The
recovery of the originals is quite accurate (mean square errors 3.3e-05, 1.4e-05,
0.0040, 0.0019 for the four loaded patterns). Note that the ﬁrst two and the last
two patterns are rather similar to each other. The ﬁltering aﬀorded by the re-
spective conceptors is “sharp” enough to separate these twin pairs. I will later
demonstrate that in this way a remarkably large number of patterns can be faith-
fully re-generated by a single reservoir.
Morphing and generalization.
Given a reservoir loaded with K patterns pj,
the associated conceptors Cj can be linearly combined by creating mixture con-
ceptors M = µ1C1 + . . . + µKCK, where the mixing coeﬃcients µj must sum to 1.
10
When the reservoir is run under the control of such a morphed conceptor M, the
resulting generated pattern is a morph between the original “pure” patterns pj. If
all µj are non-negative, the morph can be considered an interpolation between the
pure patterns; if some µj are negative, the morph extrapolates beyond the loaded
pure patterns. I demonstrate this with the four patterns used in the example
above, setting µ1 = (1−a)b, µ2 = ab, µ3 = (1−a)(1−b), µ4 = a(1−b), and letting
a, b vary from −0.5 to 1.5 in increments of 0.25. Fig. 2 shows plots of observer
signals y(n) obtained when the reservoir is generating patterns under the control
of these morphed conceptors. The innermost 5 by 5 panels show interpolations
between the four pure patterns, all other panels show extrapolations.
In machine learning terms, both interpolation and extrapolation are cases of
generalization. A standard opinion in the ﬁeld states that generalization by in-
terpolation is what one may expect from learning algorithms, while extrapolation
beyond the training data is hard to achieve.
Morphing and generalizing dynamical patterns is a common but nontrivial task
for training motor patterns in robots. It typically requires training demonstrations
of numerous interpolating patterns [88, 17, 68]. Conceptor-based pattern morphing
appears promising for ﬂexible robot motor pattern learning from a very small
number of demonstrations.
Aperture adaptation.
Choosing the aperture α appropriately is crucial for re-
generating patterns in a stable and accurate way. To demonstrate this, I loaded
a 500-neuron reservoir with signals p1 – p4 derived from four classical chaotic
attractors: the Lorenz, R¨ossler, Mackey-Glass, and H´enon attractors. Note that
it used to be a challenging task to make an RNN learn any single of these attractors
[56]; to my knowledge, training a single RNN to generate several diﬀerent chaotic
attractors has not been attempted before.
After loading the reservoir, the re-
generation was tested using conceptors C(pj, α) where for each attractor pattern
pj a number of diﬀerent values for α were tried. Fig. 3 A shows the resulting re-
generated patterns for ﬁve apertures for the Lorenz attractor. When the aperture
is too small, the reservoir-conceptor feedback loop becomes too constrained and the
produced patterns de-diﬀerentiate. When the aperture is too large, the feedback
loop becomes over-excited.
An optimal aperture can be found by experimentation, but this will not be
an option in many engineering applications or in biological neural systems. An
intrinsic criterion for optimizing α is aﬀorded by a quantity that I call attenuation:
the damping ratio which the conceptor imposes on the reservoir signal. Fig. 3
C plots the attenuation against the aperture for the four chaotic signals. The
minimum of this curve marks a good aperture value: when the conceptor dampens
out a minimal fraction of the reservoir signal, conceptor and reservoir are in good
“resonance”. The chaotic attractor re-generations shown in Fig. 3 B were obtained
by using this minimum-attenuation criterion.
The aperture range which yields visibly good attractor re-generations in this
11
b = –0.5 
0.0 
0.5 
1.0 
1.5 
a  =   –0.5            0.0 
        0.5           1.0             1.5!
Figure 2: Morphing between, and generalizing beyond, four loaded patterns. Each
panel shows a 15-step autonomously generated pattern (plot range between −1
and +1).
Panels with bold frames: the four loaded prototype patterns (same
patterns as in Fig. 1 B.)
demonstration spans about one order of magnitude.
With further reﬁnements
(zeroing small singular values in conceptors is particularly eﬀective), the viable
aperture range can be expanded to about three orders of magnitude. While setting
the aperture right is generally important, ﬁne-tuning is unnecessary.
Boolean operations and conceptor abstraction.
Assume that a reservoir
is driven by a pattern r which consists of randomly alternating epochs of two
patterns p and q. If one doesn’t know which of the two patterns is active at a
given time, all one can say is that the pattern r currently is p OR it is q. Let
C(Rp, 1), C(Rq, 1), C(Rr, 1) be conceptors derived from the two partial patterns
p, q and the “OR” pattern r, respectively. Then it holds that C(Rr, 1) = C((Rp +
Rq)/2, 1). Dropping the division by 2, this motivates to deﬁne an OR (math-
ematical notation: ∨) operation on conceptors C1(R1, 1), C2(R2, 1) by putting
C1 ∨C2 := (R1 + R2)(R1 + R2 + I)−1. The logical operations NOT (¬) and AND
(∧) can be deﬁned along similar lines. Fig. 4 shows two-dimensional examples of
applying the three operations.
12
A
11
66
4e+02
2.4e+03
1.4e+04
B
1e+03
1.3e+03
6.3e+02
C
1
2
3
4
5
ï5
0
Lorenz
log10 aperture
1
2
3
4
5
ï5
0
Roessler
1
2
3
4
5
ï5
0
MackeyïGlass
1
2
3
4
5
ï5
0
Hénon
Figure 3: Aperture adaptation for re-generating four chaotic attractors. A Lorenz
attractor. Five versions re-generated with diﬀerent apertures (values inserted in
panels) and original attractor (gray background). B Best re-generations of the
other three attractors (from left to right: R¨ossler, Mackey-Glass, and H´enon, orig-
inals on gray background). C Log10 of the attenuation criterion plotted against
the log10 of aperture. Dots mark the apertures used for plots in A and B.
ï1
0
1
ï1
0
1
ï1
0
1
ï1
0
1
ï1
0
1
ï1
0
1
Figure 4: Boolean operations on conceptors. Red/blue (thin) ellipses represent
source conceptors C1, C2. Magenta (thick) ellipses show C1 ∨C2, C1 ∧C2, ¬C1
(from left to right).
Boolean logic is the mathematical theory of ∨, ∧, ¬. Many laws of Boolean
logic also hold for the ∨, ∧, ¬ operations on conceptors: the laws of associativ-
ity, commutativity, double negation, de Morgan’s rules, some absorption rules.
Furthermore, numerous simple laws connect aperture adaptation to Boolean op-
erations. Last but not least, by deﬁning C1 ≤C2 if and only if there exists a
13
conceptor B such that C2 = C1 ∨B, an abstraction ordering is created on the set
of all conceptors of dimension N.
Neural memory management.
Boolean conceptor operations aﬀord unprece-
dented ﬂexibility of organizing and controlling the nonlinear dynamics of recurrent
neural networks. Here I demonstrate how a sequence of patterns p1, p2, . . . can be
incrementally loaded into a reservoir, such that (i) loading a new pattern pj+1
does not interfere with previously loaded p1, . . . , pj; (ii) if a new pattern pj+1 is
similar to already loaded ones, the redundancies are automatically detected and
exploited, saving memory capacity; (iii) the amount of still “free” memory space
can be logged.
Let Cj be the conceptor associated with pattern pj. Three ideas are combined
to implement the memory management scheme. First, keep track of the “already
used” memory space by maintaining a conceptor Aj = C1 ∨. . . ∨Cj. The sum of
all singular values of Aj, divided by the reservoir size, gives a number that ranges
between 0 and 1. It is an indicator of the portion of reservoir “space” which has
been used up by loading C1, . . . , Cj, and I call it the quota claimed by C1, . . . , Cj.
Second, characterize what is “new” about Cj+1 (not being already represented by
previously loaded patterns) by considering the conceptor N j+1 = Cj+1 \ Aj. The
logical diﬀerence operator \ can be re-written as A \ B = A ∧¬B. Third, load
only that which is new about Cj+1 into the still unclaimed reservoir space, that is,
into ¬Aj. These three ideas can be straightforwardly turned into a modiﬁcation
of the basic pattern loading algorithm.
For a demonstration, I created a series of periodic patterns p1, p2, . . . whose
integer period lengths were picked randomly between 3 and 15, some of these
patterns being sines, others random patterns. These patterns were incrementally
loaded in a 100-neuron reservoir, one by one. Fig. 5 shows the result. The “used
space” panels monitor the successive ﬁlling-up of reservoir space. Since patterns
j = 6, 7, 8 were identical replicas of patterns j = 1, 2, 3, no additional space was
consumed when these patterns were (re-)loaded. The “driver and y” panels doc-
ument the accuracy of autonomously re-generating patterns using conceptors Cj.
Accuracy was measured by the normalized root mean square error (NRMSE), a
standard criterion for comparing the similarity between two signals. The NRMSE
jumps from very small values to a high value when the last pattern is loaded; the
quota of 0.98 at this point indicates that the reservoir is “full”. The re-generation
testing and NRMSE computation was done after all patterns had been loaded.
An attempt to load further patterns would be unsuccessful, but it also would not
harm the re-generation quality of the already loaded ones.
This ability to load patterns incrementally solves a notorious problem in neu-
ral network training, known as catastrophic forgetting, which manifests itself in
a disruption of previously learnt functionality when learning new functionality.
Although a number of proposals have been made which partially alleviate the
problem in special circumstances [32, 42], catastrophic forgetting was still listed
14
ï1
0
1
0.0025
j = 1
0.1
0.0076
j = 2
0.25
0.00084
j = 3
0.29
0.0013
j = 4
0.35
ï1
0
1
0.0028
j = 5
0.42
0.0025
j = 6
0.42
0.0073
j = 7
0.42
0.00084
j = 8
0.42
ï1
0
1
0.0082
j = 9
0.54
0.016
j = 10
0.59
0.007
j = 11
0.65
0.0033
j = 12
0.73
1
10
20
ï1
0
1
0.0035
j = 13
0.8
1
10
20
0.0011
j = 14
0.83
1
10
20
0.022
j = 15
0.94
1
10
20
1.3
j = 16
0.98
Figure 5: Incremental pattern storing in a neural memory.
Each panel shows
a 20-timestep sample of the correct training pattern pj (black line) overlaid on
its reproduction (green line).
The memory fraction used up until pattern j is
indicated by the panel fraction ﬁlled in red; the quota value is printed in the left
bottom corner of each panel.
as an open challenge in an expert’s report solicited by the NSF in 2007 [21] which
collected the main future challenges in learning theory.
Recognizing dynamical patterns.
Boolean conceptor operations enable the
combination of positive and negative evidence in a neural architecture for dynam-
ical pattern recognition. For a demonstration I use a common benchmark, the
Japanese vowel recognition task [60]. The data of this benchmark consist in pre-
processed audiorecordings of nine male native speakers pronouncing the Japanese
di-vowel /ae/. The training data consist of 30 recordings per speaker, the test data
consist of altogether 370 recordings, and the task is to train a recognizer which
has to recognize the speakers of the test recordings. This kind of data diﬀers from
the periodic or chaotic patterns that I have been using so far, in that the patterns
are non-stationary (changing in their structure from beginning to end), multi-
dimensional (each recording consisting of 12 frequency band signals), stochastic,
and of ﬁnite duration. This example thus also demonstrates that conceptors can
be put to work with data other than single-channel stationary patterns.
A small (10 neurons) reservoir was created. It was driven with all training
recordings from each speaker j in turn (j = 1, . . . , 9), collecting reservoir response
signals, from which a conceptor Cj characteristic of speaker j was computed. In
addition, for each speaker j, a conceptor N j = ¬ (C1 ∨. . . ∨Cj−1 ∨Cj+1 ∨. . . C9)
was computed. N j characterizes the condition “this speaker is not any of the
other eight speakers”. Patterns need not to be loaded into the reservoir for this
application, because they need not be re-generated.
15
In testing, a recording p from the test set was fed to the reservoir, collect-
ing a reservoir response signal x. For each of the conceptors, a positive evidence
E+(p, j) = x′Cjx was computed. E+(p, j) is a non-negative number indicating
how well the signal x ﬁts into the ellipsoid of Cj.
Likewise, the negative evi-
dence E−(p, j) = x′N jx that the sample p was not uttered by any of the eight
speakers other than speaker j was computed.
Finally, the combined evidence
E(p, j) = E+(p, i) + E−(p, i) was computed. This gave nine combined evidences
E(p, 1), . . . , E(p, 9). The pattern p was then classiﬁed as speaker j by choosing
the speaker index j whose combined evidence E(p, j) was the greatest among the
nine collected evidences.
In order to check for the impact of the random selection of the underlying
reservoir, this whole procedure was repeated 50 times, using a freshly created
random reservoir in each trial. Averaged over these 50 trials, the number of test
misclassiﬁcations was 3.4. If the classiﬁcation would have been based solely on the
positive or negative evidences, the average test misclassiﬁcation numbers would
have been 8.4 and 5.9 respectively.
The combination of positive and negative
evidence, which was enabled by Boolean operations, was crucial.
State-of-the-art machine learning methods achieve between 4 and 10 misclassi-
ﬁcations on the test set (for instance [91, 97, 79, 15]). The Boolean-logic-conceptor-
based classiﬁer thus compares favorably with existing methods in terms of clas-
siﬁcation performance.
The method is computationally cheap, with the entire
learning procedure taking a fraction of a second only on a standard notebook
computer. The most distinctive beneﬁt however is incremental extensibility. If
new training data become available, or if a new speaker would be incorporated
into the recognition repertoire, the additional training can be done using only the
new data without having to re-run previous training data. This feature is highly
relevant in engineering applications and in cognitive modeling and missing from
almost all state-of-the-art classiﬁcation methods.
Autoconceptors and content-addressable memories.
So far I have been
describing examples where conceptors Cj associated with patterns pj were com-
puted at training time, to be later plugged in to re-generate or classify patterns.
A conceptor C matrix has the same size as the reservoir connection matrix W.
Storing conceptor matrices means to store network-sized objects. This is implau-
sible under aspects of biological modeling. Here I describe how conceptors can be
created on the ﬂy, without having to store them, leading to content-addressable
neural memories.
If the system has no pre-computed conceptors at its disposal, loaded patterns
can still be re-generated in a two-stage process. First, the target pattern p is
selected by driving the system with a brief initial “cueing” presentation of the
pattern (possibly in a noisy version). During this phase, a preliminary conceptor
Ccue is created by an online adaptation process. This preliminary Ccue already en-
ables the system to re-generate an imperfect version of the pattern p. Second, after
16
the cueing phase has ended, the system continues to run in an autonomous mode
(no external cue signal), initially using Ccue, to continuously generate a pattern.
While this process is running, the conceptor in the loop is continuously adapted
by a simple online adaptation rule. This rule can be described in geometrical
terms as “adapt the current conceptor C(n) such that its ellipsoid matches better
the shape of the point cloud of the current reservoir state dynamics”. Under this
rule one obtains a reliable convergence of the generated pattern toward a highly
accurate replica of the target pattern p that was given as a cue.
A
0
1
Singular Values
!1
0
1
y and p
0
1
!1
0
1
0
10
20
0
1
0
5
10
!1
0
1
B
1
2
#
4
5
!1.5
!1
!0.5
0
()**+,-./-0+1
23410.56789
6+:3-;*,<:*/3-.9,,3,
C
2 " # $
1&2# #0 100
!1(#
!1
!0(#
)*+,-+.,/010+2/331*45
.,610+)789:
;./55+<1/*4=46+:--1>3
D
M 
Figure 6: Content-addressable memory.
A First three of ﬁve loaded patterns.
Left panels show the leading 20 singular values of Ccue (black) and Cauto (gray).
Right panels show an overlay of the original driver pattern (black, thin) and the
reconstruction at the end of auto-adaptation (gray, thick).
B Pattern reconstruc-
tion errors directly after cueing (black squares) and at end of auto-adaptation (gray
crosses).
C Reconstruction error of loaded patterns (black) and novel patterns
drawn from the same parametric family (gray) versus the number of loaded pat-
terns, averaged over 5 repetitions of the entire experiment and 10 patterns per
plotting point. Error bars indicate standard deviations.
D Autoconceptor adap-
tation dynamics described as evolution toward a plane attractor M (schematic).
Results of a demonstration are illustrated in Figure 6. A 200-neuron reservoir
was loaded with 5 patterns consisting of a weighted sum of two irrational-period
sines, sampled at integer timesteps. The weight ratio and the phaseshift were
chosen at random; the patterns thus came from a family of patterns parametrized
17
by two parameters. The cueing time was 30 timesteps, the free-running auto-
adaptation time was 10,000 timesteps, leading to an auto-adapted conceptor Cauto
at the end of this process. On average, the reconstruction error improved from
about -0.4 (log10 NRMSE measured directly after the cueing) to -1.1 (at the
end of auto-adaptation). It can be shown analytically that the auto-adaptation
process pulls many singular values down to zero. This eﬀect renders the combined
reservoir-conceptor loop very robust against noise, because all noise components in
the directions of the nulled singular values become completely suppressed. In fact,
all results shown in Figure 6 were obtained with strong state noise (signal-to-noise
ratio equal to 1) inserted into the reservoir during the post-cue auto-adaptation.
The system functions as a content-addressable memory (CAM): loaded items
can be recalled by cueing them. The paradigmatic example of a neural CAM are
auto-associative neural networks (AANNs), pioneered by Palm [80] and Hopﬁeld
[48]. In contrast to conceptor-based CAM, which store and re-generate dynamical
patterns, AANNs store and cue-recall static patterns. Furthermore, AANNs do not
admit an incremental storing of new patterns, which is possible in conceptor-based
CAMs. The latter thus represent an advance in neural CAMs in two fundamental
aspects.
To further elucidate the properties of conceptor CAMs, I ran a suite of simu-
lations where the same reservoir was loaded with increasing numbers of patterns,
chosen at random from the same 2-parametric family (Figure 6 C). After loading
with k = 2, 3, 5, . . . , 100 patterns, the reconstruction accuracy was measured at
the end of the auto-adaptation. Not surprisingly, it deteriorated with increasing
memory load k (black line). In addition, I also cued the loaded reservoir with
patterns that were not loaded, but were drawn from the same family. As one
would expect, the re-construction accuracy of these novel patterns was worse than
for the loaded patterns – but only for small k. When the number of loaded pat-
terns exceeded a certain threshold, recall accuracy became essentially equal for
loaded and novel patterns. These ﬁndings can be explained in intuitive terms as
follows. When few patterns are loaded, the network memorizes individual patterns
by “rote learning”, and subsequently can recall these patterns better than other
patterns from the family. When more patterns are loaded, the network learns
a representation of the entire parametric class of patterns. I call this the class
learning eﬀect.
The class learning eﬀect can be geometrically interpreted in terms of a plane
attractor [24] arising in the space of conceptor matrices C (Figure 6 D). The
learnt parametric class of patterns is represented by a d-dimensional manifold M
in this space, where d is the number of deﬁning parameters for the pattern family
(in our example, d = 2). The cueing procedure creates an initial conceptor Ccue
in the vicinity of M, which is then attracted toward M by the auto-adaptation
dynamics. While an in-depth analysis of this situation reveals that this picture is
not mathematically correct in some detail, the plane attractor metaphor yields a
good phenomenal description of conceptor CAM class learning.
18
Plane attractors have been invoked as an explanation for a number of biological
phenomena, most prominently gaze direction control [24]. In such phenomena,
points on the plane attractor correspond to static ﬁxed points (for instance, a
direction of gaze). In contrast, points on M correspond to conceptors which in turn
deﬁne temporal patterns. Again, the conceptor framework “dynamiﬁes” concepts
that have previously been worked out for static patterns only.
Toward biological feasibility: random feature conceptors.
Several com-
putations involved in adapting conceptor matrices are non-local and therefore bio-
logically infeasible. It is however possible to approximate matrix conceptors with
another mechanism which only requires local computations. The idea is to project
(via random projection weights F) the reservoir state into a random feature space
which is populated by a large number of neurons zi; execute the conceptor opera-
tions individually on each of these neurons by multiplying a conception weight ci
into its state; and ﬁnally to project back to the reservoir by another set of random
projection weights G (Figure 7).
The original reservoir-internal random connection weigths W are replaced by a
dyade of two random projections of ﬁrst F, then G, and the original reservoir state
x segregates into a reservoir state r and a random feature state z. The conception
weights ci assume the role of conceptors. They can be learnt and adapted by
procedures which are directly analog to the matrix conceptor case. What had to
be non-local matrix computations before now turns into local, one-dimensional
(scalar) operations. These operations are biologically feasible in the modest sense
that any information needed to adapt a synaptic weight is locally available at that
synapse. All laws and constructions concerning Boolean operations and aperture
carry over.
G 
F' 
reservoir 
feature 
space 
p 
z 
r 
W in 
ci 
Figure 7: Random feature conceptors. This neural architecture has two pools of
neurons, the reservoir and the feature space.
A set of conception weights ci corresponding to a particular pattern can be neu-
rally represented and “stored” in the form of the connections of a single neuron to
the feature space. A dynamical pattern thus can be represented by a single neu-
19
ron. This enables a highly compact neural representation of dynamical patterns.
A machine learning application is presented below.
I re-ran with such random feature conceptors a choice of the simulations that I
did with matrix conceptors, using a number of random features that was two to ﬁve
times as large as the reservoir. The outcome of these simulations: the accuracy
of pattern re-generation is essentially the same as with matrix conceptors, but
setting the aperture is more sensitive.
A hierarchical classiﬁcation and de-noising architecture.
Here I present
a system which combines in a multi-layer neural architecture many of the items
introduced so far. The input to this system is a (very) noisy signal which at a
given time is being generated by one out of a number of possible candidate pattern
generators. The task is to recognize the current generator, and simultaneously to
re-generate a clean version of the noisy input pattern.
A
G 
F' 
!"#$%&#!'()&
y[2]
auto 
G 
F' 
u[2] 
y[1] 
G 
F' 
u[3] 
y[3] 
c[2] 
![12] 
c[1] 
![23] 
c[3] 
y[2] 
![23] 
![12] 
c[1]
auto 
c[2]
auto 
c1, c2, c3, c4 
u[1] 
y[3]
auto 
B
!
!"#
$
%&'()*+*,'-./,(0(0
!
!"#
$
%&'()*1*,'-./,(0(0
!
!"#
$
%&'()*$*,'-./,(0(0
!
!"#
$
/)20/*3&)4&5%(0
!
6!!!
7!!!
!1
!$
!
%.8$!*9:;<=0
!
$!
1!
!$
!
$
-&//()>*0&?-%(0
!
$!
1!
Figure 8: Simultaneous signal de-noising and classiﬁcation. A. Schema of archi-
tecture. B. Simulation results. Panels from above: ﬁrst three panels: hypothesis
vectors γj
[l](n) in the three layers. Color coding: p1 blue, p2 green, p3 red, p4 cyan.
Fourth panel: trust variables τ[1,2](n) (blue) and τ[2,3](n) (green). Fifth panel: sig-
nal reconstruction errors (log10 NRMSE) of y[1] (blue), y[2] (green) and y[3] (red)
versus clean signal pj. Black line: linear baseline ﬁlter. Bottom panels: 20-step
samples from the end of the two presentation periods. Red: noisy input; black:
clean input; thick gray: cleaned output signal y[3].
20
I explain the architecture with an example. It uses three processing layers to
de-noise an input signal u[1](n) = pj(n) + noise, with pj being one of the four
patterns p1, . . . , p4 used before in this report (shown for instance in Figure 1 B).
The architecture implements the following design principles (Figure 8 A). (i) Each
layer is a random feature based conceptor system (as in Figure 7 B). The four
patterns p1, . . . , p4 are initially loaded into each of the layers, and four prototype
conceptor weight vectors c1, . . . , c4 corresponding to the patterns are computed
and stored.
(ii) In a bottom-up processing pathway, the noisy external input
signal u[1](n) = pj(n) + noise is stagewise de-noised, leading to signals y[1], y[2], y[3]
on layers l = 1, 2, 3, where y[3] should be a highly cleaned-up version of the input
(subscripts [l] refer to layers, bottom layer is l = 1). (iii) The top layer auto-
adapts a conceptor c[3] which is constrained to be a weighted OR combination of
the four prototype conceptors. In a suggestive notation this can be written as
c[3](n) = γ1
[3](n) c1 ∨. . . ∨γ4
[3](n) c4. The four weights γj
[3] sum to one and represent
a hypothesis vector expressing the system’s current belief about the current driver
pj. If one of these γj
[3] approaches 1, the system has settled on a ﬁrm classiﬁcation
of the current driving pattern. (iv) In a top-down pathway, conceptors c[l] from
layers l are passed down to the respective layers l−1 below. Because higher layers
should have a clearer conception of the current noisy driver pattern than lower
layers, this passing-down of conceptors “primes” the processing in layer l −1 with
valuable contextual information. (v) Between each pair of layers l, l + 1, a trust
variable τ[l,l+1](n) is adapted by an online procedure. These trust variables range
between 0 and 1. A value of τ[l,l+1](n) = 1 indicates maximal conﬁdence that the
signal y[l+1](n) comes closer to the clean driver pj(n) than the signal y[l](n) does,
that is, the stage-wise denoising actually functions well when progressing from
layer l to l + 1. The trust τ[l,l+1](n) evolves by comparing certain noise ratios that
are observable locally in layers l and l + 1. (vi) Within layer l, an internal auto-
adaptation process generates a candidate de-noised signal yauto
[l]
and a candidate
local autoconceptor cauto
[l]
. The local estimate yauto
[l]
is linearly mixed with the signal
y[l−1], where the trust τ[l−1,l] sets the mixing rate. The mixture u[l] = τ[l−1,l] yauto
[l]
+
(1−τ[l−1,l]) y[l−1] is the eﬀective signal input to layer l. If the trust τ[l−1,l] reaches its
maximal value of 1, layer l will ignore the signal from below and work entirely by
self-generating a pattern. (vii) In a similar way, the eﬀective conceptor in layer l
is a trust-negotiated mixture c[l] = (1−τ[l,l+1]) cauto
[l]
+τ[l,l+1] c[l+1]. Thus if the trust
τ[l,l+1] is maximal, layer l will be governed entirely by the passed-down conceptor
c[l+1].
Summarizing, the higher the trusts inside the hierarchy, the more will the
system be auto-generating conceptor-shaped signals, or conversely, at low trust
values the system will be strongly permeated from below by the outside driver. If
the trust variables reach their maximum value of 1, the system will run in a pure
“confabulation” mode and generate an entirely noise-free signal y[3] – at the risk
of doing this under an entirely misguided hypothesis c[3]. The key to make this
architecture work thus lies in the trust variables. It seems to me that maintaining
21
a measure of trust (or call it conﬁdence, certainty, etc.) is an intrinsically necessary
component in any signal processing architecture which hosts a top-down pathway
of guiding hypotheses (or call them context, priors, bias, etc.).
Figure 8 B shows an excerpt from a simulation run. The system was driven ﬁrst
by an initial 4000 step period of p1 + noise, followed by 4000 steps of p3 + noise.
The signal-to-noise ratio was 0.5 (noise twice as strong as signal). The system
successfully settles on the right hypothesis (top panel) and generates very clean
de-noised signal versions (bottom panel). The crucial item in this ﬁgure is the
development of the trust variable τ[2,3]. At the beginning of each 4000 step period
it brieﬂy drops, allowing the external signal to permeate upwards through the
layers, thus informing the local auto-adaptation loops about “what is going on
outside”. After these initial drops the trust rises to almost 1, indicating that the
system ﬁrmly “believes” to have detected the right pattern. It then generates
pattern versions that have almost no mix-in from the noisy external driver.
As a baseline comparison I also trained a standard linear transversal ﬁlter which
computed a de-noised input pattern point based on the preceding K = 2600 input
values. The ﬁlter length K was set equal to the number of trainable parameters in
the neural architecture. The performance of this linear de-noising ﬁlter (black line
in Figure 8) is inferior to the architecture’s performance both in terms of accuracy
and response time.
It is widely believed that top-down hypothesis-passing through a processing hi-
erarchy plays a fundamental role in biological cognitive systems [33, 16]. However,
the current best artiﬁcial pattern recognition systems [39, 59] use purely bottom-up
processing – leaving room for further improvement by including top-down guid-
ance. A few hierarchical architectures which exploit top-down hypothesis-passing
have been proposed [33, 47, 43, 35]. All of these are designed for recognizing static
patterns, especially images. The conceptor-based architecture presented here ap-
pears to be the ﬁrst hierarchical system which targets dynamical patterns and uses
top-down hypothesis-passing. Furthermore, in contrast to state-of-the-art pattern
recognizers, it admits an incremental extension of the pattern repertoire.
Intrinsic conceptor logic.
In mathematical logics the semantics (“meaning”)
of a symbol or operator is formalized as its extension. For instance, the symbol cow
in a logic-based knowledge representation system in AI is semantically interpreted
by the set of all (physical) cows, and the OR-operator ∨is interpreted as set union:
cow ∨horse would refer to the set comprising all cows and horses. Similarly, in
cognitive science, concepts are semantically referring to their extensions, usually
called categories in this context [74]. Both in mathematical logic and cognitive
science, extensions need not be conﬁned to physical objects; the modeler may also
deﬁne extensions in terms of mathematical structures, sensory perceptions, hypo-
thetical worlds, ideas or facts. But at any rate, there is an ontological diﬀerence
between the two ends of the semantic relationship.
This ontological gap dissolves in the case of conceptors. The natural account
22
of the “meaning” of a matrix conceptor C is the shape of the neural state cloud
it is derived from.
This shape is given by the correlation matrix R of neural
states. Both C and R have the same mathematical format: positive semi-deﬁnite
matrices of identical dimension. Figure 9 visualizes the diﬀerence between clas-
sical extensional semantics of logics and the system-internal conceptor semantics.
The symbol |= is the standard mathematical notation for the semantical meaning
relationship.
Figure 9: Contrasting the extensional semantics of classical knowledge represen-
tation formalisms (upper half of graphics) with conceptor semantics (lower half).
I have cast these intuitions into a formal speciﬁcation of an intrinsic conceptor
logic (ICL), where the semantic relationship outlined above is formalized within
the framework of institutions [37]. This framework has been developed in mathe-
matics and computer science to provide a uniﬁed view on the multitude of existing
“logics”. By formalizing ICL as an institution, conceptor logic can be rigorously
compared to other existing logics. I highlight two ﬁndings. First, an ICL cast as
an institution is a dynamcial system in its own right: the symbols used in this
logic evolve over time. This is very much diﬀerent from traditional views on logic,
where symbols are static tokens. Second, it turns out that ICL is a logic which
is decidable. Stated in intuitive terms, in a decidable logic it can be calculated
whether a “concept” ψ subsumes a concept ϕ (as in “a cow is an animal”). De-
ciding concept subsumption is a core task in AI systems and human cognition.
In most logic-based AI systems, deciding concept subsumption can become com-
putationally expensive or even impossible. In ICL it boils down to determining
whether all components of a certain conception weight vector ci are smaller or
equal to the corresponding components c′
i of another such vector, which can be
23
done in a single processing step. This may help explaining why humans can make
classiﬁcation judgements almost instantaneously.
Discussion.
The human brain is a neurodynamical system which evidently sup-
ports logico-rational reasoning [49]. Since long this has challenged scientists to
ﬁnd computational models which connect neural dynamics with logic. Very dif-
ferent solutions have been suggested. At the dawn of computational neuroscience,
McCulloch and Pitts have already interpreted networks of binary-state neurons as
carrying out Boolean operations [73]. Logical inferences of various kinds have been
realized in localist connectionist networks where neurons are labelled by concept
names [82, 96]. In neurofuzzy modeling, feedforward neural networks are trained
to carry out operations of fuzzy logic on their inputs [12]. In a ﬁeld known as
neuro-symbolic computation, deduction rules of certain formal logic systems are
coded or trained into neural networks [8, 65, 10]. The combinatorial/compositional
structure of symbolic knowledge has been captured by dedicated neural circuits to
enable tree-structured representations [83] or variable-binding functionality [104].
All of these approaches require interface mechanisms. These interface mecha-
nisms are non-neural and code symbolic knowledge representations into the numer-
ical activation values of neurons and/or the topological structure of networks. One
could say, previous approaches code logic into specialized neural networks, while
conceptors instantiate the logic of generic recurrent neural networks. This novel,
simple, versatile, computationally eﬃcient, neurally not infeasible, bi-directional
connection between logic and neural dynamics opens new perspectives for compu-
tational neuroscience and machine learning.
24
2
Introduction
In this section I expand on the brief characterization of the scientiﬁc context given
in Section 1, and introduce mathematical notation.
2.1
Motivation
Intelligent behavior is desired for robots, demonstrated by humans, and studied
in a wide array of scientiﬁc disciplines. This research unfolds in two directions. In
“top-down” oriented research, one starts from the “higher” levels of cognitive per-
formance, like rational reasoning, conceptual knowledge representation, planning
and decision-making, command of language. These phenomena are described in
symbolic formalisms developed in mathematical logic, artiﬁcial intelligence (AI),
computer science and linguistics. In the “bottom-up” direction, one departs from
“low-level” sensor data processing and motor control, using the analytical tools
oﬀered by dynamical systems theory, signal processing and control theory, statis-
tics and information theory. For brevity I will refer to these two directions as
the conceptual-symbolic and the data-dynamical sets of phenomena, and levels
of description. The two interact bi-directionally. Higher-level symbolic concepts
arise from low-level sensorimotor data streams in short-term pattern recognition
and long-term learning processes. Conversely, low-level processing is modulated,
ﬁltered and steered by processes of attention, expectations, and goal-setting in a
top-down fashion.
Several schools of thought (and strands of dispute) have evolved in a decades-
long quest for a uniﬁcation of the conceptual-symbolic and the data-dynamical
approaches to intelligent behavior. The nature of symbols in cognitive processes
has been cast as a philosophical issue [95, 29, 44]. In localist connectionistic mod-
els, symbolically labelled abstract processing units interact by nonlinear spreading
activation dynamics [22, 96]. A basic tenet of behavior-based AI is that higher
cognitive functions emerge from low-level sensori-motor processing loops which
couple a behaving agent into its environment [11, 81]. Within cognitive science, a
number of cognitive pheneomena have been described in terms of self-organization
in nonlinear dynamical systems [94, 98, 105]. A pervasive idea in theoretical neu-
roscience is to interpret attractors in nonlinear neural dynamics as the carriers of
conceptual-symbolic representations. This idea can be traced back at least to the
notion of cell assemblies formulated by Hebb [45], reached a ﬁrst culmination in
the formal analysis of associative memories [80, 48, 4], and has since then diver-
siﬁed into a range of increasingly complex models of interacting (partial) neural
attractors [109, 103, 87, 101]. Another pervasive idea in theoretical neuroscience
and machine learning is to consider hierarchical neural architectures, which are
driven by external data at the bottom layer and transform this raw signal into
increasingly abstract feature representations, arriving at conceptual representa-
tions at the top layer of the hierarchy. Such hierarchical architectures mark the
state of the art in pattern recognition technology [66, 38]. Many of these systems
25
process their input data in a uni-directional, bottom-to-top fashion. Two notable
exceptions are systems where each processing layer is designed according to sta-
tistical principles from Bayes’ rule [33, 47, 16], and models based on the iterative
linear maps of map seeking circuits [35, 111], both of which enable top-down guid-
ance of recognition by expectation generation. More generally, leading actors in
theoretical neuroscience have characterized large parts of their ﬁeld as an eﬀort
to understand how cognitive phenomena arise from neural dynamics [2, 36]. Fi-
nally, I point out two singular scientiﬁc eﬀorts to design comprehensive cognitive
brain models, the ACT-R architectures developed by Anderson et al. [5] and the
Spaun model of Eliasmith et al. [26]. Both systems can simulate a broad selec-
tion of cognitive behaviors. They integrate numerous subsystems and processing
mechanisms, where ACT-R is inspired by a top-down modeling approach, starting
from cognitive operations, and Spaun from a bottom-up strategy, starting from
neurodynamical processing principles.
Despite this extensive research, the problem of integrating the conceptual-
symbolic with the data-dynamical aspects of cognitive behavior cannot be consid-
ered solved. Quite to the contrary, two of the largest current research initiatives
worldwide, the Human Brain Project [1] and the NIH BRAIN initiative [51], are
ultimately driven by this problem. There are many reasons why this question is
hard, ranging from experimental challenges of gathering relevant brain data to fun-
damental oppositions of philosophical paradigms. An obstinate stumbling block
is the diﬀerent mathematical nature of the fundamental formalisms which appear
most natural for describing conceptual-symbolic versus data-dynamical phenom-
ena: symbolic logic versus nonlinear dynamics.
Logic-oriented formalisms can
easily capture all that is combinatorially constructive and hierarchically organized
in cognition: building new concepts by logical deﬁnitions, describing nested plans
for action, organizing conceptual knowledge in large and easily extensible abstrac-
tion hierarchies. But logic is inherently non-temporal, and in order to capture
cognitive processes, additional, heuristic “scheduling” routines have to be intro-
duced which control the order in which logical rules are executed. This is how
ACT-R architectures cope with the integration problem. Conversely, dynamical
systems formalisms are predestined for modeling all that is continuously chang-
ing in the sensori-motor interface layers of a cognitive system, driven by sensor
data streams. But when dynamical processing modules have to be combined into
compounds that can solve complex tasks, again additional design elements have
to be inserted, usually by manually coupling dynamical modules in ways that
are informed by biological or engineering insight on the side of the researcher.
This is how the Spaun model has been designed to realize its repertoire of cog-
nitive functions.
Two important modeling approaches venture to escape from
the logic-dynamics integration problem by taking resort to an altogether diﬀerent
mathematical framework which can accomodate both sensor data processing and
concept-level representations: the framework of Bayesian statistics and the frame-
work of iterated linear maps mentioned above. Both approaches lead to a uniﬁed
26
formal description across processing and representation levels, but at the price of
a double weakness in accounting for the embodiment of an agent in a dynamical
environment, and for the combinatorial aspects of cognitive operations. It appears
that current mathematical methods can instantiate only one of the three: contin-
uous dynamics, combinatorial productivity, or a uniﬁed level-crossing description
format.
The conceptor mechanisms introduced in this report bi-directionally connect
the data-dynamical workings of a recurrent neural network (RNN) with a conceptual-
symbolic representation of diﬀerent functional modes of the RNN. Mathematically,
conceptors are linear operators which characterize classes of signals that are be-
ing processed in the RNN. Conceptors can be represented as matrices (convenient
in machine learning applications) or as neural subnetworks (appropriate from a
computational neuroscience viewpoint). In a bottom-up way, starting from an op-
erating RNN, conceptors can be learnt and stored, or quickly generated on-the-ﬂy,
by what may be considered the simplest of all adaptation rules: learning a regular-
ized identity map. Conceptors can be combined by elementary logical operations
(AND, OR, NOT), and can be ordered by a natural abstraction relationship.
These logical operations and relations are deﬁned via a formal semantics. Thus,
an RNN engaged in a variety of tasks leads to a learnable representation of these
operations in a logic formalism which can be neurally implemented. Conversely,
in a top-down direction, conceptors can be inserted into the RNN’s feedback loop,
where they robustly steer the RNN’s processing mode. Due to their linear algebra
nature, conceptors can be continuously morphed and “sharpened” or “defocussed”,
which extends the discrete operations that are customary in logics into the domain
of continuous “mental” transformations. I highlight the versatility of conceptors
in a series of demonstrations: generating and morphing many diﬀerent dynamical
patterns with a single RNN; managing and monitoring the storing of patterns
in a memory RNN; learning a class of dynamical patterns from presentations of
a small number of examples (with extrapolation far beyond the training exam-
ples); classiﬁcation of temporal patterns; de-noising of temporal patterns; and
content-addressable memory systems. The logical conceptor operations enable an
incremental extension of a trained system by incorporating new patterns without
interfering with already learnt ones. Conceptors also suggest a novel answer to a
perennial problem of attractor-based models of concept representations, namely
the question of how a cognitive trajectory can leave an attractor (which is at odds
with the very nature of an attractor). Finally, I outline a version of conceptors
which is biologically plausible in the modest sense that only local computations
and no information copying are needed.
2.2
Mathematical Preliminaries
I assume that the reader is familiar with properties of positive semideﬁnite matri-
ces, the singular value decomposition, and (in some of the analysis of adaptation
27
dynamics) the usage of the Jacobian of a dynamical system for analysing stability
properties.
[a, b], (a, b), (a, b], [a, b) denote the closed (open, half-open) interval between real
numbers a and b.
A′ or x′ denotes the transpose of a matrix A or vector x. I is the identity
matrix (the size will be clear from the context or be expressed as In×n). The ith
unit vector is denoted by ei (dimension will be clear in context). The trace of a
square matrix A is denoted by tr A. The singular value decomposition of a matrix
A is written as USV ′ = A, where U, V are orthonormal and S is the diagonal
matrix containing the singular values of A, assumed to be in descending order
unless stated otherwise. A† is the pseudoinverse of A. All matrices and vectors
will be real and this will not be explicitly mentioned.
I use the Matlab notation to address parts of vectors and matrices, for instance
M(:, 3) is the third column of a matrix M and M(2 : 4, :) picks from M the
submatrix consisting of rows 2 to 4. Furthermore, again like in Matlab, I use the
operator diag in a “toggling” mode: diag A returns the diagonal vector of a square
matrix A, and diag d constructs a diagonal matrix from a vector d of diagonal
elements. Another Matlab notation that will be used is “.∗” for the element-wise
multiplication of vectors and matrices of the same size, and “.∧” for element-wise
exponentation of vectors and matrices.
R(A) and N(A) denote the range and null space of a matrix A. For linear
subspaces S, T of Rn, S⊥is the orthogonal complement space of S and S +
T is the direct sum {x + y | x ∈S, y ∈T } of S and T .
PS is the n × n
dimensional projection matrix on a linear subspace S of Rn. For a k-dimensional
linear subspace S of Rn, BS denotes any n×k dimensional matrix whose columns
form an orthonormal basis of S. Such matrices BS will occur only in contexts
where the choice of basis can be arbitrary. It holds that PS = BS(BS)′.
E[x(n)] denotes the expectation (temporal average) of a stationary signal x(n)
(assuming it is well-deﬁned, for instance, coming from an ergodic source).
For a matrix M, ∥M∥fro is the Frobenius norm of M. For real M, it is the
square root of the summed squared elements of M. If M is positive semideﬁnite
with SVD M = USU ′, ∥M∥fro is the same as the 2-norm of the diagonal vector
of S, i.e. ∥M∥fro = ((diagS)′ (diagS))1/2. Since in this report I will exclusively use
the Frobenius norm for matrices, I sometimes omit the subscript and write ∥M∥
for simplicity.
In a number of simulation experiments, a network-generated signal y(n) will be
matched against a target pattern p(n). The accuracy of the match will be quanti-
ﬁed by the normalized root mean square error (NRMSE),
p
[(y(n) −p(n))2]/[(p(n)2],
where [·] is the mean operator over data points n.
The symbol N is reserved for the size of a reservoir (= number of neurons)
throughout.
28
3
Theory and Demonstrations
This is the main section of this report. Here I develop in detail the concepts,
mathematical analysis, and algorithms, and I illustrate various aspects in computer
simulations.
Figure 10 gives a navigation guide through the dependency tree of the compo-
nents of this section.
Basic theory and 
usage of conceptors: 
2.1 – 2.6 
Morphing 
patterns demo: 
2.7 
How conceptors relate 
to neural data; aperture; 
chaotic attractor demo: 
2.8 
Boolean conceptor 
logic and 
abstraction: 
2.9, 2.10 
Memory 
management 
demo: 
2.11 
Pattern 
recognition 
demo: 
2.12 
Autoconceptors: 
2.13 
random feature 
conceptors, biological 
plausibility: 
2.14 
Hierarchical filtering 
architecture: 
2.15 
Conceptor logic: 
2.16, 2.17 
Figure 10: Dependency tree of subsections in Section 3.
The program code (Matlab) for all simulations can be retrieved from
http://minds.jacobs-university.de/sites/default/ﬁles/uploads/...
...SW/ConceptorsTechrepMatlab.zip.
3.1
Networks and Signals
Throughout this report, I will be using discrete-time recurrent neural networks
made of simple tanh neurons, which will be driven by an input time series p(n).
In the case of 1-dimensional input, these networks consist of (i) a “reservoir”
of N recurrently connected neurons whose activations form a state vector x =
(x1, . . . , xN)′, (ii) one external input neuron that serves to drive the reservoir with
training or cueing signals p(n) and (iii) another external neuron which serves to
read out a scalar target signal y(n) from the reservoir (Fig. 11).
The system
29
operates in discrete timesteps n = 0, 1, 2, . . . according to the update equations
x(n + 1)
=
tanh(W x(n) + W in p(n + 1) + b)
(1)
y(n)
=
W out x(n),
(2)
where W is the N × N matrix of reservoir-internal connection weights, W in is the
N ×1 sized vector of input connection weights, W out is the 1×N vector of readout
weights, and b is a bias. The tanh is a sigmoidal function that is applied to the
network state x component-wise. Due to the tanh, the reservoir state space or
simply state space is (−1, 1)N.
The input weights and the bias are ﬁxed at random values and are not subject
to modiﬁcation through training. The output weights W out are learnt. The reser-
voir weights W are learnt in some of the case studies below, in others they remain
ﬁxed at their initial random values. If they are learnt, they are adapted from a
random initialization denoted by W ∗. Figure 11 A illustrates the basic setup.
I will call the driving signals p(n) patterns.
In most parts of this report,
patterns will be periodic. Periodicity comes in two variants. First, integer-periodic
patterns have the property that p(n) = p(n+k) for some positive integer k. Second,
irrational-periodic patterns are discretely sampled from continuous-time periodic
signals, where the sampling interval and the period length of the continuous-time
signal have an irrational ratio. An example is p(n) = sin(2 π n/(10
√
2)). These
two sorts of drivers will eventually lead to diﬀerent kinds of attractors trained into
reservoirs: integer-periodic signals with period length P yield attractors consisting
of P points in reservoir state space, while irrational-periodic signals give rise to
attracting sets which can be topologically characterized as one-dimensional cycles
that are homeomorphic to the unit cycle in R2.
3.2
Driving a Reservoir with Diﬀerent Patterns
A basic theme in this report is to develop methods by which a collection of diﬀerent
patterns can be loaded in, and retrieved from, a single reservoir. The key for
these methods is an elementary dynamical phenomenon: if a reservoir is driven
by a pattern, the entrained network states are conﬁned to a linear subspace of
network state space which is characteristic of the pattern. In this subsection I
illuminate this phenomenon by a concrete example. This example will be re-used
and extended on several occasions throughout this report.
I use four patterns.
The ﬁrst two are irrational periodic and the last two
are integer-periodic: (1) a sinewave of period ≈8.83 sampled at integer times
(pattern p1(n)) (2) a sinewave p2(n) of period ≈9.83 (period of p1(n) plus 1), (3)
a random 5-periodic pattern p3(n) and (4) a slight variation p4(n) thereof (Fig.
12 left column).
A reservoir with N = 100 neurons is randomly created. At creation time the
input weights W in and the bias b are ﬁxed at random values; these will never be
modiﬁed thereafter. The reservoir weights are initialized to random values W ∗;
30
! 
xl
j(n "1)
! 
xi
j(n)
! 
p j(n)
! 
x
! 
y
! 
p
! 
W in
! 
W
! 
Wi
*
! 
Wi
in
! 
Wi
!"
#"
$"
! 
W out
! 
xi
j(n)
! 
xm
j (n "1)
! 
xk
j(n "1)
! 
xl
j(n "1)
! 
xm
j (n "1)
! 
xk
j(n "1)
Figure 11: A. Basic system setup. Through input connections W in, an input neu-
ron feeds a driving signal p to a “reservoir” of N = 100 neurons which are recur-
rently connected to each other through connections W. From the N-dimensional
neuronal activation state x, an output signal y is read out by connections W out.
All broken connections are trainable. B. During the initial driving of the reservoir
with driver pj, using initial random weights W ∗, neuron xi produces its signal
(thick gray line) based on external driving input p and feeds from other neurons x
from within the reservoir (three shown). C. After training new reservoir weights
W, the same neuron should produce the same signal based only on the feeds from
other reservoir neurons.
in this ﬁrst demonstration they will not be subsequently modiﬁed either. The
readout weights are initially undeﬁned (details in Section 4.1).
In four successive and independent runs, the network is driven by feeding the
respective pattern pj(n) as input (j = 1, . . . , 4), using the update rule
xj(n + 1) = tanh(W ∗xj(n) + W in pj(n + 1) + b).
After an initial washout time, the reservoir dynamics becomes entrained to the
driver and the reservoir state xj(n) exhibits an involved nonlinear response to the
driver pj. After this washout, the reservoir run is continued for L = 1000 steps,
and the obtained states xj(n) are collected into N × L = 100 × 1000 sized state
collection matrices Xj for subsequent use.
The second column in Fig. 12 shows traces of three randomly chosen reservoir
neurons in the four driving conditions. It is apparent that the reservoir has become
entrained to the driving input. Mathematically, this entrainment is captured by
the concept of the echo state property: any random initial state of a reservoir
is “forgotten”, such that after a washout period the current network state is a
31
function of the driver. The echo state property is a fundamental condition for
RNNs to be useful in learning tasks [54, 13, 46, 110, 99, 71]. It can be ensured by
an appropriate scaling of the reservoir weight matrix. All networks employed in
this report possess the echo state property.
!1
0
1
driver and y
0.0064
reservoir states
0
50
100
!20
!10
0
10
log10 PC energy
0
5
10
0
20
40
leading PC energy
!1
0
1
0.0059
0
50
100
!20
!10
0
10
0
5
10
0
20
40
!1
0
1
0.091
0
50
100
!20
!10
0
10
0
5
10
0
20
40
0
10
20
!1
0
1
0.088
0
10
20
0
50
100
!20
!10
0
10
0
5
10
0
20
40
Figure 12: The subspace phenomenon. Each row of panels documents situation
when the reservoir is driven by a particular input pattern. “Driver and y”: the
driving pattern (thin black line) and the signals retrieved with conceptors (broad
light gray line). Number inset is the NRMSE between original driver and retrieved
signal. “Reservoir states”: activations of three randomly picked reservoir neurons.
“Log10 PC energy”: log10 of reservoir signal energies in the principal component
directions. “Leading PC energy”: close-up on ﬁrst ten signal energies in linear
scale.
Notice that the ﬁrst two panels in each row show discrete-time signals;
points are connected by lines only for better visual appearance.
A principal component analysis (PCA) of the 100 reservoir signals reveals
that the driven reservoir signals are concentrated on a few principal directions.
Concretely, for each of the four driving conditions, the reservoir state correlation
matrix was estimated by Rj = Xj (Xj)′/L, and its SVD U jΣj(U j)′ = Rj was com-
puted, where the columns of U j are orthonormal eigenvectors of Rj (the principal
component (PC) vectors), and the diagonal of Σj contains the singular values of
Rj, i.e. the energies (mean squared amplitudes) of the principal signal components.
Figure 12 (third and last column) shows a plot of these principal component ener-
gies. The energy spectra induced by the two irrational-period sines look markedly
diﬀerent from the spectra obtained from the two 5-periodic signals. The latter lead
to nonzero energies in exactly 5 principal directions because the driven reservoir
dynamics periodically visits 5 states (the small but nonzero values in the log10
32
plots in Figure 12 are artefacts earned from rounding errors in the SVD compu-
tation). In contrast, the irrational-periodic drivers lead to reservoir states which
linearly span all of RN (Figure 12, upper two log10 plots). All four drivers however
share a relevant characteristic (Figure 12, right column): the total reservoir energy
is concentrated in a quite small number of leading principal directions.
When one inspects the excited reservoir dynamics in these four driving con-
ditions, there is little surprise that the neuronal activation traces look similar to
each other for the ﬁrst two and in the second two cases (Figure 12, second col-
umn). This “similarity” can be quantiﬁed in a number of ways. Noting that the
geometry of the “reservoir excitation space” in driving condition j is characterized
by a hyperellipsoid with main axes U j and axis lengths diag Σj, a natural way to
deﬁne a similarity between two such ellipsoids i, j is to put
simR
i,j = (Σi)1/2 (U i)′U j(Σj)1/2
∥diagΣi∥∥diagΣj∥.
(3)
The measure simR
i,j ranges in [0, 1].
It is 0 if and only if the reservoir signals
xi, xj populate orthogonal linear subspaces, and it is 1 if and only if Ri = a Rj
for some scaling factor a. The measure simR
i,j can be understood as a generalized
squared cosine between Ri and Rj.
Figure 13 A shows the similarity matrix
(simR
i,j)i,j obtained from (3). The similarity values contained in this matrix appear
somewhat counter-intuitive, inasmuch as the reservoir responses to the sinewave
patterns come out as having similarities of about 0.6 with the 5-periodic driven
reservoir signals; this does not agree with the strong visual dissimilarity apparent
in the state plots in Figure 12. In Section 3.5 I will introduce another similarity
measure which agrees better with intuitive judgement.
A
B
C
Figure 13:
Matrix plots of pairwise similarity between the subspaces excited
in the four driving conditions.
Grayscale coding: 0 = black, 1 = white.
A:
similarity simR
i,j based on the data correlation matrices Ri. B,C: similarities based
on conceptors C(Ri, α) for two diﬀerent values of aperture α. For explanation see
text.
33
3.3
Storing Patterns in a Reservoir, and Training the Read-
out
One of the objectives of this report is a method for storing several driving patterns
in a single reservoir, such that these stored patterns can later be retrieved and
otherwise be controlled or manipulated.
In this subsection I explain how the
initial “raw” reservoir weights W ∗are adapted in order to “store” or “memorize”
the drivers, leading to a new reservoir weight matrix W.
I continue with the
four-pattern-example used above.
The guiding idea is to enable the reservoir to re-generate the driven responses
xj(n) in the absence of the driving input. Consider any neuron xi (Fig. 11B).
During the driven runs j = 1, . . . , 4, it has been updated per
xj
i(n + 1) = tanh(W ∗
i xj(n) + W in
i pj(n + 1) + bi),
where W ∗
i is the i-th row in W ∗, W in
i
is the i-th element of W in, and bi is the
ith bias component. The objective for determining new reservoir weights W is
that the trained reservoir should be able to oscillate in the same four ways as in
the external driving conditions, but without the driving input. That is, the new
weights Wi leading to neuron i should approximate
tanh(W ∗
i xj(n) + W in
i pj(n + 1) + bi) ≈tanh(Wi xj(n) + bi)
as accurately as possible, for j = 1, . . . , 4. Concretely, we optimize a mean square
error criterion and compute
Wi = argmin ˜
Wi
X
j=1,...,K
X
n=1,...,L
(W ∗
i xj(n) + W in
i pj(n + 1) −˜Wi xj(n))2,
(4)
where K is the number of patterns to be stored (in this example K = 4). This is
a linear regression task, for which a number of standard algorithms are available.
I employ ridge regression (details in Section 4.1).
The readout neuron y serves as passive observer of the reservoir dynamics.
The objective to determine its connection weights W out is simply to replicate the
driving input, that is, W out is computed (again by ridge regression) such that it
minimizes the squared error (pj(n)−W outxj(n))2, averaged over time and the four
driving conditions.
I will refer to this preparatory training as storing patterns pj in a reservoir,
and call a reservoir loaded after patterns have been stored.
3.4
Conceptors: Introduction and Basic Usage in Retrieval
How can these stored patterns be individually retrieved again?
After all, the
storing process has superimposed impressions of all patterns on all of the re-
computed connection weights W of the network – very much like the pixel-wise
34
addition of diﬀerent images would yield a mixture image in which the individual
original images are hard to discern. One would need some sort of ﬁlter which can
disentangle again the superimposed components in the connection weights. In this
section I explain how such ﬁlters can be obtained.
The guiding idea is that for retrieving pattern j from a loaded reservoir, the
reservoir dynamics should be restricted to the linear subspace which is characteris-
tic for that pattern. For didactic reasons I start with a simplifying assumption (to
be dropped later). Assume that there exists a (low-dimensional) linear subspace
Sj ⊂RN such that all state vectors contained in the driven state collection Xj lie
in Sj. In our example, this is actually the case for the two 5-periodic patterns. Let
PSj be the projector matrix which projects RN on Sj. We may then hope that
if we run the loaded reservoir autonomously (no input), constraining its states to
Sj using the update rule
x(n + 1) = PSj tanh(W x(n) + b),
(5)
it will oscillate in a way that is closely related to the way how it oscillated when
it was originally driven by pj.
However, it is not typically the case that the states obtained in the original
driving conditions are conﬁned to a proper linear subspace of the reservoir state
space. Consider the sine driver p1 in our example. The linear span of the reservoir
response state is all of RN (compare the log10 PC energy plots in Figure 12).
The associated projector would be the identity, which would not help to single
out an individual pattern in retrieval. But actually we are not interested in those
principal directions of reservoir state space whose excitation energies are negligibly
small (inspect again the quick drop of these energies in the third column, top panel
in Figure 12 – it is roughly exponential over most of the spectrum, except for an
even faster decrease for the very ﬁrst few singular values). Still considering the
sinewave pattern p1: instead of PRN we would want a projector that projects on
the subspace spanned by a “small” number of leading principal components of
the “excitation ellipsoid” described by the sine-driver-induced correlation matrix
R1. What qualiﬁes as a “small” number is, however, essentially arbitrary. So we
want a method to shape projector-like matrices from reservoir state correlation
matrices Rj in a way that we can adjust, with a control parameter, how many of
the leading principal components should become registered in the projector-like
matrix.
At this point I give names to the projector-like matrices and the adjustment
parameter. I call the latter the aperture parameter, denoted by α. The projector-
like matrices will be called conceptors and generally be denoted by the symbol
C. Since conceptors are derived from the ellipsoid characterized by a reservoir
state corrlation matrix Rj, and parametrized by the aperture parameter, I also
sometimes write C(Rj, α) to make this dependency transparent.
There is a natural and convenient solution to meet all the intuitive objectives
for conceptors that I discussed up to this point. Consider a reservoir driven by
35
a pattern pj(n), leading to driven states xj(n) collected (as columns) in a state
collection matrix Xj, which in turn yields a reservoir state correlation matrix
Rj = Xj(Xj)′/L. We deﬁne a conceptor C(Rj, α) with the aid of a cost func-
tion L(C | Rj, α), whose minimization yields C(Rj, α). The cost function has two
components. The ﬁrst component reﬂects the objective that C should behave as
a projector matrix for the states that occur in the pattern-driven run of the reser-
voir. This component is En[∥xj(n) −Cxj(n)∥2], the time-averaged deviation of
projections Cxj from the state vectors xj. The second component of L adjusts how
many of the leading directions of Rj should become eﬀective for the projection.
This component is α−2∥C∥2
fro. This leads to the following deﬁnition.
Deﬁnition 1 Let R = E[xx′] be an N × N correlation matrix and α ∈(0, ∞).
The conceptor matrix C = C(R, α) associated with R and α is
C(R, α) = argminC E[∥x −Cx∥2] + α−2 ∥C∥2
fro.
(6)
The minimization criterion (6) uniquely speciﬁes C(R, α). The conceptor ma-
trix can be eﬀectively computed from R and α. This is spelled out in the following
proposition, which also lists elementary algebraic properties of conceptor matrices:
Proposition 1 Let R = E[x x′] be a correlation matrix and α ∈(0, ∞). Then,
1. C(R, α) can be directly computed from R and α by
C(R, α) = R (R + α−2 I)−1 = (R + α−2 I)−1 R,
(7)
2. if R = UΣU ′ is the SVD of R, then the SVD of C(R, α) can be written as
C = USU ′, i.e. C has the same principal component vector orientation as
R,
3. the singular values si of C relate to the singular values σi of R by si =
σi/(σi + α−2),
4. the singular values of C range in [0, 1),
5. R can be recovered from C and α by
R = α−2 (I −C)−1 C = α−2 C (I −C)−1.
(8)
The proof is given in Section 5.1. Notice that all inverses appearing in this
proposition are well-deﬁned because α > 0 is assumed, which implies that all
singular values of C(R, α) are properly smaller than 1.
I will later generalize
conceptors to include the limiting cases α = 0 and α = ∞(Section 3.8.1).
In practice, the correlation matrix R = E[xx′] is estimated from a ﬁnite sample
X, which leads to the approximation ˆR = XX′/L, where X = (x(1), . . . , x(L)) is
a matrix containing reservoir states x(n) collected during a learning run.
36
Figure 14 shows the singular value spectra of C(R, α) for various values of α,
for our example cases of R = R1 (irrational-period sine driver) and R = R3 (5-
periodic driver). We ﬁnd that the nonlinearity inherent in (7) makes the conceptor
matrices come out “almost” as projector matrices: the singular values of C are
mostly close to 1 or close to 0. In the case of the 5-periodic driver, where the
excited network states populate a 5-dimensional subspace of RN, increasing α lets
C(R, α) converge to a projector onto that subspace.
0
50
100
0
1
sine (pattern 1)
0
50
100
0
1
10ïperiodic random (pattern 3)
 
 
_ = 1
_ = 10
_ = 100
_ = 1000
_ = 10000
Figure 14: How the singular values of a conceptor depend on α. Singular value
spectra are shown for the ﬁrst sinewave pattern and the ﬁrst 5-periodic random
pattern. For explanation see text.
If one has a conceptor matrix Cj = C(Rj, α) derived from a pattern pj through
the reservoir state correlation matrix Rj associated with that pattern, the concep-
tor matrix can be used in an autonomous run (no external input) using the update
rule
x(n + 1) = Cj tanh(W x(n) + b),
(9)
where the weight matrix W has been shaped by storing patterns among which
there was pj. Returning to our example, four conceptors C1, . . . , C4 were computed
with α = 10 and the loaded reservoir was run under rule (9) from a random initial
state x(0). After a short washout period, the network settled on stable periodic
dynamics which were closely related to the original driving patterns. The network
dynamics was observed through the previously trained output neuron. The left
column in Figure 12 shows the autonomous network output as a light bold gray line
underneath the original driver. To measure the achieved accuracy, the autonomous
output signal was phase-aligned with the driver (details in Section 4.1) and then
the NRMSE was computed (insets in Figure panels). The NRMSEs indicate that
the conceptor-constrained autonomous runs could successfully separate from each
other even the closely related pattern pairs p1 versus p2 and p3 versus p4.
A note on terminology.
Equation (9) shows a main usage of conceptor ma-
trices: they are inserted into the reservoir state feedback loop and cancel (respec-
tively, dampen) those reservoir state components which correspond to directions
in state space associated with zero (or small, respectively) singular values in the
37
conceptor matrix. In most of this report, such a direction-selective damping in
the reservoir feedback loop will be eﬀected by way of inserting matrices C like
in Equation (9). However, inserting a matrix is not the only way by which such
a direction-selective damping can be achieved. In Section 3.14, which deals with
biological plausibility issues, I will propose a neural circuit which achieves a sim-
ilar functionality of direction-speciﬁc damping of reservoir state components by
other means and with slightly diﬀering mathematical properties. I understand the
concept of a “conceptor” as comprising any mechanism which eﬀects a pattern-
speciﬁc damping of reservoir signal components. Since in most parts of this report
this will be achieved with conceptor matrices, as in (9), I will often refer to these C
matrices as “conceptors” for simplicity. The reader should however bear in mind
that the notion of a conceptor is more comprehensive than the notion of a concep-
tor matrix. I will not spell out a formal deﬁnition of a “conceptor”, deliberately
leaving this concept open to become instantiated by a variety of computational
mechanisms of which only two are formally deﬁned in this report (via conceptor
matrices, and via the neural circuit given in Section 3.14).
3.5
A Similarity Measure for Excited Network Dynamics
In Figure 13 A a similarity matrix is presented which compares the excitation
ellipsoids represented by the correlation matrices Rj by the similarity metric (3).
I remarked at that time that this is not a fully satisfactory metric, because it
does not agree well with intuition. We obtain a more intuitively adequate simil-
iarity metric if conceptor matrices are used as descriptors of “subspace ellipsoid
geometry” instead of the raw correlation matrices, i.e. if we employ the metric
simα
i,j = (Si)1/2 (U i)′U j(Sj)1/2
∥diagSi∥∥diagSj∥,
(10)
where USjU ′ is the SVD of C(Rj, α). Figure 13 B,C shows the similarity matri-
ces arising in our standard example for α = 10 and α = 10, 000. The intuitive
dissimilarity between the sinewave and the 5-periodic patterns, and the intuitive
similarity between the two sines (and the two 5-periodic pattern versions, respec-
tively) is revealed much more clearly than on the basis of simR
i,j.
When interpreting similarities simR
i,j or simα
i,j one should bear in mind that one
is not comparing the original driving patterns but the excited reservoir responses.
3.6
Online Learning of Conceptor Matrices
The minimization criterion (6) immediately leads to a stochastic gradient online
method for adapting C:
Proposition 2 Assume that a stationary source x(n) of N-dimensional reservoir
states is available. Let C(1) be any N × N matrix, and λ > 0 a learning rate.
38
Then the stochastic gradient adaptation
C(n + 1) = C(n) + λ
 (x(n) −C(n) x(n)) x′(n) −α−2 C(n)

(11)
will lead to limλ↓0 limn→∞C(n) = C(E[x x′], α).
The proof is straightforward if one employs generally known facts about stochastic
gradient descent and the fact that E[∥x −Cx∥2] + α−2 ∥C∥2
fro is positive deﬁnite
quadratic in the N 2-dimensional space of elements of C (shown in the proof of
Proposition 1), and hence provides a Lyapunov function for the gradient descent
(11). The gradient of E[∥x −Cx∥2] + α−2 ∥C∥2
fro with respect to C is
∂
∂C E[∥x −Cx∥2] + α−2 ∥C∥2
fro = (I −C) E[xx′] −α−2 C,
(12)
which immediately yields (11).
The stochastic update rule (11) is very elementary. It is driven by two com-
ponents, (i) an error signal x(n) −C(n) x(n) which simply compares the current
state with its C-mapped value, and (ii) a linear decay term. We will make heavy
use of this adaptive mechanism in Sections 3.13.1 ﬀ. This observation is also illu-
minating the intuitions behind the deﬁnition of conceptors. The two components
strike a compromise (balanced by α) between (i) the objective that C should leave
reservoir states from the target pattern unchanged, and (ii) C should have small
weights. In the terminology of machine learning one could say, “a conceptor is a
regularized identity map”.
3.7
Morphing Patterns
Conceptor matrices oﬀer a way to morph RNN dynamics. Suppose that a reser-
voir has been loaded with some patterns, among which there are pi and pj with
corresponding conceptors Ci, Cj. Patterns that are intermediate between pi and
pj can be obtained by running the reservoir via (9), using a linear mixture between
Ci and Cj:
x(n + 1) =
 (1 −µ)Ci + µCj
tanh(W x(n) + b).
(13)
Still using our four-pattern example, I demonstrate how this morphing works out
for morphing (i) between the two sines, (ii) between the two 5-periodic patterns,
(iii) between a sine and a 5-periodic pattern.
Frequency Morphing of Sines
In this demonstration, the morphing was done
for the two sinewave conceptors C1 = C(R1, 10) and C2 = C(R2, 10). The morph-
ing parameter µ was allowed to range from −2 to +3 (!). The four-pattern-loaded
reservoir was run from a random initial state for 500 washout steps, using (13)
with µ = −2. Then recording was started. First, the run was continued with the
intial µ = −2 for 50 steps. Then, µ was linearly ramped up from µ = −2 to µ = 3
during 200 steps. Finally, another 50 steps were run with the ﬁnal setting µ = 3.
39
Note that morph values µ = 0 and µ = 1 correspond to situations where the
reservoir is constrained by the original conceptors C1 and C2, respectively. Values
0 ≤µ ≤1 correspond to interpolation. Values −2 ≤µ < 0 and 1 < µ ≤3
correspond to extrapolation. The extrapolation range on either side is twice as
long as the interpolation range.
In addition, for eight equidistant values µk in −2 ≤µ < 3, the reservoir was
run with a mixed conceptor C = (1−µk)C1+µkC2 for 500 steps, and the obtained
observation signal y(n) was plotted in a delay-embedded representation, yielding
“snapshots” of the reservoir dynamics at these µ values (a delay-embedding plot
of a 1-dimensional signal y(n) creates a 2-dimensional plot by plotting value pairs
(y(n), y(n −d)) with a delay d chosen to yield an appealing visual appearance).
!
"!
#!!
#"!
$!!
$"!
%!!
!#
!
#
"
#!
#"
Figure 15: Morphing between (and beyond) two sines. The morphing range was
−2 ≤µ ≤3.
Black circular dots in the two bottom panels mark the points
µ = 0 and µ = 1, corresponding to situations where the two original conceptors
C1, C2 were active in unadulterated form. Top: Delay-embedding plots of network
observation signal y(n) (delay = 1 step). Thick points show 25 plotted points,
thin points show 500 points (appearing as connected line). The eight panels have
a plot range of [−1.4, 1.4] × [−1.4, 1.4]. Triangles in center panel mark the morph
positions corresponding to the delay embedding “snapshots”. Center: the network
observation signal y(n) of a morph run. Bottom: Thin black line: the period length
obtained from morphing between (and extrapolating beyond) the original period
lengths. Bold gray line: period lengths measured from the observation signal y(n).
Figure 15 shows the ﬁndings.
The reservoir oscillates over the entire in-
ter/extrapolation range with a waveform that is approximately equal to a sampled
sine. At the morph values µ = 0 and µ = 1 (indicated by dots in the Figure), the
system is in exactly the same modes as they were plotted earlier in the ﬁrst two
panels of the left column in Figure 12. Accordingly the ﬁt between the original
40
driver’s period lenghtes and the autonomously re-played oscillations is as good as
it was reported there (i.e. corresponding to a steady-state NRMSE of about 0.01).
In the extrapolation range, while the linear morphing of the mixing parameter µ
does not lead to an exact linear morphing of the observed period lengths, still the
obtained period lengths steadily continue to decrease (going left from µ = 0) and
to increase (going right from µ = 1).
In sum, it is possible to use conceptor-morphing to extend sine-oscillatory
reservoir dynamics from two learnt oscillations of periods ≈8.83, 9.83 to a range
between ≈7.5 −11.9 (minimal and maximal values of period lengths shown in
the Figure). The post-training sinewave generation thus extrapolated beyond the
period range spanned by the two training samples by a factor of about 4.4. From
a perspective of machine learning this extrapolation is remarkable.
Generally
speaking, when neural pattern generators are trained from demonstration data
(often done in robotics, e.g. [52, 89]), interpolation of recallable patterns is what
one expects to achieve, while extrapolation is deemed hard.
From a perspective of neurodynamics, it is furthermore remarkable that the
dimension of interpolation/extrapolation was the speed of the oscillation. Among
the inﬁnity of potential generalization dimensions of patterns, speedup/slowdown
of pattern generation has a singular role and is particularly diﬃcult to achieve.
The reason is that speed cannot be modulated by postprocessing of some under-
lying generator’s output – the prime generator itself must be modulated [108].
Frequency adaptation of neural oscillators is an important theme in research on
biological pattern generators (CPGs) (reviews: [40, 50]). Frequency adaptation
has been modeled in a number of ways, among which (i) to use a highly abstracted
CPG model in the form of an ODE, and regulate speed by changing the ODE’s
time constant; (ii) to use a CPG model which includes a pacemaker neuron whose
pace is adaptive; (iii) to use complex, biologically quite detailed, modular neu-
ral architectures in which frequency adapatation arises from interactions between
modules, sensor-motoric feedback cycles, and tonic top-down input.
However,
the fact that humans can execute essentially arbitrary motor patterns at diﬀerent
speeds is not explained by these models. Presumably this requires a generic speed
control mechanism which takes eﬀect already at higher (cortical, planning) layers
in the motor control hierarchy. Conceptor-controlled frequency adaptation might
be of interest as a candidate mechanism for such a “cognitive-level” generic speed
control mechanism.
Shape Morphing of an Integer-Periodic Pattern
In this demonstration,
the conceptors C(R3, 1000) and C(R4, 1000) from the 5-periodic patterns p3 and
p4 were morphed, again with −2 ≤µ ≤3. Figure 16 depicts the network observer
y(n) for a morph run of 95 steps which was started with µ = −2 and ended with
µ = 3, with a linear µ ramping in between. It can be seen that the diﬀerences
between the two reference patterns (located at the points marked by dots) become
increasingly magniﬁed in both extrapolation segments. At each of the diﬀerent
41
points in each 5-cycle, the “sweep” induced by the morphing is however neither
linear nor of the same type across all 5 points of the period (right panel). A simple
algebraic rule that would describe the geometric characteristics of such morphings
cannot be given. I would like to say, it is “up to the discretion of the network’s
nonlinear dynamics” how the morphing command is interpreted; this is especially
true for the extrapolation range. If reservoirs with a diﬀerent initial random W ∗
are used, diﬀerent morphing geometries arise, especially at the far ends of the
extrapolation range (not shown).
!"
#"
$"
%"
&"
'"
("
)"
*"
!!
"
!
Figure 16: Morphing between, and extrapolating beyond, two versions of a 5-
periodic random pattern. The morphing range was −2 ≤µ ≤3. Bottom: Network
observation from a morphing run.
Dots mark the points µ = 0 and µ = 1,
corresponding to situations where the two original conceptors C1, C2 were active
in unadulterated form. The network observation signal y(n) is shown. Top: Delay-
embedding “snapshots”. Figure layout similar to Figure 15.
The snapshots displayed in Figure 16 reveal that the morphing sweep takes
the reservoir through two bifurcations (apparent in the transition from snapshot 2
to 3, and from 7 to 8). In the intermediate morphing range (snapshots 3 – 7), we
observe a discrete periodic attractor of 5 points. In the ranges beyond, on both
sides the attracting set becomes topologically homomorphic to a continuous cycle.
From a visual inspection, it appears that these bifurcations “smoothly” preserve
some geometrical characteristics of the observed signal y(n).
A mathematical
characterisation of this phenomenological continuity across bifurcations remains
for future investigations.
Heterogeneous Pattern Morphing
Figure 17 shows a morph from the 5-
periodic pattern p3 to the irrational-periodic sine p2 (period length ≈9.83). This
time the morphing range was 0 ≤µ ≤1, (no extrapolation). The Figure shows
a run with an initial 25 steps of µ = 0, followed by a 50-step ramp to µ = 1 and
a tail of 25 steps at the same µ level. One observes a gradual change of signal
shape and period along the morph.
From a dynamical systems point of view
this gradual change is unexpected. The reservoir is, mathematically speaking, an
42
autonomous system under the inﬂuence of a slowly changing control parameter µ.
On both ends of the morph, the system is in an attractor. The topological nature
of the attractors (seen as subsets of state space) is diﬀerent (5 isolated points vs. a
homolog of a 1-dim circle), so there must be a at least one bifurcation taking place
along the morphing route. Such a bifurcations would usually be accompanied by
a sudden change of some qualitative characteristic of the system trajectory. We
ﬁnd however no trace of a dynamic rupture, at least not by visual inspection of
the output trajectory. Again, a more in-depth formal characterization of what
geometric properties are “smoothly” carried through these bifurcations is left for
future work.
Figure 2 in Section 1 is a compound demonstration of the three types of pattern
morphing that I here discussed individually.
A possible application for the pattern morphing by conceptors is to eﬀect
smooth gait changes in walking robots, a problem that is receiving some attention
in that ﬁeld.
!
"!
#!
$!
%!
&!
'!
(!
)!
*!
"!!
!"
!
"
&
"!
Figure 17: Morphing from a 5-periodic random pattern to an irrational-periodic
sine. The morphing range was 0 ≤µ ≤1. Figure layout otherwise is as in Figure
15.
3.8
Understanding Aperture
3.8.1
The Semantics of α as “Aperture”
Here I show how the parameter α can be interpreted as a scaling of signal energy,
and motivate why I call it “aperture”.
We can rewrite C(R, α) = C(E[xx′], α) as follows:
43
C(E[xx′], α)
=
E[xx′](E[xx′] + α−2I)−1 = E[(αx)(αx)′](E[(αx)(αx)′] + I)−1
=
C(E[(αx)(αx)′], 1) = C(α2 E[xx′], 1).
(14)
Thus, changing from C(R, 1) to C(R, α) can be interpreted as scaling the
reservoir data by a factor of α, or expressed in another way, as scaling the signal
energy of the reservoir signals by a factor of α2. This is directly analog to what
adjusting the aperture eﬀects in an optical camera. In optics, the term aperture
denotes the diameter of the eﬀective lens opening, and the amount of light energy
that reaches the ﬁlm is proportional to the squared aperture. This has motivated
the naming of the parameter α as aperture.
3.8.2
Aperture Adaptation and Final Deﬁnition of Conceptor Matrices
It is easy to verify that if Cα = C(R, α) and Cβ = C(R, β) are two versions of a
conceptor C diﬀering in their apertures 0 < α, β < ∞, they are related to each
other by
Cβ = Cα
 
Cα +
α
β
2
(I −Cα)
!−1
,
(15)
where we note that Cα +(α/β)2(I −Cα) is always invertible. Cβ is thus a function
of Cα and the ratio γ = β/α. This motivates to introduce an aperture adaptation
operation ϕ on conceptors C, as follows:
ϕ(C, γ) = C (C + γ−2(I −C))−1,
(16)
where ϕ(C, γ) is the conceptor version obtained from C by adjusting the aperture
of C by a factor of γ. Speciﬁcally, it holds that C(R, α) = ϕ(C(R, 1), α).
We introduce the notation RC = C(I −C)−1, which leads to the following
easily veriﬁed data-based version of (16):
Rϕ(C,γ) = γ2 RC.
(17)
When we treat Boolean operations further below, it will turn out that the NOT
operation will ﬂip zero singular values of C to unit singular values. Because of this
circumstance, we admit unit singular values in conceptors and formally deﬁne
Deﬁnition 2 A conceptor matrix is a positive semideﬁnite matrix whose singular
values range in [0, 1]. We denote the set of all N × N conceptor matrices by CN.
Note that Deﬁnition 1 deﬁned the concept of a conceptor matrix associated
with a state correlation matrix R and an aperture α, while Deﬁnition 2 speciﬁes
the more general class of conceptor matrices. Mathematically, conceptor matrices
(as in Deﬁnition 2) are more general than the conceptor matrices associated with
a state correlation matrix R, in that the former may contain unit singular values.
44
Furthermore, in the context of Boolean operations it will also become natural to
admit aperture adaptations of sizes γ = 0 and γ = ∞. The inversion in Equation
(16) is not in general well-deﬁned for such γ and/or conceptors with unit singular
values, but we can generalize those relationships to the more general versions of
conceptors and aperture adaptations by a limit construction:
Deﬁnition 3 Let C be a conceptor and γ ∈[0, ∞]. Then
ϕ(C, γ) =



C (C + γ−2(I −C))−1
for 0 < γ < ∞
limδ↓0 C (C + δ−2(I −C))−1
for γ = 0
limδ↑∞C (C + δ−2(I −C))−1
for γ = ∞
(18)
It is a mechanical exercise to show that the limits in (18) exist, and to cal-
culate the singular values for ϕ(C, γ). The results are collected in the following
proposition.
Proposition 3 Let C = USU ′ be a conceptor and (s1, . . . , sN)′ =
diagS the
vector of its singular values.
Let γ ∈[0, ∞].
Then ϕ(C, γ) = USγU ′ is the
conceptor with singular values (sγ,1, . . . , sγ,N)′, where
sγ,i =











si/(si + γ−2(1 −si))
for
0 < si < 1, 0 < γ < ∞
0
for
0 < si < 1, γ = 0
1
for
0 < si < 1, γ = ∞
0
for
si = 0, 0 ≤γ ≤∞
1
for
si = 1, 0 ≤γ ≤∞
(19)
Since aperture adaptation of C = USU ′ only changes the singular values of C,
the following fact is obvious:
Proposition 4 If V is orthonormal, then ϕ(V CV ′, γ) = V ϕ(C, γ) V ′.
Iterated application of aperture adaptation corresponds to multiplying the
adaptation factors:
Proposition 5 Let C be a conceptor and γ, β ∈[0, ∞]. Then ϕ(ϕ(C, γ), β) =
ϕ(C, γβ).
The proof is a straightforward algebraic veriﬁcation using (19).
Borrowing again terminology from photography, I call a conceptor with SVD
C = USU ′ hard if all singular values in S are 0 or 1 (in photography, a ﬁlm
with an extremely “hard” gradation yields pure black-white images with no gray
tones.) Note that C is hard if and only if it is a projector matrix. If C is hard,
the following holds:
C
=
C† = C′ = CC,
(20)
ϕ(C, γ)
=
C
for γ ∈[0, ∞].
(21)
45
The ﬁrst claim amounts to stating that C is a projection operator, which is obvi-
ously the case; the second claim follows directly from (19).
Besides the aperture, another illuminating characteristic of a conceptor ma-
trix is the mean value of its singular values, i.e. its normalized trace q(C) =
trace(C)/N. It ranges in [0, 1]. Intuitively, this quantity measures the fraction of
dimensions from the N-dimensional reservoir state space that is claimed by C. I
will call it the quota of C.
3.8.3
Aperture Adaptation: Example
In applications one will often need to adapt the aperture to optimize the quality of
C. What “quality” means depends on the task at hand. I present an illustrative
example, where the reservoir is loaded with very fragile patterns. Retrieving them
requires a prudent choice of α. Speciﬁcally, I loaded a reservoir of size N = 500
with four chaotic patterns, derived from the well-known R¨ossler, Lorenz, Mackey-
Glass, and H´enon attractors (details of this example are given in Section 4.2). Four
conceptors CR, CL, CMG, CH were computed, one for each attractor, using α = 1.
Then, in four retrieval experiments, the aperture of each of these was adapted using
(16) in a geometric succession of ﬁve diﬀerent γ, yielding ﬁve versions of each of
the CR, CL, CMG, CH. Each of these was used in turn for a constrained run of the
reservoir according to the state update rule x(n+1) = C tanh(W x(n)+W in p(n+
1) + b), and the resulting output observation was plotted in a delay-embedding
format.
Figure 18 displays the ﬁndings. Per each attractor, the ﬁve apertures were
hand-selected such that the middle one (the third) best re-generated the original
chaotic signal, while the ﬁrst failed to recover the original. One should mention
that it is not trivial in the ﬁrst place to train an RNN to stably generate any single
chaotic attractor timeseries, but here we require the loaded network to be able to
generate any one out of four such signals, only by constraining the reservoir by a
conceptor with a suitably adapted aperture. Number insets in the panels of ﬁgure
18 indicate the apertures and quotas used per run.
3.8.4
Guides for Aperture Adjustment
The four chaotic attractors considered in the previous subsection were “best” (ac-
cording to visual inspection) reconstructed with apertures between 630 and 1000.
A well-chosen aperture is clearly important for working with conceptors. In all
demonstrations reported so far I chose a “good” aperture based on experimenta-
tion and human judgement. In practice one will often need automated criteria for
optimizing the aperture which do not rely on human inspection. In this subsection
I propose two measures which can serve as such a guiding criterion.
A criterion based on reservoir-conceptor interaction. Introducing an
interim state variable z(n) by splitting the conceptor-constrained reservoir update
46
A
11
0.053
66
0.12
4e+02
0.23
2.4e+03
0.38
1.4e+04
0.58
B
20
0.033
1.4e+02
0.07
1e+03
0.13
7e+03
0.21
4.9e+04
0.33
C
26
0.083
1.8e+02
0.18
1.3e+03
0.34
8.8e+03
0.56
6.2e+04
0.82
D
13
0.031
90
0.066
6.3e+02
0.13
4.4e+03
0.24
3.1e+04
0.42
Figure 18: Invoking conceptors to retrieve four chaotic signals from a reservoir.
A Lorenz, B R¨ossler, C Mackey-Glass, and D H´enon attractor.
All four are
represented by delay-embedding plots of the reservoir observation signal y(n).
The plot range is [0, 1] × [0, 1] in every panel. A – C are attractors derived from
diﬀerential equations, hence subsequent points are joined with lines; D derives
from an iterated map where joining lines has no meaning. Each 6-panel block
shows ﬁve patterns generated by the reservoir under the control of diﬀerently
aperture-adapted versions of a conceptor (blue, ﬁrst ﬁve panels) and a plot of the
original chaotic reference signal (green, last panel). Empty panels indicate that
the y(n) signal was outside the [0, 1] range. The right upper panel in each block
shows a version which, judged by visual inspection, comes satisfactorily close to
the original. First number given in a panel: aperture α; second number: quota
q(C).
equation (9) into
z(n + 1) = tanh(W x(n) + b),
x(n + 1) = C(R, α) z(n + 1),
(22)
I deﬁne the attenuation measurable a as
aC,α = E[∥z(n) −x(n)∥2]/E[∥z(n)∥2],
(23)
where the states x(n), z(n) are understood to result from a reservoir constrained
by C(R, α). The attenuation is the fraction of the reservoir signal energy which
is suppressed by applying the conceptor. Another useful way to conceive of this
47
quantity is to view it as noise-to-signal ratio, where the “noise” is the compo-
nent z(n) −x(n) which is ﬁltered away from the unconstrained reservoir signal
z(n). It turns out in simulation experiments that when the aperture is varied, the
attenuation aC,α passes through a minimum, and at this minimum, the pattern
reconstruction performance peaks.
In Figure 19A the log10 of aC,α is plotted for a sweep through a range of
apertures α, for each of the four chaotic attractor conceptors (details in Section
4.2). As α grows, the attenuation aC,α ﬁrst declines roughly linearly in the log-log
plots, that is, by a power law of the form aC,α ∼α−K. Then it enters or passes
through a trough. The aperture values that yielded visually optimal reproductions
of the chaotic patterns coincide with the point where the bottom of the trough is
reached.
A
1
2
3
4
5
ï8
ï6
ï4
ï2
0
Lorenz
1
2
3
4
5
ï8
ï6
ï4
ï2
0
Roessler
1
2
3
4
5
ï8
ï6
ï4
ï2
0
MackeyïGlass
log10 aperture
log10 attenuation
1
2
3
4
5
ï8
ï6
ï4
ï2
0
Hénon
B
!10
!5
0!2
0
2
4
6
!3
!2
!1
0
1
!10
!5
0!2
0
2
4
6
!4
!2
0
!10
!5
0
log10 aperture
log10 attenuation
!2
0
2
4
6
!2
!1
0
1
!10
!5
0!2
0
2
4
6
!2
!1
0
1
log10 NRMSE
Figure 19: Using attenuation to locate optimal apertures.
A Dependancy of
attenuation on aperture for the four chaotic attractors. The blue dots mark the
apertures used to generate the plots in Figure 18. B Dependancy of attenuation on
aperture for the two sinewaves (top panels) and the two 5-point periodic patterns
(bottom) used in Sections 3.2ﬀ. These plots also provide the NRMSEs for the
accuracy of the reconstructed patterns (gray). For explanation see text.
Figure 19B gives similar plots for the two irrational-period sines and the two 5-
point periodic patterns treated in earlier sections. The same reservoir and storing
procedures as described at that place were utilized again here. The dependence
of attenuation on aperture is qualitatively the same as in the chaotic attractor
example. The attenuation plots are overlaid with the NRMSEs of the original
drivers vs. the conceptor-constrained reservoir readout signals. Again, the “best”
aperture – here quantiﬁed by the NRMSE – coincides remarkably well with the
trough minimum of the attenuation.
Some peculiarities visible in the plots B deserve a short comment. (i) The ini-
tial constant plateaus in all four plots result from C(R, α) ≈0 for the very small
apertures in this region, which leads to x(n) ≈0, z(n) ≈tanh(b). (ii) The jittery
climb of the attenuation towards the end of the plotting range in the two bottom
panels is an artefact due to roundoﬀerrors in SVD computations which blows up
singular values in conceptors which in theory should be zero. Without rounding
48
error involved, the attenuation plots would remain at their bottom value once it
is reached. (iii) In the top two panels, some time after having passed through the
trough the attenuation value starts to decrease again. This is due to the fact that
for the irrational-period sinewave signals, all singular values of the conceptors are
nonzero. As a consequence, for increasingly large apertures the conceptors will
converge to the identity matrix, which would have zero attenuation.
A criterion based on conceptor matrix properties. A very simple cri-
terion for aperture-related “goodness” of a conceptor can be obtained from moni-
toring the gradient of the squared Frobenius norm
∇(γ) =
d
d log(γ) ∥φ(C, γ)∥2
(24)
with respect to the logarithm of γ. To get an intuition about the semantics of this
criterion, assume that C has been obtained from data with a correlation matrix
R with SVD R = UΣU ′.
Then φ(C, γ) = R(R + γ−2I)−1 and ∥φ(C, γ)∥2 =
∥Σ(Σ + γ−2I)−1∥2 = ∥γ2Σ(γ2Σ + I)−1∥2. That is, φ(C, γ) can be seen as obtained
from data scaled by a factor of γ compared to φ(C, 1) = C. The criterion ∇(γ)
therefore measures the sensitivity of (the squared norm of) C on (expontential)
scalings of data. Using again photography as a metaphor: if the aperture of a
lens is set to the value where ∇(γ) is maximal, the sensitivity of the image (=
conceptor) to changes in brightness (= data scaling) is maximal.
Figure 20 shows the behavior of this criterion again for the standard example of
loading two irrational sines and two integer-periodic random patterns. Its maxima
coincide largely with the minima of the attenuation criterion, and both with what
was “best” performance of the respective pattern generator. The exception is the
two integer-periodic patterns (Figure 20 B bottom panels) where the ∇criterion
would suggest a slightly too small aperture.
Comments on criteria for guiding aperture selection:
• The two presented criteria based on attenuation and norm gradient are
purely heuristic. A theoretical analysis would require a rigorous deﬁnition
of “goodness”. Since tasks vary in their objectives, such an analysis would
have to be carried out on a case-by-case basis for varying “goodness” charac-
terizations. Other formal criteria besides the two presented here can easily
be construed (I experimented with dozens of alternatives (not documented),
some of which performed as well as the two instances reported here). Alto-
gether this appears to be a wide ﬁeld for experimentation.
• The attenuation-based criterion needs trial runs with the reservoir to be
calculated, while the norm-gradient criterion can be computed oﬄine. The
former seems to be particularly suited for pattern-generation tasks where
the conceptor-reservoir feedback loop is critical (for instance, with respect
to stability). The latter may be more appropriate in machine learning tasks
49
0
2
4
6!2
0
2
4
6
!3
!2
!1
0
1
0
2
4
6!2
0
2
4
6
!4
!2
0
!0.5
0
0.5
1
1.5
log10 aperture
norm2 gradient
!2
0
2
4
6
!2
!1
0
1
!0.5
0
0.5
1
1.5!2
0
2
4
6
!2
!1
0
1
log10 NRMSE
Figure 20: The norm-gradient based criterion to determine “good” apertures for
the basic demo example from Sections 3.2 and 3.4. Plots show ∇(γ) against the
log1 0 of aperture γ. Figure layout similar as in Figure 19 B. For explanation see
text.
where conceptors are used for classifying reservoir dynamics in a “passive”
way without coupling the conceptors into the network updates. I will give
an example in Section 3.12.
3.9
Boolean Operations on Conceptors
3.9.1
Motivation
Conceptor matrices can be submitted to operations that can be meaningfully called
AND, OR, and NOT. There are two justiﬁcations for using these classical logical
terms:
Syntactical / algebraic: Many algebraic laws governing Boolean algebras are
preserved; for hard conceptor matrices the preservation is exact.
Semantical: These operations on conceptor matrices correspond dually to oper-
ations on the data that give rise to the conceptors via (7). Speciﬁcally, the
OR operation can be semantically interpreted on the data level by merging
two datasets, and the NOT operation by inverting the principal component
weights of a dataset. The AND operation can be interpreted on the data
level by combining de Morgan’s rule (which states that x ∧y = ¬(¬x ∨¬y))
with the semantic interpretations of OR and NOT.
The mathematical structures over conceptors that arise from the Boolean op-
erations are richer than standard Boolean logic, in that aperture adaptation oper-
ations can be included in the picture. One obtains a formal framework which one
might call “adaptive Boolean logic”.
50
There are two major ways how such a theory of conceptor logic may be useful:
A logic for information processing in RNNs (cognitive and neuroscience):
The dynamics of any N-dimensional RNN (of any kind, autonomously ac-
tive or driven by external input), when monitored for some time period L,
yields an N ×L sized state collection matrix X and its corresponding N ×N
correlation matrix R, from which a conceptor matrix C = R(R + I)−1 can
be obtained which is a “ﬁngerprint” of the activity of the network for this
period. The Boolean theory of conceptors can be employed to analyse the
relationships between such “activity ﬁngerprints” obtained at diﬀerent in-
tervals, diﬀerent durations, or from diﬀerent driving input. An interesting
long-term research goal for cognitive neuroscience would be to map the log-
ical structuring described on the network data level, to Boolean operations
carried out by task-performing subjects.
An algorithmical tool for RNN control (machine learning): By controlling
the ongoing activity of an RNN in a task through conceptors which are de-
rived from logical operations, one can implement “logic control” strategies for
RNNs. Examples will be given in Section 3.11, where Boolean operations
on conceptors will be key for an eﬃcient memory management in RNNs;
in Section 3.12, where Boolean operations will enable to combine positive
and negative evidences for ﬁnite-duration pattern recognition; and in Sec-
tion 3.15, where Boolean operations will help to simultaneously de-noise and
classify signals.
3.9.2
Preliminary Deﬁnition of Boolean Operations
Deﬁning Boolean operators through their data semantics is transparent and simple
when the concerned data correlation matrices are nonsingular. In this case, the re-
sulting conceptor matrices are nonsingular too and have singular values ranging in
the open interval (0, 1). I treat this situation in this subsection. However, concep-
tor matrices with a singular value range of [0, 1] frequently arise in practice. This
leads to technical complications which will be treated in the next subsection. The
deﬁnitions given in the present subsection are preliminary and serve expository
purposes.
In the remainder of this subsection, conceptor matrices C, B are assumed to
derive from nonsingular correlation matrices.
I begin with OR. Recall that a conceptor matrix C (with aperture 1) derives
from a data source (network states) x through R = E[xx′], C = C(R, 1) = R(R +
I)−1. Now consider a second conceptor B of the same dimension N as C, derived
from another data source y by Q = E[yy′], B = B(Q, 1) = Q(Q + I)−1. I deﬁne
C ∨B := (R + Q)(R + Q + I)−1,
(25)
51
and name this the OR operation. Observe that R + Q = E[[x, y][x, y]′], where
[x, y] is the N × 2 matrix made of vectors x, y. C ∨B is thus obtained by a merge
of the two data sources which previously went into C and B, respectively. This
provides a semantic interpretation of the OR operation.
Using (7), it is straightforward to verify that C ∨B can be directly computed
from C and B by
C ∨B =

I +
 C(I −C)−1 + B(I −B)−1−1−1
,
(26)
where the assumption of nonsingular R, Q warrants that all inverses in this equa-
tion are well-deﬁned.
I now turn to the NOT operation.
For C = C(R, 1) = R(R + I)−1 with
nonsingular R I deﬁne it by
¬C := R−1(R−1 + I)−1.
(27)
Again this can be semantically interpreted on the data level. Consider the
SVDs R = UΣU ′ and R−1 = UΣ−1U ′.
R and R−1 have the same principal
components U, but the variances Σ, Σ−1 of data that would give rise to R and
R−1 are inverse to each other. In informal terms, ¬C can be seen as arising from
data which co-vary inversely compared to data giving rise to C.
Like in the case of OR, the negation of C can be computed directly from C.
It is easy to see that
¬C = I −C.
(28)
Finally, I consider AND. Again, we introduce it on the data level. Let again
C = R(R + I)−1, B = Q(Q + I)−1. The OR operation was introduced as addition
on data correlation matrices, and the NOT operation as inversion. Guided by
de Morgan’s law a ∧b = ¬(¬a ∨¬b) from Boolean logic, we obtain a correlation
matrix (R−1 + Q−1)−1 for C ∧B. Via (7), from this correlation matrix we are led
to
C ∧B := (R−1 + Q−1)−1  (R−1 + Q−1)−1 + I
−1 .
(29)
Re-expressing R, Q in terms of C, B in this equation, elementary transforma-
tions (using (7)) again allow us to compute AND directly:
C ∧B = (C−1 + B−1 −I)−1.
(30)
By a routine transformation of equations, it can be veriﬁed that the de Mor-
gan’s laws C ∨B = ¬(¬C ∧¬B) and C ∧B = ¬(¬C ∨¬B) hold for the direct
computation expressions (26), (28) and (30).
52
3.9.3
Final Deﬁnition of Boolean Operations
We notice that the direct computations (26) and (30) for OR and AND are only
well-deﬁned for conceptor matrices whose singular values range in (0, 1). I now
generalize the deﬁnitions for AND and OR to cases where the concerned conceptors
may contain singular values 0 or 1. Since the direct computation (30) of AND is
simpler than the direct computation (26) of OR, I carry out the generalization for
AND and then transfer it to OR through de Morgan’s rule.
Assume that C = USU ′, B = V TV ′ are the SVDs of conceptors C, B, where S
and/or T may contain zero singular values. The direct computation (30) is then
not well-deﬁned.
Speciﬁcally, assume that diag(S) contains l ≤N nonzero singular values and
that diag(T) contains m ≤N nonzero singular values, i.e. diag(S) = (s1, . . . , sl, 0, . . . , 0)′
and diag(T) = (t1, . . . , tm, 0, . . . , 0)′. Let δ be a positive real number. Deﬁne Sδ
to be the diagonal matrix which has a diagonal (s1, . . . , sl, δ, . . . , δ)′, and similarly
Tδ to have diagonal (t1, . . . , tm, δ, . . . , δ)′. Put Cδ = USδU ′, Bδ = V TδV ′. Then
Cδ ∧Bδ = (C−1
δ
+ B−1
δ
−I)−1 is well-deﬁned. We now deﬁne
C ∧B = lim
δ→0(C−1
δ
+ B−1
δ
−I)−1.
(31)
The limit in this equation is well-deﬁned and can be resolved into an eﬃcient
algebraic computation:
Proposition 6 Let BR(C)∩R(B) be a matrix whose columns form an arbitrary
orthonormal basis of R(C) ∩R(B).
Then, the matrix B′
R(C)∩R(B)(C† + B† −
I)BR(C)∩R(B) is invertible, and the limit (31) exists and is equal to
C ∧B = lim
δ→0(C−1
δ
+ B−1
δ
−I)−1 =
=
BR(C)∩R(B)
 B′
R(C)∩R(B) (C† + B† −I) BR(C)∩R(B)
−1 B′
R(C)∩R(B). (32)
Equivalently, let PR(C)∩R(B) = BR(C)∩R(B) B′
R(C)∩R(B) be the projector matrix on
R(C) ∩R(B). Then C ∧B can also be written as
C ∧B =
 PR(C)∩R(B) (C† + B† −I) PR(C)∩R(B)
† .
(33)
The proof and an algorithm to compute a basis matrix BR(C)∩R(B) are given
in Section 5.2.
The formulas (32) and (33) not only extend the formula (30) to cases where C
or B are non-invertible, but also ensures numerical stability in cases where C or
B are ill-conditioned. In that situation, the pseudoinverses appearing in (32), (33)
should be computed with appropriate settings of the numerical tolerance which
one can specify in common implementations (for instance in Matlab) of the SVD.
One should generally favor (32) over (30) unless one can be sure that C and B are
well-conditioned.
53
The direct computation (28) of NOT is well-deﬁned for C with a singular value
range [0, 1], thus nothing remains to be done here.
Having available the general and numerically robust computations of AND via
(32) or (33) and of NOT via (28), we invoke de Morgan’s rule C∨B = ¬(¬C∧¬B)
to obtain a general and robust computation for OR on the basis of (32) resp. (33)
and (28). Summarizing, we obtain the ﬁnal deﬁnitions for Boolean operations on
conceptors:
Deﬁnition 4
¬ C
:=
I −C,
C ∧B
:=
 PR(C)∩R(B) (C† + B† −I) PR(C)∩R(B)
† ,
C ∨B
:=
¬ (¬ C ∧¬ B),
where PR(C)∩R(B) is the projector matrix on R(C) ∩R(B).
This deﬁnition is consistent with the preliminary deﬁnitions given in the pre-
vious subsection. For AND this is clear: if C and B are nonsingular, PR(C)∩R(B)
is the identity and the pseudoinverse is the inverse, hence (30) is recovered (fact
1). We noted in the previous subsection that de Morgan’s laws hold for concep-
tors derived from nonsingular correlation matrices (fact 2). Furthermore, if C is
derived from a nonsingular correlation matrix, then ¬C also corresponds to a non-
singular correlation matrix (fact 3). Combining facts 1 – 3 yields that the way of
deﬁning OR via de Morgan’s rule from AND and NOT in Deﬁnition 4 generalises
(25)/(26).
For later use I state a technical result which gives a characterization of OR in
terms of a limit over correlation matrices:
Proposition 7 For a conceptor matrix C with SVD C = USU ′ let S(δ) be a
version of S where all unit singular values (if any) have been replaced by 1 −δ,
and let C(δ) = US(δ)U ′.
Let R(δ)
C
= C(δ)(I −C(δ))−1.
Similarly, for another
conceptor B let R(δ)
B = B(δ)(I −B(δ))−1. Then
C ∨B = I −lim
δ↓0 (R(δ)
C + R(δ)
B + I)−1 = lim
δ↓0 (R(δ)
C + R(δ)
B ) (R(δ)
C + R(δ)
B + I)−1. (34)
The proof is given in Section 5.3. Finally I note that de Morgan’s rule also
holds for AND (proof in Section 5.4):
Proposition 8
C ∧B = ¬ (¬ C ∨¬ B).
54
3.9.4
Facts Concerning Subspaces
For an N × N matrix M, let I(M) = {x ∈RN | Mx = x} be the identity space
of M. This is the eigenspace of M to the eigenvalue 1, a linear subspace of RN.
The identity spaces, null spaces, and ranges of conceptors are related to Boolean
operations in various ways. The facts collected here are technical, but will be
useful in deriving further results later.
Proposition 9 Let C, B be any conceptor matrices, and H, G hard conceptor ma-
trices of the same dimension. Then the following facts hold:
1. I(C) ⊆R(C).
2. I(C†) = I(C) and R(C†) = R(C) and N(C†) = N(C).
3. R(¬C) = I(C)⊥and I(¬C) = N(C) and N(¬C) = I(C).
4. R(C ∧B) = R(C) ∩R(B) and R(C ∨B) = R(C) + R(B).
5. I(C ∧B) = I(C) ∩I(B) and I(C ∨B) = I(C) + I(B).
6. N(C ∧B) = N(C) + N(B) and N(C ∨B) = N(C) ∩N(B).
7. I(ϕ(C, γ)) = I(C) for γ ∈[0, ∞) and R(ϕ(C, γ)) = R(C) for γ ∈(0, ∞]
and N(ϕ(C, γ)) = N(C) for γ ∈(0, ∞].
8. A = A ∧C ⇐⇒R(A) ⊆I(C) and A = A ∨C ⇐⇒I(A)⊥⊆N(C).
9. ϕ(C, 0) and ϕ(C, ∞) are hard.
10. ϕ(C, 0) = PI(C) and ϕ(C, ∞) = PR(C).
11. H = H† = PI(H).
12. I(H) = R(H) = N(H)⊥.
13. ¬H = PN(H) = PI(H)⊥.
14. H ∧G = PI(H) ∩I(G).
15. H ∨G = PI(H) + I(G).
The proof is given in Section 5.5.
55
3.9.5
Boolean Operators and Aperture Adaptation
The Boolean operations are related to aperture adaptation in a number of ways:
Proposition 10 Let C, B be N × N sized conceptor matrices and γ, β ∈[0, ∞].
We declare ∞−1 = ∞−2 = 0 and 0−1 = 0−2 = ∞. Then,
1. ¬ϕ(C, γ) = ϕ(¬C, γ−1),
2. ϕ(C, γ) ∨ϕ(B, γ) = ϕ(C ∨B, γ),
3. ϕ(C, γ) ∧ϕ(B, γ) = ϕ(C ∧B, γ),
4. ϕ(C, γ) ∨ϕ(C, β) = ϕ(C,
p
γ2 + β2),
5. ϕ(C, γ) ∧ϕ(C, β) = ϕ(C, (γ−2 + β−2)−1/2).
The proof can be found in Section 5.6. Furthermore, with the aid of aperture
adaptation and OR it is possible to implement an incremental model extension, as
follows. Assume that conceptor C has been obtained from a dataset X comprised
of m data vectors x, via R = XX′/m,
C = R(R + α−2I)−1. Then, n new data
vectors y become available, collected as columns in a data matrix Y . One wishes
to update the original conceptor C such that it also incorporates the information
from Y , that is, one wishes to obtain
˜C = ˜R( ˜R + α−2I)−1,
(35)
where ˜R is the updated correlation matrix obtained by Z = [XY ], ˜R = ZZ′/(m +
n). But now furthermore assume that the original training data X are no longer
available. This situation will not be uncommon in applications. The way to a
direct computation of (35) is barred. In this situation, the extended model ˜C can
be computed from C, Y, m, n as follows. Let CY = Y Y ′(Y Y ′ + I)−1. Then,
˜C
=
ϕ
 ϕ(C, m1/2α−1) ∨CY , (m + n)1/2α

(36)
=
I −

m
m + n(I −C)−1C +
n
m + nα2Y Y ′ + I
−1
.
(37)
These formulas can be veriﬁed by elementary transformations using (7), (8),
(25) and (17), noting that C cannot have unit singular values because it is obtained
from a bounded correlation matrix R, thus (I −C) is invertible.
3.9.6
Logic Laws
Many laws from Boolean logic carry over to the operations AND, OR, NOT deﬁned
for conceptors, sometimes with modiﬁcations.
56
Proposition 11 Let I be the N × N identity matrix, 0 the zero matrix, and
B, C, D any conceptor matrices of size N × N (including I or 0). Then the fol-
lowing laws hold:
1. De Morgan’s rules: C ∨B = ¬ (¬ C ∧¬ B) and C ∧B = ¬ (¬ C ∨¬ B).
2. Associativity: (B ∧C) ∧D = B ∧(C ∧D) and (B ∨C) ∨D = B ∨(C ∨D).
3. Commutativity: B ∧C = C ∧B and B ∨C = C ∨B.
4. Double negation: ¬(¬C) = C.
5. Neutrality of 0 and I: C ∨0 = C and C ∧I = C.
6. Globality of 0 and I: C ∨I = I and C ∧0 = 0.
7. Weighted self-absorption for OR: C ∨C = ϕ(C,
√
2) and ϕ(C,
p
1/2) ∨
ϕ(C,
p
1/2) = C.
8. Weighted self-absorption for AND: C ∧C = ϕ(C, 1/
√
2) and ϕ(C,
√
2) ∧
ϕ(C,
√
2) = C.
The proofs are given in Section 5.7. From among the classical laws of Boolean
logic, the general absorption rules A = A ∧(A ∨B) = A ∨(A ∧B) and the laws of
distributivity do not hold in general for conceptors. However, they hold for hard
conceptors, which indeed form a Boolean algebra:
Proposition 12 The set of hard N ×N conceptor matrices, equipped with the op-
erations ∨, ∧, ¬ deﬁned in Deﬁnition 4, is a Boolean algebra with maximal element
IN×N and minimal element 0N×N.
The proof is obvious, exploiting the fact that hard conceptors can be identiﬁed
with (projectors on) their identity subspaces (Proposition 9 11.), and that the
operations ∨, ∧, ¬ correspond to subspace operations +, ∩, ·⊥(Proposition 9 13.
– 15.). It is well known that the linear subspaces of RN form a Boolean algebra
with respect to these subspace operations.
While the absorption rules A = A ∧(A ∨B) = A ∨(A ∧B) are not valid for
conceptor matrices, it is possible to “invert” ∨by ∧and vice versa in a way that
is reminiscent of absorption rules:
Proposition 13 Let A, B be conceptor matrices of size N × N. Then,
1. C =
 PR(A)
 I + A† −(A ∨B)†
PR(A)
† is a conceptor matrix and
A = (A ∨B) ∧C.
(38)
2. C = I −
 PI(A)⊥
 I + (I −A)† −(I −(A ∧B))†
PI(A)⊥
† is a conceptor
matrix and
A = (A ∧B) ∨C.
(39)
The proof is given in Section 5.8.
57
3.10
An Abstraction Relationship between Conceptors
The existence of (almost) Boolean operations between conceptors suggests that
conceptors may be useful as models of concepts (extensive discussion in Section
3.16). In this subsection I add substance to this interpretation by introducing an
abstraction relationship between conceptors, which allows one to organize a set of
conceptors in an abstraction hierarchy.
In order to equip the set CN of N × N conceptors with an “abstraction” rela-
tionship, we need to identify a partial ordering on CN which meets our intuitive
expectations concerning the structure of “abstraction”. A natural candidate is the
partial order ≤deﬁned on the set of N × N real matrices by X ≤Y if Y −X is
positive semideﬁnite. This ordering is often called the L¨owner ordering. I will in-
terpret and employ the L¨owner ordering as an abstraction relation. The key facts
which connect this ordering to Boolean operations, and which justify to interpret
≤as a form of logical abstraction, are collected in the following
Proposition 14 Let CN be the set of conceptor matrices of size N. Then the
following facts hold.
1. An N × N matrix A is a conceptor matrix if and only if 0 ≤A ≤IN×N.
2. 0N×N is the global minimal element and IN×N the global maximal element of
(CN, ≤).
3. A ≤B if and only if ¬A ≥¬B.
4. Let A, B ∈CN and B ≤A. Then
C = PR(B) (B† −PR(B) A† PR(B) + I)−1 PR(B)
is a conceptor matrix and
B = A ∧C.
5. Let again A, B ∈CN and A ≤B. Then
C = I −PI(B)⊥
 (I −B)† −PI(B)⊥(I −A)† PI(B)⊥+ I
−1 PI(B)⊥
is a conceptor matrix and
B = A ∨C.
6. If for A, B, C ∈CN it holds that A ∧C = B, then B ≤A.
7. If for A, B, C ∈CN it holds that A ∨C = B, then A ≤B.
8. For A ∈CN and γ ≥1 it holds that A ≤ϕ(A, γ); for γ ≤1 it holds that
ϕ(A, γ) ≤A.
9. If A ≤B, then ϕ(A, γ) ≤ϕ(B, γ) for γ ∈[0, ∞].
58
The proof is given in Section 5.9. The essence of this proposition can be re-
expressed succinctly as follows:
Proposition 15 For conceptors A, B the following conditions are equivalent:
1. A ≤B.
2. There exists a conceptor C such that A ∨C = B.
3. There exists a conceptor C such that A = B ∧C.
Thus, there is an equivalence between “going upwards” in the ≤ordering on
the one hand, and merging conceptors by OR on the other hand. In standard
logic-based knowledge representation formalisms, a concept (or class) B is deﬁned
to be more abstract than some other concept/class A exactly if there is some
concept/class C such that A ∨C = B. This motivates me to interpret ≤as an
abstraction ordering on CN.
3.11
Example: Memory Management in RNNs
In this subsection I demonstrate the usefulness of Boolean operations by introduc-
ing a memory management scheme for RNNs. I will show how it is possible
1. to store patterns in an RNN incrementally: if patterns p1, . . . , pk have al-
ready been stored, a new pattern pk+1 can be stored in addition without
interfering with the previously stored patterns, and without having to know
them;
2. to maintain a measure of the remaining memory capacity of the RNN which
indicates how many more patterns can still be stored;
3. to exploit redundancies: if the new pattern is similar in a certain sense to
already stored ones, loading it consumes less memory capacity than when
the new pattern is dissimilar to the already stored ones.
Recall that in the original pattern storing procedure, the initial random weight
matrix W ∗is recomputed to obtain the weight matrix W of the loaded reservoir,
such that
xj(n + 1) = tanh(W ∗xj(n) + W in pj(n + 1) + b) ≈tanh(W xj(n) + b),
where pj(n) is the j-th pattern signal and xj(n) is the reservoir state signal ob-
tained when the reservoir is driven by the j-th pattern. For a transparent memory
management, it is more convenient to keep the original W ∗and record the weight
changes into an input simulation matrix D, such that
xj(n + 1) = tanh(W ∗xj(n) + W in pj(n + 1)) ≈tanh(W ∗xj(n) + D xj(n)). (40)
59
In a non-incremental batch training mode, D would be computed by regularized
linear regression to minimize the following squared error:
D = argmin ˜D
X
j=1,...,K
X
n=1,...,L
∥W in pj(n) −˜D xj(n −1)∥2,
(41)
where K is the number of patterns and L is the length of training samples (after
subtracting an initial washout period). Trained in this way, the sum W ∗+ D
would be essentially identical (up to diﬀerences due to using another regularization
scheme) to the weight W matrix obtained in the original storing procedure. In
fact, the performance of loading a reservoir with patterns via an input simulation
matrix D as in (40) is indistinguishable from what is obtained in the original
procedure (not reported).
The present objective is to ﬁnd a way to compute D incrementally, leading to
a sequence Dj (j = 1, . . . , K) of input simulation matrices such that the following
conditions are satisﬁed:
1. When the j-th input simulation matrix Dj is used in conjunction with a
conceptor Ci associated with an already stored pattern pi (i.e., i ≤j), the
autonomous dynamics
x(n + 1) = Ci tanh(W ∗x(n) + Dj x(n) + b)
(42)
re-generates the i-th pattern.
2. In order to compute Dj+1, one must not use explicit knowledge of the already
stored patterns or their conceptors, and one only needs to drive the network
a single time with the new pattern pj+1 in order to obtain the requisite
training data.
3. If two training patterns pi = pj are identical (where i > j), Dj = Dj−1 is
obtained. The network already has stored pi and does not need to change
in order to accomodate to this pattern when it is presented a second time.
We will see that, as a side eﬀect, the third condition also allows the network
to exploit redundancies when pj is similar but not identical to an earlier pi; the
additional memory consumption is reduced in such cases.
The key idea is to keep track of what parts of the reservoir memory space
have already been claimed by already stored patterns, or conversely, which parts
of the memory space are still freely disposable.
Each pattern pj is associated
with a conceptor Cj. The “used-up” memory space after having stored patterns
p1, . . . , pj will be characterized by the disjunction Aj = W{C1, . . . , Cj}, and the
“still disposable” space by its complement ¬Aj.
Let a raw reservoir with an initial weight matrix W ∗be given, as well as
training pattern timeseries pj(n) where j = 1, . . . , K and n = 1, . . . , L. Then the
incremental storing algorithm proceeds as follows.
60
Initialization (no pattern yet stored): D0 = A0 = 0N×N. Choose an aper-
ture α to be used for all patterns.
Incremental storage of patterns: For j = 1, . . . , K do:
1. Drive reservoir with pattern pj for L timesteps using state updates
xj(n + 1) = tanh(W ∗xj(n) + W inpj(n + 1) + b)
and collect the states xj(1), . . . , xj(L−1) into a N ×(L−1) sized state
collection matrix Xj. Put Rj = Xj (Xj)′ / (L−1). Likewise, collect the
patterns pj(2), . . . , pj(L) into a row vector P j of length L −1. (Here,
like elsewhere in this report, I tacitly assume that before one collects
data, the network state has been purged by driving through an initial
washout period).
2. Compute a conceptor for this pattern by Cj = Rj (Rj + α−2 I)−1.
3. Compute an N × N matrix Dj
inc (subsequently to be added as an in-
crement to Dj−1, yielding Dj = Dj−1 + Dj
inc ) by putting
(a) F j−1 = ¬Aj−1 (comment: this conceptor characterizes the “still
disposable” memory space for learning pj),
(b) T = W in P j −Dj−1 Xj (comment: this N × (L −1) sized matrix
contains the targets for a linear regression to compute Dinc),
(c) S = F j−1 Xj (comment: this N ×(L−1) sized matrix contains the
arguments for the linear regression),
(d) Dj
inc =
 (SS′/(L −1) + α−2I)† ST ′/(L −1)
′ (comment: carry out
the regression, regularized by α−2).
4. Update D: Dj = Dj−1 + Dj
inc.
5. Update A: Aj = Aj−1 ∨Cj.
Here is an intuitive explanation of the core ideas in the update step (j −1) →j
in this algorithm:
1. Step 3(a): Aj−1 = W{C1, . . . , Cj−1} keeps track of the subspace directions
that have already been “consumed” by the patterns already stored, and
its complement F j−1 = ¬Aj−1 represents the “still unused” directions of
reservoir state space.
2. Step 3(b): The regression targets for Dj
inc are given by the state contribu-
tions of the driving input (W in P j), minus what the already installed input
simulation matrix already contributes to predicting the input eﬀects from
the previous network state (−Dj−1Xj). Setting the regression target in this
way gives rise to the desired exploitation of redundancies. If pattern pj had
been learnt before (i.e., a copy of it was already presented earlier), T = 0
will result in this step, and hence, Dj
inc = 0.
61
3. Step 3(c): The arguments for the linear regression are conﬁned to the projec-
tion F j−1 Xj of the pj-driven network states Xj on the still unused reservoir
directions F j−1. In this way, Dj
inc is decoupled from the already installed
Dj−1: Dj
inc exploits for generating its output state only such information
that was not used by Dj−1.
4. Step 3(d): This is the well-known Tychonov-regularized Wiener-Hopf for-
mula for computing a linear regression of targets T on arguments S, also
known as “ridge regression” [28, 106]. Setting the Tychonov regularizer to
the inverse squared aperture α−2 is not accidental. It results from the math-
ematically identical roles of Tychonov regularizers and apertures.
The sketched algorithm provides a basic format. It can be easily extended/reﬁned,
for instance by using diﬀerent apertures for diﬀerent patterns, or multidimensional
input.
It is interesting to note that it is intrinsically impossible to unlearn patterns se-
lectively and “decrementally”. Assume that patterns p1, . . . , pK have been trained,
resulting in DK. Assume that one wishes to unlearn again pK. As a result of
this unlearning one would want to obtain DK−1. Thus one would have to com-
pute DK
inc from DK, AK and pK (that is, from DK and CK), in order to recover
DK−1 = DK −DK
inc. However, the way to identify DK
inc from DK, AK and CK is
barred because of the redundancy exploitation inherent in step 3(b). Given only
DK, and not knowing the patterns p1, . . . , pK−1 which must be preserved, there is
no way to identify which directions of reservoir space must be preserved to preserve
those other patterns. The best one can do is to put ˜AK−1 = AK −CK = AK ∧¬CK
and re-run step 3 using ˜AK−1 instead of AK−1 and putting T = W in P j in step
3(b). This leads to a version ˜DK
inc which coincides with the true DK
inc only if there
was no directional overlap between CK and the other Cj, i.e. if DK−1XK = 0 in
the original incremental learning procedure. To the extent that pK shared state
directions with the other patterns, i.e. to the extent that there was redundancy,
unlearning pK will degrade or destroy patterns that share state directions with
pK.
The incremental pattern learning method oﬀers the commodity to measure
how much “memory space” has already been used after the ﬁrst j patterns have
been stored. This quantity is the quota q(Aj). When it approaches 1, the reservoir
is “full” and an attempt to store another pattern will fail because the F matrix
from step 3(a) will be close to zero.
Two demonstrations illustrate various aspects of the incremental storing proce-
dure. Both demonstrations used an N = 100 sized reservoir, and K = 16 patterns
were stored. In the ﬁrst demonstration, the patterns (sines or random patterns)
were periodic with integer period lengths ranging between 3 and 15. In the second
demonstration, the patterns were sinewaves with irrational period lengths chosen
between 4 and 20. Details are documented in Section 4.3. Figures 21 and 22 plot
characteristic impressions from these demonstrations.
62
ï1
0
1
0.0025
j = 1
0.1
0.0076
j = 2
0.25
0.00084
j = 3
0.29
0.0013
j = 4
0.35
ï1
0
1
0.0028
j = 5
0.42
0.0025
j = 6
0.42
0.0073
j = 7
0.42
0.00084
j = 8
0.42
ï1
0
1
0.0082
j = 9
0.54
0.016
j = 10
0.59
0.007
j = 11
0.65
0.0033
j = 12
0.73
1
10
20
ï1
0
1
0.0035
j = 13
0.8
1
10
20
0.0011
j = 14
0.83
1
10
20
0.022
j = 15
0.94
1
10
20
1.3
j = 16
0.98
Figure 21: Incremental storing, ﬁrst demonstration (ﬁgure repeated from Section
1 for convenience). 13 patterns with integer period lengths ranging between 3 and
15 were stored. Patterns were sinewaves with integer periods or random. Patterns
j = 6, 7, 8 are identical to j = 1, 2, 3. Each panel shows a 20-timestep sample of the
correct training pattern pj (black line) overlaid on its reproduction (green line).
The memory fraction used up until pattern j is indicated by the panel fraction
ﬁlled in red; the quota value is printed in the left bottom corner of each panel.
The red areas in each panel in fact show the singular value spectrum of Aj (100
values, x scale not shown). The NRMSE is inserted in the bottom right corners
of the panels. For detail see text.
ï1
0
1
0.1
j = 1
0.11
0.05
j = 2
0.2
0.086
j = 3
0.26
0.04
j = 4
0.29
ï1
0
1
0.064
j = 5
0.33
0.025
j = 6
0.38
0.045
j = 7
0.4
0.047
j = 8
0.41
ï1
0
1
0.3
j = 9
0.46
0.14
j = 10
0.48
0.034
j = 11
0.49
0.13
j = 12
0.5
1
10
20
ï1
0
1
0.051
j = 13
0.52
1
10
20
0.094
j = 14
0.53
1
10
20
0.11
j = 15
0.55
1
10
20
0.032
j = 16
0.56
Figure 22: Incremental storing, second demonstration. 16 sinewave patterns with
irrational periods ranging between 4 and 20 were used. Plot layout is the same as
in Figure 21. For detail see text.
63
Comments on the demonstrations:
Demonstration 1. When a reservoir is driven with a signal that has an integer
period, the reservoir states (after an initial washout time) will entrain to this
period, i.e. every neuron likewise will exhibit an integer-periodic activation
signal. Thus, if the period length of driver j is m, the correlation matrix
Rj as well as the conceptor Cj will be matrices of rank m. An aperture
α = 1000 was used in this demonstration. The large size of this aperture
and the fact that Rj has rank m leads to a conceptor Cj which comes close
to a projector matrix, i.e. it has m singular values that are close to one and
N −m zero singular values. Furthermore, if a new pattern pj+1 is presented,
the periodic reservoir state vectors arising from it will generically be linearly
independent of all state vectors that arose from earlier drivers. Both eﬀects
together (almost projector Cj and linear independence of nonzero principal
directions of these Cj) imply that the sequence A1, . . . , AK will essentially be
a sequence of projectors, where R(Aj+1) will comprise m more dimensions
than R(Aj) (where m is the period length of pj+1). This becomes clearly
apparent in Figure 21: the area under the singular value plot of Aj has
an almost rectangular shape, and the increments from one plot to the next
match the periods of the respective drivers, except for the last pattern, where
the network capacity is almost exhausted.
Patterns j = 6, 7, 8 were identical to j = 1, 2, 3. As a consequence, when the
storage procedure is run for j = 6, 7, 8, Aj remains essentially unchanged –
no further memory space is allocated.
When the network’s capacity is almost exhausted in the sense that the quota
q(Aj) approaches 1, storing another pattern becomes inaccurate.
In this
demo, this happens for that last pattern j = 16 (see Figure 21).
Demonstration 2. When the driver has an irrational period length, the excited
reservoir states will span the available reservoir space RN. Each Rj will have
only nonzero singular values, albeit of rapidly decreasing magnitude (these
tails are so small in magnitude that they are not visible in the ﬁrst few plots
of Aj in Figure 22). The fact that each driving pattern excites the reservoir
in all directions leads to the “reverse sigmoid” kind of shapes of the singular
values of the Aj visible in Figure 22.
The sinewave drivers pj were presented in an order with randomly shuﬄed
period lengths. A redundancy exploitation eﬀect becomes apparent: while
for the ﬁrst 8 patterns altogether a quota q(A8) = 0.41 was allocated, the
next 8 patterns only needed an additional quota of q(A16) −q(A8) = 0.15.
Stated in suggestive terms, at later stages of the storing sequence the net-
work had already learnt how to oscillate in sinewaves in general, and only
needed to learn in addition how to oscillate at the particular newly presented
frequencies. An aperture of size α = 3 was used in the second demonstration.
64
The two demonstrations showed that when n-point periodic patterns are stored,
each new pattern essentially claims a new n-dimensional subspace. In contrast,
each of the irrational-periodic sines aﬀected all of the available reservoir dimen-
sions, albeit to diﬀerent degrees. This leads to problems when one tries to store
n-periodic patterns after irrational-periodic patterns have already been stored.
The latter already occupy all available reservoir state directions, and the new n-
periodic storage candidates cannot ﬁnd completely “free” directions. This leads to
poor retrieval qualities for n-periodic patterns stored on top of irrational-periodic
ones (not shown). The other order – storing irrational-periodic patterns on top of
n-periodic ones – is not problematic. This problem can be mitigated with the aid
of autoconceptors, which are developed in subsequent sections. They typically lead
to almost hard conceptors with a majority of singular values being zero, and thus
– like n-point periodic patterns – only claim low-dimensional subspaces, leaving
unoccupied “dimension space” for loading further patterns.
An architecture variant.
When one numerically computes the rank of the in-
put simulation matrix D obtained after storing all patterns, one will ﬁnd that it
has rank 1. This can be readily explained as follows. The desired functionality
of D is to replace the N-dimensional signals W in pj(n + 1) by D x(n) (in a con-
dition where the network state x is governed by the conceptor Cj). The signal
W in pj(n) has a rank-1 autocorrelation matrix E[W in pj(n)(W in pj(n))′]. There-
fore, also E[Dx(n)(Dx(n))′] should have unit rank. Since the reservoir states x(n)
will span all of RN across the diﬀerent patterns j, a unit rank of E[Dx(n)(Dx(n))′]
implies a unit rank of D. In fact, the columns of D turn out to be scaled versions
of W in, i.e. D = W ind for some N-dimensional row vector d.
Furthermore, from W in d x(n) = Dx(n) ≈W in pj(n + 1) we infer d x(n) ≈
pj(n + 1): projections of network states on d predict the next input.
This suggests an alternative way to design and train pattern-storing RNNs.
The signal d x(n) is assigned to a newly introduced single neuron π. The system
equation for the new design in external driving conditions remains unchanged:
x(n + 1) = tanh(W ∗x(n) + W inpj(n + 1) + b).
However, the autonomous dynamics (42) is replaced by
π(n + 1)
=
d x(n)
(43)
x(n + 1)
=
Cj tanh(W ∗x(n) + W inπ(n + 1) + b),
(44)
(45)
or equivalently
x(n + 1) = Cj tanh(W ∗x(n) + W in d x(n) + b).
(46)
The original readout y(n) = W out x(n) becomes superﬂuous, since π(n) = y(n).
Figure 23 is a diagram of the alternative architecture. Note that d is the only
trainable item in this architecture.
65
€ 
W in
€ 
W *
€ 
π
€ 
x
€ 
d
€ 
p
Figure 23: Alternative architecture for storing patterns. A multi-functional neuron
π serves as input unit in external driving conditions: its value is then forced by
the external driver p. In autonomous runs when p is absent, the value of π is
determined by its readout weights d from the reservoir. Non-trainable connections
are drawn as solid arrows, trainable ones are broken. The bias vector and the
action of conceptors are omitted in this diagram.
The incremental storing procedure is adapted to the alternative architecture
as follows.
Initialization (no pattern yet stored): A0 = 0N×N, d0 = 01×N. Choose an
aperture α to be used for all patterns.
Incremental storage of patterns: For j = 1, . . . , K do:
1. (unchanged from original procedure)
2. (unchanged)
3. Compute an 1 × N vector dj
inc by putting
(a) F j−1 = ¬Aj−1 (unchanged),
(b) t = P j −(dj−1) Xj (the essential change),
(c) S = F j−1 Xj (unchanged),
(d) dj
inc =
 (SS′/(L −1) + α−2I)† St′/(L −1)
′.
4. Update d: dj = dj−1 + dj
inc.
5. Update A: Aj = Aj−1 ∨Cj (unchanged).
For deterministic patterns (as they were used in the two demonstrations above),
the alternative procedure should, and does, behave identically to the original one
(not shown). For stochastic patterns it can be expected to be statistically more
eﬃcient (needing less training data for achieving same accuracy) since it exploits
the valuable structural bias of knowing beforehand that D should have unit rank
and have scaled versions of W in as columns (remains to be explored).
66
Our demonstrations used 1-dimensional drivers. For m-dimensional drivers,
the alternative architecture can be designed in an obvious way using m additional
neurons πν.
The alternative architecture has maximal representational eﬃciency in the fol-
lowing sense. For storing integer-periodic signals pj, where the sum of periods of
the stored signals approaches the network size N (as in demonstration 1), only N
parameters (namely, d) have to be trained. One may object that one also has to
store the conceptors Cj in order to retrieve the patterns, i.e. one has to learn and
store also the large number of parameters contained in the conceptors. We will
however see in Section 3.13.4 that conceptors can be re-built on the ﬂy in retrieval
situations and need not be stored.
The alternative architecture also lends itself to storing arbitrarily large num-
bers of patterns on the basis of a single reservoir. If the used memory quota q(Aj)
approaches 1 and the above storing procedure starts stalling, one can add another
π neuron and continue storing patterns using it. This however necessitates an ad-
ditional switching mechanism to select between diﬀerent available such π neurons
in training and retrieval (not yet explored).
3.12
Example: Dynamical Pattern Recognition
In this subsection I present another demonstration of the usefulness of Boolean
operations on conceptor matrices.
I describe a training scheme for a pattern
recognition system which reaches (or surpasses) the classiﬁcation test perfor-
mance of state-of-the-art recognizers on a widely used benchmark task.
Most
high-performing existing classiﬁers are trained in discriminative training schemes.
Discriminative classiﬁer training exploits the contrasting diﬀerences between the
pattern classes. This implies that if the repertoire of to-be-distinguished patterns
becomes extended by a new pattern, the classiﬁer has to be re-trained on the
entire dataset, re-visiting training data from the previous repertoire. In contrast,
the system that I present is trained in a “pattern-local” scheme which admits an
incremental extension of the recognizer if new pattern types were to be included in
its repertoire. Furthermore, the classiﬁer can be improved in its exploitation phase
by incrementally incorporating novel information contained in a newly incoming
test pattern. The key to this local-incremental classiﬁer training is agin Boolean
operations on conceptors.
Unlike in the rest of this report, where I restrict the presentation to stationary
and potentially inﬁnite-duration signals, the patterns here are nonstationary and of
short duration. This subsection thus also serves as a demonstration how conceptors
function with short nonstationary patterns.
I use is the Japanese Vowels benchmark dataset. It has been donated by [60]
and is publicly available at the UCI Knowledge Discovery in Databases Archive
(http://kdd.ics.uci.edu/). This dataset has been used in dozens of articles
in machine learning as a reference demonstration and thus provides a quick ﬁrst
67
orientation about the positioning of a new classiﬁcation learning method. The
dataset consists of 640 recordings of utterances of two successive Japanese vowels
/ae/ from nine male speakers. It is grouped in a training set (30 recordings from
each of the speakers = 270 samples) and a test set (370 further recordings, with
diﬀerent numbers of recordings per speaker). Each sample utterance is given in the
form of a 12-dimensional timeseries made from the 12 LPC cepstrum coeﬃcients.
The durations of these recordings range between 7 and 29 sampling timesteps.
The task is to classify the speakers in the test data, using the training data to
learn a classiﬁer. Figure 24 (top row) gives an impression of the original data.
5
10 15 20 25
!1
0
1
2
speaker 4
5
10 15 20 25
!1
0
1
2
speaker 6
5
10 15 20 25
!1
0
1
2
speaker 8
1
2
3
4
0
0.5
1
1
2
3
4
0
0.5
1
1
2
3
4
0
0.5
1
Figure 24: Three exemplary utterance samples from Japanese Vowels dataset.
Plots show values of twelve signal channels against discrete timesteps. Top row:
raw data as provided in benchmark repository, bottom row: standardized format
after preprocessing.
I preprocessed the raw data into a standardized format by (1) shift-scaling each
of the twelve channels such that per channel, the minimum/maximum value across
all training samples was 0/1; (2) interpolating each channel trace in each sample
by a cubic polynomial; (3) subsampling these on four equidistant support points.
The same transformations were applied to the test data. Figure 24 (bottom row)
illustrates the normalized data format.
The results reported in the literature for this benchmark typically reach an
error rate (percentage of misclassiﬁcations on the test set) of about 5 – 10 test
errors (for instance, [91, 97, 79] report from 5 – 12 test misclassiﬁcations, all using
specialized versions of temporal support vector machines). The best result that I
am aware of outside my own earlier attempts [57] is reported by [15] who reaches
about 4 errors, using reﬁned hidden Markov models in a non-discriminative train-
68
ing scheme. It is however possible to reach zero errors, albeit with an extraordinary
eﬀort: in own work [57] this was robustly achieved by combining the votes of 1,000
RNNs which were each independently trained in a discriminative scheme.
Here I present a “pocket-size” conceptor-based classiﬁcation learning scheme
which can be outlined as follows:
1. A single, small (N = 10 units) random reservoir network is initially created.
2. This reservoir is driven, in nine independent sessions, with the 30 prepro-
cessed training samples of each speaker j (j = 1, . . . , 9), and a conceptor C+
j
is created from the network response (no “loading” of patterns; the reservoir
remains unchanged throughout).
3. In exploitation, a preprocessed sample s from the test set is fed to the reser-
voir and the induced reservoir states x(n) are recorded and transformed
into a single vector z. For each conceptor then the positive evidence quan-
tity z′ C+
j z is computed.
This leads to a classiﬁcation by deciding for
j = argmaxi z′ C+
i z as the speaker of s. The idea behind this procedure
is that if the reservoir is driven by a signal from speaker j, the resulting
response z signal will be located in a linear subspace of the (transformed,
see below) reservoir state space whose overlap with the ellipsoids given by
the C+
i is largest for i = j.
4. In order to further improve the classiﬁcation quality, for each speaker j
also a conceptor C−
j = ¬ W{C+
1 , . . . , C+
j−1, C+
j+1, . . . , C+
9 } is computed. This
conceptor can be understood as representing the event “not any of the other
speakers”.
This leads to a negative evidence quantity z′ C−
j z which can
likewise be used as a basis for classiﬁcation.
5. By adding the positive and negative evidences, a combined evidence is ob-
tained which can be paraphrased as “this test sample seems to be from
speaker j and seems not to be from any of the others”.
In more detail, the procedure was implemented as follows. A 10-unit reservoir
system with 12 input units and a constant bias term with the update equation
x(n + 1) = tanh(W x(n) + W ins(n) + b)
(47)
was created by randomly creating the 10 × 10 reservoir weight matrix W, the
12×10 input weight matrix W in and the bias vector b (full speciﬁcation in Section
4.5).
Furthermore, a random starting state xstart, to be used in every run in
training and testing, was created. Then, for each speaker j, the conceptor C+
j was
learnt from the 30 preprocessed training samples sk
j(n) (where j = 1, . . . , 9; k =
1, . . . , 30; n = 1, . . . , 4) of this speaker, as follows:
69
1. For each training sample sk
j (k = 1, . . . , 30) of this speaker, the system
(47) was run with this input, starting from x(0) = xstart, yielding four
network states x(1), . . . , x(4).
These states were concatenated with each
other and with the driver input into a 4 · (10 + 12) = 88 dimensional vector
zk
j = [x(1); sk
j(1); . . . ; x(4); sk
j(4)]. This vector contains the entire network
response to the input sk
j and the input itself.
2. The 30 zk
j were assembled as columns into a 88 × 30 matrix Z from which
a correlation matrix Rj = ZZ′/30 was obtained. A preliminary conceptor
˜C+
j = Rj(Rj + I)−1 was computed from Rj (preliminary because in a later
step the aperture is optimized). Note that ˜C+
j has size 88 × 88.
After all “positive evidence” conceptors ˜C+
j
had been created, preliminary
“negative evidence” conceptors ˜C−
j were computed as
˜C−
j = ¬
_
{ ˜C+
1 , . . . , ˜C+
j−1, ˜C+
j+1, . . . , ˜C+
9 }.
(48)
An important factor for good classiﬁcation performance is to ﬁnd optimal aper-
tures for the conceptors, that is, to ﬁnd aperture adaptation factors γ+, γ−such
that ﬁnal conceptors C+
j
= ϕ( ˜C+
j , γ+), C−
j
= ϕ( ˜C−
j , γ−) function well for clas-
siﬁcation. A common practice in machine learning would be to optimize γ by
cross-validation on the training data. This, however, is expensive, and more cru-
cially, it would defy the purpose to design a learning procedure which can be
incrementally extended by novel pattern classes without having to re-inspect all
training data. Instead of cross-validation I used the ∇criterion described in Sec-
tion 3.8.4 to ﬁnd a good aperture. Figure 25 shows how this criterion varies with
γ for an exemplary case of a ϕ( ˜C+
j , γ+) sweep. For each of the nine ˜C+
j , the value
˜γ+
j which maximized ∇was numerically computed, and the mean of these nine
values was taken as the common γ+ to get the nine C+
j = ϕ( ˜C+
j , γ+). A similar
procedure was carried out to arrive at C−
j = ϕ( ˜C−
j , γ−).
0
1
2
3
4
5
6
7
8
0
5
10
log2 γ
Figure 25: The criterion ∇from an exemplary conceptor plotted against the log
2 of candidate aperture adaptations γ.
The conceptors C+
j , C−
j were then used for classiﬁcation as follows. Assume z is
an 88-dimensional combined states-and-input vector as described above, obtained
70
from driving the reservoir with a preprocessed test sample. Three kinds of classi-
ﬁcation hypotheses were computed, the ﬁrst only based on C+
j , the second based
on C−
j , and one based on a combination of both. Each classiﬁcation hypothesis is
a 9-dimensional vector with “evidences” for the nine speakers. Call these evidence
vectors h+, h−, h+−for the three kinds of classiﬁcations. The ﬁrst of these was
computed by setting ˜h+(j) = z′ C+
j z, then normalizing ˜h+ to a range of [0, 1] to
obtain h+. Similarly h−was obtained from using z′ C−
j z, and h+−= (h+ + h−)/2
was simply the mean of the two former. Each hypothesis vector leads to a classiﬁ-
cation decision by opting for the speaker j corresponding to the largest component
in the hypothesis vector.
This classiﬁcation procedure was carried out for all of the 370 test cases, giving
370 hypothesis vectors of each of the three kinds. Figure 26 gives an impression.
Figure 26: Collected evidence vectors h+, h−, h+−obtained in a classiﬁcation learn-
ing experiment. Grayscale coding: white = 0, black = 1. Each panel shows 370
evidence vectors. The (mostly) black segments along the diagonal correspond to
the correct classiﬁcations (test samples were sorted by speaker). For explanation
see text.
Results:
The outlined classiﬁcation experiment was repeated 50 times with ran-
dom new reservoirs. On average across the 50 trials, the optimal apertures γ+ / γ−
were found as 25.0 / 27.0 (standard deviations 0.48 / 0.75). The number of mis-
classiﬁcations for the three types of classiﬁcation (positive, negative, combined
evidence) were 8.5 / 5.9 / 4.9 (standard deviations 1.0 / 0.91 / 0.85). The train-
ing errors for the combined classiﬁcation (obtained from applying the classiﬁcation
procedure on the training samples) was zero in all 50 trials. For comparison, a
carefully regularized linear classiﬁer based on the same z vectors (detail in Section
4.5) reached 5.1 misclassiﬁcations across the 50 trials.
While these results are at the level of state-of-the-art classiﬁers on this bench-
mark, this basic procedure can be reﬁned, yielding a signiﬁcant improvement. The
idea is to compute the evidence for speaker j based on a conceptor ¯C+
j which itself
is based on the assumption that the test sample s belongs to the class j, that is, the
71
computed evidence should reﬂect a quantity “if s belonged to class j, what evidence
can we collect under this assumption?”. Recall that C+
j is obtained from the 30
training samples through C+
j = R (R + (γ+)−2I)−1, where R = ZZ′/30 is the cor-
relation matrix of the 30 training coding vectors belonging to speaker j. Now add
the test vector z to Z, obtaining ¯Z = [Zz], ¯R = ¯Z ¯Z′/31, ¯C+
j = ¯R( ¯R + (γ+)−2I)−1,
and use ¯C+
j in the procedure outlined above instead of C+
j . Note that, in appli-
cation scenarios where the original training data Z are no longer available at test
time, ¯C+
j
can be directly computed from C+
j
and z through the model update
formulas (36) or (37). The negative evidence conceptor is accordingly obtained by
¯C−
j = ¬ W{ ¯C+
1 , . . . , ¯C+
j−1, ¯C+
j+1, . . . , ¯C+
9 }.
Results of reﬁned classiﬁcation procedure:
Averaged over 50 learn-test tri-
als with independently sampled reservoir weights, the number of misclassiﬁcations
for the three types of classiﬁcation (positive, negative, combined evidence) were 8.4
/ 5.9 / 3.4 (standard deviations 0.99 / 0.93 / 0.61). The training misclassiﬁcation
errors for the combined classiﬁcation was zero in all 50 trials.
The detection of good apertures through the ∇criterion worked well. A manual
grid search through candidate apertures found that a minimum test misclassiﬁca-
tion rate of 3.0 (average over the 50 trials) from the combined classiﬁcator was ob-
tained with an aperture α+ = 20, α−= 24 for both the positive and negative con-
ceptors. The automated aperture detection yielded apertures α+ = 25, α−= 27
and a (combined classiﬁcator) misclassiﬁcation rate of 3.4, close to the optimum.
Discussion. The following observations are worth noting.
Method also applies to static pattern classiﬁcation. In the presented clas-
siﬁcation method, temporal input samples s (short preprocessed nonstation-
ary timeseries) were transformed into static coding vectors z as a basis for
constructing conceptors. These z contained the original input signal s plus
the state response from a small reservoir driven by s. The reservoir was only
used to augment s by some random nonlinear interaction terms between the
entries in s. Conceptors were created and used in classiﬁcation without re-
ferring back to the reservoir dynamics. This shows that conceptors can also
be useful in static pattern classiﬁcation.
Extensibility. A classiﬁcation model consisting of learnt conceptors C+
j , C−
j for
k classes can be easily extended by new classes, because the computations
needed for new C+
k+1, C−
k+1 only require positive training samples of the new
class. Similarly, an existing model C+
j , C−
j can be extended by new training
samples without re-visiting original training data by an application of the
model extension formulae (36) or (37).
In fact, the reﬁned classiﬁcation
procedure given above can be seen as an ad-hoc conditional model extension
by the test sample.
Including an “other” class. Given a learnt classiﬁcation model C+
j , C−
j for k
classes it appears straightforward to include an “other” class by including
72
C+
other = ¬ W{C+
1 , . . . , C+
k } and recomputing the negative evidence concep-
tors from the set {C+
1 , . . . , C+
k , C+
other} via (48). I have not tried this out
yet.
Discriminative nature of combined classiﬁcation. The classiﬁcation of the
combined type, paraphrased above as “sample seems to be from class j
and seems not to be from any of the others”, combines information from all
classes into an evidence vote for a candidate class j. Generally, in discrimina-
tive learning schemes for classiﬁers, too, contrasting information between the
classes is exploited. The diﬀerence is that in those schemes, these diﬀerences
are worked in at learning time, whereas in the presented conceptor-based
scheme they are evaluated at test time.
Beneﬁts of Boolean operations. The three aforementioned points – extensi-
bility, “other” class, discriminative classiﬁcation – all hinge on the availabil-
ity of the NOT and OR operations, in particular, on the associativity of the
latter.
Computational eﬃciency. The computational steps involved in learning and
applying conceptors are constructive. No iterative optimization steps are
involved (except that standard implementations of matrix inversion are iter-
ative). This leads to short computation times. Learning conceptors from the
270 preprocessed data samples, including determining good apertures, took
650 ms and classifying a test sample took 0.7 ms for the basic and 64 ms for
the reﬁned procedure (on a dual-core 2GHz Macintosh notebook computer,
using Matlab).
Competitiveness. The test misclassiﬁcation rate of 3.4 is slightly better than
the best rate of about 4 that I am aware of in the literature outside own
work [57]. Given that the zero error performance in [57] was achieved with
an exceptionally expensive model (combining 1,000 independently sampled
classiﬁers), which furthermore is trained in a discriminative setup and thus is
not extensible, the attained performance level, the computational eﬃciency,
and the extensibility of the conceptor-base model render it a competitive
alternative to existing classiﬁcation learning methods. It remains to be seen
though how it performs on other datasets.
Regularization by aperture adaptation? In supervised classiﬁcation learn-
ing tasks, it is generally important to regularize models to ﬁnd the best
balance between overﬁtting and under-exploiting training data. It appears
that the role of regularization is here played by the aperture adaptation,
though a theoretical analysis remains to be done.
Early stage of research. The proposed classiﬁer learning scheme was based on
numerous ad-hoc design decisions, and quite diﬀerent ways to exploit con-
ceptors for classiﬁcation are easily envisioned. Thus, in sum, the presented
73
study should be regarded as no more than a ﬁrst demonstration of the basic
usefulness of conceptors for classiﬁcation tasks.
3.13
Autoconceptors
3.13.1
Motivation and Overview
In the preceding sections I have deﬁned conceptors as transforms C = R(R +
α−2I)−1 of reservoir state correlation matrices R. In order to obtain some con-
ceptor Cj which captures a driving pattern pj, the network was driven by pj via
x(n + 1) = tanh(W ∗x(n) + W inpj(n + 1) + b), the obtained reservoir states were
used to compute Rj, from which Cj was computed.
The conceptor Cj could
then later be exploited via the conceptor-constrained update rule x(n + 1) =
Cj tanh(Wx(n) + b) or its variant x(n + 1) = Cj tanh(W ∗x(n) + D x(n) + b).
This way of using conceptors, however, requires that the conceptor matrices Cj
are computed at learning time (when the original drivers are active), and they have
to be stored for later usage. Such a procedure is useful and feasible in engineering
or machine learning applications, where the conceptors Cj may be written to
ﬁle for later use.
It is also adequate for theoretical investigations of reservoir
dynamics, and logical analyses of relationships between reservoir dynamics induced
by diﬀerent drivers, or constrained by diﬀerent conceptors.
However, storing conceptor matrices is entirely implausible from a perspective
of neuroscience. A conceptor matrix has the same size as the original reservoir
weight matrix, that is, it is as large an entire network (up to a saving factor
of one half due to the symmetry of conceptor matrices). It is hard to envision
plausible models for computational neuroscience where learning a new pattern by
some RNN essentially would amount to creating an entire new network.
This motivates to look for ways of how conceptors can be used for constraining
reservoir dynamics without the necessity to store conceptors in the ﬁrst place.
The network would have to create conceptors “on the ﬂy” while it is performing
some relevant task. Speciﬁcally, we are interested in tasks or functionalities which
are relevant from a computational neuroscience point of view. This objective also
motivates to focus on algorithms which are not immediately biologically implau-
sible. In my opinion, this largely excludes computations which explicitly exploit
the SVD of a matrix (although it has been tentatively argued that neural net-
works can perform principal component analysis [78] using biologically observed
mechanisms).
In the next subsections I investigate a version of conceptors with associated
modes of usage where there is no need to store conceptors and where computa-
tions are online adaptive and local in the sense that the information necessary for
adapting a synaptic weight is available at the concerned neuron. I will demon-
strate the workings of these conceptors and algorithms in two functionalities, (i)
content-addressable memory (Section 3.13.3) and (ii) simultaneous de-noising and
classiﬁcation of a signal (Section 3.15).
74
In this line of modeling, the conceptors are created by the reservoir itself at the
time of usage. There is no role for an external engineer or superordinate control
algorithm to “plug in” a conceptor. I will speak of autoconceptors to distinguish
these autonomously network-generated conceptors from the conceptors that are
externally stored and externally inserted into the reservoir dynamics. In discus-
sions I will sometimes refer to those “externalistic” conceptors as alloconceptors.
Autoconceptors, like alloconceptors, are positive semideﬁnite matrices with
singular values in the unit interval. The semantic relationship to data, aperture
operations, and Boolean operations are identical for allo- and autoconceptors.
However, the way how autoconceptors are generated is diﬀerent from alloconcep-
tors, which leads to additional constraints on their algebraic characteristics. The
set of autoconceptor matrices is a proper subset of the conceptor matrices in gen-
eral, as they were deﬁned in Deﬁnition 2, i.e. the class of positive semideﬁnite
matrices with singular values ranging in the unit interval. The additional con-
straints arise from the circumstance that the reservoir states x(n) which shape an
autoconceptor C are themselves depending on C.
The treatment of autoconceptors will be structured as follows.
I will ﬁrst
introduce the basic equations of autoconceptors and their adaptation dynamics
(Section 3.13.2), demonstrate their working in a of content-addressable memory
task (Section 3.13.3) and mathematically analyse central properties of the adapta-
tion dynamics (Section 3.13.4). The adaptation dynamics however has non-local
aspects which render it biologically implausible. In order to progress toward bio-
logically feasible autoconceptor mechanisms, I will propose neural circuits which
implement autoconceptor dynamics in ways that require only local information for
synaptic weight changes (Section 3.14).
3.13.2
Basic Equations
The basic system equation for autoconceptor systems is
x(n + 1) = C(n) tanh(W ∗x(n) + W inp(n + 1) + b)
(49)
or variants thereof, like
x(n + 1) = C(n) tanh(W x(n) + b)
(50)
or
x(n + 1) = C(n) tanh(W ∗x(n) + Dx(n) + b),
(51)
the latter two for the situation after having patterns stored. The important novel
element in these equations is that C(n) is time-dependent. Its evolution will be
governed by adaptation rules that I will describe presently. C(n) need not be
positive semideﬁnite at all times; only when the adaptation of C(n) converges, the
resulting C matrices will have the algebraic properties of conceptors.
One can conceive of the system (49) as a two-layered neural network, where the
two layers have the same number of neurons, and where the layers are reciprocally
75
connected by the connection matrices C and W (Figure 27). The two layers have
states
r(n + 1)
=
tanh(W ∗z(n) + W inp(n + 1) + b)
(52)
z(n + 1)
=
C r(n + 1).
(53)
The r layer has sigmoidal (here: tanh) units and the z layer has linear ones. The
customary reservoir state x becomes split into two states r and z, which can be
conceived of as states of two pools of neurons.
! 
r
! 
z
W* 
C 
! 
W in
! 
p
Figure 27: Network representation of a basic autoconceptor system. Bias b and
optional readout mechanisms are omitted. The broken arrow indicates that C
connections are online adaptive. For explanation see text.
In order to determine an adaptation law for C(n), I replicate the line of rea-
soning that was employed to motivate the design of alloconceptors in Section 3.4.
Alloconceptors were designed to act as “regularized identity maps”, which led to
the deﬁning criterion (6) in Deﬁnition 1:
C(R, α) = argminC E[∥x −Cx∥2] + α−2 ∥C∥2
fro.
The reservoir states x that appear in this criterion resulted from the update equa-
tion x(n + 1) = tanh(W ∗x(n) + W inp(n + 1) + b). This led to the explicit solution
(7) stated in Proposition 1:
C(R, α) = R (R + α−2 I)−1,
where R was the reservoir state correlation matrix E[xx′]. I re-use this criterion
(6), which leads to an identical formula C = R (R + α−2 I)−1 for autoconceptors.
The crucial diﬀerence is that now the state correlation matrix R depends on C:
R = E[zz′] = E[Cr(Cr)′] = C E[rr′] C =: CQC,
(54)
where we introduce Q = E[rr′]. This transforms the direct computation formula
(7) to a ﬁxed-point equation:
C = CQC (CQC + α−2 I)−1,
76
which is equivalent to
(C −I)CQC −α−2C = 0.
(55)
Since Q depends on r states, which in turn depend on z states, which in turn
depend on C again, Q depends on C and should be more appropriately be written
as QC. Analysing the ﬁxed-point equation (C −I)CQCC −α−2C = 0 is a little
inconvenient, and I defer this to Section 3.13.4. When one uses autoconceptors,
however, one does not need to explicitly solve (55). Instead, one can resort to a
version of the incremental adaptation rule (11) from Proposition 2:
C(n + 1) = C(n) + λ
 (z(n) −C(n) z(n)) z′(n) −α−2 C(n)

,
which implements a stochastic gradient descent with respect to the cost function
E[∥z −Cz∥2] + α−2 ∥C∥2
fro. In the new sitation given by (49), the state z here
depends on C. This is, however, of no concern for using (11) in practice. We thus
complement the reservoir state update rule (49) with the conceptor update rule
(11) and comprise this in a deﬁnition:
Deﬁnition 5 An autoconceptive reservoir network is a two-layered RNN with
ﬁxed weights W ∗, W in and online adaptive weights C, whose dynamics are given
by
z(n + 1)
=
C(n) tanh(W ∗z(n) + W inp(n + 1) + b)
(56)
C(n + 1)
=
C(n) + λ
 (z(n) −C(n) z(n)) z′(n) −α−2 C(n)

,
(57)
where λ is a learning rate and p(n) an input signal. Likewise, when the update
equation (56) is replaced by variants of the kind (50) or (51), we will speak of
autoconceptive networks.
I will derive in Section 3.13.4 that if the driver p is stationary and if C(n)
converges under these rules, then the limit C is positive semideﬁnite with sin-
gular values in the set (1/2, 1) ∪{0}. Singular values asymptotically obtained
under the evolution (57) are either “large” (that is, greater than 1/2) or they are
zero, but they cannot be “small” but nonzero. If the aperture α is ﬁxed at in-
creasingly smaller values, increasingly many singular values will be forced to zero.
Furthermore, the analysis in Section 3.13.4 will also reveal that among the nonzero
singular values, the majority will be close to 1. Both eﬀects together mean that au-
toconceptors are typically approximately hard conceptors, which can be regarded
as an intrinsic mechanism of contrast enhancement, or noise suppression.
3.13.3
Example: Autoconceptive Reservoirs as Content-Addressable
Memories
In previous sections I demonstrated how loaded patterns can be retrieved if the
associated conceptors are plugged into the network dynamics. These conceptors
77
must have been stored beforehand. The actual memory functionality thus resides
in whatever mechanism is used to store the conceptors; furthermore, a conceptor
is a heavyweight object with the size of the reservoir itself.
It is biologically
implausible to create and “store” such a network-like object for every pattern that
is to be recalled.
In this section I describe how autoconceptor dynamics can be used to create
content-addressable memory systems. In such systems, recall is triggered by a cue
presentation of the item that is to be recalled. The memory system then should
in some way autonomously “lock into” a state or dynamics which autonomously
re-creates the cue item. In the model that will be described below, this “locking
into” spells out as running the reservoir in autoconceptive mode (using equations
(51) and (57)), by which process a conceptor corresponding to the cue pattern
shapes itself and enables the reservoir to autonomously re-generate the cue.
The archetype of content-addressable neural memories is the Hopﬁeld network
[48]. In these networks, the cue is a static pattern (technically a vector, in demon-
strations often an image), which typically is corrupted by noise or incomplete.
If the Hopﬁeld network has been previously trained on the uncorrupted complete
pattern, its recurrent dynamics will converge to an attractor state which re-creates
the trained original from the corrupted cue. This pattern completion character-
istic is the essence of the memory functionality in Hopﬁeld networks.
In the
autoconceptive model, the aspect of completion manifests itself in that the cue is
a brief presentation of a dynamic pattern, too short for a conceptor to be properly
adapted. After the cue is switched oﬀ, the autoconceptive dynamics continues to
shape the conceptor in an entirely autonomous way, until it is properly developed
and the reservoir re-creates the cue.
This autoconceptive adaptation is superﬁcially analog to the convergence to
an attractor point in Hopﬁeld networks. However, there are important conceptual
and mathematical diﬀerences between the two models. I will discuss them at the
end of this section.
Demonstration of basic architecture.
To display the core idea of a content-
addressable memory, I ran simulations according to the following scheme:
1. Loading. A collection of k patterns pj (j = 1, . . . , k) was loaded in an N-
dimensional reservoir, yielding an input simulation matrix D as described in
Equation (40), and readout weights W out, as described in Section 3.3. No
conceptors are stored.
2. Recall. For each pattern pj, a recall run was executed which consisted of
three stages:
(a) Initial washout. Starting from a zero network state, the reservoir
was driven with pj for nwashout steps, in order to obtain a task-related
reservoir state.
78
(b) Cueing. The reservoir was continued to be driven with pj for another
ncue steps. During this cueing period, Cj was adapted by using r(n +
1) = tanh(Wr(n) + W inpj(n) + b), Cj(n + 1) = Cj(n) + λcue ((r(n) −
Cj(n) r(n)) r′(n) −α−2 Cj(n)). At the beginning of this period, Cj was
initialized to the zero matrix. At the end of this period, a conceptor
Cj cue was obtained.
(c) Autonomous recall.
The network run was continued for another
nrecall steps in a mode where the input was switched oﬀand replaced
by the input simulation matrix D, and where the conceptor Cj cue was
further adapted autonomously by the autoconceptive update mecha-
nism, via z(n + 1) = Cj(n) tanh(Wz(n) + Dz(n) + b), Cj(n + 1) =
Cj(n) + λrecall ((z(n) −Cj(n) z(n)) z′(n) −α−2 Cj(n)). At the end of
this period, a conceptor Cj recall was available.
3. Measuring quality of conceptors. The quality of the conceptors Cj cue
and Cj recall was measured by separate oﬄine runs without conceptor adap-
tation using r(n) = tanh(Wz(n) + Dz(n) + b); z(n + 1) = Cj cue r(n) (or
z(n + 1) = Cj recall r(n), respectively).
A reconstructed pattern y(n) =
W outr(n) was obtained and its similarity with the original pattern pj was
quantiﬁed in terms of a NRMSE.
I carried out two instances of this experiment, using two diﬀerent kinds of
patterns and parametrizations:
4-periodic pattern. The patterns were random integer-periodic patterns of pe-
riod 4, where per pattern the four pattern points were sampled from a uni-
form distribution and then shift-scaled such that the range became [−1 1].
This normalization implies that the patterns are drawn from an essentially
3-parametric family (2 real-valued parameters for ﬁxing the two pattern
values not equal to −1 or 1; one integer parameter for ﬁxing the relative
temporal positioning of the −1 and 1 values).
Experiment parameters:
k = 10, N = 100, α = 100, nwashout = 100, ncue = 15, nrecall = 300, γcue =
0.02, γrecall = 0.01 (full detail in Section 4.4).
Mix of 2 irrational-period sines. Two sines of period lengths
√
30 and
√
30/2
were added with random phase angles and random amplitudes, where how-
ever the two amplitudes were constrained to sum to 1. This means that
patterns were drawn from a 2-parametric family. Parameters: k = 10, N =
200, α = 100, nwashout = 100, ncue = 30, nrecall = 10000, γcue = γrecall = 0.01.
Furthermore, during the auto-adaptation period, strong Gaussian iid noise
was added to the reservoir state before applying the tanh, with a signal-to-
noise rate of 1.
Figure 28 illustrates the outcomes of these two experiments. Main observations:
79
A.
0
1
Singular Values
!1
0
1
y and p
0
1
!1
0
1
0
5
10
0
1
0
5
10
!1
0
1
B.
1
2
3
4
5
6
7
8
9 10
!6
!5
!4
!3
!2
!1
0
Pattern index
log10 NRMSE
C.
0
1
Singular Values
!1
0
1
y and p
0
1
!1
0
1
0
10
20
0
1
0
5
10
!1
0
1
D.
1
2
3
4
5
6
7
8
9 10
!1.8
!1.6
!1.4
!1.2
!1
!0.8
!0.6
!0.4
!0.2
Pattern index
log10 NRMSE
Figure 28: Basic content-addressable memory demos. A, B: 4-periodic pattern,
C, D: mix of sines pattern. Panels A, C show the ﬁrst three of the 10 patterns.
The singular value plots show the ﬁrst 10 (20, respectively) singular values of Cj cue
(black) and Cj recall (light gray). The “y and p” panels show the reconstructed pat-
tern y obtained with Cj recall (bold light gray) and the original training pattern pj
(broken black), after optimal phase-alignment. B, D plot the pattern reconstruc-
tion NRMSEs in log10 scale for the reconstructions obtained from Cj cue (black
squares) and from Cj recall (gray crosses). For explanation see text.
80
1. In all cases, the quality of the preliminary conceptor Cj cue was very much
improved by the subsequent auto-adaptation (Panels B, D), leading to an
ultimate pattern reconstruction whose quality is similar to the one that
would be obtained from precomputed/stored conceptors.
2. The eﬀects of the autoconceptive adaptation are reﬂected in the singular
value proﬁles of Cj cue versus Cj recall (Panels A, C). This is especially well
visible in the case of the sine mix patterns (for the period-4 patterns the eﬀect
is too small to show up in the plotting resolution). During the short cueing
time, the online adaptation of the conceptor from a zero matrix to Cj cue
only manages to build up a preliminary proﬁle that could be intuitively
called “nascent”, which then “matures” in the ensuing network-conceptor
interaction during the autoconceptive recall period.
3. The conceptors Cj recall have an almost rectangular singular value proﬁle. In
the next section I will show that if autoconceptive adaptation converges, sin-
gular values are either exactly zero or greater than 0.5 (in fact, typically close
to 1), in agreement with what can be seen here. Autoconceptive adaptation
has a strong tendency to lead to almost hard conceptors.
4. The fact that adapted autoconceptors typically have a close to rectangular
singular value spectrum renders the auto-adaptation process quite immune
against even strong state noise.
Reservoir state noise components in di-
rections of the nulled eigenvectors are entirely suppressed in the conceptor-
reservoir loop, and state noise components within the nonzero conceptor
eigenspace do not impede the development of a “clean” rectangular pro-
ﬁle. In fact, state noise is even beneﬁcial: it speeds up the auto-adaptation
process without a noticeable loss in ﬁnal pattern reconstruction accuracy
(comparative simulations not documented here).
This noise robustness however depends on the existence of zero singular
values in the adapting autoconceptor C. In the simulations reported above,
such zeros were present from the start because the conceptor was initialized
as the zero matrix. If it had been initialized diﬀerently (for instance, as
identity matrix), the auto-adaptation would only asymptotically pull (the
majority of) singular values to zero, with noise robustness only gradually
increasing to the degree that the singular value spectrum of C becomes
increasingly rectangular. If noise robustness is desired, it can be reached
by additional adaptation mechanisms for C. In particular, it is helpful to
include a thresholding mechanism: all singular values of C(n) exceeding a
suitable threshold are set to 1, all singular values dropping below a certain
cutoﬀare zeroed (not shown).
Exploring the eﬀects of increasing memory load – patterns from a
parametrized family.
A central theme in neural memory research is the ca-
81
pacity of a neural storage system. In order to explore how recall accuracy depends
on the memory load, I carried out two further experiments, one for each pattern
type. Each of these experiments went along the following scheme:
1. Create a reservoir.
2. In separate trials, load this reservoir with an increasing number k of patterns
(ranging from k = 2 to k = 200 for the 4-period and from k = 2 to k = 100
for the mixed sines).
3. After loading, repeat the recall scheme described above, with the same pa-
rameters. Monitor the recall accuracy obtained from Cj recall for the ﬁrst 10
of the loaded patterns (if less than 10 were loaded, do it only for these).
4. In addition, per trial, also try to cue and “recall” 10 novel patterns that were
drawn randomly from the 4-periodic and mixed-sine family, respectively, and
which were not part of the collection loaded into the reservoir. Monitor the
“recall” accuracy of these novel patterns as well.
5. Repeat this entire scheme 5 times, with freshly created patterns, but re-using
always the same reservoir.
A.
2
3
5
8
12 16
25
50
100
200
!5
!4.5
!4
!3.5
!3
!2.5
!2
!1.5
!1
!0.5
0
Nr of loaded patterns
log10 NRMSE
B.
2
3
5
8
12 16
25
50
100
!1.8
!1.6
!1.4
!1.2
!1
!0.8
!0.6
!0.4
!0.2
Nr of loaded patterns
log10 NRMSE
Figure 29: Exploring the eﬀects of memory load.
A: 4-periodic patterns, B:
mix-of-sines patterns. Each diagram shows the log10 NRMSE of recalling loaded
patterns with Ccue (black solid line) and with Crecall (black broken line), as well
as of “recalling” patterns not contained in the loaded set, again obtained from
Ccue (gray solid line) and with Crecall (gray broken line). Error bars indicate 95
% conﬁdence intervals. Both axes are in logarithmic scaling. For explanation see
text.
Figure 29 shows the results of these experiments. The plotted curves are the
summary averages over the 10 recall targets and the 5 experiment repetitions.
82
Each plot point in the diagrams thus reﬂects an average over 50 NRMSE values
(except in cases where k < 10 patterns were stored; then plotted values correspond
to averages over 5k NRMSE values for recalling of loaded patterns). I list the main
ﬁndings:
1. For all numbers of stored patterns, and for both the recall loaded patterns
and recall novel patterns conditions, the autoconceptive “maturation” from
Ccue to Crecall with an improvement of recall accuracy is found again.
2. The ﬁnal Crecall-based recall accuracy in the recall loaded pattern condition
has a sigmoid shape for both pattern types.
The steepest ascent of the
sigmoid (fastest deterioration of recall accuracy with increase of memory
load) occurs at about the point where the summed quota of all Ccue reaches
the reservoir size N – the point where the network is “full” according to
this criterion (a related eﬀect was encountered in the incremental loading
study reported in Section 3.11). When the memory load is further increased
beyond this point (one might say the network becomes “overloaded”), the
recall accuracy does not break down but levels out on a plateau which still
translates into a recall performance where there is a strong similarity between
the target signal and the reconstructed one.
3. In the recall novel patterns conditions, one ﬁnds a steady improvement of
recall accuracy with increasing memory load. For large memory loads, the
accuracy in the recall novel patterns condition is virtually the same as in the
recall loaded patterns conditions.
Similar ﬁndings were obtained in other simulation studies (not documented
here) with other types of patterns, where in each study the patterns were drawn
from a parametrized family.
A crucial characteristic of these experiments is that the patterns were samples
from a parametrized family.
They shared a family resemblance.
This mutual
relatedness of patterns is exploited by the network: for large numbers k of stored
patterns, the storing/recall mechanism eﬀectively acquires a model of the entire
parametric family, a circumstance revealed by the essentially equal recall accuracy
in the recall loaded patterns and recall novel patterns conditions. In contrast, for
small k, the recall loaded patterns condition enables a recall accuracy which is
superior to the recall novel patterns condition: the memory system stores/recalls
individual patterns. I ﬁnd this worth a special emphasis:
• For small numbers of loaded patterns (before the point of network overload-
ing) the system stores and recalls individual patterns. The input simulation
matrix D represents individual patterns.
• For large numbers of loaded patterns (overloading the network), the system
learns a representation of the parametrized pattern family and can be cued
83
with, and will “recall”, any pattern from that family. The input simulation
matrix D represents the class of patterns.
At around the point of overloading, the system, in a sense, changes its nature
from a mere storing-of-individuals device to a learning-of-class mechanism. I call
this the class learning eﬀect.
Eﬀects of increasing memory load – mutually unrelated patterns.
A
precondition for the class learning eﬀect is that the parametric pattern family is
simple enough to become represented by the network. If the pattern family is too
richly structured to be captured by a given network size, or if patterns do not have
a family resemblance at all, the eﬀect cannot arise. If a network is loaded with
such patterns, and then cued with novel patterns, the “recall” accuracy will be on
chance level; furthermore, as k increases beyond the overloading region, the recall
accuracy of patterns contained in the loaded collection will decline to chance level
too.
In order to demonstrate this, I loaded the same 100-unit reservoir that was
used in the 4-periodic pattern experiments with random periodic patterns whose
periods ranged from 3 through 9. While technically this is still a parametric family,
the number of parameters needed to characterize a sample pattern is 8, which
renders this family far too complex for a 100-unit reservoir. Figure 30 illustrates
what, expectedly, happens when one loads increasingly large numbers of such
eﬀectively unrelated patterns. The NRMSE for the recall novel patterns condition
is about 1 throughout, which corresponds to entirely uncorrelated pattern versus
reconstruction pairs; and this NRMSE is also approached for large k in the recall
loaded patterns condition.
To be or not to be an attractor.
(This part addresses readers who are fa-
miliar with dynamical systems theory.) In the light of basic concepts provided by
dynamical systems theory, and in the light of how Hopﬁeld networks are known
to function as content-addressable memories, one might conjecture that the cue-
ing/recalling dynamics should be mathematically described as follows:
• The loading process creates a number of attractors (periodic attractors in
our demonstrations) in the combined conceptor/reservoir dynamics.
• Each loaded pattern becomes represented by such an attractor.
• During the cueing phase, the combined conceptor/network state becomes
located in the basin of the attractor corresponding to the cue pattern.
• In the remaining recall phase, the combined conceptor/network dynamics
converges toward this attractor.
84
4
6
8 10
14 1822 30 40 50 65 85110
!3.5
!3
!2.5
!2
!1.5
!1
!0.5
0
0.5
Nr of loaded patterns
log10 NRMSE
Figure 30: Eﬀects of memory load on recall accuracy for unrelated patterns. Figure
layout as in Figure 29. For explanation see text.
This, roughly, would be the picture if our content-addressable memory would
function in an analog way as Hopﬁeld networks, with the only diﬀerence being that
instead of point attractors (in Hopﬁeld networks) we are now witnessing cyclic at-
tractors. To submit this picture to a more in-depth test, I re-ran the 4-periodic
recall experiment documented in Figure 29 A with some modiﬁcations. Only a
single experiment was run (no averaging over diﬀerent collections of loaded pat-
terns), and only patterns contained in the loaded collection were cued/recalled.
Five of the loaded patterns were cued (or less if fewer had been loaded) and the
found accuracy NRMSEs were averaged. The recall accuracy was measured oﬄine
after 200, 2000, and 20000 steps of the post-cue autoconceptive adaptation. The
development of accuracy with recall runtime should be illustrative of convergence
characteristics. This experiment was done in two versions, ﬁrst with clean cue
signals and then again with cues that were overlaid with relatively weak noise
(sampled from the standard normal distribution, scaled by 1/20). If the auto-
conceptive adaptation would obey the attractor interpretation, one would expect
that both in the noise-free and the noisy cue conditions, a convergence to identical
good recall patterns is obtained.
The outcome of this experiment, illustrated in Figure 31, reveals that this
is not the case. For clean cues (panel A), the recall accuracies after 200, 2000,
20000 are virtually identical (and very good for small numbers of loaded patterns).
This is consistent with an interpretation of (fast) convergence to an attractor. If
this interpretation were correct, then adding a small amount of noise to the cue
should not aﬀect the system’s convergence to the same attractor. However this
is not what is obtained experimentally (panel B). The curves for increasing post-
cue runlengths cross each other and do not reach the accuracy levels from the
85
clean cue condition. While it is not clear how to interpret this outcome in terms
of dynamical systems convergence, it certainly is not convergence to the same
attractors (if they exist) as in the clean cue case.
A.
2
5
12
25
50
200
!6
!5
!4
!3
!2
!1
0
Nr of loaded patterns
log10 NRMSE
B.
!
"
#!
!"
"$
!$$
!%
!!&"
!!
!#&"
!#
!$&"
$
$&"
'()*+),*-./.)0-11/(23
,*4#$)'5678
Figure 31: A staged re-run of the 4-periodic pattern recall with noise-free cues
(A) and cues distorted by weak noise (B). Black line with square marks: NRMSE
after cue; lighter-shaded lines with cross / diamond / star marks: NRMSEs after
200, 2000, 20000 time steps of recall adaptation. The 200, 2000, 20000 curves are
almost identical in (A) and appear as a single line. For explanation see text.
A tentative explanation of these ﬁndings could be attempted as follows (a
reﬁned account will be given in Section 3.13.4). First, a simpliﬁcation. If the
adaptation rate γ in (57) is suﬃciently small, a separation of timescales between
a fast network state dynamics (56) and a slow conceptor adaptation dynamics
occurs.
It then makes sense to average over states z(n).
Let ¯zC be the fast-
timescale average of network states under the governance of a conceptor C. This
results in an autonomous continuous-time N 2-dimensional dynamical system in C
parameters,
τ ˙C = (¯zC −C) ¯z′
C −α−2C,
(58)
where τ is a time constant which is of no concern for the qualitative properties
of the dynamics. The dynamics (58) represents the adaptation dynamics of an
autoconceptor. Note that this adaptation dynamics does not necessarily preserve
C as a positive semideﬁnite matrix, that is, matrices C obtained during adaptation
need not be conceptor matrices. However, as proved in the next section, ﬁxed point
solutions of (58) are conceptor matrices.
Let us consider ﬁrst a situation where a reservoir has been “overloaded” with
a ﬁnite but large number of patterns from a simple parametric family. According
to our simulation ﬁndings, it is then possible to re-generate with high accuracy
any of the inﬁnitely many patterns from the family by cued auto-adaptation of
conceptors.
86
A natural candidate to explain this behavior is to assume that in the dynamical
system (58) governing the C evolution, the loading procedure has established an
instance of what is known as line attractor or plane attractor [25]. An attractor
of this kind is characterized by a manifold (line or plane or higher-dimensional
surface), consisting of ﬁxed points of the system dynamics, which attracts trajec-
tories from its neighborhood. Figure 32 attempts to visualize this situation. In
our case, the plane attractor would consist of ﬁxed point solutions of (58), that is,
conceptors that will ultimately be obtained when the recall adaptation converges.
The arrows in this ﬁgure represent trajectories of (58).
Figure 32: Schematic phase portraits of plane attractors. An m-dimensional in-
variant manifold consisting of neutrally stable ﬁxed points is embedded in an
n-dimensional state space of a dynamical system.
The ﬁgure shows an exam-
ple where m = 1, n = 3 (a “line attractor”, left) and an example where m = 2
(“plane attractor”). Trajectories in a neighborhood of the ﬁxed-point manifold
are attracted toward it (arrows show selected trajectories). For explanations see
text.
In cases where the reservoir has been loaded with only a small number of
patterns (from the same parametric family or otherwise), no such plane attractor
would be created. Instead, each of the loaded patterns becomes represented by an
isolated point attractor in (58). I mention again that this picture, while it is in
agreement with the simulations done so far, is preliminary and will be reﬁned in
Section 3.13.4.
Discussion.
Neural memory mechanisms – how to store patterns in, and retrieve
from, neural networks – is obviously an important topic of research. Conceptor-
based mechanisms bring novel aspects to this widely studied ﬁeld.
The paradigmatic model for content-addressable storage of patterns in a neural
network is undoubtedly the family of auto-associative neural networks (AANNs)
whose analysis and design was pioneered by Palm [80] and Hopﬁeld [48] (with a
rich history in theoretical neuroscience, referenced in [80]). Most of these models
are characterized by the following properties:
87
• AANNs with N units are used to store static patterns which are themselves
N-dimensional vectors. The activity proﬁle of the entire network coincides
with the very patterns. In many demonstrations, these patterns are rendered
as 2-dimensional images.
• The networks are typically employed, after training, in pattern completion or
restauration tasks, where an incomplete or distorted N-dimensional pattern
is set as the initial N-dimensional network state. The network then should
evolve toward a completed or restored pattern state.
• AANNs have symmetric connections and (typically) binary neurons. Their
recurrent dynamics can be formalized as a descent along an energy function,
which leads to convergence to ﬁxed points which are determined by the input
pattern.
• An auto-associative network is trained from a set of k reference patterns,
where the network weights are adapted such that the network state energy
associated with each training pattern is minimized. If successful, this leads
to an energy landscape over state space which assumes local minima at the
network states that are identical to the reference patterns.
The comprehensive and transparent mathematical theory available for AANNs
has left a strong imprint on our preconceptions of what are essential features of
a content-addressable neural memory. Speciﬁcally, AANN research has settled
the way how the task of storing items in an associative memory is framed in the
ﬁrst place: “given k reference patterns, train a network such that in exploitation,
these patterns can be reconstructed from incomplete cues”. This leads naturally
to identifying stored memory items with attractors in the network dynamics. Im-
portantly, memory items are seen as discrete, individual entities. For convenience
I will call this the “discrete items stored as attractors” (DISA) paradigm.
Beyond modeling memory functionality proper, the DISA paradigm is histor-
ically and conceptually connected to a wide range of models of neural represen-
tations of conceptual knowledge, where attractors are taken as the neural repre-
sentatives of discrete concepts. To name only three kinds of such models: point
attractors (cell assemblies and bistable neurons) in the working memory literature
[23]; spatiotemporal attractors in neural ﬁeld theories of cortical representation
[93, 30, 31]; (lobes of) chaotic attractors as richly structured object and percept
representations [109, 7].
Attractors, by deﬁnition, keep the system trajectory conﬁned within them.
Since clearly cognitive processes do not become ultimately trapped in attractors,
it has been a long-standing modeling challenge to account for “attractors that
can be left again” – that is, to partly disengage from a strict DISA paradigm.
Many answers have been proposed. Neural noise is a plausible agent to “kick”
a trajectory out of an attractor, but a problem with noise is its unspeciﬁcity
which is not easily reconciled with systematic information processing. A number
88
of alternative “attractor-like” phenomena have been considered that may arise in
high-dimensional nonlinear dynamics and oﬀer escapes from the trapping prob-
lem: saddle point dynamics or homoclinic cycles [87, 41]; chaotic itinerancy [103];
attractor relics, attractor ruins, or attractor ghosts [101]; transient attractors [53];
unstable attractors [102]; high-dimensional attractors (initially named partial at-
tractors) [70]; attractor landscapes [76].
All of these lines of work revolve around a fundamental conundrum: on the
one hand, neural representations of conceptual entities need to have some kind of
stability – this renders them identiﬁable, noise-robust, and temporally persistent
when needed. On the other hand, there must be cognitively meaningful mecha-
nisms for a fast switching between neural representational states or modes. This
riddle is not yet solved in a widely accepted way. Autoconceptive plane attractor
dynamics may lead to yet another answer. This kind of dynamics intrinsically
combines dynamical stability (in directions complementary to the plane of attrac-
tion) with dynamical neutrality (within the plane attractor).
However, in the
next section we will see that this picture, while giving a good approximation, is
too simple.
3.13.4
Analysis of Autoconceptor Adaptation Dynamics
Here I present a formal analysis of some asymptotic properties of the conceptor
adaptation dynamics.
Problem Statement
We consider the system of the coupled fast network state updates and slow con-
ceptor adaptation given by
z(n + 1) = C(n) tanh(W z(n))
(59)
and
C(n + 1)
=
C(n) + λ
 (z(n + 1) −C(n) z(n + 1)) z′(n + 1) −α−2 C(n)

=
C(n) + λ
 (I −C(n)) z(n + 1)z′(n + 1) −α−2 C(n),

(60)
where λ is a learning rate.
When λ is small enough, the instantaneous state
correlation z(n)z′(n) in (60) can be replaced by its expectation under C ﬁxed at
C(n), that is, we consider the dynamical system in time k
z(k + 1) = C(n) tanh(Wz(k))
and take the expectation of zz′ under this dynamics,
En[zz′]
:=
Ek[C(n) tanh(W z(k)) tanh(W z(k))′C(n)′]
=
C(n) Ek[tanh(W z(k)) tanh(W z(k))′]C(n)′
=:
C(n)Q(n)C(n)′,
89
where Q(n) is a positive semi-deﬁnite correlation matrix. Note that Q(n) is a
function of C and itself changes on the slow timescale of the C adaptation. For
further analysis it is convenient to change to continuous time and instead of (60)
consider
˙C(t) = (I −C(t)) C(t)Q(t)C′(t) −α−2C(t).
(61)
I now investigate the nature of potential ﬁxed point solutions under this dy-
namics.
If C is a ﬁxed point of this dynamics, Q(t) is constant.
In order to
investigate the nature of such ﬁxed point solutions, we analyse solutions in C for
the general ﬁxed point equation associated with (61), i.e. solutions in C of
0 = (I −C) CQC′ −α−2C,
(62)
where Q is some positive semideﬁnite matrix. We will denote the dimension of C
by N throughout the remainder of this section. Let V DV ′ = Q be the SVD of Q,
where D is a diagonal matrix containing the singular values of Q on its diagonal,
without loss of generality in descending order. Then (62) is equivalent to
0
=
V ′((I −C) CQC′ −α−2C)V
=
(I −V ′CV ) V ′CV D(V ′CV )′ −α−2 V ′CV.
(63)
We may therefore assume that Q is in descending diagonal form D, analyse
solutions of
0 = (I −C) CDC′ −α−2C,
(64)
and then transform these solutions C of (64) back to solutions of (62) by C →
V CV ′. In the remainder we will only consider solutions of (64). I will characterize
the ﬁxed points of this system and analyse their stability properties.
Characterizing the Fixed-point Solutions
The case α = 0.
In this degenerate case, neither the discrete-time update rule
(60) nor the dynamical equation (61) is well-deﬁned. The aperture cannot be set
to zero in practical applications where (60) is used for conceptor adaptation.
However, it is clear that (i) for any α > 0, C = 0 is a ﬁxed point solution of (61),
and that (ii) if we deﬁne B(α) = sup{∥C∥| C is a ﬁxed-point solution of (61)},
then limα→0 B(α) = 0. This justiﬁes to set, by convention, C = 0 as the unique
ﬁxed point of (61) in the case α = 0. In practical applications this could be im-
plemented by a reset mechanism: whenever α = 0 is set by some superordinate
control mechanism, the online adaptation (60) is over-ruled and C(n) is immedi-
ately set to 0.
The case α = ∞.
In the case α = ∞(i.e., α−2 = 0) our task is to characterize
the solutions C of
0 = (I −C)CDC′.
(65)
90
We ﬁrst assume that D has full rank.
Fix some k ≤N.
We proceed to
characterize rank-k solutions C of (65). CDC′ is positive semideﬁnite and has
rank k, thus it has an SVD CDC′ = UΣU ′ where Σ is diagonal nonnegative and
can be assumed to be in descending order, i.e. its diagonal is (σ1, . . . , σk, 0, . . . , 0)′
with σi > 0. Any solution C of (65) must satisfy UΣU ′ = CUΣU ′, or equivalently,
Σ = U ′CUΣ. It is easy to see that this entails that U ′CU is of the form
U ′CU =
 Ik×k
0
0
A

for some arbitrary (n−k)×(n−k) submatrix A. Requesting rank(C) = k implies
A = 0 and hence
C = U
 Ik×k
0
0
0

U ′.
(66)
Since conversely, if U is any orthonormal matrix, a matrix C of the form given
in (66) satisﬁes C2 = C, any such C solves (65). Therefore, the rank-k solutions
of (65) are exactly the matrices of type (66).
If D has rank l < N, again we ﬁx a desired rank k for solutions C. Again
let CDC′ = UΣU ′, with Σ in descending order. Σ has a rank m which satisﬁes
m ≤k, l. From considering Σ = U ′CUΣ it follows that U ′CU has the form
U ′CU =
 Im×m
0
0
A

for some (N −m) × (N −m) submatrix A. Since we prescribed C to have rank k,
the rank of A is k −m. Let U>m be the n × (N −m) submatrix of U made from
the columns with indices greater than m. We rearrange UΣU ′ = CDC′ to
Σ =
 Im×m
0
0
A

U ′ D U
 Im×m
0
0
A

,
from which it follows (since the diagonal of Σ is zero at positions greater than m)
that AU ′
>mDU>mA′ = 0. Since the diagonal of D is zero exactly on positions > l,
this is equivalent to
A (U(1 : l, m + 1 : N))′ = A U(m + 1 : N, 1 : l) = 0.
(67)
We now ﬁnd that (67) is already suﬃcient to make C = U (Im×m|0 / 0|A) U ′
solve (65), because a simple algebraic calculation yields
CD = CCD = U
 (U(1 : N, 1 : m))′
0

D.
We thus have determined the rank-k solutions of (65) to be all matrices of
form C = U (Im×m|0 / 0|A) U ′, subject to (i) m ≤l, k, (ii) rank(A) = k −m, (iii)
A (U(1 : l, m + 1 : N))′ = 0. Elementary considerations (omitted here) lead to the
following generative procedure to obtain all of these matrices:
91
1. Choose m satisfying l −N + k ≤m ≤k.
2. Choose a size N ×l matrix ˜U ′ made from orthonormal columns which is zero
in the last k −m rows (this is possible due to the choice of m).
3. Choose an arbitrary (N −m) × (N −m) matrix A of SVD form A = V ∆W ′
where the diagonal matrix ∆is in ascending order and is zero exactly on the
ﬁrst N −k diagonal positions (hence rank(A) = k −m).
4. Put
˜˜U ′ =
 Im×m
0
0
W

˜U ′.
This preserves orthonormality of columns, i.e. ˜˜U ′ is still made of orthonormal
columns. Furthermore, it holds that (0|A) ˜˜U ′ = 0.
5. Pad ˜˜U ′ by adding arbitrary N −l further orthonormal colums to the right,
obtaining an N × N orthonormal U ′.
6. We have now obtained a rank-k solution
C = U
 Im×m
0
0
A

U ′,
(68)
where we have put U to be the transpose of the matrix U ′ that was previously
constructed.
The case 0 < α < ∞.
We proceed under the assumption that α < ∞, that is,
∞> α−2 > 0.
I ﬁrst show that any solution C of (64) is a positive semideﬁnite matrix. The
matrix CDC′ is positive semideﬁnite and therefore has a SVD of the form UΣU ′ =
CDC′, where U is orthonormal and real and Σ is the diagonal matrix with the
singular values of CDC′ on its diagonal, without loss of generality in descending
order. From (I −C)UΣU ′ = α−2 C it follows that
UΣU ′
=
α−2 C + C UΣU ′ = C (α−2I + UΣU ′)
=
C (U(α−2I + Σ)U ′).
α−2I +Σ and hence U(α−2I +Σ)U ′ are nonsingular because α−2 > 0, therefore
C = UΣU ′(U(α−2I + Σ)U ′)−1 = UΣ(α−2 I + Σ)−1U ′ =: USU ′,
where S = Σ(α−2 I+Σ)−1 is a diagonal matrix, and in descending order since Σ was
in descending order. We therefore know that any solution C of (64) is of the form
C = USU ′, where U is the same as in CDC′ = UΣU ′. From S = Σ(α−2 I + Σ)−1
it furthermore follows that si < 1 for all singular values si of C, that is, C is a
conceptor matrix.
92
We now want to obtain a complete overview of all solutions C = USU ′ of
(64), expressed in terms of an orthonormal real matrix U and a nonnegative real
diagonal matrix S. This amounts to ﬁnding the solutions in S and U of
(S −S2)U ′DUS = α−2 S,
(69)
subject to S being nonnegative real diagonal and U being real orthonormal. With-
out loss of generality we furthermore may assume that the entries in S are in
descending order.
Some observations are immediate. First, the rank of S is bounded by the rank
of D, that is, the number of nonzero diagonal elements in S cannot exceed the
number of nonzero elements in D. Second, if U, S is a solution, and S∗is the
same as S except that some nonzero elements in S are nulled, then U, S∗is also
a solution (to see this, left-right multiply both sides of (69) with a thinned-out
identity matrix that has zeros on the diagonal positions which one wishes to null).
Fix some k ≤rank(D). We want to determine all rank-k solutions U, S, i.e.
where S has exactly k nonzero elements that appear in descending order in the
ﬁrst k diagonal positions. We write Sk to denote diagonal real matrices of size
k × k whose diagonal entries are all positive. Furthermore, we write Uk to denote
any N × k matrix whose columns are real orthonormal.
It is clear that if S, U solve (69) and rank(S) = k (and S is in descending
order), and if U ∗diﬀers from U only in the last N −k columns, then also S, U ∗
solve (69). Thus, if we have all solutions Sk, Uk of
(Sk −S2
k)U ′
kDUkSk = α−2 Sk,
(70)
then we get all rank-k solutions S, U to (64) by padding Sk with N −k zero
rows/columns, and extending Uk to full size N × n by appending any choice of
orthonormal columns from the orthogonal complement of Uk. We therefore only
have to characterize the solutions Sk, Uk of (70), or equivalently, of
U ′
kDUk = α−2 (Sk −S2
k)−1.
(71)
To ﬁnd such Sk, Uk, we ﬁrst consider solutions ˜Sk, Uk of
U ′
kDUk = ˜Sk,
(72)
subject to ˜Sk being diagonal with positive diagonal elements. For this we employ
the Cauchy interlacing theorem and its converse. I restate, in a simple special case
adapted to the needs at hand, this result from [27] where it is presented in greater
generality.
Theorem 1 (Adapted from Theorem 1 in [27], see remark of author at the end
of the proof of that theorem for a justiﬁcation of the version that I render here.)
Let A, B be two symmetric real matrices with dim(A) = n ≥k = dim(B), and
singular values σ1, . . . , σn and τ1, . . . , τk (in descending order). Then there exists a
real n×k matrix U with U ′U = Ik×k and U ′AU = B if and only if for j = 1, . . . , k
it holds that σi ≥τi ≥σn−k+i.
93
This theorem implies that if Uk, ˜Sk is any solution of (72), with Uk made of
k orthonormal columns and ˜Sk diagonal with diagonal elements ˜si (where j =
1, . . . , k, and the enumeration is in descending order), then the latter “interlace”
with the diagonal entries d1, . . . , dN of D per di ≥˜si ≥dN−k+i. And conversely,
any diagonal matrix ˜Sk, whose elements interlace with the diagonal elements of
D, appears in a solution Uk, ˜Sk of (72).
Equipped with this overview of solutions to (72), we revert from (72) to (71).
Solving ˜Sk = α−2 (Sk −S2
k)−1 for Sk we ﬁnd that the diagonal elements si of Sk
relate to the ˜si by
si = 1
2
 
1 ±
s
1 −4α−2
˜si
!
.
(73)
Since si must be positive real and smaller than 1, only such solutions ˜Sk to (72)
whose entries are all greater than 4α−2 yield admissible solutions to our original
problem (71). The interlacing condition then teaches us that the possible rank of
solutions C of (64) is bounded from above by the number of entries in D greater
than 4α−2.
For each value ˜si > 4α−2, (73) gives two solutions si,1 < 1/2 < si,2. We will
show further below that the solutions smaller than 1/2 are unstable while the
solutions greater than 1/2 are stable in a certain sense.
Summarizing and adding algorithmic detail, we obtain all rank-k solutions
C = USU ′ for (64) as follows:
1. Check whether D has at least k entries greater than 4α−2. If not, there are
no rank-k solutions. If yes, proceed.
2. Find a solution in Uk, ˜Sk of U ′
kDUk = ˜Sk, with ˜Sk being diagonal with
diagonal elements greater than 4α−2, and interlacing with the elements of
D. (Note: the proof of Theorem 1 in [27] is constructive and could be used
for ﬁnding Uk given ˜Sk.)
3. Compute Sk via (73), choosing between the ± options at will.
4. Pad Uk with any orthogonal complement and Sk with further zero rows and
columns to full n×n sized U, S, to ﬁnally obtain a rank-k solution C = USU ′
for (64).
Stability Analysis of Fixed-point Solutions
The case α = ∞.
Note again that α = ∞is the same as α−2 = 0. We consider
the time evolution of the quantity ∥I−C∥2 as C evolves under the zero-α−2 version
of (61):
˙C(t) = (I −C(t)) C(t)Q(t)C′(t).
(74)
94
We obtain
(∥I −C∥2)˙ =
trace((I −C)(I −C′))˙
=
trace((C −C2)QC′C′ + CCQ(C′ −C′2) −(C −C2)QC′ −CQ(C′ −C′2))
=
2 trace((C −C2)Q(C′2 −C′))
=
−2 trace((C −C2)Q(C′ −C′2)) ≤0,
(75)
where in the last line we use that the trace of a positive semideﬁnite matrix is
nonnegative. This ﬁnding instructs us that no other than the identity C = I
can be a stable solution of (74), in the sense that all eigenvalues of the associated
Jacobian are negative. If Q(t) has full rank for all t, then indeed this is the case (it
is easy to show that ∥I −C(t)∥2 is strictly decreasing, hence a Lyapunov function
in a neighborhood of C = I).
The stability characteristics of other (not full-rank) ﬁxed points of (74) are in-
tricate. If one computes the eigenvalues of the Jacobian at rank-k ﬁxed points C
(i.e. solutions of sort C = U(Ik×k|0 / 0|0)U ′, see (66)), where k < N, one ﬁnds neg-
ative values and zeros, but no positive values. (The computation of the Jacobian
follows the pattern of the Jacobian for the case α < ∞, see below, but is simpler;
it is omitted here). Some of the zeros correspond to perturbation directions of C
which change only the coordinate transforming matrices U. These perturbations
are neutrally stable in the sense of leading from one ﬁxed point solution to another
one, and satisfy C + ∆= (C + ∆)2. However, other perturbations C + ∆with the
property that C+∆̸= (C+∆)2 lead to (∥I −C∥2)˙ < 0. After such a perturbation,
the matrix C + ∆will evolve toward I in the Frobenius norm. Since the Jacobian
of C has no positive eigenvalues, this instability is non-hyperbolic. In simulations
one accordingly ﬁnds that after a small perturbation ∆is added, the divergence
away from C is initially extremely slow, and prone to be numerically misjudged
to be zero.
For rank-deﬁcient Q, which leads to ﬁxed points of sort C = U(Im×m|0 / 0|A)U ′,
the computation of Jacobians becomes involved (mainly because A may be non-
symmetric) and I did not construct them. In our context, where Q derives from a
random RNN, Q can be expected to have full rank, so a detailed investigation of
the rank-deﬁcient case would be an academic exercise.
The case 0 < α < ∞.
This is the case of greatest practical relevance, and I
spent a considerable eﬀort on elucidating it.
Note that α < ∞is equivalent to α−2 > 0. Let C0 = USU ′ be a rank-k
ﬁxed point of ˙C = (I −C)CDC′ −α−2C, where α−2 > 0, USU ′ is the SVD of
C and without loss of generality the singular values s1, . . . , sN in the diagonal
matrix S are in descending order, with s1, . . . , sk > 0 and si = 0 for i > k (where
1 ≤k ≤N). In order to understand the stability properties of the dynamics ˙C in
a neighborhood of C0, we compute the eigenvalues of the Jacobian JC = ∂˙C/∂C
at point C0. Notice that C is an N × N matrix whose entries must be rearranged
95
into a vector of size N 2 × 1 in order to arrive at the customary representation of a
Jacobian. JC is thus an N 2 × N 2 matrix which should be more correctly written
as JC(µ, ν) = ∂vec ˙C(µ)/∂vec C(ν), where vec is the rearrangement operator
(1 ≤µ, ν ≤N 2 are the indices of the matrix JC). Details are given in Section 5.10
within the proof of the following central proposition:
Proposition 16 The Jacobian JC(µ, ν) = ∂vec ˙C(µ)/∂vec C(ν) of a rank-k ﬁxed
point of (61) has the following multiset of eigenvalues:
1. k(N −k) instances of 0,
2. N(N −k) instances of −α−2,
3. k eigenvalues α−2(1 −2sl)/(1 −sl), where l = 1, . . . , k,
4. k(k −1) eigenvalues which come in pairs of the form
λ1,2 = α−2
2


sl
sm −1 +
sm
sl −1 ±
s
sl
sm −1 −
sm
sl −1
2
+ 4

,
where m < l < k.
An inspection of sort 3. eigenvalues reveals that whenever one of the sl is
smaller than 1/2, this eigenvalue is positive and hence the ﬁxed point C0 is unsta-
ble.
If some sl is exactly equal to 1/2, one obtains additional zero eigenvalues by
3. I will exclude such cases in the following discussion, considering them to be
non-generic.
If all sl are greater than 1/2, it is straightforward to show that the values of
sorts 3. and 4. are negative. Altogether, JP thus has k(N −k) times the eigenvalue
0 and otherwise negative ones. I will call such solutions 1/2-generic. All solutions
that one will eﬀectively obtain when conceptor auto-adaptation converges are of
this kind.
This characterization of the eigenvalue spectrum of 1/2-generic solutions does
not yet allow us to draw ﬁrm conclusions about how such a solution will react
to perturbations. There are two reasons why Proposition 16 aﬀords but a partial
insight in the stability of 1/2-generic solutions. (A) The directions connected to
zero eigenvalues span a k(N −k)-dimensional center manifold whose dynamics
remains un-analysed. It may be stable, unstable, or neutral. (B) When a 1/2-
generic solution is perturbed, the matrix D which reﬂects the conceptor-reservoir
interaction will change: D is in fact a function of C and should be more correctly
written D = D(C). In our linearization around ﬁxed point solutions we implicitly
considered D to be constant. It is unclear whether a full treatment using D =
D(C) would lead to a diﬀerent qualitative picture. Furthermore, (A) and (B) are
liable to combine their eﬀects. This is especially relevant for the dynamics on the
96
center manifold, because its qualitative dynamics is determined by components
from higher-order approximations to (61) which are more susceptible to become
qualitatively changed by non-constant D than the dynamical components of (61)
orthogonal to the center manifold.
Taking (A) and (B) into account, I now outline a hypothetical picture of the
dynamics of (61) in the vicinity of 1/2-generic ﬁxed-point solutions. This picture
is based only on plausibility considerations, but it is in agreement with what I
observe in simulations.
First, a dimensional argument sheds more light on the nature of the dynamics in
the k(N −k)-dimensional center manifold. Consider a 1/2-generic rank-k solution
C = USU ′ of (69). Recall that the singular values si in S were derived from ˜si
which interlace with the diagonal elements d1, . . . , dn of D by di ≥˜si ≥dN−k+i,
and where U ′
kDUk = ˜Sk (Equation (72)).
I call C a 1/2&interlacing-generic
solution if the interlacing is proper, i.e. if di > ˜si > dN−k+i. Assume furthermore
that D(C) is constant in a neighborhood of C. In this case, diﬀerential changes to
Uk in (72) will lead to diﬀerential changes in ˜Sk. If these changes to Uk respect the
conditions (i) that Uk remains orthonormal and (ii) that ˜Sk remains diagonal, the
changes to Uk lead to new ﬁxed point solutions. The ﬁrst constraint (i) allows us
to change Uk with (N −1) + (N −2) + . . . + (N −k) = kN −k(k + 1)/2 degrees of
freedom. The second constraint (ii) reduces this by (k −1)k/2 degrees of freedom.
Altogether we have kN −k(k+1)/2−(k−1)k/2 = k(N −k) diﬀerential directions
of change of C that lead to new ﬁxed points. This coincides with the dimension of
the center manifold associated with C. We can conclude that the center manifold
of a 1/2&interlacing-generic C extends exactly in the directions of neighboring
ﬁxed point solutions. This picture is based however on the assumption of constant
D. If the dependency of D on C is included in the picture, we would not expect
to ﬁnd any other ﬁxed point solutions at all in a small enough neighborhood of
C. Generically, ﬁxed point solutions of an ODE are isolated. Therefore, in the
light of the considerations made so far, we would expect to ﬁnd isolated ﬁxed
point solutions C, corresponding to close approximations of stored patterns. In a
local vicinity of such solutions, the autoconceptor adaptation would presumably
progress on two timescales: a fast convergence toward the center manifold K
associated with the ﬁxed point C, superimposed on a slow convergence toward C
within K (Figure 33 A).
The situation becomes particularly interesting when many patterns from a d-
parametric class have been stored. When I ﬁrst discussed this situation (Section
3.13.3), I tentatively described the resulting autoadaptation dynamics as conver-
gence toward a plane attractor (Figure 32). However, plane attractors cannot be
expected to exist in generic dynamical systems. Taking into account what the
stability analysis above has revealed about center manifolds of ﬁxed points C, I
would like to propose the following picture as a working hypothesis for the geom-
etry of conceptor adaptation dynamics that arises when a d-parametric pattern
class has been stored by overloading:
97
! 
" 
" 
A 
B 
Figure 33: Hypothetical phase portraits of C autoadaptation in the parameter
space of C (schematic). Blue points show stable ﬁxed point solutions C. Gray
plane represents the merged center manifold K. Green arrows represent sample
trajectories of C adaptation.
A. When a small number of patterns has been
loaded, individual stable ﬁxed point conceptors C are created. B. In the case
of learning a d-parametric pattern class, ﬁxed point solutions Ci become located
within a d-dimensional pattern manifold M (bold magenta line). For explanation
see text.
• The storing procedure leads to a number of stable ﬁxed point solutions Ci
for the autoconceptor adaptation (blue dots in Figure 33 B). These Ci are
associated with patterns from the pattern family, but need not coincide with
the sample patterns that were loaded.
• The k(N −k)-dimensional center manifolds of the Ci merge into a com-
prehensive manifold K of the same dimension.
In the vicinity of K, the
autoadaptive C evolution leads to a convergence toward K.
• Within K a d-dimensional submanifold M is embedded, representing the
learnt class of patterns. Notice that we would typically expect d << k(N−k)
(examples in the previous section had d = 2 or d = 3, but k(N −k) in
the order of several 100). Conceptor matrices located on M correspond to
patterns from the learnt class.
• The convergence of C adaptation trajectories toward K is superimposed with
a slower contractive dynamics within K toward the class submanifold M.
• The combined eﬀects of the attraction toward K and furthermore toward M
appear in simulations as if M were acting as a plane attractor.
• On an even slower timescale, within M there is an attraction toward the
isolated ﬁxed point solutions Ci. This timescale is so slow that the motion
within M toward the ﬁxed points Ci will be hardly observed in simulations.
98
In order to corroborate this reﬁned picture, and especially to conﬁrm the last
point from the list above, I carried out a long-duration content-addressable mem-
ory simulation along the lines described in Section 3.13.3. Ten 5-periodic patterns
were loaded into a small (50 units) reservoir. These patterns represented ten stages
of a linear morph between two similar patterns p1 and p10, resulting in a morph
sequence p1, p2, . . . , p10 where pi = (1 −(i −1)/9) p1 + ((i −1)/9) p10, thus rep-
resenting instances from a 1-parametric family. Considering what was found in
Section 3.13.3, loading these ten patterns should enable the system to re-generate
by auto-adaptation any linear morph ptest between p1 and p10 after being cued
with ptest.
Figure 34: Numerical exploration of ﬁxed point solutions under C auto-adaptation.
Each panel shows pairwise distances of 20 conceptor matrices obtained after n
auto-adaptation steps, after being cued along a 20-step morph sequence of cue
signals. Color coding: blue – zero distance; red – maximum distance. For expla-
nation see text.
After loading, the system was cued with 20 diﬀerent cues. In each of these
j = 1, . . . , 20 conditions, the cueing pattern pj
test was the j-th linear interpolation
between the stored p1 and p10. The cueing was done for 20 steps, following the
procedure given at the beginning of Section 3.13.3. At the end of the cueing, the
system will be securely driven into a state z that is very accurately connected to
re-generating the pattern pj
test, and the conceptor matrix that has developed by
the end of the cueing would enable the system to re-generate a close simile of pj
test
(a post-cue log10 NRMSE of about −2.7 was obtained in this simulation).
After cueing, the system was left running in conceptor auto-adaptation mode
using (57) for 1 Mio timesteps, with an adaptation rate of λ = 0.01.
At times n = 1, 1000, 10000, 1e6 the situation of convergence was assessed
as follows.
The pairwise distances between the current twenty autoconceptors
Cj(n) were compared, resulting in a 20 × 20 distance matrix D(n) = (∥Ck(n) −
Cl(n)∥fro)k,l=1,...,20. Figure 34 shows color plots of these distance matrices. The
outcome: at the beginning of autoadaptation (n = 1), the 20 autoconceptors are
spaced almost equally widely from each other. In terms of the schematic in Figure
33 B, they would all be almost equi-distantly lined up close to M. Then, as the
adaptation time n grows, they contract toward three point attractors within M
(which would correspond to a version of 33 B with three blue dots). These three
99
point attractors correspond to the three dark blue squares on the diagonal of the
last distance matrix shown in Figure 34.
This singular simulation cannot, of course, provide conclusive evidence that
the qualitative picture proposed in Figure 33 is correct. A rigorous mathematical
characterization of the hypothetical manifold M and its relation to the center
manifolds of ﬁxed point solutions of the adaptation dynamics needs to be worked
out.
Plane attractors have been proposed as models for a number of biological neural
adaptation processes (summarized in [24]). A classical example is gaze direction
control. The fact that animals can ﬁx their gaze in arbitrary (continuously many)
directions has been modelled by plane attractors in the oculomotoric neural control
system. Each gaze direction corresponds to a (controlled) constant neural activa-
tion proﬁle. In contrast to and beyond such models, conceptor auto-adaptation
organized along a manifold M leads not to a continuum of constant neural ac-
tivity proﬁles, but explains how a continuum of dynamical patterns connected by
continuous morphs can be generated and controlled.
In sum, the ﬁrst steps toward an analysis of autoconceptor adaptation have
revealed that this adaptation dynamics is more involved than either the classical
ﬁxed-point dynamics in autoassociative memories or the plane attractor models
suggested in computational neuroscience. For small numbers of stored patterns,
the picture bears some analogies with autoassociative memories in that stable ﬁxed
points of the autonomous adaptation correspond to stored patterns. For larger
numbers of stored patterns (class learning), the plane attractor metaphor captures
essential aspects of phenomena seen in simulations of not too long duration.
3.14
Toward Biologically Plausible Neural Circuits: Ran-
dom Feature Conceptors
The autoconceptive update equations
z(n + 1)
=
C(n) tanh(Wz(n) + Dz(n) + b)
C(n + 1)
=
C(n) + λ (z(n) −C(n)z(n)) z′(n) −α−2C(n)
could hardly be realized in biological neural systems. One problem is that the
C update needs to evaluate C(n)z(n), but z(n) is not an input to C(n) in the
z update but the outcome of applying C. The input to C is instead the state
r(n) = tanh(Wz(n) + Dz(n) + b). In order to have both computations carried out
by the same C, it seems that biologically hardly feasible schemes of installing two
weight-sharing copies of C would be required. Another problem is that the update
of C is nonlocal: the information needed for updating a “synapse” Cij (that is, an
element of C) is not entirely contained in the presynaptic or postsynaptic signals
available at this synapse.
Here I propose an architecture which solves these problems, and which I think
has a natural biological “feel”.
The basic idea is to (i) randomly expand the
100
reservoir state r into a (much) higher-dimensional random feature space, (ii) carry
out the conceptor operations in that random feature space, but in a simpliﬁed
version that only uses scalar operations on individual state components, and (iii)
project the conceptor-modulated high-dimensional feature space state back to the
reservoir by another random projection. The reservoir-conceptor loop is replaced
by a two-stage loop, which ﬁrst leads from the reservoir to the feature space
(through connection weight vectors fi, collected column-wise in a random neural
projection matrix F), and then back to the reservoir through a likewise random
set of backprojection weights G (Figure 35 A). The reservoir-internal connection
weights W are replaced by the combination of F and G, and the original reservoir
state x known from the basic matrix conceptor framework is split into a reservoir
state vector r and a feature space state vector z with components zi = cif ′
ir. The
conception weights ci take over the role of conceptors. In full detail,
1. expand the N-dimensional reservoir state r = tanh(Wz + W inp + b) into
the M-dimensional random feature space by a random feature map F ′ =
(f1, . . . , fM)′ (a synaptic connection weight matrix of size M × N) by com-
puting the M-dimensional feature vector F ′ r,
2. multiply each of the M feature projections f ′
i r with an adaptive conception
weight ci to get a conceptor-weighted feature state z = diag(c) F ′ r, where
the conception vector c = (c1, . . . , cM)′ is made of the conception weights,
3. project z back to the reservoir by a random N × M backprojection matrix
˜G, closing the loop.
Since both W and ˜G are random, they can be joined in a single random map
G = W ˜G. This leads to the following consolidated state update cycle of a random
feature conception (RFC) architecture:
r(n + 1)
=
tanh(G z(n) + W inp(n) + b),
(76)
z(n + 1)
=
diag(c(n)) F ′ r(n + 1),
(77)
where r(n) ∈RN and z(n), c(n) ∈RM.
From a biological modeling perspective there exist a number of concrete can-
didate mechanisms by which the mathematical operation of multiplying-in the
conception weights could conceivably be realized. I will discuss these later and for
the time being remain on this abstract mathematical level of description.
The conception vector c(n) is adapted online and element-wise in a way that
is analog to the adaptation of matrix autoconceptors given in Deﬁnition 5. Per
each element ci of c, the adaptation aims at minimizing the objective function
E[(zi −cizi)2] + α−2c2
i ,
(78)
which leads to ﬁxed point solutions satisfying
ci = E[z2
i ](E[z2
i ] + α−2)−1
(79)
101
A
! 
r
! 
z
G 
F' 
! 
W in
! 
u
! 
"ci
reservoir 
feature space 
D 
! 
p
B
!!
"
!
!!
"
!
fi 
!i fi 
ci  fi 
Figure 35: An alternative conceptor architecture aiming at greater biological plau-
sibility. A Schematic of random feature space architecture. The reservoir state
r is projected by a feature map F ′ into a higher-dimensional feature space with
states z, from where it is back-projected by G into the reservoir. The conceptor
dynamics is realized by unit-wise multiplying conception weights ci into f ′
ir to
obtain the z state. The input unit u is fed by external input p or by learnt input
simulation weights D. B Basic idea (schematic). Black ellipse: reservoir state r
correlation matrix R. Magenta dumbbell: scaling sample points fi from the unit
sphere by their mean squared projection on reservoir states.
Green dumbbell:
feature vectors fi scaled by auto-adapted conception weights ci. Red ellipse: the
resulting virtual conceptor CF. For detail see text.
and a stochastic gradient descent online adaptation rule
ci(n + 1) = ci(n) + λi
 z2
i (n) −ci(n) z2
i (n) −α−2 ci(n)

,
(80)
where i = 1, . . . , M, λi is an adaptation rate, and zi is the i-the component of z.
In computer simulations one will implement this adaptation not element-wise but
in an obvious vectorized fashion.
If (80) converges, the converged ﬁxed point is either ci = 0, which always is a
possible and stable solution, or it is of the form
ci = 1/2 +
p
(α2φi −4)/4α2φi,
(81)
which is another possible stable solution provided that α2φi −4 > 0.
In this
formula, φi denotes the expectation φi = Er[(f ′
i r)2], the mean energy of the feature
signal f ′
i r. These possible values of stable solutions can be derived in a similar
way as was done for the singular values of autoconceptive matrix C in Section
3.13.4, but the derivation is by far simpler (because it can be done element-wise
for each ci and thus entails only scalars, not matrices) and is left as an exercise.
Like the singular values of stable autoconceptors C, the possible stable value range
for conception weights obtainable through (80) is thus {0} ∪(1/2, 1).
Some geometric properties of random feature conceptors are illustrated in Fig-
ure 35 B. The black ellipse represents the state correlation matrix R = E[rr′] of a
hypothetical 2-dimensional reservoir. The random feature vectors fi are assumed
102
to have unit norm in this schematic and therefore sample from the surface of the
unit sphere. The magenta-colored dumbbell-shaped surface represents the weigth-
ing of the random feature vectors fi by the mean energies φi = E[(f ′
i r)2] of the
feature signals f ′
i r. Under the autoconception adaptation they give rise to concep-
tion weights ci according to (81) (green dumbbell surface). For values α2 φi−4 < 0
one obtains ci = 0, which shows up in the illustration as the wedge-shaped inden-
tation in the green curve. The red ellipse renders the virtual conceptor CF (see
below) which results from the random feature conception weights.
Two properties of this RFC architecture are worth pointing out.
First, conceptor matrices C for an N-dimensional reservoir have N(N + 1)/2
degrees of freedom. If, using conception vectors c instead, one wishes to attain a
performance level of pattern reconstruction accuracy that is comparable to what
can be achieved with conceptor matrices C, one would expect that M should be
in the order of N(N + 1)/2. At any rate, this is an indication that M should be
signiﬁcantly larger than N. In the simulations below I used N = 100, M = 500,
which worked robustly well. In contrast, trying M = 100 (not documented), while
likewise yielding good accuracies, resulted in systems that were rather sensitive to
parameter settings.
Second, the individual adaptation rates λi can be chosen much larger than the
global adaptation rate λ used for matrix conceptors, without putting stability at
risk. The reason is that the original adpatation rate λ in the stochastic gradient
descent formula for matrix conceptors given in Deﬁnition 5 is constrained by the
highest local curvature in the gradient landscape, which leads to slow convergence
in the directions of lower curvature. This is a notorious general characteristic of
multidimensional gradient descent optimization, see for instance [28]. This prob-
lem becomes irrelevant for the individual ci updates in (80). In the simulations
presented below, I could safely select the λi as large as 0.5, whereas when I was
using the original conceptor matrix autoadaption rules, λ = 0.01 was often the
fastest rate possible. If adaptive individual adaptation rates λi would be imple-
mented (not explored), very fast convergence of (80) should become feasible.
Geometry of feature-based conceptors.
Before I report on simulation ex-
periments, it may be helpful to contrast geometrical properties of the RFC archi-
tecture and with the geometry of matrix autoconceptors.
For the sake of discussion, I split the backprojection N × M matrix G in (76)
into a product G = W F where the “virtual” reservoir weight matrix W := GF †
has size N×N. That is, I consider a system z(n+1) = F diag(c(n)) F ′ tanh(Wz(n))
equivalent to (76) and (77), where c is updated according to (80).
For the
sake of simplicity I omit input terms and bias in this discussion.
The map
F ◦diag(c) ◦F ′ : RN →RN then plugs into the place that the conceptor ma-
trix C held in the conceptor systems z(n + 1) = C tanh(Wz(n)) discussed in
previous sections. The question I want to explore is how F ◦diag(c)◦F ′ compares
to C in geometrical terms. A conceptor matrix C has an SVD C = USU ′, where
103
U is orthonormal. In order to make the two systems directly comparable, I assume
that all feature vectors fi in F have unit norm. Then CF = ∥F∥−2
2 F ◦diag(c) ◦F ′
is positive semideﬁnite with 2-norm less or equal to 1, in other words it is an N ×N
conceptor matrix.
Now furthermore assume that the adaptation (80) has converged. The adap-
tation loop (76, 77, 80 ) is then a stationary process and the expectations φi =
Er[(f ′
i r)2] are well-deﬁned. Note that these expectations can equivalently be writ-
ten as φi = f ′
i R fi, where R = E[rr′]. According to what I remarked earlier, after
convergence to a stable ﬁxed point solution we have, for all 1 ≤i ≤M,
ci ∈
 {0},
if α2φi −4 ≤0,
{0, 1/2 +
p
(α2φi −4)/4α2φi},
if α2φi −4 > 0.
(82)
Again for the sake of discussion I restrict my considerations to converged solu-
tions where all ci that can be nonzero (that is, α2φi −4 > 0) are indeed nonzero.
It would be desirable to have an analytical result which gives the SVD of the
N × N conceptor CF = ∥F∥−2
2 F ◦diag(c) ◦F ′ under these assumptions. Unfortu-
nately this analysis appears to be involved and at this point I cannot deliver it. In
order to still obtain some insight into the geometry of CF, I computed a number of
such matrices numerically and compared them to matrix-based autoconceptors C
that were derived from the same assumed stationary reservoir state process. The
outcome is displayed in Figure 36.
Concretely, these numerical investigations were set up as follows. The reser-
voir dimension was chosen as N = 2 to admit plotting. The number of features
was M = 200. The feature vectors fi were chosen as (cos(i 2 π/M), sin(i 2 π/M))′
(where i = 1, . . . , M), that is, the unit vector (1 0)′ rotated in increments of
(i/M) 2 π. This choice mirrors a situation where a very large number of fi would
be randomly sampled; this would likewise result in an essentially uniform coverage
of the unit circle. The conception weights ci (and hence CF) are determined by the
reservoir state correlation matrix R = E[rr′]. The same holds for autoconceptor
matrices C. For an exploration of the CF versus C geometries, I thus systemat-
ically varied R = UΣU ′. The principal directions U were randomly chosen and
remained the same through all variations. The singular values Σ = diag(σ1 σ2)
were chosen as σ1 ≡10, σ2 ∈{0, 1, 5}, which gave three versions of R. The aper-
ture α was selected in three variants as α ∈{1, 2, 3}, which altogether resulted in
nine (R, α) combinations.
For each of these combinations, conception weights ci were computed via (82),
from which CF were obtained. Each of these maps the unit circle on an ellipse,
plotted in Figure 36 in red. The values of the ci are represented in the ﬁgure as
the dumbbell-shaped curve (green) connecting the vectors ci fi. The wedge-shaped
constriction to zero in some of these curves corresponds to angular values of fi
where ci = 0.
For comparison, for each of the same (R, α) combinations also an autocon-
ceptor matrix C was computed using the results from Section 3.13.4. We saw
104
!1
0
1
!2 = 0
aperture = 1
!2 = 1
!2 = 5
!1
0
1
aperture = 2
!1
0
1
!1
0
1
aperture = 3
!1
0
1
!1
0
1
Figure 36:
Comparing matrix-based autoconceptors (bold blue ellipses) with
feature-based autoconceptors CF (red ellipses). Broken lines mark principal di-
rections of the reservoir state correlation matrix R.
Each panel corresponds
to a particular combination of aperture and the second singular value σ2 of R.
The dumbbell-shaped surfaces (green line) represent the values of the conception
weights ci. For explanation see text.
on that occasion that nonzero singular values of C are not uniquely determined
by R; they are merely constrained by certain interlacing bounds. To break this
indeterminacy, I selected those C that had the maximal admissible singular val-
ues.
According to (73), this means that the singular values of C were set to
si = 1/2 +
p
(α2σi −4)/4α2σi (where i = 1, 2) provided the root argument was
positive, else si = 0.
Here are the main ﬁndings that can be collected from Figure 36:
1. The principal directions of CF, C and R coincide. The fact that CF and R
have the same orientation can also be shown analytically, but the argument
that I have found is (too) involved and not given here. This orientation of
CF hinges on the circumstance that the fi were chosen to uniformly sample
the unit sphere.
105
2. The CF ellipses are all non-degenerate, that is, CF has no zero singular
values (although many of the ci may be zero as becomes apparent in the
wedge constrictions in the dumbbell-shaped representation of these values).
In particular, the CF are also non-degenerate in cases where the matrix au-
toconceptors C are (panels in left column and center top panel). The ﬁnding
that CF has no zero singular values can be regarded as a disadvantage com-
pared to matrix autoconceptors, because it implies that no signal direction
in the N-dimensional reservoir signal space can be completely suppressed
by CF. However, in the M-dimensional feature signal space, we do have
nulled directions. Since the experiments reported below exhibit good sta-
bility properties in pattern reconstruction, it appears that this “purging” of
signals in the feature space segment of the complete reservoir-feature loop is
eﬀective enough.
3. Call the ratio of the largest over the smallest singular value of CF or C
the sharpness of a conceptor (also known as eigenvalue spread in the signal
processing literature). Then sometimes CF is sharper than C, and sometimes
the reverse is true. If sharpness is considered a desirable feature of concepors
(which I think it often is), then there is no universal advantage of C over CF
or vice versa.
System initialization and loading patterns:
generic description.
Re-
turning from this inspection of geometrical properties to the system (76) – (80), I
proceed to describe the initial network creation and pattern loading procedure in
generic terms. Like with the matrix conceptor systems considered earlier in this
report, there are two variants which are the analogs of (i) recomputing the reser-
voir weight matrix W ∗, as in Section 3.3, as opposed to (ii) training an additional
input simulation matrix D, as in Section 3.11. In the basic experiments reported
below I found that both work equally well. Here I document the second option.
A readout weight vector W out is likewise computed during loading. Let K target
patterns pj be given (j = 1, . . . , K), which are to be loaded. Here is an outline:
Network creation. A random feature map F, random input weights W in, a ran-
dom bias vector b, and a random backprojection matrix G∗are generated. F
and G∗are suitably scaled such that the combined N ×N map G∗F ′ attains
a prescribed spectral radius. This spectral radius is a crucial system param-
eter and plays the same role as the spectral radius in reservoir computing in
general (see for instance [106, 110]). All conception weights are initialized
to cj
i = 1, that is, diag(cj) = IM×M.
Conception weight adaptation. The system is driven with each pattern pj in
turn for nadapt steps (discarding an initial washout), while cj is being adapted
106
per
zj(n + 1)
=
diag(cj(n)) F ′ tanh(G∗zj(n) + W inpj(n) + b),
cj
i(n + 1)
=
cj
i(n) + λi
 zj
i (n)2 −cj
i(n) zj
i (n)2 −α−2 cj
i(n)

,
leading to conception vectors cj at the end of this period.
State harvesting for computing D and W out, and for recomputing G. The
conception vectors cj obtained from the previous step are kept ﬁxed, and for
each pattern pj the input-driven system rj(n) = tanh(G∗zj(n)+W inpj(n)+
b); zj(n+1) = diag(cj) F ′ rj(n) is run for nharvest time steps, collecting states
rj(n) and zj(n).
Computing weights. The N × M input simulation matrix D is computed by
solving the regularized linear regression
D = argmin ˜D
X
n,j
∥W inpj(n) −˜Dzj(n −1)∥2 + β2
D ∥˜D∥2
fro
(83)
where βD is a suitably chosen Tychonov regularizer. This means that the au-
tonomous system update zj(n+1) = diag(cj) F ′ tanh(G∗zj(n)+W in D zj(n)+
b) should be able to simulate input-driven updates zj(n) =
diag(cj) F ′ tanh(G∗zj(n) + W inpj(n) + b). W out is similarly computed by
solving
W out = argmin ˜
W
X
n,j
∥pj(n) −˜Wrj(n)∥2 + β2
Wout ∥˜W∥2.
Optionally one may also recompute G∗by solving the trivial regularized
linear regression
G = argmin ˜G
X
n,j
∥G∗zj(n) −˜Gzj(n)∥2 + β2
G ∥˜G∥2
fro.
for a suitably chosen Tychonov regularizer βG.
While G∗and G should
behave virtually identically on the training inputs, the average absolute size
of entries in G will be (typically much) smaller than the original weights in
G∗as a result of the regularization. Such regularized auto-adaptations have
been found to be beneﬁcial in pattern-generating recurrent neural networks
[90], and in the experiments to be reported presently I took advantage of
this scheme.
The feature vectors fi that make up F can optionally be normalized such that
they all have the same norm. In my experiments this was not found to have a
noticeable eﬀect.
107
If a stored pattern pj is to be retrieved, the only item that needs to be changed
is the conception vector cj. This vector can either be obtained by re-activating
that cj which was adapted during the loading (which implies that it needs to
be stored in some way).
Alternatively, it can be obtained by autoadaptation
without being previously stored, as in Sections 3.13.1 – 3.13.4. I now describe
two simulation studies which demonstrate how this scheme functions (simulation
detail documented in Section 4.6). The ﬁrst study uses stored conception vectors,
the second demonstrates autoconceptive adaptation.
Example 1: pattern retrieval with stored conception vectors cj.
This
simulation re-used the N = 100 reservoir from Sections 3.2 ﬀ. and the four driver
patterns (two irrational-period sines, two very similar 5-periodic random patterns).
The results are displayed in Figure 37.
Figure 37: Using stored random feature coded conceptors in a replication of the
basic pattern retrieval experiment from Section 3.4, with M = 500 random feature
vectors fi. First column: sorted conception vectors cj. Second column: spectra
of virtual conceptors CF. Third column: reconstructed patterns (bold light gray)
and original patterns (thin black) after phase alignment. NRMSEs are given in
insets. Last column: The adaptation of cj during the 2000 step runs carried out
in parallel to the loading process. 50 of 500 traces are shown. For explanation see
text.
The loading procedure followed the generic scheme described above (details
in Section 4), with nadapt = 2000, nharvest = 400, λi = 0.5, β2
G = β2
D = 0.01 and
β2
Wout = 1. The aperture was set to α = 8. The left column in Figure 37 shows
the resulting cj spectra, and the right column shows the evolution of cj during this
adaptation. Notice that a considerable portion of the conception weights evolved
toward zero, and that none ended in the range (0 1/2), in agreement with theory.
108
For additional insight into the dynamics of this system I also computed “vir-
tual” matrix conceptors Cj
F by Rj = E[(rj)′rj], Cj
F = Rj (Rj + α−2)−1 (second
column). The singular value spectrum of Cj
F reveals that the autocorrelation spec-
tra of rj signals in RFC systems is almost identical to the singular value spectra
obtained with matrix conceptors on earlier occasions (compare Figure 14).
The settings of matrix scalings and aperture were quite robust; variations in a
range of about ±50% about the chosen values preserved stability and accuracy of
pattern recall (detail in Section 4.6).
For testing the recall of pattern pj, the loaded system was run using the update
routine
rj(n)
=
tanh(G zj(n) + W in D zj(n) + b),
yj(n)
=
W out rj(n),
zj(n + 1)
=
diag(cj) F ′ rj(n),
starting from a random starting state zj(0) which was sampled from the normal
distribution, scaled by 1/2. After a washout of 200 steps, the reconstructed pattern
yj was recorded for 500 steps and compared to a 20-step segment of the target
pattern pj. The second column in Figure 37 shows an overlay of yj with pj and
gives the NRMSEs. The reconstruction is of a similar quality as was found in
Section 3.4 where full conceptor matrices C were used.
Example 2: content-addressed pattern retrieval.
For a demonstration of
content-addressed recall similar to the studies reported in Section 3.13.3, I re-
used the M = 500 system described above. Reservoir scaling parameters and the
loading procedure were identical except that conception vectors cj were not stored.
Results are collected in Figure 38. The cue and recall procedure for a pattern pj
was carried out as follows:
1. Starting from a random reservoir state, the loaded reservoir was driven with
the cue pattern for a washout time of 200 steps by zj(n+1) = F ′ tanh(G zj(n)+
W in pj(n) + b).
2. Then, for a cue period of 800 steps, the system was updated with cj adap-
tation by
zj(n + 1)
=
diag(cj(n)) F ′ tanh(G zj(n) + W in pj(n) + b),
cj
i(n + 1)
=
cj
i(n) + λi
 zj
i (n)2 −cj
i(n) zj
i (n)2 −α−2 cj
i(n)

, (1 ≤i ≤M)
starting from an all-ones cj, with an adaptation rate λi = 0.5 for all i. At
the end of this period, a conception vector cj,cue was obtained.
109
3. To measure the quality of cj,cue, a separate run of 500 steps without c adap-
ation was done using
rj(n)
=
tanh(G zj(n) + W in D zj(n) + b),
yj(n)
=
W out rj(n),
zj(n + 1)
=
diag(cj,cue) F ′ rj(n)
obtaining a pattern reconstruction yj(n). This was phase-aligned with the
original pattern pj and an NRMSE was computed (Figure 38, third column).
4. The recall run was resumed after the cueing period and continued for another
10,000 steps in auto-adaptation mode, using
zj(n + 1)
=
diag(cj(n)) F ′ tanh(G zj(n) + W in D zj(n) + b),
cj
i(n + 1)
=
cj
i(n) + λi
 zj
i (n)2 −cj
i(n) zj
i (n)2 −α−2 cj
i(n)

, (1 ≤i ≤K)
leading to a ﬁnal cj,adapted at the end of this period.
5. Another quality measurement run was done identical to the post-cue mea-
surement run, using cj,adapted (NRMSE results in Figure 38, third column).
Like in the matrix-C-based content-addressing experiments from Section 3.13.3,
the recall quality directly after the cue further improved during the autoconcep-
tive adaption afterwards, except for the ﬁrst pattern. Pending a more detailed
investigation, this may be attributed to the “maturation” of the cj during autoad-
aption which reveals itself in the convergence of a number of cj
i to zero during
autoadaptation (ﬁrst and last column in Figure 38). We have seen similar eﬀects
in Section 3.13.3.
An obvious diﬀerence to those earlier experiments is that the cueing period is
much longer now (800 versus 15 – 30 steps). This is owed to the circumstance that
now the conceptor adaptation during cueing started from an all-ones cj, whereas
in Section 3.13.3 it was started from a zero C. In the latter case, singular values
of C had to grow away from zero toward one during cueing, whereas here they
had to sink away from one toward zero. The eﬀects of this mirror situation are
not symmetrical. In an “immature” post-cue conceptor matrix C started from
a zero C, all the singular values which eventually should converge to zero are
already at zero at start time and remain there. Conversely, the post-cue feature
projection weights cj,cue, which should eventually become zero, have not come
close to this destination even after the 800 cue steps that were allotted here (left
panels in Figure 38). This tail of “immature” nonzero elements in cj,cue leads to
an insuﬃcient ﬁltering-out of reservoir state components which do not belong to
the target pattern dynamics.
The development of the cj
i during the autonomous post-cue adapation is not
monotonous (right panels). Some of these weights meander for a while before they
110
Figure 38: Content-addressed recall using RFC conceptors with M = 500 fea-
ture vectors fi. First column: sorted feature projection weight vectors cj after
the cue phase (black) and after 10,000 steps of autoadaptation (gray). Second
column: spectra of virtual conceptors CF after cue (black) and at the end of au-
tonomous adaptation (gray). Both spectra are almost identical. Third column:
reconstructed patterns (bold light gray: after cue, bold dark gray: after autoad-
aptation; the latter are mostly covered by the former) and original patterns (thin
black). NRMSEs are given in insets (top: after cue, bottom: after autoadapta-
tion). Fourth column: The adaptation of cj during the cueing period. Last column:
same, during the 10000 autoadaptation steps. 50 of 500 traces are shown. Note
the diﬀerent timescales in column 4 versus column 5. For explanation see text.
settle to what appear stable ﬁnal values. This is due to the transient nonlinear
reservoir–cj interactions which remain to be mathematically analyzed.
A potentially important advantage of using random feature conceptors c rather
than matrix conceptors C in machine learning applications is the faster conver-
gence of the former in online adaptation scenarios. While an dedicated comparison
of convergence properties between c and C conceptors remains to be done, one
may naturally expect that stochastic gradient descent works more eﬃciently for
random feature conceptors than for matrix conceptors, because the gradient can
be followed individually for each coordinate ci, unencumbered by the second-order
curvature interactions which notoriously slow down simple gradient descent in
multidimensional systems. This is one of the reasons why in the complex hierar-
chical signal ﬁltering architecture to be presented below in Section 3.15 I opted
for random feature conceptors.
Algebraic and logical rules for conception weights.
The various deﬁnitions
and rules for aperture adaptation, Boolean operations, and abstraction introduced
previously for matrix conceptors directly carry over to random feature conceptor.
111
The new deﬁnitions and rules are simpler than for conceptor matrices because they
all apply to the individual, scalar conception weights. I present these items without
detailed derivations (easy exercises). In the following, let c = (c1, . . . , cM)′, b =
(b1, . . . , bM)′ be two conception weight vectors.
Aperture adaptation (compare Deﬁnition 3) becomes
Deﬁnition 6
ϕ(ci, γ)
:=
ci/(ci + γ−2(1 −ci))
for 0 < γ < ∞,
ϕ(ci, 0)
:=
 0
if
ci < 1,
1
if
ci = 1,
ϕ(ci, ∞)
:=
 1
if
ci > 0,
0
if
ci = 0.
Transferring the matrix-based deﬁnition of Boolean operations (Deﬁnition 4)
to conception weight vectors leads to the following laws:
Deﬁnition 7
¬ ci
:=
1 −ci,
ci ∧bi
:=
 cibi/(ci + bi −cibi)
if not
ci = bi = 0,
0
if
ci = bi = 0,
ci ∨bi
:=
 (ci + bi −2cibi)/(1 −cibi)
if not
ci = bi = 1,
1
if
ci = bi = 1.
The matrix-conceptor properties connecting aperture adaptation with Boolen
operations (Proposition 10) and the logic laws (Propositions 11, 12) remain valid
after the obvious modiﬁcations of notation.
We deﬁne c ≤b if for all i = 1, . . . , M it holds that ci ≤bi. The main elements
of Proposition 14 turn into
Proposition 17 Let a = (a1, . . . , aM)′, b = (b1, . . . , bM)′ be conception weight
vectors. Then the following facts hold.
1. If b ≤a, then b = a ∧c, where c is the conception weight vector with entries
ci =
 0
if
bi = 0,
(b−1
i
−a−1
i
+ 1)−1
if
bi > 0.
2. If a ≤b, then b = a ∨c, where c is the conception weight vector with entries
ci =
 1
if
bi = 1,
1 −((1 −bi)−1 −(1 −ai)−1 + 1)−1
if
bi < 1.
3. If a ∧c = b, then b ≤a.
112
4. If a ∨c = b, then a ≤b.
Note that all of these deﬁnitions and rules can be considered as restrictions
of the matrix conceptor items on the special case of diagonal conceptor matrices.
The diagonal elements of such diagonal conceptor matrices can be identiﬁed with
conception weights.
Aspects of biological plausibility.
“Biological plausibility” is a vague term
inviting abuse.
Theoretical neuroscientists develop mathematical or computa-
tional models of neural systems which range from ﬁne-grained compartment mod-
els of single neurons to abstract ﬂowchart models of cognitive processes. Assessing
the methodological role of formal models in neuroscience is a complex and some-
times controversial issue [2, 36]. When I speak of biological plausibility in connec-
tion with conceptor models, I do not claim to oﬀer a blueprint that can be directly
mapped to biological systems. All that I want to achieve in this section is to show
that conceptor systems may be conceived which do not have characteristics that
are decidedly not biologically feasible. In particular, (all) I wanted is a conceptor
system variant which can be implemented without state memorizing or weight
copying, and which only needs locally available information for its computational
operations. In the remainder of this section I explain how these design goals may
be satisﬁed by RFC conceptor architectures.
I will discuss only the adaptation of conception weights and the learning of
the input simulation weights D, leaving cueing mechanisms aside.
The latter
was implemented in the second examples above in an ad-hoc way just to set the
stage and would require a separate treatment under the premises of biological
plausibility.
In my discussion I will continue to use the discrete-time update dynamics that
was used throughout this report. Biological systems are not updated according
to a globally clocked cycle, so this clearly departs from biology.
Yet, even a
synchronous-update discrete-time model can oﬀer relevant insight. The critical
issues that I want to illuminate – namely, no state/weight copying and locality of
computations – are independent of choosing a discrete or continuous time setting.
I ﬁrst consider the adaptation of the input simulation weights D. In situated,
life-long learning systems one may assume that at given point in (life-)time, some
version of D is already present and active, reﬂecting the system’s learning history
up to that point. In the content-addressable memory example above, at the end of
the cueing period there was an abrupt switch from driving the system with external
input to an autonomous dynamics using the system’s own input simulation via
D. Such binary instantaneous switching can hardly be expected from biological
systems.
It seems more adequate to consider a gradual blending between the
input-driven and the autonomous input simulation mode, as per
zj(n + 1) =
(84)
=
diag(cj(n)) F ′ tanh
 G zj(n) + W in  τ(n) D zj(n) + (1 −τ(n)) p(n)

+ b

,
113
where a mixing between the two modes is mediated by a “slide ruler” parameter
τ which may range between 0 and 1 (a blending of this kind will be used in the
architecture in Section 3.15).
As a side remark, I mention that when one considers comprehensive neural
architectures, the question of negotiating between an input-driven and an au-
tonomous processing mode arises quite generically. A point in case are “Bayesian
brain” models of pattern recognition and control, which currently receive much
attention [33, 16]. In those models, a neural processing layer is driven both from
“lower” (input-related) layers and from “higher” layers which autonomously gen-
erate predictions. Both inﬂuences are merged in the target layer by some neural
implementation of Bayes’ rule. Other approaches that I would like to point out in
this context are layered restricted Boltzmann machines [47], which likewise can be
regarded as a neural implementation of Bayes’ rule; hierarchical neural ﬁeld mod-
els of object recognition [111] which are based on Arathorn’s “map seeking circuit”
model of combining bottom-up and top-down inputs to a neural processing layer
[35]; and mixture of experts models for motor control (for example [107]) where a
“responsibility” signal comparable in its function to the τ parameter negotiates a
blending of diﬀerent control signals.
+ 
_ 
1 
1 
1 
1 
d 
d' 
p 
p' 
e 
!"
!' 
u 
W in 
D 
zi 
to reservoir 
from external 
from feature space 
Figure 39: An abstract circuit which would implement the τ negotiation between
driving a reservoir with external input p versus with simulated input Dz. Abstract
neurons are marked by ﬁlled gray circles. Connections that solely copy a neural
state forward are marked with “1”. Connections marked −• refer to multiplicative
modulation. Connections that are inhibitory by their nature are represented by
⊣. Broken arrows indicate a controlling inﬂuence on the weight adaptation of D.
For explanation see text.
Returning to (84), the mathematical formula could be implemented in an ab-
stract neural circuit as drawn in Figure 39. Explanation of this diagram: gray dots
represent abstract ﬁring-rate neurons (biologically realized by individual neurons
or collectives). All neurons are linear. Activation of neuron d: simulated input
d(n) = Dz(n); of p: external driver p(n). Neuron e maintains the value of the
114
“error” p(n) −d(n). d and p project their activation values to d′ and p′, whose
activations are multiplicatively modulated by the activations of neurons τ and τ ′.
The latter maintain the values of τ and τ ′ = 1 −τ from (84). The activations
d′(n) = τ(n) Dz(n) and p′(n) = (1 −τ(n)) p(n) are additively combined in u,
which ﬁnally feeds to the reservoir through W in.
For a multiplicative modulation of neuronal activity a number of biological
mechanisms have been proposed, for example [92, 14]. The abstract model given
here is not committed to a speciﬁc such mechanism. Likewise I do not further
specify the biological mechanism which balances between τ and τ ′, maintaining a
relationship τ ′ = 1 −τ; it seems natural to see this as a suitable version of mutual
inhibition.
An in-depth discussion by which mechanisms and for which purposes τ is ad-
ministered is beyond the scope of this report. Many scenarios are conceivable. For
the speciﬁc purpose of content-addressable memory recall, the setting considered
in this section, a natural option to regulate τ would be to identify it with the
(0-1-normalized and time-averaged) error signal e. In the architecture presented
in Section 3.15 below, regulating τ assumes a key role and will be guided by novel
principles.
The sole point that I want to make is that this abstract architecture (or similar
ones) requires only local information for the adaptation/learning of D. Consider
a synaptic connection Di from a feature neuron zi to d. The learning objective
(83) can be achieved, for instance, by the stochastic gradient descent mechanism
Di(n + 1) = Di(n) + λ
 (p(n) −Di(n)zi(n)) zi(n) −α−2 Di(n)

,
(85)
where the “error” p(n) −Di(n)zi(n) is available in the activity of the e neuron.
The learning rate λ could be ﬁxed, but a more suggestive option would be to
scale it by τ ′(n) = 1 −τ(n), as indicated in the diagram. That is, Di would be
adapted with an eﬃcacy proportional to the degree that the system is currently
being externally driven.
I now turn to the action and the adaptation of the conception weights, stated in
mathematical terms in equations (77) and (80). There are a number of possibilities
to implement these formulae in a model expressed on the level of abstract ﬁring-
rate neurons. I inspect three of them. They are sketched in Figure 40.
The simplest model (Figure 40 A) represents the quantity zi(n) = ci(n) f ′
i r(n)
by the activation of a single neuron ϕi. It receives synaptic input f ′
i r(n) through
connections fi and feeds to the reservoir (or to an input gating circuit as discussed
above) through the single synaptic connection Di. The weighting of f ′
i r(n) with
the factor ci is eﬀected by some self-regulated modulation of synaptic gain. Taking
into account that ci changes on a slower timescale than f ′
i r(n), the information
needed to adapt the strength ci of this modulation (80) is a moving average of the
neuron’s own activation energy z2
i (n) and the current synaptic gain ci(n), which
are characteristics of the neuron ϕi itself and thus are trivially locally available.
In the next model (Figure 40 B), there is a division of labor between a neuron
ϕi which again represents ci(n) f ′
i r(n) and a preceding neuron ζi which represents
115
Di 
A 
fi 
!i    
Di 
fi 
ci 
!i    
"i    
Di 
fi 
!i 
!i    
"i    
B 
C 
1 
1 
Figure 40: Three candidate neural implementations of conception weight mech-
anisms. In each diagram, ϕi is an abstract neuron whose activation is ϕ(n) =
zi(n) = ci(n) f ′
i r(n). In B and C, ζi has activation f ′
i r(n). In C, γi has activation
ci. For explanation see text.
f ′
i r(n).
The latter feeds into the former through a single synaptic connection
weighted by ci. The adaptation of the synaptic strength ci here is based on the
(locally time-averaged) squared activity of the postsynaptic neuron ϕi, which again
is information locally available at the synaptic link ci.
Finally, the most involved circuit oﬀered in Figure 40 C delegates the repre-
sentation of ci to a separate neuron γi. Like in the second model, a neuron ζi
which represents f ′
i r(n) feeds to ϕi, this time copying its own activation through
a unit connection. The γi neuron multiplicatively modulates ϕi by its activation
ci. Like in the D adaptation proposal described in Figure 39, I do not commit to
a speciﬁc biological mechanism for such a multiplicative modulation. The infor-
mation needed to adapt the activation ci of neuron γi according to (80) is, besides
ci itself, the quantity zi = ci(n) f ′
i r(n). The latter is represented in ϕi which is
postsynaptic from the perspective of γi and therefore not directly accessible. How-
ever, the input f ′
i r(n) from neuron ζi is available at γi, from which the quantity
zi = ci f ′
i r(n) can be inferred by neuron γi. The neuron γi thus needs to instan-
tiate an intricate activation dynamics which combines local temporal averaging
of (f ′
i r(n))2 with an execution of (80). A potential beneﬁt of this third neural
circuit over the preceding two is that a representation of ci by a neural activa-
tion can presumably be biologically adapted on a faster timescale than the neuron
auto-modulation in system A or the synaptic strength adaptation in B.
When I ﬁrst considered content-addressable memories in this report (Section
3.13.3), an important motivation for doing so was that storing entire conceptor
matrices C for later use in retrieval is hardly an option for biological systems. This
may be diﬀerent for conception vectors: it indeed becomes possible to “store” con-
ceptors without having to store network-sized objects. Staying with the notation
used in Figure 40: a single neuron γj might suﬃce to represent and “store” a
conception vector cj associated with a pattern pj. The neuron γj would project to
all ϕi neurons whose states correspond to the signals zi, with synaptic connection
weights cj
i, and eﬀecting a multiplicative modulation of the activation of the ϕi
neurons proportional to these connection weights. I am not in a position to judge
whether this is really an option in natural brains. For applications in machine
116
learning however, using stored conception vectors cj in conjunction with RFC sys-
tems may be a relevant alternative to using stored matrix conceptors, because
vectors cj can be stored much more cheaply in computer systems than matrices.
A speculative outlook. I allow myself to indulge in a brief speculation of how
RFC conceptor systems might come to the surface – literally – in mammalian
brains.
The idea is to interpret the activations of (groups of) neurons in the
neocortical sheet as representing conception factors ci or zi values, in one of the
versions shown in Figure 40 or some other concrete realization of RFC conceptors.
The “reservoir” part of RFC systems might be found in deeper brain structures.
When some patches of the neocortical sheet are activated and others not (revealed
for instance through fMRI imaging or electro-sensitive dyes), this may then be in-
terpreted as a speciﬁc cj vector being active. In geometrical terms, the surface of
the hyperellipsoid of the “virtual” conceptor would be mapped to the neocortical
sheet. Since this implies a reduction of dimension from a hypothetical reservoir
dimension N to the 2-dimensional cortical surface, a dimension folding as in self-
organizing feature maps [58, 77, 34] would be necessary. What a cognitive scientist
would call an “activation of a concept” would ﬁnd its neural expression in such an
activation of a dimensionally folded ellipsoid pertaining to a “virtual” conceptor
CF in the cortical sheet. An intriguing further step down speculation road is to
think about Boolean operations on concepts as being neurally realized through
the conceptor operations described in Sections 3.9 – 3.12. All of this is still too
vague. Still, some aspects of this picture have already been explored in some de-
tail in other contexts. Speciﬁcally, the series of neural models for processing serial
cognitive tasks in primate brains developed by Dominey et al. [19, 20] combine
a reservoir dynamics located in striatal nuclei with cortical context-providing ac-
tivation patterns which shares some characteristics with the speculations oﬀered
here.
3.15
A Hierarchical Filtering and Classiﬁcation Architec-
ture
A reservoir equipped with some conceptor mechanism does not by itself serve
a purpose.
If this computational-dynamical principle is to be made useful for
practical purposes like prediction, classiﬁcation, control or others, or if it is to
be used in cognitive systems modeling, conceptor-reservoir modules need to be
integrated into more comprehensive architectures. These architectures take care of
which data are fed to the conceptor modules, where their output is channelled, how
apertures are adapted, and everything else that is needed to manage a conceptor
module for the overall system purpose.
In this section I present a particular
architecture for the purpose of combined signal denoising and classiﬁcation as an
example. This (still simple) example introduces a number of features which may
be of more general use when conceptor modules are integrated into architectures:
Arranging conceptor systems in bidirectional hierarchies: a higher concep-
117
tor module is fed from a lower one by the output of the latter (bottom-up
data ﬂow), while at the same time the higher module co-determines the con-
ceptors associated with the lower one (top-down “conceptional bias” control).
Neural instantiations of individual conceptors: Using random feature con-
ceptors, it becomes possible to economically store and address individual
conceptors.
Self-regulating balance between perception and action modes: a conceptor-
reservoir module is made to run in any mixture of two fundamental process-
ing modes, (i) being passively driven by external input and (ii) actively gener-
ating an output pattern. The balance between these modes is autonomously
steered by a criterion that arises naturally in hierarchical conceptor systems.
A personal remark: the ﬁrst and last of these three items constituted the
original research questions which ultimately guided me to conceptors.
The task. The input to the system is a timeseries made of alternating sections
of the four patterns p1, . . . , p4 used variously before in this report: two sines of
irrational period lengths, and two slightly diﬀering 5-periodic patterns. This signal
is corrupted by strong Gaussian i.i.d. noise (signal-to-noise ratio = 0.5) – see
bottom panels in Fig. 42. The task is to classify which of the four patterns is
currently active in the input stream, and generate a clean version of it.
This
task is a simple instance of the generic task “classify and clean a signal that
intermittently comes from diﬀerent, but familiar, sources”.
Architecture. The basic idea is to stack copies of a reservoir-conceptor loop,
giving a hierarchy of such modules (compare Figure 41). Here I present an example
with three layers, having essentially identical modules M[1] on the lowest, M[2]
on the middle, and M[3] on the highest layer (I use subscript square brackets [l]
to denote levels in the hierarchy).
Each module is a reservoir-conceptor loop. The conceptor is implemented here
through the M-dimensional feature space expansion described in the previous
section, where a high-dimensional conception weight vector c is multiplied into the
feature state (as in Figure 35). At exploitation time the state update equations
are
u[l](n + 1)
=
(1 −τ[l−1,l](n)) y[l−1](n + 1) + τ[l−1,l](n) D z[l](n),
r[l](n + 1)
=
tanh(G z[l](n) + W in u[l](n + 1) + b),
z[l](n + 1)
=
c[l](n) .∗F ′r[l](n + 1),
y[l](n + 1)
=
W out r[l](n + 1),
where u[l] is the eﬀective signal input to module M[l], y[l] is the output signal from
that module, and the τ are mixing parameters which play a crucial role here and
will be detailed later. In addition to these fast timescale state updates there are
several online adaptation processes, to be described later, which adapt τ’s and c’s
118
!"#"!$%&!'
(")*+!"'#,)-"'
G 
F' 
Win 
z[1] 
u[1] 
p
.%&#/'&.,+*'
0%1+2"'
''''![1]'
Dz[2] 
r[1] 
&.&*&)22/'1".%&#"1''
,)**"!.'
G 
F' 
Win 
y[1] 
z[2] 
r[2] 
u[2] 
D 
0%1+2"'
'''![2]'
y[1] 
Dz[3] 
3%!"'1".%&#"1''
,)**"!.'
G 
F' 
Win 
y[2] 
z[3] 
r[3] 
u[3] 
D 
0%1+2"'
'''![3]'
y[3] 
(&.)2'1".%&#"1''
,)**"!.'
Wout 
Wout 
Wout 
c[2] 
![12] 
c[1] 
![23] 
c[3] 
y[2] 
![23] 
![12] 
c[1]
aut 
c[2]
aut 
c1, c2, c3, c4 
Figure 41: Schematic of 3-layer architecture for signal ﬁltering and classiﬁcation.
For explanation see text.
on slower timescales. The weight matrices D, G, F, W in, W out are identical in all
modules. F, W in are created randomly at design time and remain unchanged. D
and W out are trained on samples of clean “prototype” patterns in an initial pattern
loading procedure. G is ﬁrst created randomly as G∗and then is regularized using
white noise (all detail in Section 4.7).
The eﬀective input signal u[l](n + 1) to M[l] is thus a mixture mediated by
a “trust” variable τ[l−1,l] of a module-external external input y[l−1](n + 1) and a
module-internal input simulation signal D z[l](n). On the bottom layer, τ[01] ≡0
and y[0](n) = p(n), that is, this layer has no self-feedback input simulation and
is entirely driven by the external input signal p(n). Higher modules receive the
output y[l−1](n + 1) of the respective lower module as their external input. Both
input mix components y[l−1] and D z[l] represent partially denoised versions of the
external pattern input. The component y[l−1] from the module below will typically
be noisier than the component Dz[l] that is cycled back within the module, because
each module is supposed to de-noise the signal further in its internal reservoir-
conceptor feedback loop. If τ[l−1,l] were to be 1, the module would be running in
an autonomous pattern generation mode and would be expected to re-generate a
very clean version of a stored pattern – which might however be a wrong one. If
119
τ[l−1,l] were to be 0, the module would be running in an entirely externally driven
mode, with no “cleaning” in eﬀect. It is crucial for the success of this system that
these mixing weights τ[l−1,l] are appropriately set. They reﬂect a “trust” of the
system in its current hypothesis about the type of the driving pattern p, hence I
call them trust variables. I mention at this point that when the external input p
changes from one pattern type to another, the trust variables must quickly decrease
in order to temporarily admit the architecture to be in an altogether more input-
driven, and less self-generating, mode. All in all this constitutes a bottom-up ﬂow
of information, whereby the raw input p is cleaned stage-wise with an amount of
cleaning determined by the current trust of the system that it is applying the right
conceptor to eﬀect the cleaning.
The output signals y[l] of the three modules are computed from the reservoir
states r[l] by output weights W out, which are the same on all layers. These output
weights are initially trained in the standard supervised way of reservoir computing
to recover the input signal given to the reservoir from the reservoir state. The 3-rd
layer output y[3] also is the ultimate output of the entire architecture and should
give a largely denoised version of the external driver p.
Besides this bottom-up ﬂow of information there is a top-down ﬂow of infor-
mation. This top-down pathway aﬀects the conception weight vectors c[l] which
are applied in each module. The guiding idea here is that on the highest layer
(l = 3 in our example), the conceptor c[3] is of the form
c[3](n) =
_
j=1,...,4
ϕ(cj, γj(n)),
(86)
where c1, . . . , c4 are prototype conception weight vectors corresponding to the four
training patterns. These prototype vectors are computed and stored at training
time. In words, at the highest layer the conception weight vector is constrained to
be a disjunction of aperture-adapted versions of the prototype conception weight
vectors. Imposing this constraint on the highest layer can be regarded as inserting
a qualitative bias in the ensuing classiﬁcation and denoising process. Adapting
c[3](n) amounts to adjusting the aperture adaptation factors γj(n).
At any time during processing external input, the current composition of c[3] as
a γj-weighted disjunction of the four prototypes reﬂects the system’s current hy-
pothesis about the type of the current external input. This hypothesis is stagewise
passed downwards through the lower layers, again mediated by the trust variables.
This top-down pathway is realized as follows. Assume that c[3](n) has been
computed. In each of the two modules below (l = 1, 2), an (auto-)conception
weight vector caut
[l] (n) is computed by a module-internal execution of the standard
autoconception adaptation described in the previous section (Equation (80)). To
arrive at the eﬀective conception weight vector c[l](n), this caut
[l] (n) is then blended
with the current conception weight vector c[l+1](n) from the next higher layer,
again using the respective trust variable as mixing coeﬃcient:
c[l](n) = (1 −τ[l,l+1](n)) caut
[l] (n) + τ[l,l+1](n) c[l+1](n).
(87)
120
In the demo task reported here, the raw input p comes from either of four
sources p1, . . . , p4 (the familiar two sines and 5-periodic patterns), with additive
noise. These four patterns are initially stored in each of the modules M[l] by
training input simulation weights D, as described in Section 3.14. The same D is
used in all layers.
In the exploitation phase (after patterns have been loaded into D, output
weights have been learnt, and prototype conceptors cj have been learnt), the ar-
chitecture is driven with a long input sequence composed of intermittent periods
where the current input is chosen from the patterns pj in turn. While being driven
with pj, the system must autonomously determine which of the four stored pat-
tern is currently driving it, assign trusts to this judgement, and accordingly tune
the degree of how strongly the overall processing mode is autonomously generative
(high degree of cleaning, high danger of “hallucinating”) vs. passively input-driven
(weak cleaning, reliable coupling to external driver).
Summing up, the overall functioning of the trained architecture is governed by
two pathways of information ﬂow,
• a bottom-up pathway where the external noisy input p is successively de-
noised,
• a top-down pathway where hypotheses about the current pattern type, ex-
pressed in terms of conception weight vectors, are passed downwards,
and by two online adaptation processes,
• adjusting the trust variables τ[l−1,l], and
• adjusting the conception weight vector c[l] in the top module.
I now describe the two adaptation processes in more detail.
Adapting the trust variables. Before I enter technicalities I want to emphasize
that here we are confronted with a fundamental problem of information processing
in situated intelligent agents (“SIA”: animals, humans, robots). A SIA continu-
ously has to “make sense” of incoming sensor data, by matching them to the
agent’s learnt/stored concepts.
This is a multi-faceted task, which appears in
many diﬀerent instantiations which likely require specialized processing strate-
gies. Examples include online speech understanding, navigation, or visual scene
interpretation. For the sake of this discussion I will lump them all together and
call them “online data interpretation” (ODI) tasks. ODI tasks variously will in-
volve subtasks like de-noising, ﬁgure-ground separation, temporal segmentation,
attention control, or novelty detection. The demo architecture described in this
section only addresses de-noising and temporal segmentation. In many cognitive
architectures in the literature, ODI tasks are addressed by maintaining an online
representation of a “current best” interpretation of the input data. This repre-
sentation is generated in “higher” levels of a processing hierarchy and is used in a
top-down fashion to assist lower levels, for instance by way of providing statistical
121
bias or predictions (discussion in [16]). This top-down information then tunes the
processing in lower levels in some way that enables them to extract from their
respective bottom-up input speciﬁc features while suppressing others – generally
speaking, by making them selective. An inherent problem in such architectures
is that the agent must not grow overly conﬁdent in its top-down pre-conditioning
of lower processing layers. In the extreme case of relying entirely on the current
interpretation of data (instead of on the input data themselves), the SIA will be
hallucinating. Conversely, the SIA will perform poorly when it relies entirely on
the input data: then it will not “understand” much of it, becoming unfocussed and
overwhelmed by noise. A good example is semantic speech or text understanding.
Linguistic research has suggested that fast forward inferences are involved which
predispose the SIA to interpret next input in terms of a current representation
of semantic context (for instance, [96]). As long as this current interpretation is
appropriate, it enables fast semantic processing of new input; but when it is inap-
propriate, it sends the SIA on erroneous tracks which linguists call “garden paths”.
Generally speaking, for robust ODI an agent should maintain a reliable measure
of the degree of trust that the SIA has in its current high-level interpretation.
When the trust level is high, the SIA will heavily tune lower levels by higher-level
interpretations (top-down dominance), while when trust levels are low, it should
operate in a bottom-up dominated mode.
Maintaining an adaptive measure of trust is thus a crucial objective for an
SIA. In Bayesian architectures (including Kalman ﬁlter observers in control engi-
neering), such a measure of trust is directly provided by the posterior probability
p(interpretation | data). A drawback here is that a number of potentially complex
probability distributions have to be learnt beforehand and may need extensive
training data, especially when prior hyperdistributions have to be learnt instead
of being donated by an oracle. In mixture of predictive expert models (for instance
[107]), competing interpretation models are evaluated online in parallel and are
assigned relative trust levels according to how precisely they can predict the cur-
rent input. A problem that I see here is computational cost, besides the biological
implausibility of executing numerous predictors in parallel. In adaptive resonance
theory [43], the role of a trust measure is ﬁlled by the ratio between the norm of a
top-down pattern interpretation over the norm of an input pattern; the functional
eﬀects of this ratio for further processing depends on whether that ratio is less
or greater than a certain “vigilance” parameter. Adaptive resonance theory how-
ever is primarily a static pattern processing architecture not designed for online
processing of temporal data.
Returning to our demo architecture, here is how trust variables are computed.
They are based on auxiliary quantities δ[l](n) which are computed within each mod-
ule l. Intuitively, δ[l](n) measures the (temporally smoothed) discrepancy between
the external input signal fed to the module and the self-generated, conceptor-
cleaned version of it. For layers l > 1 the external input signal is the bottom-up
passed output y[l−1] of the lower layer. The conceptor-cleaned, module-generated
122
version is the signal Dz[l](n) extracted from the conception-weighted feature space
signal z[l](n) = c[l](n) .∗F ′r[l](n) by the input simulation weights D, where r[l](n)
is the reservoir state in layer l. Applying exponential smoothing with smoothing
rate σ < 1, and normalizing by the likewise smoothed variance of y[l−1], gives
update equations
¯y[l−1](n + 1)
=
σ ¯y[l−1](n) + (1 −σ) y[l−1](n + 1), (running average) (88)
var y[l−1](n + 1)
=
σ var y[l−1](n + 1) + (1 −σ) (y[l−1](n + 1) −¯y[l−1](n + 1))2, (89)
δ[l](n + 1)
=
σ δ[l](n) + (1 −σ) (y[l−1](n + 1) −Dz[l](n + 1))2
var y[l−1](n + 1)
(90)
for the module-internal detected discrepancies δ[l]. In the bottom module M[1],
the same procedure is applied to obtain δ[1] except that the module input is here
the external driver p(n) instead the output y[l−1](n) from the level below.
From these three discrepancy signals δ[l](n) two trust variables τ[12], τ[23] are
derived. The intended semantics of τ[l,l+1] can be stated as “measuring the degree
by which the discrepancy is reduced when going upwards from level l to level l+1”.
The rationale behind this is that when the currently active conception weights in
modules l and l + 1 are appropriate for the current drive entering module i from
below (or from the outside when l = 1), the discrepancy should decrease when
going from level l to level l+1, while if the the currently applied conception weights
are the wrong ones, the discrepancy should increase when going upwards. The
core of measuring trust is thus the diﬀerence δ[l](n) −δ[l+1](n), or rather (since we
want the same sensitivity across all levels of absolute values of δ) the diﬀerence
log(δ[l](n)) −log(δ[l+1](n)). Normalizing this to a range of (0, 1) by applying a
logistic sigmoid with steepness d[l,l+1] ﬁnally gives
τ[l,l+1](n) =
 
1 +
δ[l+1](n)
δ[l](n)
d[l,l+1]!−1
.
(91)
The steepness d[l,l+1] of the trust sigmoid is an important design parameter, which
currently I set manually. Stated in intuitive terms it determines how “decisively”
the system follows its own trust judgement. It could be rightfully called a “meta-
trust” variable, and should itself be adaptive. As will become clear in the demon-
strations below, large values of this decisiveness leads the system to make fast
decisions regarding the type of the current driving input, at an increased risk of set-
tling down prematurely on a wrong decision. Low values of d[l,l+1] allow the system
to take more time for making a decision, consolidating information acquired over
longer periods of possibly very noisy and only weakly pattern-characteristic input.
My current view on the regulation of decisiveness is that it cannot be regulated
on the sole basis of the information contained in input data, but reﬂects higher
cognitive capacities (connected to mental attitudes like “doubt”, “conﬁdence”, or
even “stubbornness”...) which are intrinsically not entirely data-dependent.
123
Adapting the top-level conception weight vectors c[l]. For clarity of notation
I will omit the level index [l] in what follows, assuming throughout l = 3. By
equation (86), the eﬀective conception weight vector used in the top module will
be constrained to be a disjunction c(n) = W
j=1,...,4 ϕ(cj, γj(n)), where the cj are
prototype conception weight vectors, computed at training time. Adapting c(n)
amounts to adjusting the apertures of the disjunctive components cj via γj(n).
This is done indirectly.
The training of the prototype conception weights (and of the input simulation
matrix D and of the readout weights W out) is done with a single module that is
driven by the clean patterns pj. Details of the training procedure are given in the
Experiments and Methods Section 4.7. The prototype conception weight vectors
can be written as
cj = E[(zj).∧2] .∗(E[(zj).∧2] + α−2).∧−1,
where zj(n) = cj .∗F ′ rj(n) is the M-dimensional signal fed back from the feature
space to the reservoir while the module is being driven with pattern j during
training, and the aperture α is a design parameter. Technically, we do not actually
store the cj but their constitutents α and the corresponding mean signal energy
vectors E[(zj).∧2], the latter of which are collected in an M × 4 prototype matrix
P = (E[(zj
i )2])i=1,...,M; j=1,...,4.
(92)
I return to the conceptor adaptation dynamics in the top module at exploitation
time. Using results from previous sections, equation (86) can be re-written as
c(n) =
 X
j
(γj(n)).∧2 .∗E[(zj).∧2]
!
.∗
 X
j
(γj(n)).∧2 .∗E[(zj).∧2] + α−2
!.∧−1
,
(93)
where the +α−2 operation is applied component-wise to its argument vector. The
strategy for adapting the factors γj(n) is to minimize the loss function
L{γ1, . . . , γ4} = ∥
X
j
(γj).∧2 E[(zj).∧2] −E[z.∧2]∥2,
(94)
where z is the feature space output signal z(n) = c(n) .∗F ′r(n) available during
exploitation time in the top module. In words, the adaptation of c aims at ﬁnding
a weighted disjunction of prototype vectors which optimally matches the currently
observed mean energies of the z signal.
It is straightforward to derive a stochastic gradient descent adaptation rule for
minimizing the loss (94). Let γ = (γ1, . . . , γ4) be the row vector made from the
γj, and let ·.2 denote element-wise squaring of a vector. Then
γ(n + 1) = γ(n) + λγ
 z(n + 1).∧2 −P (γ′(n)).∧2′ P diag(γ(n))
(95)
124
implements the stochastic gradient of L with respect to γ, where λγ is an adapta-
tion rate. In fact I do not use this formula as is, but add two helper mechanisms,
eﬀectively carrying out
γ∗(n + 1)
=
γ(n) +
λγ
 z(n + 1).∧2 −P (γ′(n)).∧2′ P diag(γ(n)) + d (1/2 −γ(n))

(96)
γ(n + 1)
=
γ∗(n + 1)/sum(γ∗(n + 1)).
(97)
The addition of the term d (1/2−γ(n)) pulls the γj away from the possible extremes
0 and 1 toward 1/2 with a drift force d, which is a design parameter. This is helpful
to escape from extreme values (notice that if γj(n) = 0, then γj would forever
remain trapped at that value in the absence of the drift force). The normalization
(97) to a unit sum of the γj greatly reduces adaptation jitter.
I found both
amendments crucial for a reliable performance of the γ adaptation.
Given the γ(n) vector, the top-module c(n) is obtained by
c(n) =
 P (γ′(n)).∧2
.∗
 P (γ′(n)).∧2 + α−2.∧−1 .
Simulation Experiment 1: Online Classiﬁcation and De-Noising. Please con-
sult Figure 42 for a graphical display of this experiment. Details (training, inital-
ization, parameter settings) are provided in the Experiments and Methods section
4.7.
The trained 3-layer architecture was driven with a 16,000 step input signal
composed of four blocks of 4000 steps each. In these blocks, the input was gen-
erated from the patterns p1, p3, p2, p4 in turn (black lines in bottom row of Figure
42), with additive Gaussian noise scaled such that a signal-to-noise ratio of 1/2
was obtained (red lines in bottom row of Figure 42). The 3-layer architecture was
run for the 16,000 steps without external intervention.
The evolution of the four γj weights in the top layer represent the system’s
classiﬁcation hypotheses concerning the type of the current driver (top row in
Figure 42). In all four blocks, the correct decision is reached after an initial “re-
thinking” episode. The trust variable τ[23] quickly approaches 1 after taking a drop
at the block beginnings (green line in fourth row of Figure). This drop allows the
system to partially pass through the external driver signal up to the top module,
de-stabilizing the hypothesis established in the preceding block. The trust variable
τ[12] (blue line in fourth row) oscillates more irregularly but also takes its steepest
drops at the block beginnings.
For diagnostic purposes, γj
[l] weights were also computed on the lower layers
l = 2, 3, using (96) and (97).
These quantities, which were not entering the
system’s processing, are indicators of the “classiﬁcation belief states” in lower
modules (second and third row in the Figure). A stagewise consolidation of these
hypotheses can be observed as one passes upwards through the layers.
The four patterns fall in two natural classes, “sinewaves” and “period-5”. In-
specting the top-layer γj it can be seen that within each block, the two hypothesis
125
0
0.5
1
0
0.5
1
0
0.5
1
0
0.5
1
0
4000
8000
12000
16000
!2
!1
0
0
10
20
!2
!1
0
1
2
0
10
20
0
10
20
0
10
20
Figure 42: Denoising and classiﬁcation of prototype patterns. Noisy patterns were
given as input in the order p1, p3, p2, p4 for 4000 timesteps each. Top row: evolution
of γ1 (blue), γ2 (green), γ3 (red), γ4 (cyan) in top layer module. Rows 2 and 3:
same in modules 2 and 1. Fourth row: trust variables τ[12] (blue) and τ[23] (green).
Fifth row: NRMSEs for reconstructed signals y[1] (blue), y[2] (green) and y[3] (red).
Black line shows the linear ﬁlter reference NRMSE. Thin red line: NRMSE of
phase-aligned y[3]. The plotting scale is logarithmic base 10. Bottom: pattern
reconstruction snapshots from the last 20 timesteps in each pattern presentation
block, showing the noisy input (red), the layer-3 output y[3] (thick gray) and the
clean signal (thin black). For explanation see text.
126
indicators associated with the “wrong” class are quickly suppressed to almost zero,
while the two indicators of the current driver’s class quickly dominate the picture
(summing to close to 1) but take a while to level out to their relative ﬁnal values.
Since the top-level c(n) is formally a γj-weighted disjunction of the four prototype
cj conception vectors, what happens in the ﬁrst block (for instance) can also be
rephrased as, “after the initial re-thinking, the system is conﬁdent that the current
driver is p1 OR p3, while it is quite sure that it is NOT (p2 OR p4)”. Another
way to look at the same phenomena is to say, “it is easy for the system to quickly
decide between the two classes, but it takes more time to distinguish the rather
similar patterns within each class”.
The ﬁfth row in Figure 42 shows the log10 NRMSE (running smoothed average)
of the three module outputs y[l](n) with respect to a clean version of the driver
(thick lines; blue = y[1], green = y[2], red = y[3]).
For the 5-periodic patterns
(blocks 2 and 4) there is a large increase in accuracy from y[1] to y[2] to y[3]. For
the sinewave patterns this is not the case, especially not in the ﬁrst block. The
reason is that the re-generated sines y[2] and y[3] are not perfectly phase-aligned
to the clean version of the driver. This has to be expected because such relative
phase shifts are typical for coupled oscillator systems; each module can be regarded
as an oscillator. After optimal phase-alignment (details in the Experiments and
Methods section), the top-level sine re-generation matches the clean driver very
accurately (thin red line). The 5-periodic signal behaves diﬀerently in this respect.
Mathematically, an 5-periodic discrete-time dynamical system attractor is not an
oscillation but a ﬁxed point of the 5-fold iterated map, not admitting anything
like a gradual phase shift.
As a baseline reference, I also trained a linear transversal ﬁlter (Wiener ﬁlter,
see [28] for a textbook treatment) on the task of predicting the next clean input
value (details in the Experiments and Methods section). The length of this ﬁlter
was 2600, matching the number of trained parameters in the conceptor architecture
(P has 500 ∗4 learnt parameters, D has 500, W out has 100). The smoothed log10
NRMSE of this linear predictor is plotted as a black line. It naturally can reach
its best prediction levels only after 2600 steps in each block, much more slowly
than the conceptor architecture. Furthermore, the ultimate accuracy is inferior
for all four patterns.
Simulation Experiment 2: Tracking Signal Morphs. Our architecture can be
characterized as a signal cleaning-and-interpretation systems which guides itselft
by allowing top-down hypotheses to make lower processing layers selective. An
inherent problem in such systems is that that they may erroneously lock themselves
on false hypotheses. Top-down hypotheses are self-reinforcing to a certain degree
because they cause lower layers to ﬁlter out data components that do not agree
with the hypothesis – which is the essence of de-noising after all.
In order to test how our architecture fares with respect to this “self-locking
fallacy”, I re-ran the simulation with an input sequence that was organized as a
linear morph from p1 to p2 in the ﬁrst 4000 steps (linearly ramping up the sine
127
frequency), then in the next block back to p1; this was followed by a morph from
p3 to p4 and back again. The task now is to keep track of the morph mixture in the
top-level γj. This is a greatly more diﬃcult task than the previous one because
the system does not have to just decide between 4 patterns, but has to keep track
of minute changes in relative mixtures. The signal-to-noise ratio of the external
input was kept at 0.5. The outcome reveals an interesting qualitative diﬀerence
in how the system copes with the sine morph as opposed to the 5-periodic morph.
As can be seen in Figure 43, the highest-layer hypothesis indicators γj can track
the frequency morph of the sines (albeit with a lag), but get caught in a constant
hypothesis for the 5-period morph.
This once again illustrates that irrational-period sines are treated qualitatively
diﬀerently from integer-periodic signals in conceptor systems. I cannot oﬀer a
mathematical analysis, only an intuition. In the sinewave tracking, the overall
architecture can be described as a chain of three coupled oscillators, where the
bottom oscillator is externally driven by a frequency-ramping sine. In such a driven
chain of coupled oscillators, one can expect either chaos, the occurrence of natural
harmonics, or frequency-locking across the elements of the chain.
Chaos and
harmonics are ruled out in our architecture because the prototype conceptors and
the loading of two related basic sines prevent it. Only frequency-locking remains
as an option, which is indeed what we ﬁnd. The 5-periodic morph cannot beneﬁt
from this oscillation entrainment. The minute diﬀerences in shape between the two
involved prototypes do not stand out strongly enough from the noise background
to induce a noticable decline in the trust variable τ[23]: once established, a single
hypothesis persists.
On a side note it is interesting to notice that the linear ﬁlter that was used
as a baseline cannot at all cope with the frequency sweep, but for the 5-periodic
morph it performs as well as in the previous simulation. Both eﬀects can easily
be deduced from the nature of such ﬁlters.
Only when I used a much cleaner input signal (signal-to-noise ratio of 10), and
after the decisiveness d was reduced to 0.5, it became possible for the system to
also track the 5-period pattern morph, albeit less precisely than it could track the
sines (not shown).
Variants and Extensions. When I ﬁrst experimented with architectures of the
kind proposed above, I computed the module-internal conception weight vectors
caut
[l]
(compare Equation (87)) on the two lower levels not via the autoconception
mechanism, but in a way that was similar to how I computed the top-level con-
ception weight vector c[3], that is, optimizing a ﬁt to a disjunction of the four
prototypes. Abstractly speaking, this meant that a powerful piece of prior infor-
mation, namely of knowing that the driver was one of the four prototypes, was
inserted in all processing layers. This led to a better system performance than
what I reported above (especially, faster decisions in the sense of faster conver-
gence of the γ[3]).
However I subsequently renounced this “trick” because the
diﬀerences in performance were only slight, and from a cognitive modelling per-
128
0
0.5
1
0
0.5
1
0
0.5
1
0
0.5
1
0
4000
8000
12000
16000
!2
!1
0
0
10
20
!2
!1
0
1
2
0
10
20
0
10
20
0
10
20
Figure 43: Morph-tracking. First 4000 steps: morphing from p1 to p2, back again
in next 4000 steps; steps 8000 – 16,000: morphing from p3 to p4 and back again.
Figure layout same as in Figure 42.
spective I found it more appealing to insert such a valuable prior only in the top
layer (motto: the retina does not conceptually understand what it sees).
Inspecting again the top row in Figure 42, one ﬁnds fast initial decision between
the alternatives “pattern 1 or 2” versus “pattern 3 or 4”, followed by a much slower
diﬀerentation within these two classes. This suggests architecture variants where
all layers are informed by priors of the kind as in Equation (86), that is, the local
conceptor on a layer is constrained to an aperture-weighted disjunction of a ﬁnite
number of prototype conceptors. However, the number of prototype conceptors
would shrink as one goes upwards in the hierarchy. The reduction in number would
be eﬀected by merging several distinct prototype conception vectors cj1, . . . , cjk in
layer l into a single prototype vector cj = W{cj1, . . . , cjk}. In terms of classical AI
129
knowledge representation formalisms this would mean to implement an abstraction
hierarchy. A further reﬁnement that suggests itself would be to install a top-down
processing pathway by which the current hypothesis on layer l + 1 selects which
ﬁner-grained disjunction of prototypes on layer l is chosen. For instance, when
c[l+1](n) = cj and cj = W{cj1, . . . , cjk}, then the conception weight vector c[l](n) is
constrained to be of the form W
i=1,...,k ϕ(cji, γji
[l](n)). This remains for future work.
The architecture presented above is replete with ad-hoc design decisions. Nu-
merous details could have been realized diﬀerently. There is no uniﬁed theory
which could inform a system designer what are the “right” design decisions. A
complete SIA architecture must provide a plethora of dynamical mechanisms for
learning, adaptation, stabilization, control, attention and so forth, and each of
them in multiple versions tailored to diﬀerent subtasks and temporal scales. I do
not see even a theoretical possibility for an overarching, principled theory which
could aﬀord us with rigorous design principles for all of these. The hierarchical
conceptor architecture presented here is far from realizing a complete SIA system,
but repercussions of that under-constrainedness of design already show.
Discussion. Hierarchical neural learning architectures for pattern recognition
have been proposed in many variants (examples: [66, 38, 33, 47, 35, 111]), al-
beit almost always for static patterns. The only example of hierarchical neural
architectures for temporal pattern recognition that I am aware of are the localist-
connectionistic SHRUTI networks for text understanding [96]. Inherently tem-
poral hierarchical pattern classiﬁcation is however realized in standard hidden-
Markov-model (HMM) based models for speech recognition.
There is one common characteristic across all of these hierarchical recognition
systems (neural or otherwise, static or temporal). This shared trait is that when
one goes upward through the processing layers, increasingly “global” or “coarse-
grained” or “compound” features are extracted (for instance, local edge detection
in early visual processing leading through several stages to object recognition in
the highest layer).
While the concrete nature of this layer-wise integration of
information diﬀers between approaches, at any rate there is change of represented
categories across layers. For the sake of discussion, let me refer to this as the
feature integration principle.
From the point of view of logic-based knowldege representation, another impor-
tant trait is shared by hierarchical pattern recognition systems: abstraction. The
desired highest-level output is a class labelling of the input data. The recognition
architecture has to be able to generalize from the particular input instance. This
abstraction function is not explicitly implemented in the layers of standard neural
(or HMM) recognizers. In rule-based decision-tree classiﬁcation systems (text-
book: [75]), which can also be regarded as hierarchical recognition systems, the
hierarchical levels however directly implement a series of class abstractions. I will
refer to the abstraction aspect of pattern recognition as the categorical abstraction
principle.
130
The conceptor-based architecture presented in this section implements categor-
ical abstraction through the γ variables in the highest layer. They yield (graded)
class judgements similar to what is delivered by the class indicator variables in
the top layers of typical neural pattern recognizers.
The conceptor-based architecture is diﬀerent from the typical neural recog-
nition systems in that it does not implement the feature integration principle.
As one progresses upwards through the layers, always the same dynamic item is
represented, namely, the current periodic pattern, albeit in increasingly denoised
versions.
I will call this the pattern integrity principle.
The pattern integrity
principle is inherently conﬂicting with the feature integration principle.
By decades of theoretical research and successful pattern recognition applica-
tions, we have become accustomed to the feature integration principle. I want
to argue that the pattern integrity principle has some cognitive plausibility and
should be considered when one designs SIA architectures.
Consider the example of listening to a familiar piece of music from a CD player
in a noisy party environment. The listener is capable of two things. Firstly, s/he
can classify the piece of music, for instance by naming the title or the band.
This corresponds to performing categorical abstraction, and this is what standard
pattern recognition architectures would aim for. But secondly, the listener can
also overtly sing (or whistle or hum) along with the melody, or s/he can mentally
entrain to the melody. This overt or covert accompaniment has strong de-noising
characteristics – party talk fragments are ﬁltered out in the “mental tracing” of
the melody. Furthermore, the mental trace is temporally entrained to the source
signal, and captures much temporal and dynamical detail (single-note-level of
accuracy, stressed versus unstressed beats, etc). That is an indication of pattern
integrity.
Another example of pattern integrity: viewing the face of a conversation part-
ner during a person-to-person meeting, say Anne meeting Tom. Throughout the
conversation, Anne knows that the visual impression of Tom’s face is indeed Tom’s
face: categorical abstraction is happening. But also, just like a listener to noise-
overlaid music can trace online the clean melody, Anne maintains a “clean video”
representation of Tom’s face as it undergoes a succession of facial expressions and
head motions. Anne experiences more of Tom’s face than just the top-level ab-
straction that it is Tom’s; and this online experience is entrained to Anne’s visual
input stream.
Pattern integrity could be said to be realized in standard hierarchical neural
architectures to the extent that they are generative. Generative models aﬀord of
mechanisms by which example instances of a recognition class can be actively pro-
duced by the system. Prime examples are the architectures of adaptive resonance
theory [43], the Boltzmann Machine [3] and the Restricted Boltzmann Machine /
Deep Belief Networks [47]. In systems of this type, the reconstruction of pattern
instances occurs (only) in the input layer, which can be made to “confabulate” or
“hallucinate” (both terms are used as technical terms in the concerned literature)
131
pattern instances when primed with the right bias from higher layers.
Projecting such architectures to the human brain (a daring enterprise) and
returning to the two examples above, this would correspond to re-generating
melodies in the early auditory cortex or facial expressions in the early visual cortex
(or even in the retina). But I do not ﬁnd this a convincing model of what happens
in a human brain. Certainly I am not a neuroscientist and not qualiﬁed to make
scientiﬁc claims here. My doubts rest on introspection (forbidden! I know) and
on a computational argument. Introspection: when I am mentally humming along
with a melody at a party, I still do hear the partytalk – I dare say my early audi-
tory modules keep on being excited by the entire auditory input signal. I don’t feel
like I was hallucinating a clean version of the piece of music, making up an audi-
tory reality that consists only of clean music. But I do not listen to the talk noise,
I listen only to the music components of the auditory signal. The reconstruction
of a clean version of the music happens – as far as I can trust my introspection
– “higher up” in my brain’s hierarchy, closer to the quarters where consciously
controllable cognition resides. The computational argument: generative models,
such as the ones mentioned, cannot (in their current versions at least) generate
clean versions of noisy input patterns while the input is presented. They either
produce a high-level classiﬁcation response while being exposed to input (bottom-
up processing mode), or they generate patterns in their lowest layer while being
primed to a particular class in their highest layer (top-down mode). They can’t
do both at the same time. But humans can: while being exposed to input, a
cleaned-up version of the input is being maintained. Furthermore, humans (and
the conceptor architecture) can operate in an online-entrained mode when driven
by temporal data, while almost all existing recognition architectures in machine
learning are designed for static patterns.
Unfortunately I cannot oﬀer a clear deﬁnition of “pattern integrity”. An aspect
of pattern integrity that I ﬁnd important if not deﬁning is that some temporal
and spatial detail of a recognized pattern is preserved across processing layers.
Even at the highest layers, a “complete” representation of the pattern should
be available. This seems to agree with cognitive theories positing that humans
represent concepts by prototypes, and more speciﬁcally, by exemplars (critical
discussion [63]). However, these cognitive theories relate to empirical ﬁndings on
human classiﬁcation of static, not temporal, patterns. I am aware that I am vague.
One reason for this is that we lack a scientiﬁc terminology, mathematical models,
and standard sets of examples for discussing phenomena connected with pattern
integrity. All I can bring to the table at this point is just a new architecture that
has some extravagant processing characteristics. This is, I hope, relevant, but it
is premature to connect this in any detail to empirical cognitive phenomena.
132
3.16
Toward a Formal Marriage of Dynamics with Logic
In this subsection I assume a basic acquaintance of the reader with Boolean and
ﬁrst-order predicate logic.
So far, I have established that conceptor matrices can be combined with (al-
most) Boolean operations, and can be ordered by (a version of) abstraction. In this
subsection I explore a way to extend these observations into a formal “conceptor
logic”.
Before I describe the formal apparatus, I will comment on how I will be un-
derstanding the notions of “concept” and “logic”. Such a preliminary clariﬁcation
is necessary because these two terms are interpreted quite diﬀerently in diﬀerent
contexts.
“Concepts” in the cognitive sciences. I start with a quote from a recent survey
on research on concepts and categories in the cognitive sciences [74]: “The concept
of concepts is diﬃcult to deﬁne, but no one doubts that concepts are fundamental
to mental life and human communication. Cognitive scientists generally agree that
a concept is a mental representation that picks out a set of entities, or a category.
That is, concepts refer, and what they refer to are categories. It is also commonly
assumed that category membership is not arbitrary but rather a principled matter.
What goes into a category belongs there by virtue of some law-like regularities.
But beyond these sparse facts, the concept CONCEPT is up for grabs.”
Within
this research tradition, one early strand [85, 18] posited that the overall organi-
zation of a human’s conceptual representations, his/her semantic memory, can be
formally well captured by AI representation formalisms called semantic networks
in later years. In semantic network formalisms, concepts are ordered in abstrac-
tion hierarchies, where a more abstract concept refers to a more comprehensive
category. In subsequent research this formally clear-cut way of deﬁning and or-
ganizing concepts largely dissolved under the impact of multi-faceted empirical
ﬁndings. Among other things, it turned out that human concepts are graded,
adaptive, and depend on features which evolve by learning.
Such ﬁndings led
to a diversity of enriched models of human concepts and their organization (my
favourites: [62, 22, 64]), and many fundamental questions remain controversial.
Still, across all diversity and dispute, the basic conception of concepts spelled
out in the initial quote remains largely intact, namely that concepts are mental
representations of categories, and categories are deﬁned extensionally as a set of
“entities”. The nature of these entities is however “up to grabs”. For instance,
the concept named “Blue” might be referring to the set of blue physical objects,
to a set of wavelengths of light, or to a set of sensory experiences, depending on
the epistemic approach that is taken.
“Concepts” in logic formalisms. I ﬁrst note that the word “concept” is not
commonly used in logics. However, it is quite clear what elements of logical systems
are equated with concepts when such systems are employed as models of semantic
memory in cognitive science, or as knowledge representation frameworks in AI.
There is a large variety of logic formalisms, but almost all of them employ typed
133
symbols, speciﬁcally unary predicate symbols, relation symbols of higher arity,
constant symbols, and function symbols. In the model-theoretic view on logic,
such symbols become extensionally interpreted by sets of elements deﬁned over the
domain set of a set-theoretic model. Unary predicate symbols become interpreted
by sets of elements; n-ary relation symbols become interpreted by n-tuples of such
elements; function symbols by sets of argument-value pairs; constant symbols by
individual elements. A logic theory uses a ﬁxed set of such symbols called the
theory’s signature. Within a theory, the interpretation of the signature symbols
becomes constrained by the axioms of the theory. In AI knowledge representation
systems, this set of axioms can be very large, forming a world model and situation
model (sometimes called “T-Box” and “A-Box”). In the parlance of logic-oriented
AI, the extension of unary predicate symbols are often called classes instead of
“categories”.
In AI applications, the world model is often implemented in the structure of a
semantic network [67], where the classes are represented by nodes labelled by pred-
icate symbols. These nodes are arranged in a hierarchy with more abstract class
nodes in higher levels. This allows the computer program to exploit inheritance
of properties and relations down the hierarchy, reducing storage requirements and
directly enabling many elementary inferences. Class nodes in semantic networks
can be laterally linked by relation links, which are labelled by relation symbols.
At the bottom of such a hierarchy one may locate individual nodes labelled by
constant symbols. A cognitive scientist employing such a semantic network rep-
resentation would consider the class nodes, individual nodes, and relation links
as computer implementations or formal models of class concepts, individual con-
cepts, and relational concepts, respectively. Also, semantic network speciﬁcation
languages are sometimes called concept description languages in AI programming.
On this background, I will understand the symbols contained in a logic signature
as names of concepts.
Furthermore, a logical expression ϕ[x1, . . . , xn] containing n free (ﬁrst-order)
variables can be interpreted by the set of all n-tuples satisfying this expression.
ϕ[x1, . . . , xn] thus deﬁnes an n-ary relation.
For example, ϕ[x] = Fruit(x) ∧
Yellow(x) ∧Longish(x) ∧Curved(x) would represent a class (seems to be the
class of bananas). Quite generally, logical expressions formed according to the
syntax of a logic formalism can build representations of new concepts from given
ones.
There is an important diﬀerence between how “concepts” are viewed in cog-
nitive modeling versus logic-based AI. In the latter ﬁeld, concepts are typically
named by symbols, and the formal treatment of semantics is based on a refer-
ence relationship between the symbols of a signature and their interpretations.
However, even in logic-based knowledge representation formalisms there can be
un-named concepts which are formally represented as logic expressions with free
variables, as for instance the banana formula above. In cognitive science, concepts
are not primarily or necessarily named, although a concept can be optionally la-
134
belled with a name. Cognitive modeling can deal with conceptual systems that
have not a single symbol, for instance when modeling animal cognition. By con-
trast, AI-style logic modeling typically is strongly relying on symbols (the only
exception being mathematical theories built on the empty signature; this is of
interest only for intra-mathematical investigations).
Remarks on “Logics”. In writing this paragraph, I follow the leads of the PhD
thesis [86] of Florian Rabe which gives a comprehensive and illuminating account
of today’s world of formal logic research. The ﬁeld of mathematical logics has
grown and diversiﬁed enormously in the last three decades. While formal logic
historically has been developed within and for pure mathematics, much of this
recent boost was driven by demands from theoretical computer science, AI, and
semantic web technologies. This has led to a cosmos populated by a multitude of
“logics” which sometimes diﬀer from each other even in basic premises of what,
actually, qualiﬁes a formal system as a “logic”. In turn, this situation has led to
meta-logical research, where one develops formal logical frameworks in order to
systematically categorize and compare diﬀerent logics.
Among the existing such logical frameworks, I choose the framework of in-
stitutions [37], because it has been devised as an abstraction of model-theoretic
accounts of logics, which allows me to connect quite directly to concepts, cate-
gories, and the semantic reference link between these two. Put brieﬂy, a formal
system qualiﬁes as a logic within this framework if it can be formulated as an
institution. The framework of institutions is quite general: all logics used in AI,
linguistics and theoretical cognitive sciences can be characterized as institutions.
The framework of institutions uses tools from category theory. In this section
I do not assume that the reader is familiar with category theory, and therefore
will give only an intuitive account of how “conceptor logic” can be cast as an
institution. A full categorical treatment is given in Section 3.17.
An institution is made of three main components, familiar from the model
theory of standard logics like ﬁrst-order predicate logic:
1. a collection Sign of signatures, where each signature Σ is a set of symbols,
2. for each signature Σ, a set Sen(Σ) of Σ-sentences that can be formed using
the symbols of Σ,
3. again for each signature Σ, a collection Mod(Σ) of Σ-models, where a Σ-model
is a mathematical structure in which the symbols from Σ are interpreted.
Furthermore, for every signature Σ there is a model relation |=Σ ⊆Mod(Σ) ×
Sen(Σ). For a Σ-model m and a Σ-sentence χ, we write inﬁx notation m |=Σ χ
for (m, χ) ∈|=Σ, and say “m is a model of χ”, with the understanding that the
sentence χ makes a true statement about m.
The relationsships between the main elements of an institution can be visual-
ized as in Figure 44.
135
Σ 
Sen(Σ) 
Mod(Σ) 
Sen 
Mod 
Figure 44: How the elements of an institution relate to each other. For explanation
see text.
The full deﬁnition of an institution includes a mechanism for symbol re-naming.
The intuitive picture is the following. If a mathematician or an AI engineer writes
down a set of axioms, expressed as sentences in a logic, the choice of symbols
should be of no concern whatsoever. As Hilbert allegedly put it, the mathemati-
cal theory of geometry should remain intact if instead of “points, lines, surfaces”
one would speak of “tables, chairs, beer mugs”. In the framework of institutions
this is reﬂected by formalizing how a signature Σ may be transformed into another
signature Σ′ by a signature morphism φ : Σ →Σ′, and how signature morphisms
are extended to sentences (by re-naming symbols in a sentence according to the
signature morphism) and to models (by interpreting the re-named symbols by the
same elements of a model that were previously used for interpreting the origi-
nal symbols). Then, if m′, χ′ denote the re-named model m and sentence χ, an
institution essentially demands that m |=Σ χ if and only if m′ |=Σ′ χ′.
For example, ﬁrst-order logic (FOL) can be cast as an institution by taking
for Sign the class of all FOL signatures, that is the class of all sets containing
typed predicate, relation, function and constant symbols; Sen maps a signature
Σ to the set of all closed (that is, having no free variables) Σ-expressions (usually
called sentences); Mod assigns to each signature the class of all set-theoretic Σ-
structures; and |= is the satisfaction relation of FOL (also called model relation).
For another example, Boolean logic can be interpreted as an institution in several
ways, for instance by declaring Sign as the class of all totally ordered countable
sets (the elements of which would be seen as Boolean variables); for each signature
Σ of Boolean variables, Sen(Σ) is the set of all Boolean expressions ϕ[Xi1, . . . , Xin]
over Σ and Mod(Σ) is the set of all truth value assignments τ : Σ →{T, F} to the
Boolean variables in Σ; and τ |=Σ ϕ if ϕ evaluates to T under the assignment τ.
In an institution, one can deﬁne logical entailment between Σ-sentences in the
familiar way, by declaring that χ logically entails χ′ (where χ, χ′ are Σ-sentences)
if and only if for all Σ-models m it holds that m |=Σ χ implies m |=Σ χ′. By a
standard abuse of notation, this is also written as χ |=Σ χ′ or χ |= χ′.
I will sketch two entirely diﬀerent approaches to deﬁne a “conceptor logic”.
The ﬁrst follows in the footsteps of familiar logics. Conceptors can be named by
arbitrary symbols, sentences are built by an inductive procedure which speciﬁes
how more complex sentences can be constructed from simpler ones by similar syn-
tax rules as in ﬁrst-order logic, and models are designated as certain mathematical
136
structures built up from named conceptors. This leads to a logic that essentially
represents a version of ﬁrst-order logic constrained to conceptor domains. It would
be a logic useful for mathematicians to investigate “logical” characteristics of con-
ceptor mathematics, especially whether there are complete calculi that allow one
to systematically prove all true facts concerning conceptors. I call such logics ex-
trinsic conceptor logics. Extrinsic conceptor logics are tools for mathematicians to
reason about conceptors. A particular extrinsic conceptor logic as an institution
is detailed in Section 3.17.
The other approach aims at a conceptor logic that, instead of being a tool for
mathematicians to reason about conceptors, is a model of how a situated intelligent
agent does “logical reasoning” with conceptors. I call this intrinsic conceptor logic
(ICL). An ICL has a number of unconventional properties:
• An ICL should function as a model of a situated agent’s conceptor-based
information processing. Agents are bound to diﬀer widely in their structure
and their concrete lifetime learning histories. Therefore I do not attempt
to design a general “ﬁts-all-agents” ICL. Instead, for every single, concrete
agent life history there will be an ICL, the private ICL of that agent life.
• An agent with a personal learning history is bound to develop its private
“logic” over time. The ICL of an agent life thus becomes a dynamical system
in its own right.
The framework of institutions was not intended by its
designers to model temporally evolving objects. Specifying an institution
such that it can be considered a dynamical system leads to some particularly
unconventional characteristics of an agent life ICL. Speciﬁcally, signatures
become time-varying objects, and signature morphisms (recall that these
model the “renaming” of symbols) are used to capture the temporal evolution
of signatures.
An agent’s lifetime ICL is formalized diﬀerently according to whether the agent
is based on matrix conceptors or random feature conceptors. Here I work out only
the second case.
In the following outline I use the concrete three-layer de-noising and classiﬁ-
cation architecture from Section 3.15 as a reference example to ﬁll the abstract
components of ICL with life. Even more concretely, I use the speciﬁc “lifetime his-
tory” of the 16000-step adaptation run illustrated in Figure 42 as demonstration
example. For simplicity I will refer to that particular de-noising and classiﬁcation
architecture run as “DCA”.
Here is a simpliﬁed sketch of the main components of an agent’s lifetime ICL
(full treatment in Section 3.17):
1. An ICL is designed to model a particular agent lifetime history. A speciﬁ-
cation of such an ICL requires that a formal model of such an agent life is
available beforehand. The core part of an agent life model AL is a set of
137
m conceptor adaptation sequences {a1(n), . . . , am(n)}, where each ai(n) is
an M-dimensional conception weight vector. It is up to the modeler’s dis-
cretion which conceptors in a modeled agent become included in the agent
life model AL. In the DCA example I choose the four prototype conception
weight vectors c1, . . . , c4 and the two auto-adapted caut
[l] on layers l = 1, 2. In
this example, the core constitutent of the agent life model AL is thus the set
of m = 6 conceptor adaptation trajectories c1(n), . . . , c4(n), caut
[1] (n), caut
[2] (n),
where 1 ≤n ≤16000. The ﬁrst four trajectories c1(n), . . . , c4(n) are con-
stant over time because these prototype conceptors are not adapted; the last
two evolve over time. Another part of an agent life model is the lifetime T,
which is just the interval of timepoints n for which the adaptation sequences
ai(n) are deﬁned. In the DCA example, T = (1, 2, . . . , 16000).
2. A signature is a ﬁnite non-empty set Σ(n) = {A(n)
1 , . . . , A(n)
m } of m time-
indexed symbols Ai. For every n ∈T there is a signature Σ(n).
DCA example: In the ICL of this example agent life, the collection Sign of
signatures is made of 16000 signatures {C(n)
1 , . . . , C(n)
4 , A(n)
1 , A(n)
2 } containing
six symbols each, with the understanding that the ﬁrst four symbols refer
to the prototype conceptors c1(n), . . . , c4(n) and the last two refer to the
auto-adapted conceptors caut
[1] (n), caut
[2] (n).
3. For every pair Σ(n+k), Σ(n) of signatures, where k ≥0, there is a signature
morphism φ(n+k,n) : Σ(n+k) →Σ(n) which maps A(n+k)
i
to A(n)
i . These signa-
ture morphisms introduce a time arrow into Sign. This time arrow “points
backwards”, leading from later times n + k to earlier times n. There is a
good reason for this backward direction. Logic is all about describing facts.
In a historically evolving system, facts χ(n+k) established at some later time
n + k can be explained in terms of facts ζ(n) at preceding times n, but not
vice versa. Motto: “the future can be explained in terms of the past, but the
past cannot be reduced to facts from the future”. Signature morphisms are a
technical vehicle to re-formulate descriptions of facts. They must point back-
wards in time in order to allow facts at later times to become re-expressed
in terms of facts stated for earlier times. Figure 45 illustrates the signatures
and their morphisms in an ICL.
4. Given a signature Σ(n) = {A(n)
1 , . . . , A(n)
m }, the set of sentences Sen(Σ(n))
which can be expressed with the symbols of this signature is the set of
syntactic expressions deﬁned inductively by the following rules (incomplete,
full treatment in next subsection):
(a) A(n)
1 , . . . , A(n)
m are sentences in Sen(Σ(n)).
(b) For k ≥0 such that n, n + k ∈T, for A(n)
i
∈Σ(n), δ(n)
k
A(n)
i
is in
Sen(Σ(n)).
138
time!n 
A(n – 1) 
A(n) 
A(n + 1) 
A(n – 1) 
A(n) 
A(n + 1) 
!(n – 1) 
!(n +1) 
!(n) 
" (n, n – 1) 
" (n+1, n) 
... 
... 
" (n+1, n – 1) 
1 
1 
1 
2 
2 
2 
... 
... 
... 
Figure 45: Signatures and their morphisms in an ICL (schematic). For explanation
see text.
(c) If ζ, ξ ∈Sen(Σ(n)), then (ζ ∨ξ), (ζ ∧ξ), ¬ζ ∈Sen(Σ(n)).
(d) If ζ ∈Sen(Σ(n)), then ϕ(ζ, γ) ∈Sen(Σ(n)) for every γ ∈[0, ∞] (this
captures aperture adaptation).
(e) If ζ, ξ ∈Sen(Σ(n)) and 0 ≤b ≤1, then βb(ζ, ξ) ∈Sen(Σ(n)) (this will
take care of linear blends bζ + (1 −b)ξ).
In words, sentences express how new conceptors can be built from existing
ones by Boolean operations, aperture adaptation, and linear blends. The
“seed” set for these inductive constructions is provided by the conceptors
that can be directly identiﬁed by the symbols in Σ(n).
The sentences of form δ(n)
k
A(n)
i
deserve a special comment. The operators
δ(n)
k
are time evolution operators. A sentence δ(n)
k
A(n)
i
will be made to refer
to the conceptor version ai(n + k) at time n + k which has evolved from
ai(n).
5. For every time n, the set Mod(Σ(n)) of Σ(n)-models is the set Z of M-
dimensional nonnegative vectors.
Remarks: (i) The idea for these models is that they represent mean energy
vectors E[z.∧.2] of feature space states. (ii) The set of models Mod(Σ(n)) is
the same for every signature Σ(n).
DCA example: Such feature space signal energy vectors occur at various
places in the DCA, for instance in Equations (92), (93), and conception
weight vectors which appear in the DCA evolution are all deﬁned or adapted
in one way or other on the basis of such feature space signal energy vectors.
139
6. Every Σ(n)-sentence χ is associated with a concrete conception weight vector
ι(χ) by means of the following inductive deﬁnition:
(a) ι(A(n)
i ) = ai(n).
(b) ι(δ(n)
k A(n)
i ) = ai(n + k).
(c) Case χ = (ζ ∨ξ): ι(χ) = ι(ζ) ∨ι(ξ) (compare Deﬁnition 7).
(d) Case χ = (ζ ∧ξ): ι(χ) = ι(ζ) ∧ι(ξ).
(e) Case χ = ¬ζ: ι(χ) = ¬ι(ζ).
(f) Case χ = ϕ(ζ, γ): ι(χ) = ϕ(ι(ζ), γ) (compare Deﬁnition 6).
(g) Case χ = βb(ζ, ξ): ι(χ) = b ι(ζ) + (1 −b)ι(ξ).
Remark: This statement of the interpretation operator ι is suggestive only.
The rigorous deﬁnition (given in the next section) involves additional non-
trivial mechanisms to establish the connection between the symbol A(n)
i
and
the concrete conceptor version ai(n) in the agent life. Here I simply appeal
to the reader’s understanding that symbol Ai refers to object ai.
7. For z.∧2 ∈Z and χ ∈Sen(Σ(n)), the model relationship is deﬁned by
z.∧2 |=Σ(n) χ
iﬀ
z.∧2 .∗(z.∧2 + 1).∧−1 ≤ι(χ).
(98)
Remark: This deﬁnition in essence just repeats how a conception weight
vector is derived from a feature space signal energy vector.
When all category-theoretical details are ﬁlled in which I have omitted here,
one obtains a formal deﬁnition of an institution which represents the ICL of an
agent life AL. It can be shown that in an ICL, for all ζ, ξ ∈Sen(Σ(n)) it holds
that
ζ |=Σ(n) ξ
iﬀ
ι(ζ) ≤ι(ξ).
By virtue of this fact, logical entailment becomes decidable in an ICL: if one wishes
to determine whether ξ is implied by ζ, one can eﬀectively compute the vectors
ι(ζ), ι(ξ) and then check in constant time whether ι(ζ) ≤ι(ξ).
Returning to the DCA example (with lifetime history shown in Figure 42), its
ICL identieﬁes over time the four prototype conceptors c1, . . . , c4 and the two auto-
adapted conceptors cauto
[1] , cauto
[2]
by temporally evolving symbols {C(n)
1 , . . . , C(n)
4 ,
A(n)
1 , A(n)
2 }.
All other conceptors that are computed in this architecture can
be deﬁned in terms of these six ones.
For instance, the top-level conceptor
c[3](n) can be expressed in terms of the identiﬁable four prototype conceptors
by c[3](n) = W
j=1,...,4 ϕ(cj, γj(n)) by combining the operations of disjunction and
aperture adaptation. In ICL syntax this construction would be expressible by a
Σ(n) sentence, for instance by
(((ϕ(C(n)
1 , γ1(n)) ∨ϕ(C(n)
2 , γ2(n))) ∨ϕ(C(n)
3 , γ3(n))) ∨ϕ(C(n)
4 , γ4(n))).
140
A typical adaptation objective of a conception vector c(n) occurring in an agent
life is to minimize a loss of the form (see Deﬁnition 78)
Ez[∥z −c(n) .∗z∥2] + α−2 ∥c(n)∥2,
or equivalently, the objective is to converge to
c(n) = E[α2z.∧2] .∗(E[α2z.∧2] + 1).∧−1.
This can be re-expressed in ICL terminology as “adapt c(n) such that α2E[z.∧2] |=Σ(n)
χc(n), and such that not z.∧2 |=Σ(n) χc(n) for any z.∧2 > α2E[z.∧2]” (here χc(n) is
an adhoc notation for an ICL sentence χc(n) ∈Sen(Σ(n)) specifying c(n)).
In
more abstract terms, the typical adaptation of random feature conceptors can be
understood as an attempt to converge toward the conceptor that is maximally
|=-speciﬁc under a certain constraint.
Discussion. I started this section by a rehearsal of how the notion of “con-
cept” is understood in cognitive science and logic-based AI formalisms. According
to this understanding, a concept refers to a category (terminology of cognitive
science); or a class symbol or logical expression with free variables is interpreted
by its set-theoretical extension (logic terminology). Usually, but not necessarily,
the concepts/logical expressions are regarded as belonging to an “ontological” do-
main that is diﬀerent from the domain of their respective referents. For instance,
consider a human maintaining a concept named cow in his/her mind. Then many
cognitive scientists would identiﬁy the category that is referred to by this con-
cept with the some set of physical cows. Similarly, an AI expert system set up
as a farm management system would contain a symbol cow in its signature, and
this symbol would be deemed to refer to a collection of physical cows. In both
cases, the concept / symbolic expression cow is ontologically diﬀerent from a set of
physical cows. However, both in cognitive science and AI, concepts / symbolic ex-
pressions are sometimes brought together with their referents much more closely.
In some perspectives taken in cognitive science, concepts are posited to refer to
other mental items, for instance to sensory perceptions. In most current AI proof
calculi (“inference engines”), models of symbolic expressions are created which
are assembled not from external physical objects but from symbolic expressions
(“Herbrand universe” constructions). Symbols from a signature Σ then refer to
sets of Σ-terms. In sum, ﬁxing the ontological nature of referents is ultimately left
to the modeling scientist in cognitive science or AI.
In contrast, ICL is committed to one particular view on the semantic relation-
ship: Σ(n)-sentences are always describing conception weight vectors, and refer
to neural activation energy vectors z.∧2. In the case of matrix conceptor based
agents, Σ(n)-sentences describe conceptor matrices and refer to neural activation
correlation matrices R by the following variant of (98):
R |=Σ(n) χ
iﬀ
R(R + I)−1 ≤ι(χ).
(99)
141
In Figure 46 I try to visualize this diﬀerence between the classical, extensional
view on symbols and their referents, and the view adopted by ICL. This ﬁgure
contrasts how classical logicians and cognitive scientists would usually model an
agent’s representation of farm livestock, as opposed to how ICL renders that sit-
uation. The semantic relation is here established between the physical world on
the one side, and symbols and logical expressions on the other side. The world is
idealized as a set of individuals (individual animals in this example), and symbols
for concepts (predicate symbols in logic) are semantically interpreted by sets of
individuals. In the farmlife example, a logician might introduce a symbol lifestock
which would denote the set of all economically relevant animals grown in farms,
and one might introduce another symbol poultry to denote the subset of all feath-
ered such animals. The operator that creates “meaning” for concept symbols is
the grouping of individuals into sets (the bold “{ }” in Figure 46).
With conceptors, the semantic relation connects neural activity patterns trig-
gered by perceiving animals on the one side, with conceptors acting on neural
dynamics on the other side. The core operator that creates meaning is the con-
densation of the incoming data into a neural activation energy pattern z.∧2 (or
correlation matrix R for matrix conceptors) from which conceptors are generated
via the fundamental construction c = E[z.∧2] .∗(E[z.∧2]+1).∧−1 or C = R(R+I)−1
(Figure 46 depicts the latter case).
ICL, as presented here, cannot claim to be a model of all “logical reasoning”
in a neural agent. Speciﬁcally, humans sometimes engage in reasoning activities
which are very similar to how syntactic logic calculi are executed in automated
theorem proving. Such activities include the build-up and traversal of search trees,
creating and testing hypotheses, variable binding and renaming, and more. A
standard example is the step-by-step exploration of move options done by a human
chess novice. ICL is not designed to capture such conscious combinatorial logical
reasoning. Rather, ICL is intended to capture the automated aspects of neural
information processing of a situated agent, where incoming (sensor) information
is immediately transformed into perceptions and maybe situation representations
in a tight dynamical coupling with the external driving signals.
The material presented in this and the next section is purely theoretical and
oﬀers no computational add-on beneﬁts over the material presented in earlier
sections. There are three reasons why nonetheless I invested the eﬀort of deﬁninig
ICLs:
• By casting conceptor logic rigorously as an institution, I wanted to substan-
tiate my claim that conceptors are “logical” in nature, beyond a mere appeal
to the intuition that anything admitting Boolean operations is logic.
• The institutional deﬁnition given here provides a consistent formal picture
of the semantics of conceptors. A conceptor c identiﬁed by an ICL sentence
χc “means” neural activation energy vectors z.∧2.
Conceptors and their
meanings are both neural objects of the same mathematical format, M-
142
Figure 46: Contrasting the extensional semantics of classical knowledge representa-
tion formalisms (upper half of graphics) with the system-internal neurodynamical
semantics of conceptors (lower half). In both modeling approaches, abstraction
hierarchies of “concepts” arise. For explanation see text.
dimensional nonnegative vectors. Having a clear view on this circumstance
helps to relate conceptors to the notions of concepts and their referents,
which are so far from being fully understood in the cognitive sciences.
• Some of the design ideas that went into casting ICLs as institutions may
be of more general interest for mathematical logic research.
Speciﬁcally,
making signatures to evolve over time – and hence, turn an institution into
a dynamical system – might be found a mechanism worth considering in
scenarios, unconnected with conceptor theory or neural networks, where one
wants to analyse complex dynamical systems by means of formal logics.
3.17
Conceptor Logic as Institutions: Category-Theoretical
Detail
In this section I provide a formal speciﬁcation of conceptor logic as an institution.
This section addresses only readers with a dedicated interest in formal logic. I
assume that the reader is familiar with the institution framework for representing
logics (introduced in [37] and explained in much more detail in Section 2 in [86])
and with basic elements of category theory. I ﬁrst repeat almost verbatim the
143
categorical deﬁnition of an institution from [37].
Deﬁnition 8 An institution I consists of
1. a category Sign, whose objects Σ are called signatures and whose arrows are
called signature morphisms,
2. a functor Sen : Sign →Set, giving for each signature a set whose elements
are called sentences over that signature,
3. a functor Mod : Sign →Catop, giving for each signature Σ a category
Mod(Σ) whose objects are called Σ-models, and whose arrows are called Σ-
(model) morphisms, and
4. a relation |=Σ ⊆Mod(Σ)×Sen(Σ) for each Σ ∈Sign, called Σ-satisfaction,
such that for each morphism φ : Σ1 →Σ2 in Sign, the Satisfaction Condition
m2 |=Σ2 Sen(φ)(χ1)
iﬀ
Mod(φ)(m2) |=Σ1 χ1
(100)
holds for each m2 ∈Mod(Σ2) and each χ1 ∈Sen(Σ1).
The interrelations of these items are visualized in Figure 47.
Set 
Catop 
Sen 
Mod 
Sign 
!1"
!2"
!
Mod(!1) 
Mod(!2) 
Mod(!) 
Sen(!1) 
Sen(!2) 
Sen(!) 
!2 
!1 
Figure 47: Relationships between the constituents of an institution (redrawn from
[37]).
Remarks:
1. The morphisms in Sign are the categorical model of re-naming the symbols
in a logic. The essence of the entire apparatus given in Deﬁnition 8 is to
capture the condition that the model-theoretic semantics of a logic is invari-
ant to renamings of symbols, or, as Goguen and Burstall state it, “Truth is
invariant under change of notation”.
144
2. The intuition behind Σ-model-morphisms, that is, maps µ : IΣ
1 →IΣ
2 , where
IΣ
1 , IΣ
2 are two Σ-models, is that µ is an embedding of IΣ
1 in IΣ
2 . If we take
ﬁrst-order logic as an example, with IΣ
1 , IΣ
2 being two Σ-structures, then
µ : IΣ
1 →IΣ
2 would be a map from the domain of IΣ
1 to the domain of IΣ
2
which preserves functional and relational relationships speciﬁed under the
interpretations of IΣ
1 and IΣ
2 .
3. In their original 1992 paper [37], the authors show how a number of stan-
dard logics can be represented as institutions. In the time that has passed
since then, institutions have become an important “workhorse” for software
speciﬁcation in computer science and for semantic knowledge management
systems in AI, especially for managing mathematical knowledge, and several
families of programming toolboxes have been built on institutions (overview
in [86]). Alongside with the model-theoretic spirit of institutions, this proven
usefulness of institutions has motivated me to adopt them as a logical frame-
work for conceptor logic.
Logical entailment between sentences is deﬁned in institutions in the traditional
way:
Deﬁnition 9 Let χ1, χ2 ∈Sen(Σ). Then χ1 entails χ2, written χ1 |=Σ χ2, if for
all m ∈Ob(Mod(Σ)) it holds that m |=Σ χ1 →m |=Σ χ2.
Institutions are ﬂexible and oﬀering many ways for deﬁning logics. I will frame
two entirely diﬀerent kinds of conceptor logics. The ﬁrst kind follows the intuitions
behind the familiar ﬁrst-order predicate logic, and should function as a formal
tool for mathematicians to reason about (and with) conceptors. Since it looks
at conceptors “from the outside” I will call it extrinsic conceptor logic (ECL).
Although ECL follows the footsteps of familiar logics in many respects, in some
aspects it deviates from tradition. The other kind aims at modeling the “logical”
operations that an intelligent neural agent can perform whose “brain” implements
conceptors. I ﬁnd this the more interesting formalization; certainly it is the more
exotic one. I will call it intrinsic conceptor logic (ICL).
Extrinsic conceptor logic. I ﬁrst give an intuitive outline. I treat only the case
of matrix-based conceptors. An ECL concerns conceptors of a ﬁxed dimension
N and their logical interrelationships, so one should more precisely speak of N-
dimensional ECL. I assume some N is ﬁxed. Sentences of ECL should enable
a mathematician to talk about conceptors in a similar way as familiar predicate
logics allow a mathematician to describe facts about other mathematical objects.
For example, “for all conceptors X, Y it holds that X ∧Y ≤X and X ∧Y ≤
Y ” should be formalizable as an ECL sentence. A little notational hurdle arises
because Boolean operations appear in two roles: as operators acting on conceptors
(the “∧” in the sentence above), and as constitutents of the logic language (the
“and” in that sentence). To keep these two roles notationally apart, I will use
AND, OR, NOT (allowing inﬁx notation) for the role as operators, and ∧, ∨, ¬
145
for the logic language. The above sentence would then be formally written as
“∀x∀y (x AND y ≤x) ∧(x AND y ≤y)”.
The deﬁnition of signatures and ECL-sentences in many respects follows stan-
dard customs (with signiﬁcant simpliﬁcations to be explained after the deﬁnition)
and is the same for any conceptor dimension N:
Deﬁnition 10 Let Var = {x1, x2, ...} be a ﬁxed countable indexed set of variables.
1. (ECL-signatures) The objects (signatures) of Sign are all countable sets,
whose elements are called symbols. For signatures Σ1, Σ2, the set of mor-
phisms hom (Σ1, Σ2) is the set of all functions φ : Σ1 →Σ2.
2. (ECL-terms) Given a signature Σ, the set of Σ-terms is deﬁned inductively
by
(a) Every variable xi, every symbol a ∈Σ, and I is a Σ-term.
(b) For Σ-terms t1, t2 and γ ∈[0, ∞], the following are Σ-terms: NOT t1,
(t1 AND t2), (t1 OR t2), and ϕ(t1, γ).
3. (ECL-expressions) Given a signature Σ, the set Exp(Σ) of Σ-expressions is
deﬁned inductively by
(a) If t1, t2 are Σ-terms, then t1 ≤t2 is a Σ-expression.
(b) If e1, e2 are Σ-expressions, and xi a variable, then the following are
Σ-expressions: ¬e1, (e1 ∧e2), (e1 ∨e2), ∀xi e1.
4. (ECL-sentences) A Σ-expression that contains no free variables is a Σ-
sentence (free occurrence of variables to be deﬁned as usual, omitted here.)
Given a Σ-morphism φ : Σ1 →Σ2, its image Sen(φ) under the functor Sen is
the map which sends every Σ1-sentence χ1 to the Σ2-sentence χ2 obtained from
χ1 by replacing all occurrences of Σ1 symbols in χ1 by their images under φ. I
omit the obvious inductive deﬁnition of this replacement construction.
Notes:
• ECL only has a single sort of symbols with arity 0, namely constant symbols
(which will be made to refer to conceptors later). This renders the categorical
treatment of ECL much simpler than it is for logics with sorted symbols of
varying arities.
• The operator symbols NOT, AND, OR, the parametrized operation sym-
bol ϕ(·, γ) and the relation symbol ≤are not made part of signatures, but
become universal elements in the construction of sentences.
146
The models of ECL are quite simple. For a signature Σ, the objects of Mod(Σ)
are the sets of Σ-indexed N-dimensional conceptor matrices
Ob(Mod(Σ)) = {m ⊂CN×N × Σ | ∀σ ∈Σ ∃=1C ∈CN×N : (C, σ) ∈m}
where CN×N is the set of all N-dimensional conceptor matrices. The objects of
Mod(Σ) are thus the graph sets of the functions from Σ to the set of N-dimensional
conceptor matrices. The model morphisms of Mod(Σ) are canonically given by
the index-preserving maps
hom({(C1, σ)}, {(C2, σ)}) = {µ : {(C1, σ)} →{(C2, σ)} | µ : (C1, σ) 7→(C2, σ)}.
Clearly, hom({(C, σ)}, {(C′, σ)}) contains exactly one element.
Given a signature morphism Σ1
φ→Σ2, then Mod(φ) is deﬁned to be a map
from Mod(Σ2) to Mod(Σ1) as follows. For a Σ2-model m2 = {(C2, σ2)} let [[σ2]]m2
denote the interpretation of σ2 in m2, that is, [[σ2]]m2 is the conceptor matrix C2
for which (C2, σ2) ∈m2.
Then Mod(φ) assigns to to m2 the Σ1-model m1 =
Mod(φ)(m2) = {([[φ(σ1)]]m2, σ1)} ∈Mod(Σ1).
The model relations |=Σ are deﬁned in the same way as in the familiar ﬁrst-
order logic. Omitting some detail, here is how:
Deﬁnition 11 Preliminaries: A map β : Var →CN×N is called a variable as-
signment. B is the set of all variable assignments. We denote by β C
xi the variable
assignment that is identical to β except that xi is mapped to C. A Σ-interpretation
is a pair I = (m, β) consisting of a Σ-model m and a variable assignment β. By
I C
xi we denote the interpretation (m, β C
xi).
For a Σ-term t, the interpretation
I(t) ∈CN×N is deﬁned in the obvious way. Then |=∗
Σ ⊆(Mod(Σ) × B) × Exp(Σ)
is deﬁned inductively by
1. I |=∗
Σ t1 ≤t2
iﬀ
I(t1) ≤I(t2),
2. I |=∗
Σ ¬e
iﬀ
not I |=∗
Σ e,
3. I |=∗
Σ (e1 ∧e2)
iﬀ
I |=∗
Σ e1 and I |=∗
Σ e2,
4. I |=∗
Σ (e1 ∨e2)
iﬀ
I |=∗
Σ e1 or I |=∗
Σ e2,
5. I |=∗
Σ ∀xi e
iﬀ
for all C ∈CN×N it holds that I C
xi |=∗
Σ e.
|=Σ then is the restriction of |=∗
Σ on sentences.
This completes the deﬁnition of ECL as an institution. The satisfaction con-
dition obviously holds. While in many respects ECL follows the role model of
ﬁrst-order logic, the associated model theory is much more restricted in that only
N-dimensional conceptors are admitted as interpretations of symbols. The natural
next step would be to design calculi for ECL and investigate whether this logic
is complete or even decidable. Clarity on this point would amount to an insight
147
in the computational tractability of knowledge representation based on matrix
conceptors with Boolean and aperture adaptation opertors.
Intrinsic conceptor logic. I want to present ICL as a model of the “logics”
which might unfold inside a neural agent.
All constitutents of ICL should be
realizable in terms of neurodynamical processes, giving a logic not for reasoning
about conceptors, but with conceptors.
Taking the idea of placing “logics” inside an agent seriously has a number of
consequences which lead quite far away from traditional intuitions about “logics”:
• Diﬀerent agents may have diﬀerent logics. I will therefore not try to deﬁne
a general ICL that would ﬁt any neural agent. Instead every concrete agent
with a concrete lifetime learning history will need his/her/its own individual
conceptor logic. I will use the signal de-noising and classiﬁcation architecture
from Section 3.15 as an example “agent” and describe how an ICL can be
formulated as an institution for this particular case. Some general design
principles will however become clear from this case study.
• Conceptors are all about temporal processes, learning and adaptation. An
agent’s private ICL will have to possess an eminently dynamical character.
Concepts will change their meaning over time in an agent. This “personal
history dynamics” is quintessential for modeling an agent and should become
reﬂected in making an ICL a dynamical object itself – as opposed to intro-
ducing time through descriptive syntactical elements in an otherwise static
logic, like it is traditionally done by means of modal operators or axioms de-
scribing a timeline. In my proposal of ICLs, time enters the picture through
the central constitutent of an institution, signature morphisms. These maps
between signatures (all commanded by the same agent) will model time, and
an agent’s lifetime history of adaptation will be modeled by an evolution of
signatures. Where the original core motif for casting logics as institutions
was that “truth is invariant under change of notation” ([37]), the main point
of ICLs could be contrasted as “concepts and their meaning change with
time”. The role of signature morphisms in ICLs is fundamentally diﬀerent
in ICLs compared to customary formalizations of logics. In the latter, signa-
ture changes should leave meaning invariant; in the former, adaptive changes
in conceptors are reﬂected by temporally indexed changes in signature.
• Making an ICL private to an agent implies that the model relation |= be-
comes agent-speciﬁc. An ICL cannot be speciﬁed as an abstract object in
isolation. Before it can be deﬁned, one ﬁrst needs to have a formal model of
a particular agent with a particular lifelong adaptation history.
In sum, an ICL (formalized as institution) itself becomes a dynamical system,
deﬁned relative to an existing (conceptor-based) neural agent with a particular
adaptation history. The “state space” of an ICL will be the set of signatures. A
“trajectory” of the temporal evolution of an ICL will essentially be a sequence of
148
signatures, enriched with information pertaining to forming sentences and models.
For an illustration, assume that a neural agent adapts two random feature concep-
tors a(n), b(n). These are named by two temporally indexed symbols A(n), B(n).
A signature will be a timeslice of these, Σ(n) = {A(n), B(n)}. For every pair of
integer timepoints (n + k, n) (where k ≥0) there will be a signature morphism
φ(n+k,n) : Σ(n+k) →Σ(n). The (strong) reason why signature morphisms point
backwards in time will become clear later. Figure 45 visualizes the components of
this example. The dotted lines connecting the A(n)
i
are suggestive graphical hints
that the symbols A(n)
i
all name the “same” conceptor ai. How this “sameness of
identity over time” can be captured in the institution formalism will become clear
presently.
Formal deﬁnitions of ICLs will vary depending on what kind of conceptors are
used (for instance, matrix or random feature based), or whether time is taken to
be discrete or continuous. I give a deﬁnition for discrete-time, random feature
conceptor based ICLs.
Because ICLs will be models of an agent’s private logic which evolves over the
agent’s lifetime, the deﬁnition of an ICL is stated relative to an agent’s lifetime
conceptor adaptation history. The only property of such an agent that is needed
for deﬁning an ICL is the existence of temporally adapted conceptors owned by
the agent. Putting this into a formal deﬁnition:
Deﬁnition 12 An agent life (here: M-dimensional random feature conceptor
based, discrete time) is a structure AL = (T, Σ, ιAL), where
1. T ⊆Z is an interval (ﬁnite or inﬁnite) of the integers, the lifetime of AL,
2. Σ = {A1, . . . , Am} is a ﬁnite nonempty set of conceptor identiﬁers,
3. ιAL : Σ × T →[0, 1]M, (Ai, n) 7→ai(n) assigns to every time point and
conceptor identiﬁer an adaptation version ai(n) of the conceptor identiﬁed
by the symbol Ai.
As an example of an agent consider the signal de-noising and classiﬁcation
architecture (DCA) presented in Section 3.15, with a “life” being the concrete
16000-step adaptation run illustrated in Figure 42. In this example, the lifetime
is T = {1, . . . , 16000}. I will identify by symbols the four prototype conceptors
c1, . . . , c4 and the two auto-adapted conceptors cauto
[1] , cauto
[2] . Accordingly I choose Σ
to be {C1, . . . , C4, A1, A2}. The map ιAL is constant in time for the four protype
conceptors: ιAL(n, Cj) = cj for all n ∈T, j = 1, . . . , 4. For the remaining two
conceptors, ιAL(n, Ai) = cauto
[i] (n).
The stage is now prepared to spell out the deﬁnition of an agent’s lifetime ICL
(for the case of an agent based on M-dimensional random feature conceptor and
discrete time):
Deﬁnition 13 The intrinsic conceptor logic (ICL) of an agent lifeAL = (T, Σ, ιAL)
is an institution whose components obey the following conditions:
149
1. The objects (signatures) of Sign are the pairs Σ(n) = ({A(n)
1 , . . . , A(n)
m }, σ(n)),
where n ∈T, and σ(n) : Σ →{A(n)
1 , . . . , A(n)
m } is a bijection.
DCA example: The lifetime of this example is T = {1, . . . , 16000}. A
signature Σ(n) = ({C(n)
1 , . . . , C(n)
4 , A(n)
1 , A(n)
2 }, σ(n)) at time n ∈T will later
be employed to denote some of the conceptors in the DCA in their adapted
versions at time n. These conceptor adaptation versions will thus become
identiﬁable by symbols from Σ(n).
For σ(n) I take the natural projection
Cj 7→C(n)
j
, Ai 7→A(n)
i .
2. For every n, n + k ∈T (where k ≥0), φ(n+k,n) : Σ(n+k) →Σ(n), A(n+k)
i
7→
(σ(n) ◦(σ(n+k))−1)(A(n+k)
i
) is a morphism in
Sign.
There are no other
morphisms in Sign besides these. Remark: At ﬁrst sight this might seem
unneccessarily complicated. Why not simply require φ(n+k,n) : A(n+k)
i
7→A(n)
i ?
The reason is that the set of symbols {A(n)
1 , . . . A(n)
m } of Σ(n) is just that, a
set of symbols.
That over time A(n)
i
should correspond to A(n+k)
i
is only
visually suggested to us, the mathematicians, by the chosen notation for
these symbols, but by no means does it actually follow from that notation.
3. Sen(Σ(n)) is inductively deﬁned as follows:
(a) A(n)
1 , . . . , A(n)
m and I and 0 are sentences in Sen(Σ(n)).
(b) For k ≥0 such that n, n + k ∈T, for A(n)
i
∈Σ(n), δ(n)
k
A(n)
i
is in
Sen(Σ(n)). Remark: the δ operators capture the temporal adaptation of
conceptors. The symbol A(n)
i
will be used to denote a conceptor ai(n)
in its adaptation version at time n, and the sentence δ(n)
k
A(n)
i
will be
made to refer to ai(n + k).
(c) If ζ, ξ ∈Sen(Σ(n)), then (ζ ∨ξ), (ζ ∧ξ), ¬ζ ∈Sen(Σ(n)). Remark:
unlike in ECL there is no need for a notational distinction between ∧
and AND etc.
(d) If ζ ∈Sen(Σ(n)), then ϕ(ζ, γ) ∈Sen(Σ(n)) for every γ ∈[0, ∞] (this
captures aperture adaptation).
(e) If ζ, ξ ∈Sen(Σ(n)) and 0 ≤b ≤1, then βb(ζ, ξ) ∈Sen(Σ(n)) (this will
take care of linear blends bζ + (1 −b)ξ).
Remark: Including I and 0 in the sentence syntax is a convenience item. 0
could be deﬁned in terms of any A(n)
i
by 0
∧= (ϕ(A(n)
i , ∞) ∧¬ϕ(A(n)
i , ∞)),
and I by I
∧= ¬0. Likewise, ∨(or ∧) could be dismissed because it can be
expressed in terms of ∧and ¬ (∨and ¬, respectively).
4. For a signature morphism φ(n+k,n) : Σ(n+k) →Σ(n), Sen(φ(n+k,n)) : Sen(Σ(n+k)) →
Sen(Σ(n)) is the map deﬁned inductively as follows:
(a) Sen(φ(n+k,n)) : I 7→I, 0 7→0.
150
(b) Sen(φ(n+k,n)) : A(n+k)
i
7→δ(n)
k
φ(n+k,n)(A(n+k)
i
). Remark 1: When we use
the natural projections σ(n) : Ai 7→A(n)
i , this rule could be more simply
written as Sen(φ(n+k,n)) : A(n+k)
i
7→δ(n)
k
A(n)
i . Remark 2: This is the
pivotal point in this entire deﬁnition, and the point where the diﬀerence
to customary views on logics comes to the surface most conspicuously.
Usually signature morphisms act on sentences by simply re-naming all
signature symbols that occur in a sentence. The structure of a sentence
remains unaﬀected, in agreement with the motto “truth is invariant un-
der change of notation”. By contrast, here a signature symbol A(n+k)
i
is replaced by an temporal change operator term δ(n)
k
A(n)
i , reﬂecting the
new motto “meaning changes with time”. The fact that φ(n+k,n) leads
from A(n+k)
i
to A(n)
i
establishes “sameness of identity over time” be-
tween A(n+k)
i
and A(n)
i . Usually one would formally express sameness
of identity of some mathematical entity by using the same symbol to
name that entity at diﬀerent time points. Here diﬀerent symbols are
used, and thus another mechanism has to be found in order to establish
that an entity named by diﬀerent symbols at diﬀerent times remains
“the same”. The dotted “identity” lines in Figure 45 are ﬁxed by the
signature morphisms φ(n+k,n), not by using the same symbol over time.
Remark 3: At this point it also becomes clear why the signature mor-
phisms φ(n+k,n) : Σ(n+k) →Σ(n) lead backwards in time. A conceptor
ai(n + k) in its time-(n + k) version can be expressed in terms of the
earlier version ai(n) with the aid of the temporal evolution operator δ,
but in general an earlier version ai(n) cannot be expressed in terms of
a later ai(n + k). This reﬂects the fact that, seen as a trajectory of an
input-driven dynamical system, an agent life is (typically) irreversible.
To put it into everyday language, “the future can be explained from the
past, but not vice versa”.
(c) Sen(φ(n+k,n)) : δ(n+k)
l
A(n+k)
i
7→δ(n)
(k+l) φ(n+k,n)(A(n+k)
i
).
(d) For ζ, ξ ∈Sen(Σ(n+k)), put
Sen(φ(n+k,n)) :
(ζ ∨ξ)
7→
(Sen(φ(n+k,n))(ζ) ∨Sen(φ(n+k,n))(ξ)),
(ζ ∧ξ)
7→
(Sen(φ(n+k,n))(ζ) ∧Sen(φ(n+k,n))(ξ)),
¬ζ
7→
¬Sen(φ(n+k,n))(ζ),
ϕ(ζ, γ)
7→
ϕ(Sen(φ(n+k,n))(ζ), γ),
βb(ζ, ξ)
7→
βb(Sen(φ(n+k,n))(ζ), Sen(φ(n+k,n))(ξ)).
5. For every signature Σ(n) ∈Sign, Mod(Σ(n)) is always the same category
Z with objects all non-negative M-dimensional vectors z.∧2. There are no
model morphisms except the identity morphisms z.∧2 id
→z.∧2.
151
6. For every morphism φ(n+k,n) ∈Sign, Mod(φ(n+k,n)) is the identity morphism
of Z.
7. As a preparation for deﬁning the model relationships |=Σ(n) we assign by
induction to every sentence χ ∈Sen(Σ(n)) an M-dimensional conception
weight vector ι(χ) as follows:
(a) ι(I) = (1, . . . , 1)′ and ι(0) = (0, . . . , 0)′.
(b) ι(A(n)
i ) = ιAL(n, (σ(n))−1A(n)
i ).
(c) ι(δ(n)
k A(n)
i ) = ιAL(n + k, (σ(n))−1A(n)
i ).
(d) Case χ = (ζ ∨ξ): ι(χ) = ι(ζ) ∨ι(ξ) (compare Deﬁnition 7).
(e) Case χ = (ζ ∧ξ): ι(χ) = ι(ζ) ∧ι(ξ).
(f) Case χ = ¬ζ: ι(χ) = ¬ι(ζ).
(g) Case χ = ϕ(ζ, γ): ι(χ) = ϕ(ι(ζ), γ) (compare Deﬁnition 6).
(h) Case χ = βb(ζ, ξ): ι(χ) = b ι(ζ) + (1 −b)ι(ξ).
8. For z.∧2 ∈Z = Mod(Σ(n)) and χ ∈Sen(Σ(n)), the model relationship is
deﬁned by
z.∧2 |=Σ(n) χ
iﬀ
z.∧2 .∗(z.∧2 + 1).∧−1 ≤ι(χ).
The satisfaction condition (100) requires that for χ(n+k) ∈Sen(Σ(n+k)) and
z.∧2 ∈Mod(Σ(n)) it holds that
z.∧2 |=Σ(n) Sen(φ(n+k,n))(χ(n+k)) iﬀMod(φ(n+k,n))(z.∧2) |=Σ(n+k) χ(n+k),
which is equivalent to
z.∧2 |=Σ(n) Sen(φ(n+k,n))(χ(n+k)) iﬀz.∧2 |=Σ(n+k) χ(n+k)
(101)
because Mod(φ(n+k,n)) is the identity morphism on Z. This follows directly from
the following fact:
Lemma 1 ι
 Sen(φ(n+k,n))(χ(n+k))

= ι(χ(n+k)),
which in turn can be established by an obvious induction on the structure of
sentences, where the crucial steps are the cases (i) χ(n+k) = A(n+k)
i
and (ii) χ(n+k) =
δ(n+k)
l
A(n+k)
i
.
In case (i), ι(Sen(φ(n+k,n))(A(n+k)
i
)) = ι(δ(n)
k φ(n+k,n)(A(n+k)
i
)) = ι(δ(n)
k (σ(n) ◦
(σ(n+k))−1)(A(n+k)
i
)) = ιAL(n + k, (σ(n)))−1(σ(n) ◦(σ(n+k))−1)(A(n+k)
i
)) = ιAL(n +
k, (σ(n+k))−1(A(n+k)
i
)) = ι(A(n+k)
i
).
152
In case (ii), conclude ι(Sen(φ(n+k,n))(δ(n+k)
l
A(n+k)
i
)) = ι(δ(n)
(k+l) φ(n+k,n)(A(n+k)
i
)) =
ι(δ(n)
(k+l) (σ(n)◦(σ(n+k))−1)(A(n+k)
i
)) = ιAL(n+k+l, (σ(n)))−1(σ(n)◦(σ(n+k))−1)(A(n+k)
i
)) =
ιAL(n + k + l, (σ(n+k))−1(A(n+k)
i
)) = ι(δ(n+k)
l
A(n+k)
i
).
An important diﬀerence between “traditional” logics and the ICL of an agent
AL concerns diﬀerent intuitions about semantics. Taking ﬁrst-order logic as an
example of a traditional logic, the “meaning” of a ﬁrst-order sentence χ with
signature Σ is the class of all of its models. Whether a set is a model of χ depends
on how the symbols from Σ are extensionally interpreted over that set. First-
order logic by itself does not prescribe how the symbols of a signature have to be
interpreted over some domain. In contrast, an ICL is deﬁned with respect to a
concrete agent AL, which in turn uniquely ﬁxes how the symbols from an ICL
signature Σ(n) must be interpreted – this is the essence of points 7. (a – c) in
Deﬁnition 13.
Logical entailment in an ICL coincides with abstraction of conceptors:
Proposition 18 In an ICL, for all ζ, ξ ∈Sen(Σ(n)) it holds that
ζ |=Σ(n) ξ
iﬀ
ι(ζ) ≤ι(ξ).
(102)
The simple proof is given in Section 5.11.
In an agent’s lifetime ICL, for
any sentence χ ∈Sen(Σ(n)) the concrete interpretation ι(χ) ∈[0, 1]M can be
eﬀectively computed via the rules stated in Nr. 7 in Deﬁnition 13, provided one
has access to the identiﬁable conceptors ai(n) in the agent’s life. Since for two
vectors a, b ∈[0, 1]M it can be eﬀectively checked whether a ≤b, it is decidable
whether ι(ζ) ≤ι(ξ). An ICL is therefore decidable.
Seen from a categorical point of view, an ICL is a particularly small-size in-
stitution. Since the (only) category in the image of Mod is a set, we can regard
the functor Mod as having codomain Setop instead of Catop. Also the category
Sign is small, that is, a set. Altogether, an ICL institution nowhere needs proper
classes.
The ICL deﬁnition I gave here is very elementary. It could easily be augmented
in various natural ways, for instance by admitting permutations and/or projections
of conceptor vector components as model morphisms in Z, or allowing conceptors
of diﬀerent dimensions in the makeup of an agent life. Likewise it is straightforward
to spell out deﬁnitions for an agent life and its ICL for matrix-based conceptors.
In the latter case, the referents of sentences are correlation matrices R, and the
deﬁning equation for the model relationship appears as
R |=Σ(n) χ
iﬀ
R(R + I)−1 ≤ι(χ).
In traditional logics and their applications, an important role is played by cal-
culi. A calculus is a set of syntactic transformation rules operating on sentences
(and expressions containing free variables) which allows one to derive purely syn-
tactical proofs of logical entailment statements ζ |= ξ. Fundamental properties of
153
familiar logics, completeness and decidability in particular, are deﬁned in terms
of calculi.
While it may be possible and mathematically interesting to design
syntactical calculi for ICLs, they are not needed because ζ |= ξ is decidable via
the semantic equivalent ι(ζ) ≤ι(ξ). Furthermore, the ICL decision procedure
is computationally very eﬀective: only time O(1) is needed (admitting a parallel
execution) to determine whether some conception vector is at most as large as
another in all components. This may help to explain why humans can so quickly
carry out many concept subsumption judgements (“this looks like a cow to me”).
3.18
Final Summary and Outlook
Abstracting from all technical detail, here is a summary account of the conceptor
approach:
From neural dynamics to conceptors. Conceptors capture the shape of a neu-
ral state cloud by a positive semi-deﬁnite operator.
From conceptors to neural dynamics. Inserting a conceptor into a neurody-
namical state update loop allows to select and stabilize a previously stored
neural activation pattern.
Matrix conceptors are useful in machine learning applications and as mathe-
matical tools for analysing patterns emerging in nonlinear neural dynamics.
Random feature conceptors are not biologically apriori implausible and can
be neurally coded by single neurons.
Autoconceptor adaptation dynamics leads to content-addressable neural mem-
ories of dynamical patterns and to signal ﬁltering and classiﬁcation systems.
Boolean operations and the abstraction ordering on conceptors establish
a bi-directional connection between logic and neural dynamics.
From static to dynamic models. Conceptors allow to understand and control
dynamical patterns in scenarios that previously have been mostly restricted
to static patterns. Speciﬁcally this concerns content-addressable memories,
morphing, ordering concepts in logically structured abstraction hierarchies,
and top-down hypothesis control in hierarchical architectures.
The study of conceptors is at an early stage.
There are numerous natural
directions for next steps in conceptor research:
Aﬃne conceptor maps. Conceptors constrain reservoir states by applying a
positive semi-deﬁnite map. This map is adapted to the “soft-bounded” linear
subspace visited by a reservoir when it is driven through a pattern. If the
pattern-driven reservoir states do not have zero mean, it seems natural to
154
ﬁrst subtract the mean before applying the conceptor. If the mean is µ, this
would result in an update loop of the form x(n + 1) = µ + C(tanh(...) −µ).
While this may be expected to improve control characteristics of pattern
re-generation, it is not immediately clear how Boolean operations transfer
to such aﬃne conceptors which are characterized by pairs (C, µ).
Nonlinear conceptor ﬁlters. Even more generally, one might conceive of con-
ceptors which take the form of nonlinear ﬁlters, for instance instantiated as
feedforward neural networks. Such ﬁlters F could be trained on the same ob-
jective function as I used for conceptors, namely minimizing E[∥z−F(z)∥2]+
α−2∥F∥2, where ∥F∥is a suitably chosen norm deﬁned for the ﬁlter. Like
with aﬃne conceptor maps, the pattern speciﬁcity of such nonlinear con-
ceptor ﬁlters would be greater than for our standard matrix conceptors, but
again it is not clear how logical operations would extend to such ﬁlters.
Basic mathematical properties. From a mathematics viewpoint, there are some
elementary questions about conceptors which should be better understood,
for instance:
• What is the relationship between Boolean operations, aperture adap-
tation, and linear blending? in particular, can the latter be expressed
in terms of the two former?
• Given a dimension N, what is the minimial number of “basis” concep-
tors such that the transitive closure under Boolean operations, aperture
adaptation, and/or linear blending is the set of all N-dimensional con-
ceptors?
• Are there normal forms for expressions which compose conceptors by
Boolean operations, aperture adaptation, and/or linear blending?
• Find conceptor analogs of standard structure-building mathematical
operations, especially products. These would be needed to design archi-
tectures with several conceptor modules (likely of diﬀerent dimension)
where the “soft subspace constraining” of the overall dynamics works
on the total architecture state space. Presumably this leads to tensor
variants of conceptors. This may turn out to be a challenge because
a mathematical theory of “semi positive-deﬁnite tensors” seems to be
only in its infancy (compare [84]).
Neural realization of Boolean operations. How can Boolean operations be
implemented in biologically not impossible neural circuits?
Complete analysis of autoconceptor adaptation. The analysis of autocon-
ceptor adaptation in Section 3.13.4 is preliminary and incomplete. It only
characterizes certain aspects of ﬁxed-point solutions of this adaptation dy-
namics but remains ignorant about the eﬀects that the combined, nonlinear
155
conceptor-reservoir update dynamics may have when such a ﬁxed-point solu-
tion is perturbed. Speciﬁcally, this reservoir-conceptor interaction will have
to be taken into account in order to understand the dynamics in the center
manifold of ﬁxed-point solutions.
Applications. The usefulness of conceptors as a practical tool for machine learn-
ing and as a modeling tool in the cognitive and computational neurosciences
will only be established by a suite of successful applications.
156
4
Documentation of Experiments and Methods
In this section I provide details of all simulation experiments reported in Sections
1 and 3.
4.1
General Set-Up, Initial Demonstrations (Section 1 and
Section 3.2 - 3.4)
A reservoir with N = 100 neurons, plus one input unit and one output unit was
created with a random input weight vector W in, a random bias b and preliminary
reservoir weights W ∗, to be run according to the update equations
x(n + 1)
=
tanh(W ∗x(n) + W in p(n + 1) + b),
(103)
y(n)
=
W out x(n).
Initially W out was left undeﬁned.
The input weights were sampled from a
normal distribution N(0, 1) and then rescaled by a factor of 1.5. The bias was
likewise sampled from N(0, 1) and then rescaled by 0.2. The reservoir weight ma-
trix W ∗was ﬁrst created as a sparse random matrix with an approximate density
of 10%, then scaled to obtain a spectral radius (largest absolute eigenvalue) of 1.5.
These scalings are typical in the ﬁeld of reservoir computing [69] for networks to
be employed in signal-generation tasks.
For each of the four driving signals pj a training time series of length 1500
was generated. The reservoir was driven with these pj(n) in turn, starting each
run from a zero initial reservoir state. This resulted in reservoir state responses
xj(n). The ﬁrst 500 steps were discarded in order to exclude data inﬂuenced by
the arbitrary starting condition, leading to four 100-dimensional reservoir state
time series of length L = 1000, which were recorded into four 100 × 1000 state
collection matrices Xj, where Xj(:, n) = xj(n + 500) (j = 1, . . . , 4). Likewise,
the corresponding driver signals were recorded into four pattern collection (row)
vectors P j of size 1 × 1000. In addition to this, a version ˜Xj of Xj was built,
identical to Xj except that it was delayed by one step:
˜Xj(:, n) = xj(n + 499).
These collections were then concatenated to obtain X = [X1|X2|X3|X4], ˜X =
[ ˜X1| ˜X2| ˜X3| ˜X4], P = [P 1|P 2|P 3|P 4].
The “PC energy” plots in Figure 12 render the singular values of the correlation
matrices Xj(Xj)′/L.
The output weights W out were computed as the regularized Wiener-Hopf so-
lution (also known as ridge regression, or Tychonov regularization)
W out = ((XX′ + ϱoutIN×N)−1 X P ′)′,
(104)
where the regularizer ϱout was set to 0.01.
Loading: After loading, the reservoir weights W should lead to the approximate
equality Wxj(n) ≈W ∗xj(n) + W in pj(n + 1), across all patterns j, which leads to
157
the objective of minimizing the squared error ϵj(n + 1) = ((tanh−1(xj(n + 1)) −
b) −Wxj(n))2, averaged over all four j and training time points. Writing B for
the 100 × (4 ∗1000) matrix whose columns are all identical equal to b, this has the
ridge regression solution
W = (( ˜X ˜X′ + ϱWIN×N)−1 ˜X (tanh−1(X) −B))′,
(105)
where the regularizer ϱW was set to 0.0001. To assess the accuracy of the weight
computations, the training normalized root mean square error (NRMSE) was com-
puted. For the readout weights, the NRMSE between y(n) = W outx(n) and the
target P was 0.00068. For the reservoir weights, the average (over reservoir neu-
rons i, times n and patterns j) NRMSE between W(i, :)xj(n) and the target
W ∗(i, :) xj(n) + W in(i) pj(n + 1) was 0.0011.
In order to determine the accuracy of ﬁt between the original driving signals
pj and the network observation outputs yj(n) = W out Cj tanh(W x(n −1) + b) in
the conceptor-constrained autonomous runs, the driver signals and the yj signals
were ﬁrst interpolated with cubic splines (oversampling by a factor of 20). Then
a segment length 400 of the oversampled driver (corresponding to 20 timesteps
before interpolation) was shifted over the oversampled yj in search of a position of
best ﬁt. This is necessary to compensate for the indeterminate phaseshift between
the driver data and the network outputs. The NRMSEs given in Figure 12 were
calculated from the best-ﬁt phaseshift position, and the optimally phase-shifted
version of yj was also used for the plot.
4.2
Aperture Adaptation (Sections 3.8.3 and 3.8.4)
Data generation. For the R¨ossler attractor, training time series were obtained
from running simple Euler approximations of the following ODEs:
˙x
=
−(y + z)
˙y
=
x + a y
˙z
=
b + x z −c z,
using parameters a = b = 0.2, c = 8. The evolution of this system was Euler
approximated with stepsize 1/200 and the resulting discrete time series was then
subsampled by 150. The x and y coordinates were assembled in a 2-dimensional
driving sequence, where each of the two channels was shifted/scaled to a range of
[0, 1]. For the Lorenz attractor, the ODE
˙x
=
σ(y −x)
˙y
=
r x −y −x z
˙z
=
x y −b z
158
with σ = 10, r = 28, b = 8/3 was Euler-approximated with stepsize 1/200 and
subsequent subsampling by 15. The x and z coordinates were collected in a 2-
dimensional driving sequence, again each channel normalized to a range of [0, 1].
The Mackey Glass timeseries was obtained from the delay diﬀerential equation
˙x(t) =
β x(t −τ)
1 + x(t −τ)n −γ x(t)
with β = 0.2, n = 10, τ = 17, γ = 0.1, a customary setting when this attractor
is used in neural network demonstrations. An Euler approximation with stepsize
1/10 was used. To obtain a 2-dim timeseries that could be fed to the reservoir
through the same two input channels as the other attractor data, pairs x(t), x(t−τ)
were combined into 2-dim vectors. Again, these two signals were normalized to
the [0, 1] range. The H´enon attractor is governed by the iterated map
x(n + 1)
=
y(n) + 1 −a x(n)
y(n + 1)
=
b x(n),
where I used a = 1.4, b = 0.3.
The two components were ﬁled into a 2-dim
timeseries (x(n), y(n))′ with no further subsampling, and again normalization to
a range of [0, 1] in each component.
Reservoir setup. A 500-unit reservoir RNN was created with a normal dis-
tributed, 10%-sparse weight matrix W ∗scaled to a spectral radius of 0.6. The
bias vector b and input weights W in (sized 400 × 2 for two input channels) were
sampled from standard normal distribution and then scaled by 0.4 and 1.2, re-
spectively. These scaling parameters were found by a (very coarse) manual opti-
mization of the performance of the pattern storing process. The network size was
chosen large enough to warrant a robust trainability of the four chaotic patterns.
Repeated executions of the experiment with diﬀerent randomly initialized weights
(not documented) showed no signiﬁcant diﬀerences.
Pattern storing. The W ∗reservoir was driven, in turn, by 2500 timesteps of
each of the four chaotic timeseries. The ﬁrst 500 steps were discarded to account
for initial reservoir state washout, and the remaining 4 × 2000 reservoir states
were collected in a 500 × 8000 matrix X. From this, the new reservoir weights W
were computed as in (105), with a regularizer ϱW = 0.0001. The readout weights
were computed as in (104) with regularizer ϱout = 0.01. The average NRMSEs
obtained for the reservoir and readout weights were 0.0082 and 0.013, respectively.
Computing conceptors. From each of the four n = 2000 step reservoir state
sequences X recorded in the storing procedure, obtained from driving the reser-
voir with one of the four chaotic signals, a preliminary correlation matrix ˜R =
XX′/2000 and its SVD U ˜SU ′ = ˜R were computed. This correlation matrix was
then used to obtain a conceptor associated with the respective chaotic signal, using
an aperture α = 1. From these unit-aperture conceptors, versions with diﬀering
α were obtained through aperture adaptation per (16).
159
In passing I note that the overall stability and parameter robustness of this sim-
ulation can be much improved if small singular values in ˜S (for instance, with val-
ues smaller than 1e-06) are zeroed, obtaining a clipped S, from which a “cleaned-
up” correlation matrix R = USU ′ would be computed. This would lead to a range
of well-working apertures spanning three orders of magnitude (not shown). I did
not do this in the reported simulation in order to illustrate the eﬀects of too large
apertures; these eﬀects would be partly suppressed when the spectrum of R is
clipped.
Pattern retrieval and plotting. The loaded network was run using the conceptor-
constrained update rule x(n+1) = C tanh(W x(n)+W in p(n+1)+b) with various
C = ϕ(C#, γ#,i) (# = R, L, MG, H, i = 1, . . . , 5) for 800 steps each time, of which
the ﬁrst 100 were discarded to account for initial state washout. The delay embed-
ding plots in Figure 18 were generated from the remaining 700 steps. Embedding
delays of 2, 2, 3, 1 respectively were used for plotting the four attractors.
For each of the four 6-panel blocks in Figure 18, the ﬁve aperture adaptation
factors γ#,i were determined in the following way. First, by visual inspection, the
middle γ#,3 was determined to fall in the trough center of the attenuation plot of
Fig. 19 A. Then the remaining γ#,1, γ#,2, γ#,4, γ#,5 were set in a way that (i) the
entire γ sequence was a geometrical progression, and (ii) that the plot obtained
from the ﬁrst γ#,1 was visually strongly corrupted.
4.3
Memory Management (Section 3.11)
For both reported simulation studies, the same basic reservoir was used: size
N = 100, W ∗sparse with approximately 10% nonzero entries, these sampled
from a standard normal distribution and later rescaled to obtain a spectral radius
of 1.5. The input weights W in were non-sparse, sampled from a standard normal
distribution and then scaled by a factor of 1.5. The bias vector b was sampled from
a standard normal distribution and scaled by a factor of 0.25. The bias serves the
important function to break the sign symmetry of the reservoir dynamics. Without
it, in autonomous pattern generation runs a pattern pj or its sign-reversed version
−pj could be generated equally well, which is undesirable.
The driver patterns used in the ﬁrst demonstration were either sinewaves with
integer period lengths, or random periodic patterns scaled to a range of [−0.9, 0.9].
Period lengths were picked randomly between 3 and 15.
The drivers for the second demonstrations were the sinewaves sin(2∗π∗n/(3
√
2·
(1.1)i)) (where i = 1, . . . , 16), leading to periods exponentially spread over the
range [4.24, 19.49]. These signals were randomly permuted to yield the presenta-
tion order j = randperm(i).
The lengths of pattern signals used for training were L = 100 in demo 1 and
L = 1000 in demo 2, with additional washouts of the same lengths.
The readout weights W out were trained by the generic procedure detailed in
Section 4.1 after all patterns had been used to drive the network in turn for the
160
incremental storage, and thus having available all state collections Xj.
For testing, the conceptors Cj obtained during the storage procedure were used
by running the reservoir via x(n+1) = Cj tanh(W ∗x(n)+D x(n)+b) from random
initial states. After a washout time, the network outputs yj(n) were recorded for
durations of 200 and 400, respectively, in the two experiments. A 20-step portion
of the original driver pattern was interpolated by a factor of 10 and shifted over
a likewise interpolation-reﬁned version of these outputs. The best matching shift
position was used for plotting and NRMSE computation (such shift-search for a
good ﬁt is necessary because the autonomous runs are not phase-synchronized to
the original drivers, and irrational-period sinewaves need interpolation because
they may be phase-shifted by noninteger amounts).
4.4
Content-Addressable Memory (Section 3.13.3)
Network setup. Reservoir network matrices W were sampled sparsely (10% nonzero
weights) from a normal distribution, then scaled to a spectral radius of 1.5 in all
experiments of this section. Reservoir size was N = 100 for all period-4 experi-
ments and the unrelated patterns experiment, and N = 200 for all experiments
that used mixtures-of-sines patterns. For all experiments in the section, input
weights W in were randomly sampled from the standard normal distribution and
rescaled by a factor of 1.5. The bias vector b was likewise sampled from the stan-
dard normal distribution and rescaled by 0.5. These scaling parameters had been
determined by a coarse manual search for a well-working conﬁguration when this
suite experiments was set up. The experiments are remarkably insensitive to these
parameters.
Storing patterns. The storage procedure was set up identically for all experi-
ments in this section. The reservoir was driven by the k loading patterns in turn
for l = 50 steps (period-4 and unrelated patterns) or l = 500 steps (mix-of-sines)
time steps, plus a preceding 100 step initial washout. The observed network states
x(n) were concatenated into a N ×kl sized state collection matrix X, and the one-
step earlier states x(n−1) into a matrix ˜X of same size. The driver pattern signals
were concatenated into a kl sized row vector P. The readout weights W out were
then obtained by ridge regression via W out = ((XX′ + 0.01 I)−1 XP ′)′, and D by
D = (( ˜X ˜X′ + 0.001 I)−1 ˜X(W in P)′)′.
Quality measurements. After the cueing, and at the end of the recall period (or
at the ends of the three interim intervals for some of the experiments), the current
conceptor C was tested for retrieval accuracy as follows. Starting from the current
network state x(n), the reservoir network was run for 550 steps, constrained by C.
The ﬁrst 50 steps served as washout and were discarded. The states x from the
last 500 steps were transformed to patterns by applying W out, yielding a 500-step
pattern reconstruction. This was interpolated with cubic splines and then sampled
at double resolution, leading to a 999-step pattern reconstruction ˜y.
A 20-step template sample of the original pattern was similarly interpolated-
161
resampled and then passed over ˜y, detecting the best-ﬁtting position where the
the NRMSE between the target template and ˜y was minimal (this shift-search
accomodated for unknown phase shifts between the target template and ˜y). This
minimal NRMSE was returned as measurement result.
Irrational-period sines. This simulation was done exactly as the one before,
using twelve irrational-period sines as reference patterns.
4.5
The Japanese Vowels Classiﬁcation (Section 3.12)
Network setup. In each of the 50 trials, the weights in a fully connected, 10 × 10
reservoir weight matrix, a 12-dimensional input weight vector, a 10-dimensional
bias vector, and a 10-dimensional start state were ﬁrst sampled from a normal
distribution, then rescaled to a spectral radius of 1.2 for W, and by factors of 0.2,
1, 1 for W in, b, xstart respectively.
Numerical determination of best aperture. To determine γ+
i , the quantities
∥ϕ( ˜C+
i , 2g)∥2
fro were computed for g = 0, 1, . . . , 8. These values were interpolated
on a 0.01 raster with cubic splines, the support point gmax of the maximum of the
interpolation curve was detected, returning γ+
i = 2gmax.
Linear classiﬁer training. The linear classiﬁer that serves as a baseline compar-
ison was designed in essentially the same way as the Echo State Networks based
classiﬁers which in [57] yielded zero test misclassiﬁcations (when combining 1,000
such classiﬁers made from 4-unit networks) and 2 test misclassiﬁcations (when a
single such classiﬁer was based on a 1,000 unit reservoir), respectively. Thus, lin-
ear classiﬁers based on reservoir responses outperform all other reported methods
on this benchmark and therefore provide a substantive baseline information.
In detail, the linear classiﬁer was learnt from 270 training data pairs of the form
(z, yteacher), where the z were the same 88-dimensional vectors used for constructing
conceptors, and the yteacher were 9-dimensional, binary speaker indicator vectors
with a “1” in the position of the speaker of z. The classiﬁer consists in a 9 ×
88 sized weight matrix V , and the cost function was the quadratic error ∥V z −
yteacher∥2. The classiﬁer weight matrix V which minimized this cost function on
average over all training samples was computed by linear regression with Tychonov
regularization, also known as ridge regression [106].
The Tychonov parameter
which determines the degree of regularization was determined by a grid search over
a 5-fold cross-validation on the training data. Across the 50 trials it was found
to vary quite widely in a range between 0.0001 and 0.25; in a separate auxiliary
investigation it was also found that the eﬀect of variation of the regularizer within
this range was very small and the training of the linear classiﬁer can therefore be
considered robust.
In testing, V was used to determine classiﬁcation decisions by computing ytest =
V ztest and opting for the index of the largest entry in ytest as the speaker.
162
4.6
Conceptor Dynamics Based on RFC Conceptors (Sec-
tion 3.14)
Network setup. In both experiments reported in this section, the same reservoir
made from N = 100 units was used. F and G were full matrices with entries
ﬁrst sampled from the standard normal distribution. Then they were both scaled
by an identical factor a such that the product a2 G F attained a spectral radius
of 1.4.
The input weight vector W in was sampled from the standard normal
distribution and then scaled by 1.2. The bias b was likewise sampled from the
normal distribution and then scaled by 0.2.
These values were determined by coarse manual search, where the main guiding
criterion was recall accuracy. The settings were rather robust in the ﬁrst experi-
ment which used stored cj. The spectral radius could be individually varied from
0.6 to 1.45, the input weight scaling from 0.3 to 1.5, the bias scaling from 0.1 to
0.4, and the aperture from 3 to 8.5, while always keeping the ﬁnal recall NRMSEs
for all four patterns below 0.1. Furthermore, much larger combined variations of
these scalings were also possible (not documented).
In the second experiment with content-addressed recall, the functional param-
eter range was much narrower. Individual parameter variation beyond ±5% was
disruptive. Speciﬁcally, I observed a close inverse coupling between spectral radius
and aperture: if one of the two was raised, the other had to be lowered.
Loading procedure. The loading procedure is described in some detail in the
report text. The mean NRMSEs on training data was 0.00081 for recomputing G,
0.0011 for D, and 0.0029 for W out. The mean absolute size of matrix elements in
G was 0.021, about a third of the mean absolute size of elements of G∗.
Computing NRMSEs. The NRMSE comparison between the re-generated pat-
terns at the end of the c adaptation and the original drivers was done in the
same way as reported on earlier occasions (Section 4.4), that is, invoking spline
interpolation of the comparison patterns and optimal phase-alignment.
4.7
Hierarchical Classiﬁcation and Filtering Architecture
(Section 3.15)
Module setup. The three modules are identical copies of each other. The reservoir
had N = 100 units and the feature space had a dimension of M = 500. The
input weight matrix W in was sampled from the standard normal distribution and
rescaled by a factor of 1.2. The reservoir-featurespace projection and backprojec-
tion matrices F, G∗(sized N × M) were ﬁrst sampled from the standard normal
distribution, then linearly rescaled by a common factor such that the N × N ma-
trix G∗F ′ (which functionally corresponds to an internal reservoir weight matrix)
had a spectral radius of 1.4. The bias b was likewise sampled from the standard
normal distribution and then scaled by 0.2.
A regularization procedure was then applied to G∗to give G as follows. The
163
preliminary module was driven per
z(n + 1) = F ′r(n),
r(n + 1) = tanh(G∗z(n + 1) + W inu(n) + b),
with an i.i.d. input signal u(n) sampled uniformly from [−1, 1], for 1600 steps (after
discarding an initial washout). The values obtained for z(n + 1) were collected
as columns in a M × 1600 matrix Z. The ﬁnal G was then computed by a ridge
regression with a regularizer a = 0.1 by
G = ((ZZ′ + aI)−1 Z (G∗Z)′)′.
In words, G should behave as the initially sampled G∗in a randomly driven mod-
ule, but do so with minimized weight sizes. This regularization was found to be
important for a stable working of the ﬁnal architecture.
Training. The input simulation weights D (size 1 × M) and W out (size 1 × N)
were trained by driving a single module with the four target patterns, as follows.
The module was driven with clean p1, . . . , p4 in turn, with auto-adaptation of
conception weights c activated:
z(n + 1)
=
c(n) .∗F ′ r(n),
r(n + 1)
=
tanh(G z(n + 1) + W in pj(n) + b),
c(n + 1)
=
c(n) + λc
 (z(n + 1) −c(n) .∗z(n + 1)) .∗z(n + 1) −α−2 c(n)

,
where the c adaptation rate was set to λc = 0.5, and an aperture α = 8 was used.
After discarding initial washouts in each of the four driving conditions (long enough
for c(n) to stabilize), 400 reservoir state vectors r(n + 1), 400 z vectors z(n) and
400 input values pj(n) were collected for j = 1, . . . , 4, and collected column-wise in
matrices R (size N ×1600), Z (size M ×1600) and Q (size 1×1600), respectively.
In addition, the 400-step submatrices Zj of Z containing the z-responses of the
module when driven with pj were registered separately.
The output weights were then computed by ridge regression on the objective
to recover pj(n) from r(n + 1) by
W out =
 (RR′ + aI)−1 RQ′′ ,
using a regularizer a = 0.1. In a similar way, the input simulation weights were
obtained as
D =
 (ZZ′ + aI)−1 ZQ′′
with a regularizer a = 0.1 again. The training NRMSEs for W out and D were
0.0018 and 0.0042, respectively.
The M × 4 prototype matrix P was computed as follows. First, a prelimi-
nary version P ∗was constructed whose j-the column vector was the mean of the
element-wise squared column vectors in Zj. The four column vectors of P ∗were
then normalized such that the norm of each of them was the mean of the norms
164
of columns in P ∗. This gave P. This normalization is important for the per-
formance of the architecture, because without it the optimization criterion (94)
would systematically lead to smaller values for those γj that are associated with
smaller-norm columns in P.
Baseline linear ﬁlter. The transversal ﬁlter that served as a baseline was a row
vector w of size 2600. It was computed to minimize the loss function
L(w) = 1
4
4
X
j=1
E[pj(n + 1) −(pj(n −2600 + 1), . . . , pj(n))2] + a2∥w∥2,
where pj(n) were clean versions of the four patterns. 400 timesteps per pattern
were used for training, and a was set to 1.0. The setting of a was very robust.
Changing a in either direction by factors of 100 changed the resulting test NRMSEs
at levels below the plotting accuracy in Figure 42.
Parameter settings in testing. The adaptation rate λγ was set to 0.002 for the
classiﬁcation simulation and to 0.004 for the morph-tracking case study. The other
global control parameters were identical in both simulations: trust smoothing rate
σ = 0.99, decisiveness d[12] = d[23] = 8, drift d = 0.01, caut
[l] adaptation rate λ = 0.5
(compare Equation (80); I used the same adaptation rate λi ≡λ for all of the 500
feature units).
Computing and plotting running NRMSE estimates. For the NRMSE plots in
the ﬁfth rows in Figures 42 and 43, a running estimate of the NRMSE between
the module outputs y[l] and the clean input patterns p (unknown to the system)
was computed as follows. A running estimate var p(n) of the variance of the clean
pattern was maintained like it was done for var y[l](n) in (88) and (89), using an
exponential smoothing rate of σ = 0.95. Then the running NRMSE was computed
by another exponential smoothing per
nrmse y[l](n + 1) = σ nrmse y[l](n) + (1 −σ)
(p(n + 1) −y[l](n + 1))2
var p(n + 1)
1/2
.
The running NRMSE for the baseline transversal ﬁlter were obtained in a similar
fashion.
165
5
Proofs and Algorithms
5.1
Proof of Proposition 1 (Section 3.4)
Claim 1. We ﬁrst re-write the minimization quantity, using R = E[xx′]:
E[∥x −Cx∥2] + α−2 ∥C∥2
fro =
=
E[tr (x′(I −C′)(I −C)x)] + α−2 ∥C∥2
fro
=
tr ((I −C′)(I −C)R + α−2 C′C)
=
tr (R −C′R −CR + C′C(R + α−2 I))
=
X
i=1,...,N
e′
i(R −C′R −CR + C′C(R + α−2 I))ei.
This quantity is quadratic in the parameters of C and non-negative.
Because
tr C′C(R+α−2 I) = tr C(R+α−2 I)C′ and R+α−2 I is positive deﬁnite, tr C(R+
α−2 I)C′ is positive deﬁnite in the N 2-dimensional space of C elements. Therefore
E[∥x −Cx∥2] + α−2 ∥C∥2
fro has a unique minimum in C space. To locate it we
compute the derivative of the i-th component of this sum with respect to the entry
C(k, l) = Ckl:
∂
∂Ckl
e′
i(R −C′R −CR + C′C(R + α−2 I))ei =
=
−2 Rkl + ∂/∂Ckl
X
j,a=1,...,N
Cij Cja Aai
=
−2 Rkl +
X
a=1,...,N
∂/∂Ckl Ckj Cka Aai
=
−2 Rkl +
X
a=1,...,N
(Cka Aal + Cki Ali)
=
−2 Rkl + (CA)kl + N Cki Ail.
where we used the abbreviation A = R + α−2 I, observing in the last line that
A = A′. Summing over i yields
∂
∂Ckl
E[∥x −Cx∥2] + α−2 ∥C∥2
fro = −2 N Rkl + 2N (CA)kl,
which in matrix form is
∂
∂Ckl
E[∥x −Cx∥2] + α−2 ∥C∥2
fro = −2 N R + 2N C(R + α−2 I),
where we re-inserted the expression for A. Setting this to zero yields the claim 1.
stated in the proposition.
The subclaim that R(R + α−2I)−1 = (R + α−2I)−1 R can be easily seen when
R is written by its SVD R = UΣU ′: UΣU ′(UΣU ′ + α−2I)−1 = UΣU ′(U(Σ +
α−2I)U ′)−1 = UΣU ′U(Σ+α−2I)−1U ′ = UΣ(Σ+α−2I)−1U ′ = U(Σ+α−2I)−1ΣU ′ =
... = (R + α−2I)−1 R. This also proves claim 2.
Claims 3.–5. can be derived from the ﬁrst claim by elementary arguments.
166
5.2
Proof of Proposition 6 (Section 3.9.3)
C−1
δ
can be written as US−1
δ U ′, where the diagonal of S−1
δ
is (s−1
1 , . . . , s−1
l , δ−1, . . . , δ−1)′.
Putting e = δ−1 and Se := (s−1
1 , . . . , s−1
l , e, . . . , e)′, and similarly Te = (t−1
1 , . . . , t−1
m ,
e, . . . , e)′, we can express the limit limδ→0(C−1
δ
+ B−1
δ
−I)−1 equivalently as
lime→∞(USeU ′ + V TeV ′ −I)−1. Note that USeU ′ + V TeV ′ −I is invertible for
suﬃciently large e.
Let U>l be the N × (N −l) submatrix of U made from
the last N −l columns of U (spanning the null space of C), and let V>m be
N × (N −m) submatrix of V made from the last N −m columns of V . The
N × N matrix U>l(U>l)′ + V>m(V>m)′ is positive semideﬁnite.
Let WΣW ′ be
its SVD, with singular values in Σ in descending order.
Noting that C† =
Udiag(s−1
1 , . . . , s−1
l , 0, . . . , 0)′U ′, and B† = V diag(t−1
1 , . . . , t−1
m , 0, . . . , 0)′V ′, we can
rewrite
lim
δ→0 Cδ ∧Bδ
=
lim
e→∞(USeU ′ + V TeV ′ −I)−1
=
lim
e→∞(C† + B† + e WΣW ′ −I)−1
=
W

lim
e→∞(W ′C†W + W ′B†W + eΣ −I)−1
W ′.
(106)
If Σ is invertible, clearly (106) evaluates to the zero matrix. We proceed to
consider the case of non-invertible Σ = diag(σ1, . . . , σk, 0, . . . , 0), where 0 ≤k <
N.
We derive two auxiliary claims. Let W>k be the N × (N −k) submatrix of
W made from the last N −k columns. Claim 1: the (N −k) × (N −k) matrix
A = (W>k)′ (C† + B† −I) W>k is invertible. We rewrite
A = (W>k)′C†W>k + (W>k)′B†W>k −I(N−k)×(N−k),
(107)
and analyse (W>k)′C†W>k. It holds that (W>k)′U>l = 0(N−k)×(N−l), because
W





σ1 ... σk 0
...
0




W ′
=
U>l(U>l)′ + V>m(V>m)′
=⇒





σ1 ... σk 0
...
0





=
W ′U>l(U>l)′W + W ′V>m(V>m)′W
=⇒
0(N−k)×(N−k)
=
(W>k)′U>l(U>l)′W>k + (W>k)′V>m(V>m)′W>k
=⇒
(W>k)′U>l
=
0(N−k)×(N−l).
(108)
167
Let U≤l be the N × l submatrix of U made of the ﬁrst l columns. Because of
(108) it follows that
(W>k)′C†W>k
=
(W>k)′U







s−1
1
...
s−1
l
0
...
0







U ′W>k
=
(W>k)′U≤l


s−1
1
...
s−1
l

(U≤l)′W>k
(109)
The rows of the (N −k)×l sized matrix (W>k)′U≤l are orthonormal, which also
follows from (108). The Cauchy interlacing theorem (stated in Section 3.13.4 in
another context) then implies that all singular values of (W>k)′C†W>k are greater
or equal to min{s−1
1 , . . . , s−1
l }. Since all si are smaller or equal to 1, all singular
values of (W>k)′C†W>k are greater or equal to 1.
The singular values of the
matrix (W>k)′C†W>k −1/2 I(N−k)×(N−k) are therefore greater or equal to 1/2.
Speciﬁcally, (W>k)′C†W>k −1/2 I(N−k)×(N−k) is positive deﬁnite.
By a similar
argument, (W>k)′B†W>k −1/2 I(N−k)×(N−k) is positive deﬁnite.
The matrix A
from Claim 1 is therefore revealed as the sum of two positive deﬁnite matrices,
and hence is invertible.
Claim 2:
If M is a symmetric N × N matrix, and the right lower prin-
cipal submatrix M>k>k = M(k + 1 : N, k + 1 : N) is invertible, and Σ =
diag(σ1, . . . , σk, 0, . . . , 0) with all σi > 0, then
lim
e→∞(M + eΣ)−1 =
 0
0
0
M −1
>k>k

.
(110)
We exploit the following elementary block representation of the inverse of a
symmetric matrix (e.g. [9], fact 2.17.3):
 X Y
Y ′ Z
−1
=

V −1
−V −1Y Z−1
−Z−1Y ′V −1
Z−1Y ′V −1Y Z−1 + Z−1

,
where Z is assumed to be invertible and V = (X −Y Z−1Y ′) is assumed to be
invertible. Block-structuring M + eΣ analogously to this representation, where
M>k>k is identiﬁed with Z, then easily leads to the claim (110).
Applying Claims 1 and 2 to the limit expression lime→∞(W ′C†W + W ′B†W +
eΣ −I)−1 in (106), where the matrix M in Claim 2 is identiﬁed with W ′C†W +
W ′B†W −I and the matrix Z from Claim 2 is identiﬁed with the matrix A =
(W>k)′ (C† + B† −I) W>k from Claim 1, yields
lim
e→∞(W ′C†W + W ′B†W + eΣ −I)−1 =
 0
0
0
 (W>k)′ (C† + B† −I) W>k
−1

168
which, combined with (106), leads to
lim
δ→0 Cδ ∧Bδ = W>k
 (W>k)′ (C† + B† −I) W>k
−1 (W>k)′.
(111)
W>k is an N × (N −k) size matrix whose columns are orthonormal. Its range is
R(W>k)
=
N(WΣW ′)
=
N(U>l(U>l)′ + V>m(V>m)′)
=
N((U>l(U>l)′) ∩N(V>m(V>m)′)
=
N((U>l)′) ∩N((V>m)′)
=
R(U>l)⊥∩R(V>m)⊥
=
N(C)⊥∩N(B)⊥
=
R(C′) ∩R(B′)
=
R(C) ∩R(B),
(112)
that is, W>k is a matrix whose columns form an orthonormal basis of R(C)∩R(B).
Let BR(C)∩R(B) be any matrix whose columns form an orthonormal basis of
R(C)∩R(B). Then there exists a unique orthonormal matrix T of size (N −k)×
(N −k) such that BR(C)∩R(B) = W>k T. It holds that
W>k
 (W>k)′ (C† + B† −I) W>k
−1 (W>k)′ =
=
W>kT
 (W>kT)′ (C† + B† −I) W>kT
−1 (W>kT)′
=
BR(C)∩R(B)
 B′
R(C)∩R(B) (C† + B† −I) BR(C)∩R(B)
−1 B′
R(C)∩R(B),(113)
which gives the ﬁnal form of the claim in the proposition.
For showing equivalence of (33) with (113), I exploit a fact known in matrix the-
ory ([9], Fact 6.4.16): for two real matrices X, Y of sizes n×m, m×l the following
two condition are equivalent: (i) (X Y )† = Y † X†, and (ii) R(X′ X Y ) ⊆R(Y ) and
R(Y Y ′ X′) ⊆R(X′). Observing that PR(C)∩R(B) = BR(C)∩R(B)B′
R(C)∩R(B) and
that B†
R(C)∩R(B) = B′
R(C)∩R(B), setting X = BR(C)∩R(B) and Y = B′
R(C)∩R(B) (C†+
B† −I)PR(C)∩R(B) in (33) yields
 PR(C)∩R(B) (C† + B† −I) PR(C)∩R(B)
−1 =
 B′
R(C)∩R(B) (C† + B† −I) PR(C)∩R(B)
† B′
R(C)∩R(B),
where condition (ii) from the abovementioned fact is easily veriﬁed. In a second,
entirely analog step one can pull apart

B′
R(C)∩R(B) (C† + B† −I) PR(C)∩R(B)
†
into
BR(C)∩R(B)
 B′
R(C)∩R(B) (C† + B† −I) BR(C)∩R(B)
† .
Algorithm for Computing BR(C)∩R(B). Re-using ideas from this proof, a
basis matrix BR(C)∩R(B) can be computed as follows:
1. Compute the SVDs C = Udiag(s1, . . . , sl, 0, . . . , 0)U ′ and B = V diag(t1, . . . , tm,
0, . . . , 0)V ′.
2. Let U>l be the submatrix of U made from the last N −l columns in U, and
similarly let V>m consist of the last N −m columns in V .
169
3. Compute the SVD U>l(U>l)′ + V>m(V>m)′ = WΣW ′, where
Σ = diag(σ1, . . . , σk, 0, . . . , 0).
4. Let W>k be the submatrix of W consisting of the last N −k columns of W.
Then BR(C)∩R(B) = W>k.
5.3
Proof of Proposition 7 (Section 3.9.3)
.
By Deﬁnition 4, Proposition 6, and Equation (7),
C ∨B
=
¬ (¬ C ∧¬ B) = I −lim
δ↓0
 (¬C)−1
δ
+ (¬B)−1
δ
−I
−1
=
I −lim
δ↓0
 (I −C(δ))−1 + (I −B(δ))−1 −I
−1
=
I −lim
δ↓0

(I −R(δ)
C (R(δ)
C + I)−1)−1 + (I −R(δ)
B (R(δ)
B + I)−1 −I
−1
.
It is easy to check that (I−A(A+I)−1)−1 = I+A holds for any positive semideﬁnite
matrix A. Therefore,
C ∨B = I −lim
δ↓0

R(δ)
C + R(δ)
B + I
−1
.
Furthermore, for positive semideﬁnite A, B it generally holds that I −(A + B +
I)−1 = (A + B)(A + B + I)−1, and hence
C ∨B = lim
δ↓0 (R(δ)
C + R(δ)
B ) (R(δ)
C + R(δ)
B + I)−1.
5.4
Proof of Proposition 8 (Section 3.9.3)
Using (32), Proposition 7 and that fact A(δ) = R(δ)
A (R(δ)
A + I)−1 holds for any
conceptor A (which entails I −A(δ) = (R(δ)
A +I)−1), we derive the claim as follows:
C ∧B
=
lim
δ↓0
 C−1
δ
+ B−1
δ
−I
−1
=
lim
δ↓0
 (¬¬C)−1
δ
+ (¬¬B)−1
δ
−I
−1
=
lim
δ↓0
 (¬(¬C)(δ))−1 + (¬(¬B)(δ))−1 −I
−1
=
lim
δ↓0
 (I −(¬C)(δ))−1 + (I −(¬B)(δ))−1 −I
−1
=
lim
δ↓0

(R(δ)
¬C + I) + (R(δ)
¬B) + I) −I
−1
=
lim
δ↓0 (R(δ)
¬C + R(δ)
¬B + I)−1
=
I −lim
δ↓0 (I −(R(δ)
¬C + R(δ)
¬B + I)−1)
=
¬(¬C ∨¬B).
170
5.5
Proof of Proposition 9 (Section 3.9.4)
Claims 1. – 3. are elementary.
Claim 4a: R(C ∧B) = R(C) ∩R(B). By deﬁnition we have
C ∧B = BR(C)∩R(B)
 B′
R(C)∩R(B)(C† + B† −I)BR(C)∩R(B)
−1 B′
R(C)∩R(B).
Since B′
R(C)∩R(B)(C† + B† −I)BR(C)∩R(B) is invertible and R(B′
R(C)∩R(B)) =
Rdim(R(C)∩R(B)), we have R

(B′
R(C)∩R(B)(C† + B† −I)BR(C)∩R(B))−1 B′
R(C)∩R(B)

=
Rdim(R(C)∩R(B)). From this it follows that R(C ∧B) = R(BR(C)∩R(B)) = R(C) ∩
R(B).
Claim 5a: I(C ∧B) = I(C) ∩I(B). We have, by deﬁnition,
C ∧B = BR(C)∩R(B)
 B′
R(C)∩R(B)(C† + B† −I)BR(C)∩R(B)
−1 B′
R(C)∩R(B).
For shorter notation put B = BR(C)∩R(B) and X = B′
R(C)∩R(B)(C†+B†−I)BR(C)∩R(B).
Note (from proof of Proposition 6) that X is invertible. We characterize the unit
eigenvectors of C ∧B. It holds that BX−1B′ x = x if and only if BXB′ x = x.
We need to show that the conjunction Cx = x and Bx = x is equivalent to
(C ∧B) x = BX−1B x = x.
First assume that Cx = x and Bx = x. This implies x ∈R(C) ∩R(B) and
C† x = x and B† x = x, and hence PR(C)∩R(B)(C† +B† −I)PR(C)∩R(B) x = x. But
PR(C)∩R(B)(C† + B† −I)PR(C)∩R(B) = BXB′, thus (C ∧B) x = x.
Now assume conversely that not Cx = x or not Bx = x.
Case 1: x /∈R(C) ∩R(B). Then PR(C)∩R(B)(C† + B† −I)PR(C)∩R(B) x ̸= x
and hence BXB′ x ̸= x, which implies (C ∧B) x ̸= x.
Case 2: x ∈R(C) ∩R(B). We ﬁrst show an auxiliary claim: ∥(C† + B† −
I) x∥> ∥x∥.
Let C0 = C† −CC†, B0 = B† −BB†.
C0 and B0 are positive
semideﬁnite because the nonzero singular values of C†, B† are greater or equal to
1. Furthermore, CC† x = BB† x = x. Thus, (C† + B†) x = 2Ix + C0x + B0x, i.e.
(C† + B† −I) x = Ix + C0x + B0x. From not Cx = x or not Bx = x it follows
that C0x ̸= 0 or B0x ̸= 0. We infer
C0x ̸= 0 or B0x ̸= 0
=⇒
x′C0x > 0 or x′B0x > 0
=⇒
x′(C0 + B0)x > 0
=⇒
x′(C0 + B0)2x > 0.
This implies ∥Ix+C0x+B0x∥2 = ∥x∥2+2x′(C0+B0)x+x′(C0+B0)2x > ∥x∥2,
or equivalently, ∥(C† + B† −I) x∥> ∥x∥, the auxiliary claim.
Since PR(C)∩R(B) preserves vector norm on R(C) ∩R(B) and R(C† + B† −
I)PR(C)∩R(B) ⊆R(C)∩R(B), it follows that ∥PR(C)∩R(B)(C†+B†−I)PR(C)∩R(B) x∥>
∥x∥, hence (C ∧B) x ̸= x.
Altogether we have that the conjunction Cx = x and Bx = x is equivalent to
(C ∧B) x = x, which is equivalent to the claim.
171
Claim 6a: N(C ∧B) = N(C) + N(B). This follows from 4a by N(C ∧B) =
(R(C ∧B))⊥= (R(C) ∩R(B))⊥= (N(C)⊥∩N(B)⊥)⊥= N(C) + N(B).
Claims 4b, 5b, 6b: The second statements in 4., 5., 6. follow from the ﬁrst
statements and 3., exploiting de Morgan’s rule.
Claim 7 follows from Equation (19).
Claim 8: Let A = A ∧C. Then claim 4. implies R(A) ∩R(C) = R(A). By
Proposition 6 we can write
A ∧C = BR(A)
 B′
R(A) (C† + B† −I) BR(A)
−1 B′
R(A).
Let A = USU ′ be the SVD of A, and assume A has rank k ≤N, that is,
exactly the ﬁrst k singular values in S are nonzero. Let Uk be the N × k matrix
consisting of the ﬁrst k columns of U. It holds that Uk = BR(A). We obtain
S = U ′AU =
=
U ′ Uk
 U ′
k(A† + C† −I) Uk
−1 U ′
k U
=
Ik
 U ′
k(A† + C† −I) Uk
−1 I′
k,
(114)
where Ik is the N × k matrix consisting of the ﬁrst k columns of I. Let Sk be the
k × k upper left submatrix of S. Then Sk =
 U ′
k(A† + C† −I) Uk
−1 and
S−1
k
=
U ′
k(A† + C† −I) Uk
=
U ′
k A† Uk + U ′
k (C† −I) Uk
=
S−1
k
+ U ′
k (C† −I) Uk,
hence U ′
k (C†−I) Uk = 0k×k or equivalently, U ′
k C† Uk = Ik×k. This implies R(A) ⊆
I(C).
Conversely, assume R(A) ⊆I(C). Going through the above line of arguments
in reverse order establishes again S = Ik
 U ′
k(A† + C† −I) Uk
−1 I′
k which implies
A = BR(A)
 B′
R(A) (C† + B† −I) BR(A)
−1 B′
R(A).
R(A) ⊆I(C) implies R(A) ⊆R(C), which leads to
A
=
BR(A)∩R(C)
 B′
R(A)∩R(C) (C† + B† −I) BR(A)∩R(C)
−1 B′
R(A)∩R(C)
=
A ∧C.
The dual A = A ∨C ⇔I(A)⊥⊆N(C) is easily obtained from A = A ∧C ⇔
R(A) ⊆I(C) by applying de Morgan’s rules and claim 3..
Claims 9 – 13 follow from Equation (19). For Claim 14. use 11. and 4. and 6.
to ﬁrst establish that R(H ∧G) = I(H)∩I(G) and N(H ∧G) = (I(H)∩I(G))⊥,
from which 14. follows. Claim 15 is analog.
172
5.6
Proof of Proposition 10 (Section 3.9.5)
Claim 1: ¬ϕ(C, γ) = ϕ(¬C, γ−1). Notation: All of the matrices C, ¬C, ϕ(C, γ),
¬ϕ(C, γ), ϕ(¬C, γ−1) are positive semideﬁnite and have SVDs with identical prin-
cipal component matrix U. For any matrix X among these, let USXU ′ be its
SVD. We write sX
i
for the ith singular value in SX.
We have to show that
s¬ϕ(C,γ)
i
= sϕ(¬C,γ−1)
i
. In the derivations below, we use various facts from Proposi-
tion 9 and Equation (19) without explicit reference.
Case 0 < γ < ∞, 0 < sC
i < 1:
s¬ϕ(C,γ)
i
=
1 −
sC
i
sC
i + γ−2 (1 −sC
i )
=
1 −sC
i
(1 −sC
i ) + γ2 sC
i
=
s¬C
i
s¬C
i
+ (γ−1)−2 (1 −s¬C
i
)
=
sϕ(¬C,γ−1)
i
.
Case 0 < γ < ∞, sC
i = 0: Using sC
i = 0 ⇔s¬C
i
= 1 we have
s¬ϕ(C,γ)
i
= 1 −sϕ(C,γ)
i
= 1 = sϕ(¬C,γ−1)
i
.
Case 0 < γ < ∞, sC
i = 1: dual of previous case.
Case γ = 0: We show ¬ϕ(C, γ) = ϕ(¬C, γ−1) directly.
¬ϕ(C, γ)
=
¬ϕ(C, 0) = I −ϕ(C, 0) = I −PI(C) = PI(C)⊥
=
PN(¬C)⊥= ϕ(¬C, ∞) = ϕ(¬C, γ−1).
Case γ = ∞: the dual analog.
Claim 2: ϕ(C, γ) ∨ϕ(B, γ) = ϕ(C ∨B, γ).
Case 0 < γ < ∞: Using concepts and notation from Proposition 7, it is easy
to check that any conceptor A can be written as
A = lim
δ↓0 R(δ)
A (R(δ)
A + I)−1,
(115)
and its aperture adapted versions as
ϕ(A, γ) = lim
δ↓0 R(δ)
A (R(δ)
A + γ−2I)−1.
(116)
Using Proposition 7 and (115) we thus have
C ∨B
=
lim
δ↓0 (R(δ)
C + R(δ)
B ) (R(δ)
C + R(δ)
B + I)−1
(117)
=
lim
δ↓0 R(δ)
C∨B(R(δ)
C∨B + I)−1.
(118)
Furthermore, again by Proposition 7 and by (116),
ϕ(C, γ) ∨ϕ(B, γ) = lim
δ↓0 (R(δ)
ϕ(C,γ) + R(δ)
ϕ(B,γ)) (R(δ)
ϕ(C,γ) + R(δ)
ϕ(B,γ) + I)−1
(119)
173
and
ϕ(C ∨B, γ) = lim
δ↓0 (R(δ)
C∨B)(R(δ)
C∨B + γ−2I)−1.
(120)
Using (17), it follows for any conceptor A that
R(δ)
ϕ(A,γ) = γ2 R(δ/(δ+γ−2(1−δ)))
A
.
(121)
Applying this to (119) and observing that limδ↓0 δ = 0 = limδ↓0 δ/(δ + γ−2(1 −δ))
yields
ϕ(C, γ) ∨ϕ(B, γ)
=
lim
δ↓0 (γ2R(δ)
C + γ2R(δ)
B ) (γ2R(δ)
C + γ2R(δ)
B + I)−1
=
lim
δ↓0 (R(δ)
C + R(δ)
B ) (R(δ)
C + R(δ)
B + γ−2I)−1.
(122)
We now exploit the following auxiliary fact which can be checked by elemen-
tary means: If (X(δ))δ is a δ-indexed family of positive semideﬁnite matrices
whose eigenvectors are identical for diﬀerent δ, and similarly the members of the
familiy (Y (δ))δ have identical eigenvectors, and if the limits limδ↓0 X(δ)(X(δ)+I)−1,
limδ↓0 Y (δ)(Y (δ) + I)−1 exist and are equal, then the limits limδ↓0 X(δ)(X(δ) +
γ−2I)−1, limδ↓0 Y (δ)(Y (δ) + γ−2I)−1 exist and are equal, too.
Putting X(δ) =
R(δ)
C + R(δ)
B and Y (δ) = R(δ)
C∨B, combining (117), (118), (120) and (122) with this
auxiliary fact yields ϕ(C ∨B, γ) = ϕ(C, γ) ∨ϕ(B, γ).
Case γ = 0: Using various ﬁndings from Proposition 9 we have
ϕ(C, 0) ∨ϕ(B, 0)
=
PI(C) ∨PI(B)
=
PI(PI(C)) + I(PI(B))
=
PI(C)+I(B)
=
PI(C∨B)
=
ϕ(C ∨B, 0).
Case γ = ∞:
ϕ(C, ∞) ∨ϕ(B, ∞)
=
PR(C) ∨PR(B)
=
PI(PR(C)) + I(PR(B))
=
PR(C) + R(B)
=
PR(C∨B)
=
ϕ(C ∨B, ∞).
Claim 3: ϕ(C, γ) ∧ϕ(B, γ) = ϕ(C ∧B, γ): follows from Claims 1. and 2. with
de Morgan’s law.
Claim 4: ϕ(C, γ) ∨ϕ(C, β) = ϕ(C,
p
γ2 + β2):
Case 0 < γ, β < ∞: Using Proposition 9 and Equations (121), (117), (116),
174
we obtain
ϕ(C, γ) ∨ϕ(C, β)
=
lim
δ↓0 (R(δ)
ϕ(C,γ) + R(δ)
ϕ(C,β)) (R(δ)
ϕ(C,γ) + R(δ)
ϕ(C,β) + I)−1
=
lim
δ↓0 (γ2R(δ/(δ+γ−2(1−δ)))
C
+ β2R(δ/(δ+β−2(1−δ)))
C
) ·
· (γ2R(δ/(δ+γ−2(1−δ)))
C
+ β2R(δ/(δ+β−2(1−δ)))
C
+ I)−1
(∗)
=
lim
δ↓0 (γ2 + β2)R(δ)
C ((γ2 + β2)R(δ)
C + I)−1
=
lim
δ↓0 R(δ)
C (R(δ)
C + (γ2 + β2)−1 I)−1
=
ϕ(C,
p
γ2 + β2),
where in step (*) we exploit the fact that the singular values of R(δ/(δ+γ−2(1−δ)))
C
cor-
responding to eigenvectors whose eigenvalues in C are less than unity are identical
to the singular values of R(δ)
C at the analog positions.
Case γ = 0, 0 < β < ∞: Using Proposition 7, facts from Proposition 9, and
Equation (115), we obtain
ϕ(C, 0) ∨ϕ(C, β)
=
PI(C) ∨ϕ(C, β)
=
lim
δ↓0 (R(δ)
PI(C) + R(δ)
ϕ(C,β)) (R(δ)
PI(C) + R(δ)
ϕ(C,β) + I)−1
(∗)
=
lim
δ↓0 R(δ/(2−δ))
ϕ(C,β)
(R(δ/(2−δ))
ϕ(C,β)
+ I)−1 = lim
δ↓0 R(δ)
ϕ(C,β) (R(δ/)
ϕ(C,β) + I)−1
=
ϕ(C, β)
=
ϕ(C,
p
02 + β2),
where step (*) is obtained by observing I(C) = I(ϕ(C, β)) and applying the
deﬁnition of R(δ)
A given in the statement of Proposition 7.
Case γ = ∞, 0 < β < ∞: the dual analog to the previous case:
ϕ(C, ∞) ∨ϕ(C, β)
=
PR(C) ∨ϕ(C, β)
=
lim
δ↓0 (R(δ)
PR(C) + R(δ)
ϕ(C,β)) (R(δ)
PR(C) + R(δ)
ϕ(C,β) + I)−1
(∗)
=
lim
δ↓0 R(δ)
PR(C) (R(δ/)
PR(C) + I)−1
=
PR(C) = ϕ(C, ∞) = ϕ(C,
p
∞2 + β2),
where in step (*) I have omitted obvious intermediate calculations.
The cases 0 < γ < ∞, β ∈{0, ∞} are symmetric to cases already treated, and
the cases γ, β ∈{0, ∞} are obvious.
Claim 5: ϕ(C, γ)∧ϕ(C, β) = ϕ(C, (γ−2+β−2)−2): an easy exercise of applying
de Morgan’s rule in conjunction with Claims 1. and 4.
175
5.7
Proof of Proposition 11 (Section 3.9.6)
1. De Morgan’s rules: By Deﬁnition 4 and Proposition 8. 2. Associativity: From
Equations (117) and (118) it follows that for any conceptors B, C it holds that
limδ↓0 R(δ)
B∨C = limδ↓0 R(δ)
B + R(δ)
C . Employing this fact and using Proposition 7
yields associativity of OR. Applying de Morgan’s law then transfers associativity
to AND. 3.
Commutativity and 4.
double negation are clear.
5.
Neutrality:
Neutrality of I: Observing that R(C) ∩R(I) = R(C) and I† = I, starting from
the deﬁnition of ∨we obtain
C ∨I
=
(PR(C)C†PR(C))†
=
(PR(C†)C†PR(C†))†
(by Prop. 9 Nr. 2)
=
(C†)†
=
C.
Neutrality of 0 can be obtained from neutrality of I via de Morgan’s rules.
6. Globality: C ∧0 = 0 follows immediately from the Deﬁnition of ∧given in
4, observing that PR(C)∩R(0) = 0. The dual C ∨I = I is obtained by applying de
Morgan’s rule on C ∧0 = 0.
7. and 8. weighted absorptions follows from Proposition 10 items 4. and 5.
5.8
Proof of Proposition 13 (Section 3.9.6)
Let ≤denote the well-known L¨owner ordering on the set of real N × N matrices
deﬁned by A ≤B if B −A is positive semideﬁnite. Note that a matrix C is a
conceptor matrix if and only if 0 ≤C ≤I. I ﬁrst show the following
Lemma 2 Let A ≤B. Then A† ≥PR(A) B† PR(A).
Proof of Lemma. A† and B† can be written as
A† = lim
δ→0 PR(A)(A + δI)−1PR(A) and B† = lim
δ→0 PR(B)(B + δI)−1PR(B).
(123)
From A ≤B it follows that R(A) ⊆R(B), that is, PR(A)PR(B) = PR(A), which
in turn yields
PR(A) B† PR(A) = lim
δ→0 PR(A)(B + δI)−1PR(A).
(124)
A ≤B entails A + δI ≤B + δI, which is equivalent to (A + δI)−1 ≥(B + δI)−1
(see [9], fact 8.21.11), which implies
PR(A)(B + δI)−1PR(A) ≤PR(A)(A + δI)−1PR(A),
see Proposition 8.1.2 (xii) in [9]. Taking the limits (123) and (124) leads to the
claim of the lemma (see fact 8.10.1 in [9]).
176
Proof of Claim 1. Let A, B be conceptor matrices of size N ×N. According to
Proposition 15 (which is proven independently of the results stated in Proposition
13), it holds that A ≤A ∨B, which combined with the lemma above establishes
PR(A)(A† −(A ∨B)†)PR(A) ≥0,
from which it follows that I + PR(A)(A† −(A ∨B)†)PR(A) is positive semideﬁnite
with all singular values greater or equal to one. Therefore, PR(A)(I + A† −(A ∨
B)†)PR(A) is positive semideﬁnite with all nonzero singular values greater or equal
to one. Hence C =
 PR(A)
 I + A† −(A ∨B)†
PR(A)
† is a conceptor matrix. It
is furthermore obvious that R(C) = R(A).
From A ≤A ∨B it follows that R(A) ⊆R(A ∨B), which together with
R(C) = R(A) leads to R(A ∨B) ∩R(C) = R(A). Exploiting this fact, starting
from the deﬁnition of AND in Def. 4, we conclude
(A ∨B) ∧C
=
 PR(A∨B)∩R(C)
 (A ∨B)† + C† −I

PR(A∨B)∩R(C)
†
=
 PR(A)
 (A ∨B)† + C† −I

PR(A)
†
=
 PR(A)
 (A ∨B)† + PR(A)
 I + A† −(A ∨B)†
PR(A) −I

PR(A)
†
=
(PR(A)A†PR(A))†
=
A.
Proof of Claim 2.
This claim is the Boolean dual to claim 1 and can be
straightforwardly derived by transformation from claim 1, using de Morgan’s rules
and observing that R(¬A) = I(A)⊥(see Prop. 9 item 3), and that ¬A = I −A.
5.9
Proof of Proposition 14 (Section 3.10)
Claim 1: 0 ≤A is equivalent to A being positive semideﬁnite, and for a positive
semideﬁnite matrix A, the condition A ≤I is equivalent to all singular values of
A being at most one. Both together yield the claim.
Claim 2: Follows from claim 1.
Claim 3: A ≤B iﬀ−A ≥−B iﬀI −A ≥I −B, which is the same as
¬A ≥¬B.
Claim 4: We ﬁrst show an auxiliary, general fact:
Lemma 3 Let X be positive semideﬁnite, and P a projector matrix. Then
(P(PXP + I)−1P)† = PXP + P.
Proof of Lemma. Let U = R(P) be the projection space of P. It is clear that
PXP+I : U →U and PXP+I : U ⊥→U ⊥, hence PXP+I is a bijection on U.
Also P is a bijection on U. We now call upon the following well-known property
of the pseudoinverse (see [9], fact 6.4.16):
For matrices K, L of compatible sizes it holds that (KL)† = L†K† if and only
if R(K′KL) ⊆R(L) and R(LL′K) ⊆R(K′).
177
Observing that P and PXP + I are bijections on U = R(P), a twofold ap-
plication of the mentioned fact yields (P(PXP + I)−1P)† = P†(PXP + I)P† =
PXP + P which completes the proof of the lemma.
Now let A, B ∈CN and B ≤A. By Lemma 2, B† −PR(B) A† PR(B) is pos-
itive semideﬁnite, hence B† −PR(B) A† PR(B) + I is positive deﬁnite with sin-
gular values greater or equal to one, hence invertible.
The singular values of
(B† −PR(B) A† PR(B) + I)−1 are thus at most one, hence C = PR(B) (B† −
PR(B) A† PR(B) + I)−1 PR(B) is a conceptor matrix. Obviously R(C) = R(B).
Using these ﬁndings and Lemma 3 we can now infer
A ∧C
=
 PR(A)∩R(C) (A† + C† −I) PR(A)∩R(C)
†
=

PR(B) (A† +
 PR(B) (B† −PR(B) A† PR(B) + I)−1 PR(B)
† −I) PR(B)
†
=
 PR(B)
 A† + PR(B) (B† −PR(B) A†PR(B)) PR(B)

PR(B)
†
=
B.
Claim 5: This claim is the Boolean dual to the previous claim. It follows by
a straightforward transformation applying de Morgan’s rules and observing that
R(¬B) = I(B)⊥(see Prop. 9 item 3), and that ¬A = I −A.
Claim 6: Let A ∧C = B. Using the notation and claim from Prop. 6 rewrite
A ∧C = limδ→0(C−1
δ
+ A−1
δ
−I)−1.
Similarly, obviously we can also rewrite
A = limδ→0 Aδ. Since C−1
δ
≥I, conclude
C−1
δ
+ A−1
δ
−I ≥A−1
δ
⇐⇒
(C−1
δ
+ A−1
δ
−I)−1 ≤Aδ
=⇒
lim
δ→0(C−1
δ
+ A−1
δ
−I)−1 ≤lim
δ→0 Aδ
⇐⇒
A ∧C ≤A,
where use is made of the fact that taking limits preserves ≤(see [9] fact 8.10.1).
Claim 7: Using the result from the previous claim, infer A ∨C = B =⇒
¬A ∧¬C = ¬B =⇒¬A ≥¬B =⇒A ≤B.
Claim 8: Let γ =
p
1 + β2 ≥1, where β ≥0. By Proposition 10 4. we get
ϕ(A, γ) = ϕ(A, 1) ∨ϕ(A, β) = A ∨ϕ(A, β), hence A ≤ϕ(A, γ). The dual version
is obtained from this result by using Proposition 5: let γ ≤1, hence γ−1 ≥1.
Then φ(A, γ) ≤φ(φ(A, γ), γ−1) = A.
Claim 9: If γ = 0, then from Proposition 3 it is clear that ϕ(A, 0) is the
projector matrix on I(A) and ϕ(B, 0) is the projector matrix on I(B). From
A ≤B and the fact that A and B do not have singular values exceeding 1 it is
clear that I(A) ⊆I(B), thus ϕ(A, 0) ≤ϕ(B, 0).
If γ = ∞, proceed in an analog way and use Proposition 3 to conclude that
ϕ(A, ∞), ϕ(B, ∞) are the projectors on R(A), R(B) and apply that A ≤B implies
R(A) ⊆R(B).
It remains to treat the case 0 < γ < ∞. Assume A ≤B, that is, there exists
a positive semideﬁnite matrix D such that A + D = B. Clearly D cannot have
178
singular values exceeding one, so D is a conceptor matrix. For (small) δ > 0, let
A(δ) = (1−δ)A. Then A(δ) can be written as A(δ) = R(δ)(R(δ) +I)−1 for a positive
semideﬁnite R(δ), and it holds that
A = lim
δ→0 A(δ),
and furthermore
A(δ) ≤A(δ′) ≤A for δ ≥δ′.
Similarly, let D(δ) = (1 −δ)D, with D(δ) = Q(δ)(Q(δ) + I)−1, and observe again
D = lim
δ→0 D(δ)
and
D(δ) ≤D(δ′) ≤D for δ ≥δ′.
Finally, deﬁne B(δ) = (1 −δ)B, where B(δ) = P (δ)(P (δ) + I)−1. Then
B = lim
δ→0 B(δ)
and
B(δ) ≤B(δ′) ≤B for δ ≥δ′
and
B(δ) = A(δ) + D(δ).
Because of B(δ) = A(δ) + D(δ) we have
R(δ)(R(δ) + I)−1 ≤P (δ)(P (δ) + I)−1.
(125)
We next state a lemma which is of interest in its own right too.
Lemma 4 For correlation matrices R, P of same size it holds that
R(R + I)−1 ≤P(P + I)−1
iﬀ
R ≤P.
(126)
Proof of Lemma. Assume R(R + I)−1 ≤P(P + I)−1. By claim 4. of this propo-
sition, there is a conceptor matrix C such that P(P + I)−1 = R(R + I)−1 ∨C.
Since P(P + I)−1 < I, C has no unit singular values and thus can be writ-
ten as S(S + I)−1, where S is a correlation matrix. Therefore, P(P + I)−1 =
R(R + I)−1 ∨S(S + I)−1 = (R + S)(R + S + I)−1, hence P = R + S, that is,
R ≤P.
Next assume R ≤P, that is, P = R + S for a correlation matrix S. This
implies P(P + I)−1 = (R + S)(R + S + I)−1 = R(R + I)−1 ∨S(S + I)−1. By claim
6. of this proposition, R(R + I)−1 ≤P(P + I)−1 follows. This concludes the proof
of the lemma.
Combining this lemma with (125) and the obvious fact that R(δ) ≤P (δ) if and
only if γ2 R(δ) ≤γ2 P (δ) yields
γ2R(δ)(γ2R(δ) + I)−1 ≤γ2P (δ)(γ2P (δ) + I)−1.
(127)
Another requisite auxiliary fact is contained in the next
Lemma 5 Let 0 < γ < ∞. If A = USU ′ = limδ→0 R(δ)(R(δ) + I)−1 and for all δ,
R(δ) has a SVD R(δ) = UΣ(δ)U ′, then ϕ(A, γ) = limδ→0 γ2R(δ)(γ2R(δ) + I)−1.
179
Proof of Lemma. Since all R(δ) (and hence, all R(δ)(R(δ)+I)−1 and γ2R(δ)(γ2R(δ)+
I)−1) have the same eigenvectors as A, it suﬃces to show the convergence claim on
the level of individual singular values of the concerned matrices. Let s, sγ, σ(δ), s(δ), s(δ)
γ
denote a singular value of A, ϕ(A, γ), R(δ), R(δ)(R(δ) + I)−1, γ2R(δ)(γ2R(δ) + I)−1,
respectively (all these versions referring to the same eigenvector in U). For con-
venience I restate from Proposition 3 that
sγ =
( s/(s + γ−2(1 −s))
for
0 < s < 1,
0
for
s = 0,
1
for
s = 1.
It holds that s(δ) = σ(δ)/(σ(δ) + 1) and limδ→0 s(δ) = s, and similarly s(δ)
γ
=
γ2σ(δ)/(γ2σ(δ) + 1). It needs to be shown that limδ→0 s(δ)
γ
= sγ.
Case s = 0:
s = 0
=⇒
lim
δ→0 s(δ) = 0
=⇒
lim
δ→0 σ(δ) = 0
=⇒
lim
δ→0 s(δ)
γ
= 0.
Case s = 1:
s = 1
=⇒
lim
δ→0 s(δ) = 1
=⇒
lim
δ→0 σ(δ) = ∞
=⇒
lim
δ→0 s(δ)
γ
= 1.
Case 0 < s < 1:
s = lim
δ→0 s(δ)
=⇒
s = lim
δ→0 σ(δ)/(σ(δ) + 1)
=⇒
lim
δ→0 σ(δ) = s/(1 −s)
=⇒
lim
δ→0 s(δ)
γ
=
γ2s/(1 −s)
γ2s/(1 −s) + 1 = s/(s + γ−2(1 −s)).
This concludes the proof of the lemma.
After these preparations, we can ﬁnalize the proof of claim 9. as follows. From
Lemma 5 we know that
ϕ(A, γ) = lim
δ→0 γ2R(δ)(γ2R(δ) + I)−1
and
ϕ(B, γ) = lim
δ→0 γ2P (δ)(γ2P (δ) + I)−1.
From (127) and the fact that ≤is preserved under limits we obtain ϕ(A, γ) ≤
ϕ(B, γ).
180
5.10
Proof of Proposition 16 (Section 3.13.4)
We use the following notation for matrix-vector transforms. We sort the entries
of an N × N matrix M into an N 2-dimensional vector vec M row-wise (!). That
is, vec M(µ) =
M(⌈µ/N⌉, mod1(µ, N)), where the ceiling ⌈x⌉of a real number x is the smallest
integer greater or equal to x, and mod1(µ, N) is the modulus function except for
arguments of the form (lk, k), where we replace the standard value mod(lm, m) = 0
by mod1(lm, m) = m. Conversely, M(i, j) = vec M((i −1)N + j).
The Jacobian JC can thus be written as a N 2×N 2 matrix JC(µ, ν) = ∂vec ˙C(µ)/∂vec C(ν).
The natural parametrization of matrices C by their matrix elements does not
lend itself easily to an eigenvalue analysis. Assuming that a reference solution
C0 = USU ′ is ﬁxed, any N × N matrix C is uniquely represented by a parameter
matrix P through C = C(P) = U (S + P) U ′, with C(P) = C0 if and only if
P = 0. Conversely, any parameter matrix P yields a unique C(P).
Now we consider the Jacobian JP(µ, ν) = ∂vec ˙PC(µ)/∂vec PC(ν). By using
that (i) ˙P = U ′ ˙CU, (ii) vec (X′Y X) = (X ⊗X)′ vec Y for square matrices X, Y ,
and (iii) ∂A ˙x/∂Ax = A(∂˙x/∂x)A−1 for invertible A, (iv) (U ⊗U)−1 = (U ⊗U)′,
one obtains that JP = (U ⊗U)′JC(U ⊗U) and hence JP and JC have the same
eigenvalues.
Using C0 = USU ′ and ˙C = (I −C)CDC′ −α−2C and the fact that diagonal
entries in S of index greater than k are zero, yields
∂˙plm
∂pij

P=0
=
e′
l (IijU ′DUS + SU ′
kDUIji
−IijSU ′
kDUS −SIijU ′DUS −S2U ′
kDUIji −α−2Iij) em,(128)
where el is the l-th unit vector and Iij = ei e′
j. Depending on how l, m, i, j relate to
k and to each other, calculating (128) leads to numerous case distinctions. Each of
the cases concerns entries in a speciﬁc subarea of JP. These subareas are depicted
in Fig. 48, which shows JP in an instance with N = 5, k = 3.
I will demonstrate in detail only two of these cases (subareas A and B in Fig.
48) and summarize the results of the others (calculations are mechanical).
The case A concerns all entries JP(µ, ν) with µ < ν, ν ≤kN, mod1(ν, N) ≤k.
Translating indices µ, ν back to indices l, m, i, j via JP(µ, ν) = ∂vec ˙PC(µ) / ∂vec PC(ν) =
∂˙p(⌈µ/N⌉,mod1 (µ,N)) / ∂p(⌈ν/N⌉,mod1 (ν,N)) = ∂˙plm / ∂pij yields conditions (i) i ≤k
(from i = ⌈ν/N⌉and ν ≤kN), (ii) j ≤k (from j = mod1(ν, N) ≤k) and (iii.a)
l < i or (iii.b) l = i ∧m < j (from µ < ν).
I ﬁrst treat the subcase (i), (ii), (iii.a). Since l ̸= i one has e′
l Iij = 0 and eqn.
(128) reduces to the terms starting with S, leading to
181
A 
B 
C 
D 
E 
F 
G 
H 
I 
J 
K 
L 
m!
n!
Figure 48: Main case distinction areas for computing values in the matrix JP. An
instance with N = 5, k = 3 is shown. Areas are denoted by A, ..., L; same color =
same area. JP has size N 2 × N 2. Its structure is largely organized by a kN × kN-
dimensional and a (N −k)N ×(N −k)N submatrix on the diagonal (areas ABEFK
and HIJL, respectively). Column/row indices are denoted by µ, ν. Area speciﬁ-
cations: A: µ < ν, ν ≤kN, mod1(ν, N) ≤k. B: µ < ν, ν ≤kN, mod1(ν, N) > k.
C: ν > kN, µ ≤kN, mod1(ν, N) ≤k. D: ν > kN, µ ≤kN, mod1(ν, N) > k.
E: µ > ν, µ ≤kN, mod1(ν, N) ≤k. F: µ > ν, µ ≤kN, mod1(ν, N) > k. G:
µ > kN, ν ≤kN. H: ν > kN, ν < µ. I: µ > kN, µ < ν, mod1(ν, N) ≤k. J:
µ > kN, µ < ν, mod1(ν, N) > k. K: µ = ν ≤kN. L: µ = ν > kN.
∂˙plm
∂pij

P=0
=
e′
l (SU ′
kDUIji −SIijU ′DUS −S2U ′
kDUIji) em
=
slu′
lDujδim −sle′
lIijU ′DUSem −s2
l u′
lDujδim
=
slu′
lDujδim −s2
l u′
lDujδim
=
( 0,
if i ̸= m
(subcase A1)
0,
if i = m, j ̸= l
(A2)
α−2,
if i = m, j = l
(A3),
where ul is the l-th column in U and δim = 1 if and only if i = m (else 0) is
the Kronecker delta. The value α−2 noted for subcase A3 is obtained through
(sl −s2
l )u′
lDul = (sl −s2
l ) ˜dl = α−2. Note that since l = i ≤k in subcase A3 it
holds that sl > 1/2.
Next, in the subcase (i), (ii), (iii.b) one has
182
∂˙plm
∂pij

P=0
=
u′
jDumsm + slu′
lDujδim −sju′
jDumsm
−slu′
jDumsm −s2
l u′
lDujδim −α−2e′
jem
=
(sl −s2
l )u′
lDujδim
(since u′
jDum = 0 and j ̸= m)
=
0,
(A4)
(129)
because assuming i ̸= m or j ̸= l each null the last expression, and i = m ∧j = l
is impossible because condition (iii.b) would imply m = j contrary to (iii.b).
The case B concerns all entries JP(µ, ν) with µ < ν, ν ≤kN, mod1(ν, N) > k.
Like in case A above, this yields conditions on the P-matrix indices: (i) i ≤k, (ii)
j > k, (iii.a) l < i or (iii.b) l = i ∧m < j.
Again we ﬁrst treat the subcase (i), (ii), (iii.a). Since l ̸= i one has e′
l Iij = 0
and eqn. (128) reduces to the terms starting with S, leading to
∂˙plm
∂pij

P=0
=
e′
l (SU ′
kDUIji −SIijU ′DUS −S2U ′
kDUIji) em
=
slu′
lDujδim −sle′
lIijU ′DUSem −s2
l u′
lDujδim
=
slu′
lDujδim −s2
l u′
lDujδim
=

0,
if i ̸= m
(B1)
(sl −s2
l )u′
lDuj
if i = m
(B2).
where ul is the l-th column in U and δim = 1 if and only if i = m (else 0) is the
Kronecker delta. Note that since l < i ≤k it holds that sl > 1/2.
In the subcase (i), (ii), (iii.b) from (128) one obtains
∂˙plm
∂pij

P=0
=
u′
jDumsm + slu′
lDujδim −sju′
jDumsm
−slu′
jDumsm −s2
l u′
lDujδim −α−2e′
jem
=
sm(1 −sl)u′
jDum + (sl −s2
l )u′
lDujδim
=



sm(1 −sl)u′
jDum
if i ̸= m and m ≤k
(B3)
0
if i ̸= m and m > k
(B4)
sm(1 −sl)u′
jDum + (sl −s2
l )u′
lDuj
if i = m
(B5),
where in the step from the ﬁrst to the second line one exploits j > k, hence sj = 0;
and m < j, hence e′
jem = 0. Note that since l = i ≤k it holds that sl > 0; that
m > k implies sm = 0 and that m = i ≤k implies sm > 0.
Most of the other cases C – L listed in Fig. 48 divide into subcases like A and
B. The calculations are similar to the ones above and involve no new ideas. Table
1 collects all ﬁndings. It only shows subcases for nonzero entries of JP. Fig. 49
depicts the locations of these nonzero areas.
183
Subcase Index range
Cell value
A3
l = j ≤k; i = m ≤k;
α−2
B2
l < i = m ≤k < j
(sl −s2
l ) u′
lDuj
B3
l = i ≤k < j; m ≤j; m ̸= i
sm(1 −sl)u′
jDum
B5
l = i ≤k < j; m ≤j; m = i
sl(1 −sl)u′
jDul + (sl −s2
l )u′
lDuj
C1
l = j ≤k; i = m > k
α−2
D1
l ≤k; i, j > k; i = m
(sl −s2
l ) u′
lDuj
E1
i = m < l = j ≤k
α−2
F1
i = m < l ≤k < j
(sl −s2
l ) u′
lDuj
J1
i = l > k; m ≤k < j
smu′
jDum
K1
i = j = l = m ≤k
α−2(1 −2sl)/(1 −sl)
K2
i = l; j = m > k; l ̸= m
−α−2
K3
i = l; j = m ≤k; l ̸= m
−α−2 sl/(1 −sm)
L1
i = l > k; m = j > k
−α−2
Table 1: Values in nonzero areas of the Jacobian JP.
The eigenvalues of JP are now readily obtained. First observe that the eigen-
values of a matrix with block structure
 
K
L
0
M
!
,
where K and M are square, are the eigenvalues collected from the principal sub-
matrices K and M. In JP we therefore only need to consider the leading kN ×kN
and the trailing (N −k)N × (N −k)N submatrices; call them K and M.
M is upper triangular, its eigenvalues are therefore its diagonal values. We thus
can collect from M (N −k)k times eigenvalues 0 and (N −k)2 times eigenvalues
−α−2.
By simultaneous permutations of rows and columns (which leave eigenvalues
unchanged) in K we can bring it to the form Kperm shown in Fig. 49B. A block
structure argument as before informs us that the eigenvalues of Kperm fall into
three groups. There are k(N −k) eigenvalues −α−2 (corresponding to the lower
right diagonal submatrix of Kperm, denoted as area K2), k eigenvalues α−2(1 −
2sl)/(1 −sl), where l = 1, . . . , k earned from the k leading diagonal elements of
Kperm (stemming from the K1 entries in JP), plus there are the eigenvalues of
k(k −1)/2 twodimensional submatrices, each of which is of the form
184
J1 
L1 
K2 
K1 
K3 A3 
E1 
K1 
B5 
A3 
B3 
C1 
D1 
E1 
F1 
J1 
L1 
B2 
K3 
K2 
A 
C 
B 
Figure 49: A. Nonzero areas in the Jacobian matrix JP. An instance with N =
5, k = 3 is shown.
Areas denotations correspond to Table 1.
Same color =
same area. Values: areas A3, C1, E1: α−2; B2, D1, F1: (sl −s2
l ) u′
lDuj; B3:
sm(1−sl)u′
jDum; B5: sl(1−sl)u′
jDul+(sl−s2
l )u′
lDuj; J1: smu′
jDum; K1: α−2(1−
2sl)/(1 −sl); K2, L1: −α−2; K3: −α−2 sl/(1 −sm). B. The left upper principal
submatrix re-arranged by simultaneous row/column permutations. C. One of the
N ×N submatrices Cr from the diagonal of the (N −k)N ×(N −k)N right bottom
submatrix of JP. For explanations see text.
Kl,m = −α−2
 
sl/(1 −sm)
1
1
sm/(1 −sl)
!
,
where l = 1, . . . , k; m < l. By solving the characteristic polynomial of Kl,m its
eigenvalues are obtained as
λ1,2 = α−2
2


sl
sm −1 +
sm
sl −1 ±
s
sl
sm −1 −
sm
sl −1
2
+ 4

.
(130)
Summarizing, the eigenvalues of JP are constituted by the following multiset:
1. k(N −k) instances of 0,
2. N(N −k) instances of −α−2,
3. k eigenvalues α−2(1 −2sl)/(1 −sl), where l = 1, . . . , k,
4. k(k −1) eigenvalues which come in pairs of the form given in Eqn. (130).
185
5.11
Proof of Proposition 18 (Section 3.17)
ζ |=Σ(n) ξ
iﬀ
∀z.∧2 : z.∧2 |=Σ(n) ζ →z.∧2 |=Σ(n) ξ
iﬀ
∀z.∧2 : z.∧2 .∗(z.∧2 + 1)−1 ≤ι(ζ) →z.∧2 .∗(z.∧2 + 1)−1 ≤ι(ξ)
iﬀ
ι(ζ) ≤ι(ξ),
where the last step rests on the fact that all vector components of ζ, ξ are at most
1 and that the set of vectors of the form z.∧2 .∗(z.∧2+1)−1 is the set of nonnegative
vectors with components less than 1.
186
References
[1] The Human Brain Project. Report to the European Commission, HBP Con-
sortium, EPFL Lausanne, 2012. URL: http://www.humanbrainproject.eu/.
[2] L. F. Abbott. Theoretical neuroscience rising. Neuron, 60(November 6):489–
495, 2008.
[3] D.H. Ackley, G.E. Hinton, and T.J. Sejnowski. A learning algorithm for
Boltzmann machines. Cognitive Science, 9:147–169, 1985.
[4] D. J. Amit, H. Gutfreund, and H. Sompolinsky. Spin-glass models of neural
networks. Phys. Rev. A, 32:1007–1018, 1985.
[5] J. R. Anderson, D. Bothell, M. D. Byrne, S. Douglass, C. Lebiere, and Y .
Qin. An integrated theory of the mind. Psychological Review, 111(4):1036–
1060, 2004.
[6] L. Appeltant, M. C. Soriano, G. Van der Sande, J. Danckaert, S. Mas-
sar, J. Dambre, B. Schrauwen, C. R. Mirasso, and I. Fischer. Information
processing using a single dynamical node as complex system. Nature Com-
munications, 2(468), 2011. DOI: 10.1038/ncomms1476.
[7] A. Babloyantz and C. Louren¸co. Computation with chaos: A paradigm for
cortical activity. Proceedings of the National Academy of Sciences of the
USA, 91:9027–9031, 1994.
[8] S. Bader, P. Hitzler, and S. H¨olldobler. Connectionist model generation: A
ﬁrst-order approach. Neurocomputing, 71(13):2420–2432, 2008.
[9] D. S. Bernstein. Matrix Mathematics, 2nd Edition. Princeton Univ. Press,
2009.
[10] R. V. Borges, A. Garcez, and L. C. Lamb. Learning and representing tem-
poral knowledge in recurrent networks. IEEE Trans. on Neural Networks,
22(12):2409–2421, 2011.
[11] R.A. Brooks. The whole iguana. In M. Brady, editor, Robotics Science,
pages 432–456. MIT Press, Cambridge, Mass., 1989.
[12] M. Brown and C. Harris. Neurofuzzy Adaptive Modelling and Control. Pren-
tice Hall, 1994.
[13] M. Buehner and P. Young. A tighter bound for the echo state property.
IEEE Transactions on Neural Networks, 17(3):820– 824, 2006.
[14] F. S. Chance and L. F. Abbott. Divisive inhibition in recurrent networks.
Network: Comput. Neural Syst., 11:119–129, 2000.
187
[15] S. P. Chatzis. Hidden Markov models with nonelliptically contoured state
densities.
IEEE Trans. on Pattern Analysis and Machine Intelligence,
32(12):2297 – 2304, 2010.
[16] A. Clark. Whatever next? Predictive brains, situated agents, and the future
of cognitive science. Behavioral and Brain Sciences, pages 1–86, 2012.
[17] A. Coates, P. Abbeel, and A. Y. Ng. Learning for control from multiple
demonstrations. In Proc. 25th ICML, Helsinki, 2008.
[18] A. M. Collins and M. r. Quillian. Retrieval time from semantic memory.
Journal of verbal learning and verbal behavior, 8(2):240–247, 1969.
[19] P. Dominey, M. Arbib, and J.-P. Joseph. A model of corticostriatal plasticity
for learning oculomotor associations and sequences. Journal of Cognitive
Neuroscience, 7(3):311–336, 1995.
[20] P. F. Dominey.
From sensorimotor sequence to grammatical construc-
tion: Evidence from simulation and neurophysiology. Adaptive Behaviour,
13(4):347–361, 2005.
[21] R. Douglas and T. Sejnowski. Future challenges for the sciene and engineer-
ing of learning: Final workshop report. Technical report, National Science
Foundation, 2008.
[22] G.L. Drescher.
Made-up Minds: A Constructivist Approach to Artiﬁcial
Intelligence. MIT Press, Cambridge, Mass., 1991.
[23] D. Durstewitz, J. K. Seamans, and T. J. Sejnowski. Neurocomputational
models of working memory. Nature Neuroscience, 3:1184–91, 2000.
[24] C. Eliasmith. A uniﬁed approach to building and controlling spiking attrac-
tor networks. Neural Computation, 17:1276–1314, 2005.
[25] C. Eliasmith. Attractor network. Scholarpedia, 2(10):1380, 2007.
[26] C. Eliasmith, Stewart T. C., Choo X., Bekolay T., Tang Y. DeWolf T.,
and D. Rasmussen. A large-scale model of the functioning brain. Science,
338(6111):1202–1205, 2012.
[27] K. Fan and G. Pall. Imbedding conditions for Hermitian and normal matri-
ces. Canad. J. Math., 9:298–304, 1957.
[28] B. Farhang-Boroujeny. Adaptive Filters: Theory and Applications. Wiley,
1998.
[29] J.A. Fodor and Z.W. Pylyshin. Connectionism and cognitive architecture:
A critical analysis. Cognition, 28:3–71, 1988.
188
[30] W. J. Freeman.
Deﬁnitions of state variables and state space for brain-
computer interface. part 1: Multiple hierarchical levels of brain function.
Cognitive Neurodynamics, 1(1):3–14, 2007.
[31] W. J. Freeman.
Deﬁnitions of state variables and state space for brain-
computer interface. part 2. extraction and classiﬁcation of feature vectors.
Cognitive Neurodynamics, 1(2):85–96, 2007.
[32] R. M. French.
Catastrophic interference in connectionist networks.
In
L. Nadel, editor, Encyclopedia of Cognitive Science, volume 1, pages 431–
435. Nature Publishing Group, 2003.
[33] K. Friston. A theory of cortical response. Phil. Trans. R. Soc. B, 360:815–
836, 2005.
[34] M. Galtier, O. D. Faugeras, and P. C. Bressloﬀ. Hebbian learning of recurrent
connections: a geometrical perspective. Neural Computation, 24(9):2346–
2383, 2012.
[35] T. Gedeon and D. Arathorn. Convergence of map seeking circuits. J. Math.
Imaging Vis., 29:235–248, 2007.
[36] W. Gerstner, H. Sprekeler, and D. Deco. Theory and simulation in neuro-
science. Science, 338(5 Oct):60–65, 2012.
[37] J. Goguen and R. Burstall. Institutions: Abstract model theory for speciﬁ-
cation and programming. J. of the ACM, 39(1):95–146, 1992.
[38] A. Graves, M. Liwicki, S. Fernandez, R. Bertolami, H. Bunke, and J. Schmid-
huber. A novel connectionist system for unconstrained handwriting recog-
nition. IEEE Transactions on Pattern Analysis and Machine Intelligence,
31(5):855 – 868, 2009.
[39] A. Graves and J. Schmidhuber. Oﬄine handwriting recognition with mul-
tidimensional recurrent neural networks. In Proc. NIPS 2008. MIT Press,
2008.
[40] S. Grillner. Biological pattern generation: The cellular and computational
logic of networks in motion. Neuron, 52:751–766, 2006.
[41] C. Gros and G. Kaczor. Semantic learning in autonomously active recurrent
neural networks. Logic Journal of the IGPL, 18(5):686–704, 2010.
[42] S. Grossberg. Linking attention to learning, expectation, competition, and
consciousness. In L. Itti, G. Rees, and J. Tsotsos, editors, Neurobiology of
attention, chapter 107, pages 652–662. San Diego: Elsevier, 2005.
[43] S. Grossberg. Adaptive resonance theory. Scholarpedia, 8(5):1569, 2013.
189
[44] S. Harnad. The symbol grounding problem. Physica, D42:335–346, 1990.
[45] D. O. Hebb. The Organization of Behavior. New York: Wiley & Sons, 1949.
[46] M. Hermans and B. Schrauwen.
Recurrent kernel machines: Computing
with inﬁnite echo state networks. Neural Computation, 24(1):104–133, 2012.
[47] G. E. Hinton and R. R. Salakuthdinov. Reducing the dimensionality of data
with neural networks. Science, 313(July 28):504–507, 2006.
[48] J. J. Hopﬁeld. Neural networks and physical systems with emergent col-
lective computational abilities. Proc. NatL Acad. Sci. USA, 79:2554–2558,
1982.
[49] O. Houd´e and N. Tzourio-Mazoyer. Neural foundations of logical and math-
ematical cognition.
Nature Reviews Neuroscience, 4(June 2003):507–514,
2003.
[50] A. J. Ijspeert. Central pattern generators for locomotion control in animals
and robots: A review. Neural Networks, 21:642–653, 2008.
[51] T. R. Insel, S. C. Landis, and F. S. Collins. The NIH BRAIN initiative.
Science, 340(6133):687–688, 2013.
[52] M. Ito and J. Tani. Generalization in learning multiple temporal patterns
using RNNPB. In Neural Information Processing, number 3316 in LNCS,
pages 592–598. Springer Verlag, 2004.
[53] H. Jaeger. Identiﬁcation of behaviors in an agent’s phase space. Arbeitspa-
piere der GMD 951, GMD, St. Augustin, 1995.
[54] H. Jaeger. The ”echo state” approach to analysing and training recurrent
neural networks.
GMD Report 148, GMD - German National Research
Institute for Computer Science, 2001.
[55] H. Jaeger. Reservoir self-control for achieving invariance against slow input
distortions. technical report 23, Jacobs University Bremen, 2010.
[56] H. Jaeger and H. Haas. Harnessing nonlinearity: Predicting chaotic systems
and saving energy in wireless communication. Science, 304:78–80, 2004.
[57] H. Jaeger, M. Lukosevicius, D. Popovici, and U. Siewert. Optimization and
applications of echo state networks with leaky integrator neurons. Neural
Networks, 20(3):335–352, 2007.
[58] T. Kohonen and T. Honkela. Kohonen network. In Scholarpedia, volume 2,
page 1568. 2007.
190
[59] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classiﬁcation with
deep convolutional neural networks.
In Advances in Neural Information
Processing Systems 25, pages 1106–1114, 2012.
[60] M. Kudo, J. Toyama, and M. Shimbo. Multidimensional curve classiﬁcation
using passing-through regions.
Pattern Recognition Letters, 20(11):1103–
1111, 1999.
[61] R. Laje and D. V. Buonomano. Robust timing and motor patterns by taming
chaos in recurrent neural networks.
Nature Neuroscience, 16(7):925–933,
2013.
[62] G. Lakoﬀ. Women, ﬁre, and dangerous things: What categories reveal about
the mind. University of Chicago, 1987.
[63] G. Lakoﬀ. Cognitive models and prototype theory. In I. Margolis and S. Lau-
rence, editors, Concepts: Core Readings, chapter 18, pages 391–422. Brad-
ford Books / MIT Press, 1999.
[64] G. Lakoﬀand R. E. Nunez.
Where mathematics comes from: How the
embodied mind brings mathematics into being. Basic Books, 2000.
[65] L. C. Lamb. The grand challenges and myths of neural-symbolic compu-
tation. In L. De Raedt, B. Hammer, P. Hitzler, and W. Maass, editors,
Recurrent Neural Networks- Models, Capacities, and Applications, number
08041 in Dagstuhl Seminar Proceedings, Dagstuhl, Germany, 2008. Interna-
tionales Begegnungs- und Forschungszentrum f¨ur Informatik (IBFI), Schloss
Dagstuhl, Germany.
[66] Y. LeCun, L. Bottou, J. Bengio, and P. Haﬀner. Gradient-based learning
applied to document recognition.
Proceedings of the IEEE, 86(11):2278–
2324, 1998. http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf.
[67] F. Lehmann. Semantic networks in artiﬁcial intelligence. Elsevier Science,
1992.
[68] L. Lukic, J. Santos-Victor, and A. Billard.
Learning coupled dynamical
systems from human demonstration for robotic eye-arm-hand coordination.
In Proc. IEEE-RAS International Conference on Humanoid Robots, Osaka
2012, 2012.
[69] M. Lukoˇseviˇcius and H. Jaeger. Reservoir computing approaches to recurrent
neural network training. Computer Science Review, 3(3):127–149, 2009.
[70] W. Maass, P. Joshi, and E. Sontag. Computational aspects of feedback in
neural circuits. PLOS Computational Biology, 3(1):1–20, 2007.
191
[71] G. Manjunath and H. Jaeger. Echo state property linked to an input: Ex-
ploring a fundamental characteristic of recurrent neural networks. Neural
Computation, 25(3):671–696, 2013.
[72] N. M. Mayer and M. Browne. Echo state networks and self-prediction. In Bi-
ologically Inspired Approaches to Advanced Information Technology, volume
3141 of LNCS, pages 40–48. Springer Verlag Berlin / Heidelberg, 2004.
[73] W. S. McCulloch and W. Pitts. A logical calculus of the ideas immanent in
nervous activity. Bull. of Mathematical Biophysics, 5:115–133, 1943.
[74] D. L. Medin and L. J. Rips. Concepts and categories: Memory, meaning, and
metaphysics. In K. J. Holyoak and R. G. Morrison, editors, The Cambridge
Handbook of Thinking and Reasoning, chapter 3, pages 37–72. Cambridge
University Press, 2005.
[75] T. M. Mitchell. Machine Learning. McGraw-Hill, 1997.
[76] M. Negrello and F. Pasemann. Attractor landscapes and active tracking:
The neurodynamics of embodied action. Adaptive Behaviour, 16:196 – 216,
2008.
[77] K. Obermayer, H. Ritter, and K. Schulten. A principle for the formation of
the spatial structure of cortical feature maps. Proc. of the National Academy
of Sciences of the USA, 87:8345–8349, 1990.
[78] E. Oja. A simpliﬁed neuron model as a principal component analyzer. J.
Math. Biol., 15:267–273, 1982.
[79] C. Orsenigo and C. Vercellis.
Combining discrete SVM and ﬁxed cardi-
nality warping distances for multivariate time series classiﬁcation. Pattern
Recognition, 43(11):3787–3794, 2010.
[80] G. Palm. On associative memory. Biol. Cybernetics, 36(1):19–31, 1980.
[81] R. Pfeifer and Ch. Scheier. Understanding Intelligence. MIT Press, 1999.
[82] G. Pinkas. Propositional non-monotonic reasoning and inconsistency in sym-
metric neural networks. In Proc. 12th international joint conference on Ar-
tiﬁcial intelligence - Volume 1, pages 525–530, 1991.
[83] J. B. Pollack. Recursive distributed representations. Artiﬁcial Intelligence,
46(1-2):77–105, 1990.
[84] L.
Qi.
Symmetric
nonnegative
tensors
and
copositive
tensors.
arxiv.org/pdf/1211.5642, 2012.
192
[85] M. R. Quillain.
Word concepts: A theory and simulation of some basic
semantic capabilities. Behavioral Science, 12(5):410–430, 1967.
[86] F. Rabe. Representing Logics and Logic Translations. Phd thesis, School of
Engineering and Science, Jacobs University Bremen, 2008.
[87] M. I. Rabinovich, R. Huerta, P. Varona, and V. S. Afraimovich. Transient
cognitive dynamics, metastability, and decision making. PLOS Computa-
tional Biology, 4(5):e1000072, 2008.
[88] F. R. Reinhart and J. J. Steil. Recurrent neural associative learning of for-
ward and inverse kinematics for movement generation of the redundant pa-10
robot. In A. Stoica, E. Tunsel, T. Huntsberger, T. Arslan, S. Vijayakumar,
and A. O. El-Rayis, editors, Proc. LAB-RS 2008, vol. 1, pages 35–40, 2008.
[89] R. F. Reinhart, A. Lemme, and J. J. Steil. Representation and generaliza-
tion of bi-manual skills from kinesthetic teaching. In Proc. of IEEE-RAS
International Conference on Humanoid Robots, Osaka, 2012, in press.
[90] R. F. Reinhart and J. J. Steil. A constrained regularization approach for
input-driven recurrent neural networks.
Diﬀerential Equations and Dy-
namical Systems, 19(1–2):27–46, 2011 (2010 online pre-publication). DOI
10.1007/s12591-010-0067-x.
[91] J. J. Rodriguez, C. J. Alonso, and J. A. Maestro. Support vector machines
of interval-based features for time series classiﬁcation.
Knowledge-Based
Systems, 18(4-5):171–178, 2005.
[92] J. S. Rothman, L. Cathala, V. Steuber, and R. A. Silver. Synaptic depression
enables neuronal gain control. Nature, 457(19 Feb):1015–1018, 2009.
[93] G. Sch¨oner, M. Dose, and C. Engels. Dynamics of behavior: theory and
applications for autonomous robot architectures. Robotics and Autonomous
Systems, 16(2):213–246, 1995.
[94] G. Sch¨oner and J. A. Kelso. Dynamic pattern generation in behavioral and
neural systems. Science, 239(4847):1513–1520, 1988.
[95] J.R. Searle. Minds, brains, and programs. The Behavioral and Brain Sci-
ences, 3:417–457, 1980.
[96] L. Shastri. Advances in Shruti – a neurally motivated model of relational
knowledge representation and rapid inference using temporal synchrony. Ar-
tiﬁcial Intelligence, 11:79–108, 1999.
[97] K. R. Sivaramakrishnan, K. Karthik, and C. Bhattacharyya. Kernels for
large margin time-series classiﬁcation. In Proc. IJCNN 2007, pages 2746 –
2751, 2007.
193
[98] L.B. Smith and E. Thelen, editors. A Dynamic Systems Approach to Devel-
opment: Applications. Bradford/MIT Press, Cambridge, Mass., 1993.
[99] T. Strauss, W. Wustlich, and R. Labahn. Design strategies for weight ma-
trices of echo state networks. Neural Computation, 24(12):3246–3276, 2012.
[100] D. Sussillo and L. Abbott.
Transferring learning from external to inter-
nal weights in echo-state networks with sparse connectivity. PLoS ONE,
7(5):e37372, 2012.
[101] D. Sussillo and O Barak.
Opening the black box: Low-dimensional dy-
namics in high-dimensional recurrent neural networks. Neural Computation,
25(3):626–649, 2013.
[102] M. Timme, F. Wolf, and Th. Geisel.
Unstable attractors induce per-
petual synchronization and desynchronization.
Chaos, 13:377–387, 2003.
http://arxiv.org/abs/cond-mat/0209432.
[103] I. Tsuda. Towards an interpretation of dynamic neural activity in terms of
chaotic dynamical systems. Behavioural and Brain Sciences, 24(5):793–810,
2001.
[104] F. van der Velde and M. de Kamps. Neural blackboard architectures of com-
binatorial structures in cognition. Behavioural and Brain Sciences, 29(1):37–
70, 2006.
[105] T. van Gelder. The dynamical hypothesis in cognitive science. Behavioural
and Brain Sciences, 21(5):615–628, 1998.
[106] D. Verstraeten. Reservoir Computing: Computation with Dynamical Sys-
tems. Phd thesis, Electronics and Information Systems, University of Ghent,
2009.
[107] D.M. Wolpert and M. Kawato. Multiple paired forward and inverse models
for motor control. Neural Networks, 11(7-8):1317–1330, 1998.
[108] F. wyﬀels, J. Li, T. Waegeman, B. Schrauwen, and H. Jaeger. Frequency
modulation of large oscillatory neural networks. Biological Cybernetics, pub-
lished online Feb 2014.
[109] Y. Yao and W.J. Freeman. A model of biological pattern recognition with
spatially chaotic dynamics. Neural Networks, 3(2):153–170, 1990.
[110] I. B. Yildiz, H. Jaeger, and S. J. Kiebel. Re-visiting the echo state property.
Neural Networks, 35:1–20, 2012.
194
[111] S. K. U. Zibner, C. Faubel, I. Iossiﬁdis, and G. Sch¨oner. Dynamic neural
ﬁelds as building blocks of a cortex-inspired architecture for robotic scene
representation. IEEE Trans. on Autonomous Mental Development, 3(1):74–
91, 2011.
195
