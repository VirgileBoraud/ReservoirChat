royalsocietypublishing.org/journal/rstb
Review
Cite this article: Seoane LF. 2019
Evolutionary aspects of reservoir computing.
Phil. Trans. R. Soc. B 374: 20180377.
http://dx.doi.org/10.1098/rstb.2018.0377
Accepted: 22 November 2018
One contribution of 15 to a theme issue ‘Liquid
brains, solid brains: How distributed cognitive
architectures process information’.
Subject Areas:
computational biology, neuroscience,
theoretical biology, cognition, evolution,
systems biology
Keywords:
reservoir computing, liquid brains, solid brains,
evolution, evolutionary computation,
morphospace
Author for correspondence:
Luı´s F. Seoane
e-mail: luis.seoane@upf.edu
Evolutionary aspects of reservoir computing
Luı´s F. Seoane1,2
1ICREA-Complex Systems Lab, Universitat Pompeu Fabra, Barcelona 08003, Spain
2Institut de Biologia Evolutiva (CSIC-UPF), Barcelona 08003, Spain
LFS, 0000-0003-0045-8145
Reservoir computing (RC) is a powerful computational paradigm that allows
high versatility with cheap learning. While other artificial intelligence
approaches need exhaustive resources to specify their inner workings, RC
is based on a reservoir with highly nonlinear dynamics that does not require
a fine tuning of its parts. These dynamics project input signals into high-
dimensional spaces, where training linear readouts to extract input features
is vastly simplified. Thus, inexpensive learning provides very powerful tools
for decision-making, controlling dynamical systems, classification, etc. RC
also facilitates solving multiple tasks in parallel, resulting in a high through-
put. Existing literature focuses on applications in artificial intelligence and
neuroscience. We review this literature from an evolutionary perspective.
RC’s versatility makes it a great candidate to solve outstanding problems
in biology, which raises relevant questions. Is RC as abundant in nature as
its advantages should imply? Has it evolved? Once evolved, can it be
easily sustained? Under what circumstances? (In other words, is RC an
evolutionarily stable computing paradigm?) To tackle these issues, we intro-
duce a conceptual morphospace that would map computational selective
pressures that could select for or against RC and other computing
paradigms. This guides a speculative discussion about the questions above
and allows us to propose a solid research line that brings together
computation and evolution with RC as test model of the proposed
hypotheses.
This article is part of the theme issue ‘Liquid brains, solid brains: How
distributed cognitive architectures process information’.
1. Introduction
Somewhere between pre-biotic chemistry and the first complex replicators, infor-
mation assumed a paramount role in our planet’s fate [1–3]. From then onwards,
Darwinian evolution explored multiple ways to organize the information flows
that shape the biosphere [4–11]. As Hopfield argues, ‘biology looks so different’
because it is ‘physics plus information’ [12]. Central in this view is the ability of
living systems to capitalize on available external information and forecast regu-
larities from their environment [13,14], a driving force behind life’s progression
towards more complex computing capabilities [15].
We can trace computation in biology from pattern recognition in RNA and
DNA [16,17] (figure 1a), through the Boolean logic implemented by interactions
in gene regulatory networks (GRNs) [22–24] (figure 1a), to the diverse and ver-
satile circuitry implemented by nervous systems of increasing complexity
[25,26] (figure 1c–e). Computer science, often inspired by biology, has re-
invented some of these computing paradigms, usually from simplest to most
complex, or guided by their saliency in natural systems. It is no surprise that
we find some fine-tuned, sequential circuits for motor control (figure 1c) that
resemble the wiring of electrical installations. Such pipelined circuitry gets
assembled to perform parallel and more coarse-grained operations, e.g. in
assemblies of ganglion retinal cells that implement edge detection [27,28]
(figure 1d), similar to filters used in image processing [29–31]. Systems at
large often present familiar design philosophies or overall architectures, as illus-
trated by the resemblance between much of our visual cortex (figure 1e) and
deep convolutional neural networks for computer vision [19–21,32] (figure 1e).
& 2019 The Author(s) Published by the Royal Society. All rights reserved.
Such convergences suggest that chosen computational
strategies might be partly dictated by universal pressures.
We expect that specific computational tricks are readily avail-
able for natural selection to exploit them (e.g. convolving
signals with a filter is faster in Fourier space, and the visual
system could take advantage of it). Such universalities
could constrain network structure in specific ways. We also
expect that the substrate chosen for implementing those com-
putations is guided by what is needed and available. This is,
at large, one of the topics discussed in this issue. Different
authors
explore
specific
properties
of
computation
as
implemented, on the one hand, by liquid substrates with
moving components such as ants or T cells; and, on the
other hand, by solid brains such as cortical or integrated cir-
cuits.
Rather
than
this
‘thermodynamic
state’
of
the
hardware substrate, this paper reviews the reservoir comput-
ing (RC) framework [33–37], which somehow deals with a
‘solid’ or ‘liquid’ quality of the signals involved, hence
rather focusing on the ‘state’ of the software. As with other
computing architectures, tricks and paradigms, we expect
that the use of RC by nature responds to evolutionary
pressures and contingent availability of resources.
RC matters within a broader historical context because it
has helped bypass a huge problem in machine learning. The
first widely successful artificial intelligence architectures were
layered, feed-forward networks (figure 1e and [20] are
modern examples). These get a static input (e.g. a picture)
whose basic features (intensity of light in each pixel) are
read and combined by artificial neurons or units. A neuron’s
reaction, or activation, is determined by a collection of
weights that measure how much each of the features matters
to that unit. These activations are conveyed forward to newer
neurons that use their own weights to combine features non-
linearly, and thus extract complex structures (edges, shapes,
on
off
(a)
(b)
(c)
(d)
A
B
C
D
(e)
Cas9
PAM
target
5’
3’
3’
5’
5’
3’
N
GG
gRNA
AND
GFP
gene A
gene B
reporter gene
NOT
a
a
b
b
dsDNA
PA
PD
PB
LN
LN
LN
LN
LN
LN
LN
layer 1
layer 2
layer 3
layer 4
cleavage
Figure 1. Some computing devices in biology. (a) DNA and RNA perform multiple pattern recognition, starting with the simplest matching of codons in ribosomes to
synthesize proteins. More complex pattern matching (here, Cas9 matches a longer string of RNA to DNA, drawing from Marius Walter available at https://commons.
wikimedia.org/w/index.php?curid=62766587) allows cleavage and insertion of DNA snippets. (b) Boolean logic determines gene expression or silencing. In the
example, a receptor gene that expresses green fluorescent protein (GFP) is only active if genes A and B are expressed and their corresponding products dimerize.
This implements a Boolean AND gate. (Redrawn from [18].) (c) The knee jerk reflex is implemented by sequential circuits that remind us of the simple wiring of an
electrical installation. Sensory neurons (A) gather a sudden input. Information follows the arrows towards ganglia at the entrance of the spinal cord (B) and grey
matter within. There, information splits in two: it is partially sent upstream towards the brain stem (not shown) and partially used to excite motor neurons (C),
which right away pull from muscles at their end (D). (d) Retinal ganglion cells (red) pool information from rods and cones (grey) through horizontal, bipolar (black
and blue) and amacrine cells. This pooling involves excitatory (blue) and inhibitory (black) mediating connections, which make a ganglion cell responsive to light
and its absence in the so-called on- and off-centres and surround (red and blue discs, corresponding to dark and light grey cones). Receptive fields of several ganglia
are consequently pooled as well, resulting in edge detectors. (e) Specific neurons implementing average and pooling exist throughout the visual cortices [19]. These
are the building blocks of deep convolutional neural networks [20]. The overall information flow, architecture and various receptive fields of such artificial networks
resembles several processing steps found in mammal visual cortices [21], including edge detection in V1 thanks to retinal ganglia. (This image was drawn with
elements from [21] and https://commons.wikimedia.org/w/index.php?curid=1679336.) (Online version in colour.)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
2
faces, etc.). Eventually, the whole network settles into a fixed
state. A set of output units returns the static result of a com-
putation (e.g. whether Einstein is present in the picture).
Training a network consists of adjusting the weights for
every neuron such that the system at large implements a
desired computation (e.g. automatic face classification).
Solving this problem in feed-forward networks with several
layers was a challenge for decades until a widespread
solution (back-propagation [38]) was adopted. Recurrent
neural networks (RNN) brought in more computational
power to the field. RNN contemplate feedback from more
forward to earlier processing neurons. These networks do
not necessarily settle in static output states, allowing them
to produce dynamic patterns, e.g. for system control. They
are also apt to process spatiotemporal inputs (videos, voice
recordings, temporal data series, etc.), finding dynamical
patterns often with long-term dependencies. Echoing the
early challenges in feed-forward networks, full RNN training
(i.e. adjusting every weight optimally for a desired task)
presents important problems still not fully tamed [39,40].
RC is an approach that vastly simplifies the training of
RNN, thus making more viable the application of this power-
ful technology. Instead of attempting to adjust every weight
in the network, RC considers a fixed reservoir that does not
need training (figure 2a), which works as if multiple, parallel
spatiotemporal filters were simultaneously applied onto the
input signal. This effectively projects nonlinear input features
onto a huge-dimensional space. There, separating these fea-
tures becomes a simple, linear task. Despite the simplicity
of this method, RC-trained RNN have been robustly used
for a plethora of tasks including data classification [42–44],
systems control [43,45–47], time-series prediction [48,49],
uncovering grammar and other linguistic and speech features
[43,50–53], etc.
Again, we expect that nature has taken advantage of any
computing approaches available, including RC, and that
important
design
choices
are
affected
by evolutionary
constraints. These will be major topics through the paper:
how could RC be exapted by living systems and how
might evolutionary forces have shaped its implementation?
Section 2 provides a brief introduction to RC and reviews
what its operating principles (notably optimal reservoir
design) imply for biological systems. Section 3 shows inspir-
ing examples from biology and engineering. Comments on
selective forces abound throughout the paper, but §4 wraps
up the most important messages. This last part is largely
speculative in an attempt to pose relevant research questions
and strategies around RC, evolution and computation. We
will hypothesize about two important topics: (i) what explicit
evolutionary conditions might RC demand and (ii) what
evolutionary paths can transform a system into a reservoir.
2. Computational aspects of reservoir computing
(a) Reservoir computing in a nutshell
RC was simultaneously introduced by Jaeger [33] and Maass
et al. [34]. Jaeger arrived at Echo State Networks from a
machine learning approach, while Maass et al. developed
Liquid State Machines with neuroscientifically realistic spiking
neurons. The powerful operating principle is the same behind
both approaches, later unified under the RC label [35–37].
Consider a RNN consisting of N units, all connected
to
each
other,
which
receive
an
external
input
yin(t) ; {yin
i (t), i ¼ 1, . . . , N} (with yin
i (t) ¼ 0 if the ith unit
receives no input). Each unit has an internal state xi(t) that
evolves following:
xi(t þ Dt) ¼ s
X
N
j¼1
vijx j(t) þ yin
i (t)
0
@
1
A,
(2:1)
where s() represents some nonlinear function (e.g. an
hyperbolic tangent). Variations of this basic theme appear
in the literature. For example, continuous dynamics based
on differential equations could be used; or inputs could
consist of weighted linear combinations of Nfeat more funda-
mental features {uk(t), k ¼ 1, . . . , Nfeat} such that yin
i (t) ¼
PNfeat
k¼1 vin
ik uk(t). This would allow us to trade off importance
of external stimuli versus internal dynamics.
Such RNN can be trained so that Nout designated output
units produce a desired response yout(t) ; {yout
i
(t), i ¼ 1, . . . ,
Nout} when yin(t) is fed into the network. The training consists
in varying the vij until the state of the output units given the
input
xout(tjyin(t)) ; {xout
i
(tjyin(t)), i ¼ 1, . . . , Nout}
matches
the desired behaviour yout(t). A naive approach is to initialize
the vij randomly and modify all weights, e.g. using gradient
descent, to minimize some error function e(yout(t), xout
(tjyin(t))) that measures how much the network activity devi-
ates from the target. Such a training procedure is often useless
because the RNN’s recurrent dynamics introduce insurmoun-
table numerical problems [39,40].
The RC approach to RNN training still uses random vij
but does not attempt to modify them. We say then that the
units described by equation (2.1) constitute a reservoir
(figure 2a). Upon it, we append a set of Nout readout units
(figure 2b) whose activity xout(t) ; {xout
i
(t), i ¼ 1, . . . , Nout}
is just a linear combination of the reservoir activity:
xout
i
(t) ¼
X
N
j¼1
vout
ij x j(t):
(2:2)
Training proceeds on these output units alone. Only the
vout
ij
are modified; these do not feed back into the reservoir,
which remains unchanged. The absence of this feedback
during learning dissolves the grave numerical problems
that affect other RNN. Finding the right vout
ij
for a task
becomes as simple as a linear regression between reservoir
activity over time given an input x j(tjyin(t)) and the desired
target yout(t) [54]. For a review on RC training (including
issues of reservoir design discussed in §2b), see [55] and
other works [56,57]. Also, the computational capacity of RC
can be boosted if a (trainable) feedback is added from the
linear readouts to the reservoir [58–62]. This allows for
more agile context-dependent computations and longer-
term memory [58]. Controlling the network attractors (thus
the learning) that such feedback induces is still not fully
understood. These and other variations upon the RC theme
are good steps towards full RNN training.
Back to the most naive version of RC, the only task of
the reservoir is to have its dynamic internal state perturbed
by the input. In doing so, through its nonlinear, convo-
luted dynamics, the reservoir is picking up the external
signal and projecting it into the huge-dimensional space
that consists of all possible dynamic configurations of
the reservoir (figure 2c). This high-dimensional space
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
3
hopefully renders relevant features from the input more
easily separable. Ideally, such features could be separated
with a simple hyperplane that bisects this abstract space.
This is precisely what the vout
ij
implement. RC training con-
sists in finding the right hyperplane solving each task
given the reservoir.
Of course, very poor dynamics could project inputs into
boring reservoir configurations so that prominent features
cannot be picked up. The next section discusses issues of
reservoir design so that it produces optimal dynamics. How-
ever, the most important part of the training falls upon the
readouts, whose dynamics do not affect the reservoir. This
brings in the two most important advantages of RC: (i) learn-
ing is extremely easy, as just discussed and (ii) multiple
readouts can be appended to a same reservoir (thus solving
different, parallel tasks) without interference.
input
y–in(t)
x– 
1
out(t)
x– 
N
out    (t)
out
x– (t)
output
(b)
(a)
(c)
(d)
0.5
1
2 3
l
wscale
l
l
4 6 8
0.05
0.1
0.5
1
2
4
8
200
250
300
350
400
450
0.5
1
2 3 4 6 8
0.05
0.1
0.5
1
2
4
8
200
250
300
350
400
450
(e)
(g)
( f )
0.5
1
2 3 4 6 8
0.05
0.1
0.5
1
2
4
8
1
3
0
5
10
15
20
2
Figure 2. Computational basis of reservoir computing. (a) Reservoir dynamics (black nodes and arrows) are driven by external inputs (grey arrows). For RC, weights
within the reservoir are not modified. (b) A set of Nout output nodes can be appended. Each of them computes a linear combination of the reservoir state and solves
a single task. In RC training, only the weights of these output units are modified. This allows us to solve several (in this case Nout) different tasks simultaneously
without interference. (c) The key to RC computing is that the reservoir filters, and thus projects, low-dimensional input signals into a huge-dimensional space. Here
we observe three example trajectories (elicited, e.g. by three different inputs) through a representation of this abstract space. (d) Appropriate reservoir dynamics will
make it possible to separate the projection of different inputs using simple linear classifiers (implemented by the readout units from b). If we sort in a matrix the
different dynamical states reached after different inputs, the rank of this matrix tells us how linearly independent the dynamics have remained. This gives us an idea
of how many different hyperplanes (i.e. readouts and different tasks) we can draw in this abstract space. (e–g) rS, rG and rS 2 rG. Plot reproduced from [41]. (e) rS,
which captures this separability property, is plotted for reservoirs made of simulated cortical columns with varying correlation length between neurons (l) and scale
of synaptic strength (vscale). We find large rS (hence large separability) for chaotic dynamics achieved by large l and vscale. (f) When noise and redundant
information have been provided, we expect the rank of a similar matrix (rG, which measures a generalization capability) to score low. The same microcolumns
as before present good generalization property when the dynamics are ordered and similar inputs are consistently mapped into a same region of the space
of reservoir dynamics. (g) Subtracting rG from rS measures a balance between the two desired reservoir properties. Good performance also correlates with a measure
of criticality (dot marked ‘2’, where the Lyapunov exponent vanishes). Performance is bad for too ordered (dot ‘1’, negative Lyapunov exponent) or too chaotic
dynamics (dot ‘3’, positive Lyapunov exponent). Black curves on top indicate the limits where Lyapunov exponents are fairly close to 0, thus marking the edge of
chaos. (Online version in colour.)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
4
(b) Good reservoir design: separability, generalization,
criticality and chaos
When RC was first introduced, the authors proved in a series
of theorems what conditions (e.g. having ‘echo state’, ‘fading
memory’, the ‘separation property’, etc.) a reservoir needed
to fulfill to implement the paradigm properly [33,34].
While the theorems demanded these requirements in prin-
ciple, practical implementations turned out more lenient
and a wide variety of systems can trivially work as suffi-
ciently good reservoirs. In practice, a wide variety of
systems can trivially work as reservoirs. As proofs of concept,
among many others, in silico implementations have used rea-
listic
theoretical
models
of
neural
networks
[34,41,63],
networks of springs [64,65] or cellular automata [66]; and in
hardware, analogue circuits [67,68], a bucket of water [69]
and diverse photonic devices [70–72] have been tried. All
those theoretical conditions on reservoir dynamics boil
down to two desired properties: (i) reservoir dynamics
must be able to separate different input features that are
meaningful for a variety of tasks while (ii) these same
dynamics must be able to generalize to unseen examples,
thus projecting reasonably similar inputs into a reasonable
neighbourhood within the dynamical space of the reservoir.
To fulfill these conditions, we want our reservoirs to behave
somehow in between chaotic and simple dynamics, thus
resonating with ideas about criticality in complex systems,
as discussed below.
In a series of papers [41,63,73], Maass et al. explored two
elegant measures that quantify the ability of a reservoir to sep-
arate relevant features and to generalize to unseen examples.
Regarding separability, since RC works with simple
linear readouts, we can quantify straightforwardly how
many different binary features can be extracted by hyper-
planes that bisect the space of dynamic configurations of
the reservoir (figure 2d). Suppose a collection of inputs
YS ; {yS
k(t), k ¼ 1, . . . , NS} is fed to the reservoir. At time
t0 þ T after input onset (which happened at t0), the reservoir
activity x(t0 þ TjyS
k) ; {xi(t0 þ TjyS
k), i ¼ 1, . . . , N} is recorded
in an N-sized array. The collection XS(t0 þ TjYS) ; {x(t0þ
TjyS
k), k ¼ 1, . . . , NS} consists of NS such arrays sorted in a
matrix. The rank rS ; rankfXSg conveys an idea of how line-
arly independent the driven activity of the reservoir is—i.e. of
how many binary-classifiable features the reservoir can pick
apart. Given YS, there are 2NS different possible binary classi-
fication problems. If rS ¼ NS, we ensure that the dynamics of
the reservoir can pick apart the features relevant to all these
problems. Even if rS , NS, in general, the larger rS the
better a reservoir would be, since it would make more
degrees of freedom available to set up the linear readouts.
As for a reservoir’s ability to generalize, we approach
the problem similarly, but assuming now that our input
contains some redundant or tangential information (e.g.
some of its variability comes from noise). Good reservoirs
should be able to smooth this out. Assume that a larger
collection of inputs YG ; {yG
k (t), k ¼ 1, . . . , NG . NS} is fed
to the reservoir. Its dynamics are similarly captured by
the
matrix
XG(t0 þ TjYG) ; {x(t0 þ TjyG
k ), k ¼ 1, . . . , NG}.
If the reservoir is capable of classifying noisy versions of
the
input
under
a
same
class
(i.e.
of
generalizing),
we expect now the rank rG ; rankfXGg to be as small as poss-
ible. (More rigorous analysis relates rG to the VC-dimension
of the system. This is, to the volume of input space that can
be shattered by the reservoir dynamics—see [41,63,73–75] for
further explanations.)
These measures rely on the expectation that the rank of
arbitrary activity matrices (such as XS and XG) will increase
if new meaningful examples are provided, but will stall if
examples add spurious or redundant variability. What consti-
tutes meaning and noise in each case is an open question,
which still depends on the eventual task. We expect to find
more spurious information the more examples we have—
hence NG . NS. Final values of rS and rG will still depend
on NS and NG—which could stand, e.g. for test and train
set sizes.
Despite these caveats, naive applications of rS and rG
seem to capture good reservoir design that works for a var-
iety of problems. In [41,63,73], realistic models of cortical
columns were used as reservoirs. These models mimic the
three-dimensional geometric disposition of neurons in the
neocortex. Given two neurons located at x1 and x2, the likeli-
hood that they are connected decays as exp (2D2(x1, x2)/l2)
with the Euclidean distance D(x1, x2) between them. The
parameter l introduces an average geometric length of
connections, resulting in sparse or dense circuits if l is,
respectively, small or large. This in turn leads to short-lived
or more sustained dynamics. Individual neurons were
simulated with equations for realistic leaky membranes
[76], including proportions of inhibitory neurons within the
circuit. Synaptic strengths were drawn randomly to reflect
biological data [34], and they all were scaled by a common
factor vscale. Small vscale led to weakly coupled neurons in
which activity faded quickly; while large vscale implied
strong coupling between units, resulting in more active
dynamics.
The authors produce morphologically diverse reservoirs
by varying l and vscale. Too sparse a connection (owing to
low likelihood of connection or weak synapses—respectively
low l and vscale, lower-left corner in figure 2e–g) leads to
poor dynamics. This results in undesired low separability
(low rS, figure 2e) but brings about the expected large gener-
alization (small rG, figure 2f) because large classes of noisy
input result in converging reservoir dynamics. Meanwhile
strong synapses and a very dense network (large l and
vscale, upper-right corner in figures 2e–g) easily become
chaotic. These have the desired large separability (large rS,
figure 2e) but very low generalization capabilities (large rG,
figure 2f). This is so because of the high sensibility of chaotic
systems to initial conditions. Performance in arbitrary tasks is
best when rS and rG are relatively balanced (figure 2g). Reser-
voirs in that region of the l 2 vscale plane are capable
of
separating
relevant
behaviours
while
recognizing
redundancy and noise in the input data.
The quantities rS and rG capture desirable properties of
reservoir dynamics. They are also easy to measure empiri-
cally (see §3a) and hence determine if those given dynamics
are good for RC. These convenient reservoir properties
(separability and generalization) are conflicting traits—chan-
ging a circuit to improve one often degrades the performance
in the other. The authors in [41,63,73] acknowledge that
they do not have a principled way to compare rS versus
rG—figure 2e just shows the difference between them.
We propose that Pareto optimality [77–79] might be a
well-grounded framework for this problem. Pareto, or multi-
objective, optimization is the most parsimonious way to bring
together quantifiable traits that cannot be directly compared
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
5
(e.g. because they have disparate units and dimensions),
and to do so without introducing undesired biases. For the
current problem, Pareto optimality would give us the best
rS 2 rG trade-off as embodied by the subset of neural
circuits that cannot be changed to improve both quantities
simultaneously. This offers a guideline to select systems that
perform somehow optimally towards both targets. This
approach has helped identify salient designs in biological sys-
tems evolving under conflicting forces [80–83]. It can also link
the computation capabilities of reservoirs to phase transitions
[79,84–86] and, more relevant for us, criticality [87].
Indeed, criticality has long been a good candidate as a
governing principle of brain connectivity and dynamics
[73,88–94]. In statistical mechanics, critical systems are rare
configurations of matter poised between order and disorder.
Such states present long-range correlations between parts of
the system in space and time, with arguably optimal sensi-
tivity to external perturbations. A similar phenomenon was
noted in computer science studying cellular automata and
random
Boolean
networks
[23,73,95–98].
Such
systems
usually present ordered and disordered phases. In the
former (analogous, e.g. to solid matter in thermodynamics),
activity fades away quickly into a featureless attractor. No
memory is preserved about the initial state (which acts as
computational input to the system) or its internal structure.
The later, disordered phase presents chaotic dynamics with
large sensitivity to the input and its inner vagaries. Slightly
similar initial conditions differ quickly, thus erasing any cor-
relation between potentially related inputs and resulting in
trajectories without computational significance. The so-
called edge of chaos separates both behaviours, balancing the
stability of the ordered phase (thus building relatively lasting
steady states) and the versatile dynamics of the disordered
phase (which enables the mixing of relevant input features).
These properties allow systems at the edge of chaos to
optimally combine input parts and compute.
This depiction of critical systems reminds us of the
desirable design captured by rS and rG. Several authors
have used hallmark indicators of criticality to contrive opti-
mal reservoirs with enhanced performance. The measures
employed include Lyapunov exponents [41,63,98–102] and
Fisher information [103,104] of reservoir dynamics. The
former estimates how rapidly slight perturbations get ampli-
fied
(thus
diverge)
as the
reservoir
dynamics
unfold.
This divergence never happens if the dynamics are too
ordered (perturbations get dumped, Lyapunov exponents
are negative) and happens too quickly in chaotic regimes
(positive exponents). Only at the edge of chaos (Lyapunov
exponents tend to zero) can a small perturbation make a last-
ing yet meaningful difference that does not fade away or
explode over time. Similar principles lead to diverging
Fisher information as criticality is approached [105].
All these works report notable correlations between such
traces of criticality and enhanced reservoir performance—in
line with the evidence that a balanced rS 2 rG is an indicator
of good reservoir design [41,63,73]. Figure 2g also shows how
this rS 2 rG compromise degrades faster in the chaotic than in
the more ordered region. Meanwhile, Toyozumi & Abbott
[100] use Lyapunov exponents to suggest that reservoir
performance should degrade faster in the ordered phase
than in the disordered one. In the emerging picture, the
advantages of criticality are clear and suggest powerful evol-
utionary constraints to bring naturally occurring RC-based
systems towards the edge of chaos. But a critical state is
often difficult to reach and sustain, so often we settle for get-
ting as close as possible. Both [41,63,73,100] predict that it
makes a difference from which side we approach the edge
of chaos. They disagree on what side is computationally pre-
ferred. Empirical studies (see [94] for an up-to-date review)
remain inconclusive too. Further research is needed.
3. Inspiring examples in biology and engineering
(a) Reservoirs in the brain
When conceiving liquid state machines, Maass et al. drew
inspiration
from
the
cortical
microcolumn
(figure
3a).
Through RC, they proposed a plausible computing strategy
that could be the basic operating principle of these circuits
[34,111–113]. These neural motifs are roughly cylindrical
structures that convey information mostly inwards from
and perpendicularly to the neocortex surface. Contiguous
columns are saliently distinguished from each other, but
sparse lateral connections do exist. An important part of the
(c)
(b)
(a)
layer 2/3
layer 4
layer 5
x1
out
Fh(t)
x 
N
outout
Figure 3. Reservoirs in the brain and the body. (a) Cortical microcolumns have been proposed as the basic operating unit of the neocortex [106]. Liquid state
networks [34] were largely introduced in an attempt to clarify the computational basis of these circuits. Figure credit to Oberlaender et al. reconstruction methods
in [107]. (b) Schematic of connections between layers within a single cortical microcolumn. Figure reconstructed from [108–110]. Black node contour and black
arrows represent, respectively, excitatory neurons and connections, red indicates inhibitory ones. Arrow width is proportional to connection strength as estimated
from empirical data. Dashed connections indicate data are less significant. Grey-filled nodes indicate that they receive input from other areas—mostly cortex and
thalamus. Green-filled nodes indicate they act as output—mostly to other cortical areas (both output nodes) and thalamus (layer 5 output node). (c) Networks of
springs can work very efficiently as reservoirs, as long as they display heterogeneous dynamics as a response to inputs. This opens a huge potential in robotics and
elsewhere, since multiple mechanoelastical systems can be modelled as networked spring-mass structures. (Online version in colour.)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
6
internal structure, as well as external connections to and from
other columns and other parts of the brain (notably the thala-
mus), appears to be stereotypical [114]. Specific connections
and circuit motifs correlate with the different cortical layers
within each column (figure 3b). Maass et al. built upon this
known average structure when designing realistic reservoirs.
Still,
columns
vary
morphologically
and
functionally
throughout the cortex. Most of them can hardly be associated
with exclusive functionality, while others can be linked down
to specific computations. For example, the receptive field of
columns in V1 in cats was early identified by Hubel and
Wiesel as responding to visual gratings with specific incli-
nation [115]; the barrel cortex in mice consists of cortical
columns that have grown in size through evolution and
have specialized in processing the sensory stimuli from
individual whiskers [116]. It has been proposed that the
cortical microcolumn constitutes the operational unit of the
neocortex, and that the advanced cognitive success of
mammal brains is a consequence of the exhaustive use of
this versatile circuit [106]. Evidence about this remains incon-
clusive. In any case, discerning the computational basis of
cortical circuits is a relevant, open question in neuroscience
to which RC can contribute greatly.
In [110], Maass notes the heterogeneity of neuron types
in the brain (also within cortical columns). They display
great morphological diversity, varying numbers of pre-
synaptic neurons and different physiological constituency
that results in diversified time and spacial integration
scales. The recursive nature of most neural circuits is also
highlighted. This diversity and recursivity poses a challenge
to the mostly feed-forward computing paradigms engineered
into our computers, which also rely on a relative homogen-
eity of its components to make hardware modular and
reprogramable. What could then be the computational basis
of neural circuits? RC comes in to take advantage of these
features, and steps forward as a likely computational
foundation of the brain. Looking for empirical evidence,
three hallmarks have been sought in cortical circuits that
would be indicative of RC: (i) records of this morphologi-
cal heterogeneity, which in turn result in the dynamical
diversity proper of a reservoir; (ii) existence of parallel neur-
ons that gather information from a same processing centre
(akin to a repository) and project their output into distinct
areas that solve different tasks, just as RC readouts use a
same reservoir without interference; and (iii) the ability to
retrieve relevant (including highly nonlinear) input features
by training simple linear classifiers on neural recordings.
Note that all these would constitute rather circumstantial evi-
dence for RC since each of these features could be exploited
by other computing paradigms too. As of today, this is the
best that we can get regarding signs of RC in the brain. We
review a series of studies showing such indirect evidence in
the next paragraphs.
Neural heterogeneity resulting in desirable, reservoir-like
properties has been reported in cortical neurons, the retina
and the primary visual cortex, and in networks grown
in vitro from dissociated neurons [109,117–121]. The existing
diversity of neural components results in very heterogeneous
dynamics [122]—which can be exploited by RC-like compu-
tation. The emergence of such dynamical heterogeneity
in dissociated neurons reveals that this property follows
spontaneously from the wiring mechanisms. Indeed, compu-
tational
studies
show
how
input-driven
spike-timing
dependent plasticity can generate heterogeneous circuitry
that then displays good reservoir behaviour [123].
As argued above, by restricting training to the output
units, RC facilitates the use of a same reservoir to solve differ-
ent tasks by just plugging parallel readouts that do not
interfere with each other. This principle seems to be exploited
by the brain. In [124], neurons are recorded that project from
the mouse barrel cortex to other sensory and motor centres.
Both kinds of neurons retrieve the same information, but
they respond differently, with task specificity depending on
whether further sensory processing is required or whether
the motor system needs to be involved. From the RC frame-
work, the barrel cortex would be working as a reservoir
whose activity is not enough to determine which neurons
will respond and how, suggesting independent wiring for
each individual task based on a same, shared input.
All these works casually suggest that advanced neural
structures are indeed using some of the RC principles.
More compelling evidence comes from the computational
analysis of neural populations as they relate to input signals
and more precisely, from the ability to retrieve nonlinear
input features by using just linear combinations of sparse,
recorded neural activity [125–127]. In [125], it is shown
how sparse recordings from the primary auditory cortex of
ferrets (involving just between 4 and 10 neurons) are
enough to retrieve nonlinear features of auditory stimulus
in a task that involves tones that increase or decrease ran-
domly by octaves. This information can be extracted by
simple linear classifiers trained upon the recorded neural
activity. This is possible because the recorded neurons
(which act as a reservoir) already implement a sufficient, non-
linear
transformation
of
the
input.
More
complicated
methods
(e.g.
support
vector
machines)
do
not
show
a relevant performance increase. We can parsimoniously
assume that evolution would settle for simpler solutions
(i.e.
linear
readouts)
if
they
suffice—unless
unknown
selective pressures existed.
The term mixed selectivity is used in [126,127] to under-
score that neurons from the reservoir in these experiments
do not respond to simple (i.e. somehow linear) features of
the stimulus. Measures akin to the rS and rG introduced
above are computed in [126,127] for a set of neural recordings
while monkeys perform a series of tasks. It is shown how the
classification accuracy grows with the dimensionality of
the space (which corresponds to larger rS) into which the
neural recordings project the input stimuli. The underlying
reason is, again, that this larger rS allows more different
binary classifiers to be allocated among the data.
The baseline story is that biological neural systems in the
neocortex (and potentially other parts of the brain) exhibit all
the ingredients needed to implement RC. Most importantly, a
lot of meaningful information can be retrieved from real
neural activity using simple linear readouts. From an
evolutionary point of view, it would feel suboptimal to per-
form further complicated operations (especially provided
that they do not improve performance [125,126]).
(b) The body as a reservoir
In [64], Hauser et al. implemented a reservoir using two-
dimensional networks of springs (figure 3c). Inputs are
provided as horizontal forces that displace some springs
from their resting states. Such perturbations propagate
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
7
through the network similarly to activity in other reservoirs.
Simple linear readouts can be trained to pick up both vertical
and horizontal elongations (these would constitute the
internal state x(t) of the system), and thus perform all kind
of computations upon the input signals. It seems trivial that
such a reservoir will work as long as the springs present a
variety of elastic constants (hence providing the richness of
dynamics that RC demands). But a more important concep-
tual point is made in [64]: the possibility that bodies can
function as reservoirs, with springs modelling muscle fibres
and other sources of mechanical tension.
A more explicit implementation is explored in [65,128],
where the muscles of an octopus arm are simulated and
used as a reservoir. Torques at the base of the arm serve as
inputs. These forces propagate along the arm, perturbing
modules of coupled springs. A simple linear classifier reads
the elongation of the various springs. The readouts are
trained using standard RC methods until they reproduce a
desired function of the output. Alternatively, readout activity
is fed back to the arm and trained so that it displays a target
motion. Octopuses have a central brain with approximately
50 million neurons versus a distributed nervous system
with approximately 300 million neurons [65]. The compu-
tational power of nerve cells along the arms is beyond
doubt [129,130]. But this approach is telling us something
much more important: a lot of the nonlinear calculations
needed to process and control an arm’s motion could be pro-
vided for free by spurious mechanical forces picked up by
simple linear classifiers.
The nonlinearities and unpredictable behaviour of soft
tissue could have been a nuisance in robotics. They could
have been perceived as untameable systems, very costly to
simulate, that a central controller would need to oversee in
real time. But a recent trend termed morphological computation
[131–133] exploits these nonlinearities, self-organization and
in general the ability that soft tissues and compliant elements
have shown to carry out complex computations. This frame-
work includes simple behaviours such as passive walkers
[134,135], materials optimized to provide sensory feedback
[136] or collective self-organization of smaller robots [137].
The approach of the body as a reservoir (demonstrated by
the networks of springs just described [64,65,128,138–140]
or by tensegrity structures that can crawl controlled by RC-
based feedback [141,142]) offers a principled way to develop
a sound theory of morphological computation [64,133,138].
Other bodily elements besides physical tensions can work
as a reservoir. Recently, Gabalda-Sagarra et al. [143] have
shown how the GRNs in a range of cells (from bacteria to
humans) present a structure quite suited for RC. Empirically
inspired GRNs are simulated and used as reservoirs to solve
benchmark problems as well as known optimal topologies.
They also show how an evolutionary process could success-
fully train output readouts stacked on top of those GRNs.
These examples with physical bodies and gene cross-
regulation highlight a potential abundance of repertoires
in nature.
They also highlight the importance of embodied compu-
tation [132,144]—the fact that living systems develop their
behaviour within a physical reality whose elements (includ-
ing bodies) can participate in the needed calculations,
become passive processors, expand an agent’s memory, etc.
In robotics, this opens up huge possibilities [145,146]—e.g.
to outsource much of the virtual operations needed to
simulate robot bodies. From an evolutionary perspective,
the powerful and affordable computations that RC offers
through compliant bodies raise a series of questions. For
example, with animal motor control in mind: since RC is a
valid approach to the problem and it seems to provide so
much computational power for free, why is it not more
broadly used? Why would, instead, a centralized model
and simulation of our body (such as the one harboured by
the sensory-motor areas of the cortex) become so prominent
instead? What were the evolutionary forces shaping this pro-
cess,
which
somehow
displaced
computation
from
its
embodiment to favour a more virtual approach? It is still
possible that, unknown to us, RC actually takes place with
the body (or parts of the body) as a reservoir—after all, the
paradigm has only been introduced recently (and already;
see [133,147,148] for examples falling close enough). However,
the most salient features of advanced motor control (e.g. sen-
sory-motor
cortices,
the
central
pattern
generator
that
regulates gait, some peripheral circuits implementing reflexes)
do not resemble RC much. So the above questions can still
teach us something about how RC endures different selective
pressures. A possibility that we explore further in the next
section is that RC could be an unstable evolutionary solution.
4. Discussion: evolutionary paths to reservoir
computing
RC is a very cheap and versatile paradigm. By exploiting a
reservoir capable of extracting spatio-temporal, nonlinear fea-
tures from arbitrary input signals, simple linear classifiers
suffice to solve a large collection of tasks including classifi-
cation, motor control, time-series forecasting, etc. [42–53].
This approach simplifies astonishingly the problem of train-
ing RNNs, a job plagued with hard numerical and analytic
difficulties [39,40]. Furthermore, as we have seen, reservoir-
like systems abound in nature: from nonlinearities in liquids
and GRNs [69,143], through mechanoelastic forces in muscles
[64,65,128], to the electric dynamics across neural networks
[34,41,63], a plethora of systems can be exploited as reser-
voirs. Reading off relevant, highly nonlinear information
from an environment becomes as simple as plugging linear
perceptrons into such structures. Adopting the RC viewpoint,
it appears that nature presents a trove of meaningful infor-
mation ready to be exploited and coopted by Darwinian
evolution or engineers so that more complex shapes can be
built and ever-more intricate computations can be solved.
When looking at RC from an evolutionary perspective
these advantages pose a series of questions. Where and
how is RC actually employed? Why is this paradigm not as
prominent as its power and simplicity would suggest? In
biology, why is RC not exploited more often by living organ-
isms (or is it?); in engineering, why is RC only so recently
making a show? This section is a speculative exercise
around these points. We will suggest a series of factors that,
we think, are indispensable for RC to emerge and, more
importantly, to persist over evolutionary time. Based on
these factors, we propose a key hypothesis: while RC shall
emerge easily and reservoirs abound around us, these are
not evolutionarily stable designs as systems specialize or
scale up. If reservoirs evolve such that signals need to
travel longer distances (e.g. over bigger bodies), integrate
information from senses with wildly varying time scales, or
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
8
carry out very specific functions (such that the generalizing
properties of the reservoir are not needed anymore), then
the original RC paradigm might be abandoned in favour of
better options. Then, fine-tuned, dedicated circuits might
evolve from the raw material that reservoirs offer. A main
goal of this speculative section is to provide testable hypoth-
eses that can be tackled computationally through simulations,
thus suggesting open research questions at the interface
between computation and evolution.
First of all, we should not dismiss the possibility that RC
has been overlooked around us—it might actually be a fre-
quent computing paradigm in living systems. It has only
recently been introduced, which suggests that it is not as sali-
ent or intuitive as other computing approaches. There was a
lot of mutual inspiration between biology and computer
science as perceptrons [149], attractor networks [150] or
self-organized
maps
[151]
were
introduced.
Prominent
systems in our brain clearly seem to use these and other
known paradigms [19,21,32,152–154]. We expect that RC is
used as well. We have reviewed some evidence suggesting
that it is exploited by several neural circuits [109,117–
121,124–127], or by body parts using the morphological com-
putation approach [147,148]. All this evidence, while enticing,
is far from, e.g. the strikingly appealing similarity between
the structure of the visual cortices and modern, deep convo-
lutional neural networks for computer vision [20,30,32]
(figure 1e). Altogether, it seems fair to say that RC in biology
is either scarce or elusive, even if we have only recently begun
looking at biological systems through this optic.
The
two
main
advantages
brought
about
by
RC
are: (i) very cheap learning and (ii) a startling capability
for parallel processing. Its main drawback compared to
other paradigms is the amount of extra activity needed
to capture incidental input features that might never be
actually used. We can view these aspects of RC as evolution-
ary
pressures
defining
the
axes
of
a
morphospace.
Morphospaces are an insightful picture that has been used
to
relate
instances
of
natural
[155–158]
and
synthetic
[159–161] complex systems to each other guided by metrics
(sometimes rigorous, other times qualitative) that emerge
from mathematical models or empirical data. Here we lean
towards the qualitative side, but it should also be possible
to quantitatively locate RC and other computational para-
digms in the morphospace that follows. That would allow
us to compare these different paradigms, or different circuit
topologies within each paradigm, against each other under
evolutionary pressures.
A first axis is straightforwardly the dynamical cost (C,
figure 4a) of the reservoir, since RC demands so much more
activity than it eventually uses. This would prevent RC at
large organism scales with costly metabolism, but still
allows myriad smaller physical systems (such as muscles or
tiny bodies) to behave as free-floating reservoirs ready to
be exapted.
(a)
(b)
(c)
(d)
C
R
(c)
(b)
(d)
t
Figure 4. Evolutionary paths to and from RC. (a) We propose a morphospace to locate RC-based circuits (this could be extended to other computing paradigms). The
axes are determined by the circuit cost (C ) and two aspects of an underlying fitness landscape that drives circuit selection: its ruggedness (R) and the average
lifetime (t) that computational tasks contribute to fitness. RC should ensue when both C and t are low and R is high (green volume). Low t and high R demand,
respectively, cheap learning and multitasking. Possible evolutionary paths of non-RC circuits into the RC area (and vice versa) are depicted. (b) (Red curve in panel
a.) Readout units implement simple linear classifiers (indicated by the grey rectangle). Among the many computations implemented by this reservoir, the red units
are solely responsible for one specific, valuable computation that turns out to reliably contribute a lot of fitness. The other readouts and their tasks (as well as some
components of the reservoir) gradually become spurious and evolutionary dynamics get rid of them. Eliminating some nodes can prompt a reorganization of the
original units implementing the valuable computation (solid red units). Such a circuit migrates through the morphospace from the RC region into a position with
little ruggedness and stable peaks of the landscape. The cost, too, should have been decreased. (c) (Also red curve in panel a, traversed in the opposite direction.) A
simple circuit gradually becomes more complex. New tasks are implemented as components are appended. If true RC is reached, feedback between the outputs
(marked with red links) should be lost and readout units should implement linear classifiers (again, grey rectangle). At some point, this should be noted as a certain
symmetry breaking between reservoir and readout units. This might leave an empirical trace in biological systems. (d) (Blue trajectories in panel a.) A genetic
mutation produces two copies of a very specialized circuit. One of the copies keeps implementing its very specific task (blue ball in the upper left corner of
the morphospace, panel a). This represents a lasting, singled-out peak in fitness landscape and the implementation is costly because mistakes would cause a
fatal failure. The circuit copy is released from the cost of failure ( jump represented by the dashed blue arrow in panel a). It can then wander the morphospace
freely (different blue trajectories), probably becoming cheaper as some further functionality is lost. Its very rich dynamics make it a perfect candidate to be hijacked
by simple, linear readouts that start using the circuit as a reservoir. (Online version in colour.)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
9
This prompts the next question: given these freely avail-
able reservoirs (specifically supported by the spring and
GRN examples), when will they be exploited? To answer
this, let us focus on the two main advantages of RC men-
tioned above, starting with the cheap learning. Let us also
assume that there exists an underlying fitness landscape
that tells us whether a feature (e.g. solving a specific compu-
tation) contributes to the success of a living organism. We
conceive learning as a process much faster than evolutionary
dynamics. Looking at the fitness landscape, we would expect
features that offer fitness over long evolutionary periods to be
hard-wired, not learned. We would not need a reservoir to
capture these, but rather a robust, efficient and dedicated
structure fixed by evolution. Features that are rather learned,
on the other hand, offer fitness on a time-scale briefer than a
lifetime. We are talking about short-lived peaks of the land-
scape,
so
voluble
or
unpredictable
that
it
becomes
preferable to keep a learning engine rather than hardwiring
a fixed design. Some notion of an average lifetime (t, figure
4a), during which a feature contributes to fitness, defines a
second axis of our morphospace.
Similarly, to exploit the parallel processing abilities of RC,
the underlying fitness landscape should be peaked with mul-
tiple optima that represent different useful computations.
Thus ruggedness (R, figure 4a) defines the last axis of our mor-
phospace. If one or a few tasks contribute much more fitness
than others, parts of the reservoir dedicated to them would
be reinforced over evolutionary time and unimportant com-
ponents would fade away, eventually thinning down the
reservoir and dismissing the less fit computations (figure
4b). Over very short time-scales, the ruggedness and peak
lifetime axes shall become indistinguishable: a quickly shift-
ing single peak resembles a rugged landscape when looked
at from afar.
These wildly speculative hypotheses suggest a research
program bridging evolution and computation. We can craft
artificial fitness landscapes with computing tasks at their
peaks. We could then mutate and select reservoirs with
costs associated to their dynamics, and with a reward col-
lected as they solve tasks around the shifting landscape. We
expect RC to fade away if some of the conditions are
removed, for example, if a reservoir grows in physical size
so that communicating dynamical states over long distances
becomes metabolically costly. Might this have happened in
motor control for larger bodies? How would the interplay
between the advantages be afforded by RC and a growing
morphology? What would be the evolutionary fate of the
components of a reservoir? Under what conditions do reser-
voirs retain their full original architecture with redundant
dynamics? When do they thin down to a subset of dedicated
parts that capture specific signals and ignore others (figure
4b)? Could some conditions prompt the development of
more complex reservoirs? Can we observe a reservoir com-
plexity ratchet—a threshold beyond which RC becomes
evolutionarily robust? These are all issues that can be tackled
through simulations in a systematic and easy manner. Also,
these questions extend beyond RC. We used its properties
as guidelines to design morphospace axes that, we think,
could optimally distinguish this computing strategy from
others. But both the lifetime and ruggedness of tasks in our
landscape are independent of the strategy used to tackle
each problem. A dynamical cost can also be calculated in
other
computing
devices.
Hence,
we
expect
that
the
morphospace will be useful in locating RC among a larger
family of RNN and other paradigms. The eventual picture
could contain hybrids as well, thus potentially revealing a
continuum of computing options.
Back to RC: up to this point we have assumed that a
fully-fledged reservoir exists that, depending on external con-
straints, might shift to simpler computing paradigms and
lose part of its structure (figure 4b). This will be relevant
for the kind of reservoirs provided for free by nature, as dis-
cussed above. But there are alternative questions at the other
end of the spectrum: what would be plausible evolutionary
paths for non-RC circuits to progress towards RC? If cortical
microcolumns actually implement RC in our brains, we could
then wonder how they got there by building upon non-RC
elements. Some non-exhaustive possibilities include:
— Gradually, over evolutionary time, a small, specialized
circuit acquires more tasks at the same time that it becomes
more complex—e.g. by incorporating more parts and a
more convoluted topology (figure 4c). For this to result
in RC, partial computations needed for the acquired
tasks must be distributed over several circuit components
in a way that makes them difficult to disentangle. Other-
wise, the system would more likely evolve into smaller,
separate and task-specific structures. Importantly, at
some point, a kind of symmetry-breaking between the
reservoir
and
the
readouts
should
take
place.
All
the costly, nonlinear calculations should be relegated to
the parts that will constitute the reservoir. Readouts or cir-
cuit effectors, on the other hand, can become simpler—
with linear classifiers eventually sufficing to do the job.
Given the way that learning works in RC, which only
affects the readouts, this symmetry breaking between
parts should also be reflected by a specialization of feed-
backs controlling synaptic plasticity—a point that we
will discuss again below.
— An already complex, yet specialized circuit is freed of its
main task, thus becoming raw material that can be
coopted to solve other computations (figure 4d). This
evolutionary path to RC could be explored, e.g. if a com-
plex circuit gets duplicated and one of the copies keeps
implementing
the
original
task—so
that
outlandish
explorations are not penalized. This is a mechanism
exploited elsewhere in biology, e.g. by duplicated genes
with one of the copies exploring the phenotypic neigh-
bourhood of an existing peak in fitness landscape.
(Indeed, we could look at such pools of duplicated
genes as a reservoir of sorts; and we could wonder
whether the structure, variability and frequency of such
gene reservoirs could help us quantify aspects of our RC
morphospace.) The similarities between central pattern
generators (CPGs) in the brain stem and columns in the
neocortex have been noted at several levels, including
dynamical, histological, biomolecular, pathological and
structural [162]. A key difference is the versatility and
plasticity of microcolumns when compared with the
stereotypical
behaviour
of
CPGs
(a
versatility
that
would be very costly for CPGs, since they would fail to
implement their main task). RC is explicitly suggested
in [162] as a paradigm to frame the differences between
CPGs and cortical columns. The hypothesis that micro-
columns
followed this
path
to
RC from a
shared
evolutionary origin with CPGs becomes tempting.
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
10
These paths to RC, again, work as evolutionary-compu-
tational
hypotheses
that
can
be
easily
tested
through
simulations. It is perhaps also possible to derive some
mean-field models of average circuit structure and their com-
putational power and address some limit cases analytically.
These are all engaging topics for the future, but a look at
the foundation of our speculations already offers hints to
answer some of the questions above—notably, why is RC
not a more prominent paradigm?
While circuits and complex systems with the ability to
work as reservoirs abound in nature, situations that sustain
evolutionary pressures with a rugged, shifting landscape
(demanding multitasking and adaptability within times
much shorter than the lifetime of the reservoir) might not
be as common. One possibility is that a peak of the fitness
landscape
becomes
more
prominent
for
an
evolving
species—e.g. by a process of niche construction. Then, a
single task among the several solved by a reservoir can pro-
vide enough fitness. Redundant tasks and components can
be consequently lost (figure 4b). As suggested in the previous
section, against the convenience of bodies as a reservoir, we
propose that this might be a reason why nerves at large
evolved towards a more sequential and archetypal wiring.
As bodies grew bigger, peaks in the motor-control landscape
might have become more prominent (e.g. just a few coarse-
grained commands seem enough for CPGs to coordinate
gait behaviour at large [163,164]). Even though bodies as
reservoirs still serve the information needed to solve motor
control, the archetypal circuitry is more stable in the long
run. The need to integrate visual cues (which can hardly be
incorporated into a mechanoelastic reservoir) and long-term
movement planning hint at yet other evolutionary pressures
against the RC solution. This suggests that smaller or more
primitive organisms, if any, shall exploit RC more clearly.
Just as RC appears to be a relevant tool to develop a prin-
cipled theory of morphological computation [64,133,138], it
also seems a great addition to think about liquid and solid
brains along the lines of this issue, especially as embodied
and non-standard computations with small living organisms
are explored [165].
Cortical microcolumns, on the other hand, are likely to
operate based on RC or to incorporate most RC principles.
These are the circuits in charge of the more abstract and com-
plex tasks, such as language or conscious processing. These
tasks appear indeed shifting in nature, and often present a
wide variety of solutions (e.g. as indicated by the different
syntax and grammars that implement language equally
well). These features loosely correspond to rugged land-
scapes whose peaks either shift in time or cannot be
anticipated from the long evolutionary perspective—thus fit-
ting nicely in our speculation. Efforts to clarify whether RC is
actually exploited in the neocortex and other neural circuits
are underway, as reviewed earlier [109,117–121,124–127].
This work focuses mostly on the richness of the dynamics
and on the ability of simple linear classifiers to pick up non-
linear input features based on recorded neural activity alone.
We would like to add an alternative empirical approach:
as mentioned above, RC implies that training focuses on
the linear readouts. This must be reflected at several levels.
For instance, the target behaviour must be made available
to the readouts (e.g. to implement back propagation or Heb-
bian learning), but is not needed elsewhere. On the other
hand, we have explored a series of computation-theoretical
features that reservoirs should preferably display. These
include a tendency to criticality and a simultaneous maximi-
zation of the separability and generalization properties (as
measured by rS and rG). All these pose diverging evolution-
ary targets for reservoir and readout plasticity. This should
result in differences in the mechanisms guiding neural
wiring—perhaps even at the molecular level. Trying to spot
such differences empirically should be within the reach of
current technology.
We close with a reflection to link this paper with the gen-
eral theme of this issue. In §1, we anticipated that most
authors would be exploring liquid or solid brains as referring
to the thermodynamic state of the physical substrate in which
computation happens—i.e. whether computing units are
motile (ants, T-cells, etc.) or fixed (neurons, semiconductors,
etc.). Instead, this paper rather tackled aspects of the soft-
ware. RC largely resembles a liquid in the signals involved
(sometimes literally so [69]), specially through the abundance
of spurious dynamics elicited. Evolutionary costs could limit
these generous dynamics so that RC is lost (e.g. figure 4b).
This could somehow crystallize the available liquid signals to
a handful of stereotypical patterns. This could, in turn,
either lower the demands on the hardware (as it requires
less active dynamics) and otherwise free resources that
could be invested, e.g. in hardware motility. We expect
non-trivial interplays between afforded liquidity at the soft-
ware and hardware levels. It then becomes relevant to
determine whether and how the evolutionary pressures dis-
cussed above can constrain the liquid or solid nature of brain
substrates, not only of its signal repertoire.
Data accessibility. This article has no additional data.
Competing interests. We declare no competing interests.
Funding. This work has been supported by the Botı´n Foundation, by
Banco Santander through its Santander Universities Global Division,
a MINECO FIS2015-67616 fellowship, and the Secretaria d’Universi-
tats i Recerca del Departament d’Economia i Coneixement de la
Generalitat de Catalunya.
Acknowledgements. We thank members of the CSL for useful discussion,
especially Prof. Ricard Sole´, Jordi Pin˜ ero and Blai Vidiella, as well as
Dr Amor from the Gore Lab at MIT’s Physics of Living Systems. We
also thank all participants of the ‘Liquid brains, solid brains’ working
group at the Santa Fe Institute for their insights about computation,
algorithmic thinking and biology. Finally, we would like to thank
all members of RCC for their numerous insights.
References
1.
Szathma´ry E, Maynard-Smith J. 1997 From
replicators to reproducers: the first
major transitions leading to life. J. Theor.
Biol. 187, 555–571. (doi:10.1006/jtbi.1996.
0389)
2.
Joyce GF. 2002 Molecular evolution: booting up life.
Nature 420, 278. (doi:10.1038/420278a)
3.
Walker SI, Davies PC. 2013 The algorithmic origins
of life. J. R. Soc. Interface 10, 20120869. (doi:10.
1098/rsif.2012.0869)
4.
Schuster P. 1996 How does complexity arise
in evolution: nature’s recipe for mastering
scarcity, abundance, and unpredictability.
Complexity 2, 22–30. (doi:10.1002/(ISSN)
1099-0526)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
11
5.
Smith JM. 2000 The concept of information in
biology. Philos. Sci. 67, 177–194. (doi:10.1086/
392768)
6.
Jablonka E, Lamb MJ. 2006 The evolution of
information in the major transitions. J. Theor.
Biol. 239, 236–246. (doi:10.1016/j.jtbi.2005.
08.038)
7.
Nurse P. 2008 Life, logic and information. Nature
454, 424. (doi:10.1038/454424a)
8.
Joyce GF. 2012 Bit by bit: the Darwinian basis of
life. PLoS Biol. 10, e1001323. (doi:10.1371/journal.
pbio.1001323)
9.
Adami C. 2012 The use of information theory in
evolutionary biology. Ann. NY Acad. Sci. 1256,
49–65. (doi:10.1111/j.1749-6632.2011.06422.x)
10. Hidalgo J, Grilli J, Suweis S, Mun˜oz MA, Banavar JR,
Maritan A. 2014 Information-based fitness and the
emergence of criticality in living systems. Proc. Natl
Acad. Sci. USA 111, 10 095–10 100. (doi:10.1073/
pnas.1319166111)
11. Smith E, Morowitz HJ. 2016 The origin and nature of
life on earth: the emergence of the fourth geosphere.
Cambridge, UK: Cambridge University Press.
12. Hopfield JJ. 1994 Physics, computation, and why
biology looks so different. J. Theor. Biol. 171,
53–60. (doi:10.1006/jtbi.1994.1211)
13. Jacob F. 1998 Of flies, mice and man. Harvard, MA:
Harvard University Press.
14. Wagensberg J. 2000 Complexity versus uncertainty:
the question of staying alive. Biol. Phil. 15,
493–508. (doi:10.1023/A:1006611022472)
15. Seoane LF, Sole´ R. 2018 Information theory,
predictability and the emergence of complex life.
R. Soc. open sci. 5, 172221. (doi:10.1098/rsos.
172221)
16. Paun G, Rozenberg G, Salomaa A. 2005 DNA
computing: new computing paradigms. Berlin,
Germany: Springer Science & Business Media.
17. Doudna JA, Sternberg SH. 2017 A crack in creation:
gene editing and the unthinkable power to control
evolution. Boston, MA: Houghton Mifflin Harcourt.
18. Macia J, Sole R. 2014 How to make a synthetic
multicellular computer. PLoS ONE 9, e81248.
(doi:10.1371/journal.pone.0081248)
19. Fukushima K, Miyake S. 1982 Neocognitron: a self-
organizing neural network model for a mechanism
of visual pattern recognition. In Competition and
cooperation in neural nets, pp. 267–285. Berlin,
Germany: Springer.
20. Krizhevsky A, Sutskever I, Hinton GE. 2012 Imagenet
classification with deep convolutional neural
networks. In Advances in neural information
processing systems, pp. 1097–1105. Lake Tahoe,
NV: Neural Information Processing Systems
Foundation, Inc.
21. Yamins DL, Hong H, Cadieu CF, Solomon EA, Seibert
D, DiCarlo JJ. 2014 Performance-optimized
hierarchical models predict neural responses in
higher visual cortex. Proc. Natl Acad. Sci. USA 111,
8619–8624. (doi:10.1073/pnas.1403112111)
22. Thomas R. 1973 Boolean formalization of genetic
control circuits. J. Theor. Biol. 42, 563–585. (doi:10.
1016/0022-5193(73)90247-6)
23. Kauffman S. 1996 At home in the universe: the
search for the laws of self-organization and
complexity. Oxford, UK: Oxford University Press.
24. Rodrı´guez-Caso C, Corominas-Murtra B, Sole´ R. 2009
On the basic computational structure of gene
regulatory networks. Mol. Biosyst. 5, 1617–1629.
(doi:10.1039/b904960f)
25. Dayan P, Abbott LF. 2001 Theoretical neuroscience.
Cambridge, MA: MIT Press.
26. Seung S. 2012 Connectome: how the brain’s wiring
makes us who we are. Boston, MA: HMH.
27. Levick WR. 1967 Receptive fields and trigger
features of ganglion cells in the visual streak of the
rabbit’s retina. J. Physiol. 188, 285–307. (doi:10.
1113/jphysiol.1967.sp008140)
28. Russell TL, Werblin FS. 2010 Retinal synaptic
pathways underlying the response of the rabbit
local edge detector. J. Neurophysiol. 103,
2757–2769. (doi:10.1152/jn.00987.2009)
29. Marr D, Hildreth E. 1980 Theory of edge detection.
Proc. R. Soc. Lond. B 207, 187–217. (doi:10.1098/
rspb.1980.0020)
30. Marr D. 1982 Vision: a computational investigation
into the human representation and processing of
visual information. Cambridge, MA: MIT Press.
31. Stephens GJ, Mora T, Tkacˇik G, Bialek W. 2013
Statistical thermodynamics of natural images. Phys.
Rev. Let. 110, 018701. (doi:10.1103/PhysRevLett.
110.018701)
32. Khaligh-Razavi SM, Kriegeskorte N. 2014 Deep
supervised, but not unsupervised, models may
explain IT cortical representation. PLoS Comp. Biol.
10, e1003915. (doi:10.1371/journal.pcbi.1003915)
33. Jaeger H. 2001 The ‘echo state’ approach to
analysing and training recurrent neural networks–
with an erratum note. Bonn, Germany: German
National Research Center for Information Technology
GMD Technical Report, vol. 148, 13.
34. Maass W, Natschla¨ger T, Markram H. 2002 Real-time
computing without stable states: a new framework
for neural computation based on perturbations.
Neural Comput. 14, 2531–2560. (doi:10.1162/
089976602760407955)
35. JaegerH,MaassW,PrincipeJ.2007Specialissueonecho
state networks and liquid state machines. Neural Netw.
20, 287–289. (doi:10.1016/j.neunet.2007.04.001)
36. Verstraeten D, Schrauwen B, d’Haene M, Stroobandt
D. 2007 An experimental unification of reservoir
computing methods. Neural Netw. 20, 391–403.
(doi:10.1016/j.neunet.2007.04.003)
37. Lukosˇevicˇius M, Jaeger H, Schrauwen B. 2012
Reservoir computing trends. KI-K”unstliche Intelligenz
26, 365–371. (doi:10.1007/s13218-012-0204-5)
38. Rumelhart DE, Hinton GE, Williams RJ. 1986
Learning representations by back-propagating
errors. Nature 323, 533. (doi:10.1038/323533a0)
39. Bengio Y, Simard P, Frasconi P. 1994 Learning long-
term dependencies with gradient descent is
difficult. IEEE T Neural Netw. 5, 157–166. (doi:10.
1109/72.279181)
40. Pascanu R, Mikolov T, Bengio Y. 2013 On the
difficulty of training recurrent neural networks. In
Int. Conf. Machine Learning, pp. 1310–1318.
41. Legenstein R, Maass W. 2007 Edge of chaos and
prediction of computational performance for neural
circuit models. Neural Netw. 20, 323–334. (doi:10.
1016/j.neunet.2007.04.017)
42. Verstraeten D, Schrauwen B, Stroobandt D. 2006
Reservoir-based techniques for speech recognition.
In The 2006 IEEE Inter. joint Conf. on Neural
Network Proc., 16–21 July 2006, Vancouver, BC,
Canada, pp. 1050–1053. IEEE.
43. Jaeger H, Lukosˇevicˇius M, Popovici D, Siewert U.
2007 Optimization and applications of echo state
networks with leaky-integrator neurons. Neural
Netw. 20, 335–352. (doi:10.1016/j.neunet.2007.
04.016)
44. Soria DI, Soria-Frisch A, Garcı´a-Ojalvo J, Picardo J,
Garcı´a-Banda G, Servera M, Ruffini G. 2018
Hypoarousal non-stationary ADHD biomarker based
on echo-state networks. bioRxiv. (doi:10.1101/
271858)
45. Joshi P, Maass W. 2004 Movement generation and
control with generic neural microcircuits. In
Biologically inspired approaches to advanced
information technology (eds A Ijspeert, A Murata,
N Wakamiya), pp. 258–273. Berlin, Germany:
Springer.
46. Salmen M, Ploger PG. 2005 Echo state networks
used for motor control. In: ICRA 2005. Proc. 2005
IEEE Inter. Conf. on Robotics and Automation,
Barcelona, Spain, 18–22 April 2005 (1953–1958).
IEEE.
47. Burgsteiner H. 2005 Training networks of biological
realistic spiking neurons for real-time robot control.
In: Proc. of the 9th Inter. Conf. on Engineering
Applications of Neural Networks, pp. 129–136.
France: Lille.
48. Jaeger H, Haas H. 2004 Harnessing nonlinearity:
predicting chaotic systems and saving energy in
wireless communication. Science 304, 78–80.
(doi:10.1126/science.1091277)
49. Iba´nez-Soria D, Garcı´a-Ojalvo J, Soria-Frisch A,
Ruffini G. 2018 Detection of generalized
synchronization using echo state networks. Chaos
28, 033118. (doi:10.1063/1.5010285)
50. Verstraeten D, Schrauwen B, Stroobandt D, Van
Campenhout J. 2005 Isolated word recognition with
the liquid state machine: a case study. Inf. Proc.
Lett. 95, 521–528. (doi:10.1016/j.ipl.2005.05.019)
51. Tong MH, Bickett AD, Christiansen EM, Cottrell GW.
2007 Learning grammatical structure with echo
state networks. Neural Netw. 20, 424–432. (doi:10.
1016/j.neunet.2007.04.013)
52. Triefenbach F, Jalalvand A, Schrauwen B, Martens
JP. 2010 Phoneme recognition with large
hierarchical reservoirs. In Advances in neural
information processing systems (eds JD Lafferty, CKI
Williams, J Shawe-Taylor, RS Zemel, A Culotta),
pp. 2307–2315. Vancouver, Canada: Neural
Information Processing Systems Foundation, Inc.
53. Hinaut X, Dominey PF. 2013 Real-time parallel
processing of grammatical structure in the fronto-
striatal system: a recurrent network simulation
study using reservoir computing. PLoS ONE 8,
e52946. (doi:10.1371/journal.pone.0052946)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
12
54. Wyffels F, Schrauwen B, Stroobandt D. 2008 Stable
output feedback in reservoir computing using ridge
regression. In Inter. Conf. Artificial Neural Networks,
pp. 808–817. Berlin, Germany: Springer.
55. Lukosˇevicˇius M, Jaeger H. 2009 Reservoir computing
approaches to recurrent neural network training.
Comput. Sci. Rev. 3, 127–149. (doi:10.1016/j.cosrev.
2009.03.005)
56. Jaeger H. 2002 Tutorial on training recurrent neural
networks, covering BPPT, RTRL, EKF and the ‘echo
state network’ approach, vol. 5. Bonn: GMD-
Forschungszentrum Informationstechnik.
57. Lukosˇevicˇius M. 2012 A practical guide to applying
echo state networks. In Neural networks: tricks of
the trade, pp. 659–686. Berlin, Germany: Springer.
58. Maass W, Joshi P, Sontag ED. 2007 Computational
aspects of feedback in neural circuits. PLoS
Comput. Biol. 3, e165. (doi:10.1371/journal.pcbi.
0020165)
59. Sussillo D, Abbott LF. 2009 Generating coherent
patterns of activity from chaotic neural networks.
Neuron 63, 544–557. (doi:10.1016/j.neuron.2009.
07.018)
60. Dai J, Venayagamoorthy GK, Harley RG. 2009 An
introduction to the echo state network and its
applications in power system. In ISAP’09. 15th Inter.
Conf. on Intelligent System Applications to Power
Systems, Curitiba, Brazil, 8–12 November 2009,
pp. 1–7. IEEE.
61. Rivkind A, Barak O. 2017 Local dynamics in trained
recurrent neural networks. Phys. Rev. Let. 118,
258101. (doi:10.1103/PhysRevLett.118.258101)
62. Ceni A, Ashwin P, Livi L. 2018 Interpreting RNN
behaviour via excitable network attractors. (http://
arxiv.org/abs/1807.10478).
63. Maass W, Legenstein RA, Bertschinger N. 2005
Methods for estimating the computational power
and generalization capability of neural microcircuits.
In Advances in neural information processing systems
(eds Y Weiss, B Scho¨lkopf, JC Platt), pp. 865–872.
Vancouver, Canada: Neural Information Processing
Systems Foundation, Inc.
64. Hauser H, Ijspeert AJ, Fu¨chslin RM, Pfeifer R, Maass
W. 2011 Towards a theoretical foundation for
morphological computation with compliant bodies.
Biol. Cybern. 105, 355–370. (doi:10.1007/s00422-
012-0471-0)
65. Nakajima K, Hauser H, Kang R, Guglielmino E,
Caldwell DG, Pfeifer R. 2013 A soft body as a
reservoir: case studies in a dynamic model of
octopus-inspired soft robotic arm. Front. Comput.
Neurosc. 7, 91. (doi:10.3389/fncom.2013.00091)
66. Nichele S, Gundersen MS. 2017 Reservoir computing
using non-uniform binary cellular automata. (http://
arxiv.org/abs/1702.03812).
67. Soriano MC, Ortı´n S, Keuninckx L, Appeltant L,
Danckaert J, Pesquera L, Van der Sande G. 2015
Delay-based reservoir computing: noise effects in a
combined analog and digital implementation. IEEE
Trans. Neural Netw. Learning Syst. 26, 388–393.
(doi:10.1109/TNNLS.2014.2311855)
68. Du C, Cai F, Zidan MA, Ma W, Lee SH, Lu WD. 2017
Reservoir computing using dynamic memristors for
temporal information processing. Nat. Com. 8, 2204.
(doi:10.1038/s41467-017-02337-y)
69. Fernando C, Sojakka S. 2003 Pattern recognition in
a bucket. In Proc. 7th European Conf. on Artificial
Life, Dortmund, Germany, 14–17 September 2003,
pp. 588–597. Berlin, Germany: Springer.
70. Appeltant L, Soriano MC, Van der Sande G,
Danckaert J, Massar S, Dambre J, Schrauwen B,
Mirasso CR, Fischer I. 2011 Information processing
using a single dynamical node as complex system.
Nat. Commun. 2, 468. (doi:10.1038/ncomms1476)
71. Paquot Y, Duport F, Smerieri A, Dambre J,
Schrauwen B, Haelterman M, Massar S. 2012
Optoelectronic reservoir computing. Sci. Rep. 2, 287.
(doi:10.1038/srep00287)
72. Vandoorne K, Mechet P, Van Vaerenbergh T, Fiers
M, Morthier G, Verstraeten D, Schrauwen B,
Dambre J, Bienstman P. 2014 Experimental
demonstration of reservoir computing on a silicon
photonics chip. Nat. Commun. 5, 3541. (doi:10.
1038/ncomms4541)
73. Legenstein R, Maass W. 2007 What makes a
dynamical system computationally powerful. New
directions in statistical signal processing: from
systems to brain (ed. SS Haykin), pp. 127–154.
Cambridge, MA: MIT Press.
74. Vapnik V 1998 Statistical learning theory. New York,
NY: Wiley.
75. Cherkassky V, Mulier F. 1998 Learning from data:
concepts, theory, and methods. New York, NY: Wiley.
76. Markram H, Wang Y, Tsodyks M. 1998 Differential
signaling via the same axon of neocortical
pyramidal neurons. Proc. Natl Acad. Sci. USA 95,
5323–5328. (doi:10.1073/pnas.95.9.5323)
77. Coello CC. 2006 Evolutionary multi-objective
optimization: a historical view of the field. IEEE
Comput. Intell. Mag. 1, 28–36. (doi:10.1109/MCI.
2006.1597059)
78. Schuster P. 2012 Optimization of multiple criteria:
pareto efficiency and fast heuristics should be more
popular than they are. Complexity 18, 5–7. (doi:10.
1002/cplx.v18.2)
79. Seoane LF. 2016 Multiobjective optimization in
models of synthetic and natural living systems.
PhD thesis, Universitat Pompeu Fabra, Barcelona,
Spain.
80. Shoval O, Sheftel H, Shinar G, Hart Y, Ramote O,
Mayo A, Dekel E, Kavanagh K, Alon U. 2012
Evolutionary trade-offs, Pareto optimality, and the
geometry of phenotype space. Science 336,
1157–1160. (doi:10.1126/science.1217405)
81. Hart Y, Sheftel H, Hausser J, Szekely P, Ben-Moshe
NB, Korem Y, Tendler A, Mayo AE, Alon U. 2015
Inferring biological tasks using Pareto analysis of
high-dimensional data. Nat. Methods 12, 233–235.
(doi:10.1038/nmeth.3254)
82. Szekely P, Korem Y, Moran U, Mayo A, Alon U. 2015
The mass-longevity triangle: Pareto optimality and
the geometry of life-history trait space. PLoS Comp.
Biol. 11, e1004524. (doi:10.1371/journal.pcbi.
1004524)
83. Tendler A, Mayo A, Alon U. 2015 Evolutionary
tradeoffs, Pareto optimality and the morphology of
ammonite shells. BMC Syst. Biol. 9, 12. (doi:10.
1186/s12918-015-0149-z)
84. Seoane LF, Sole´ R. 2013 A multiobjective
optimization approach to statistical mechanics.
(http://arxiv.org/abs/1310.6372).
85. Seoane LF, Sole´’ R. 2015 Phase transitions in Pareto
optimal complex networks. Phys. Rev. E 92, 032807.
(doi:10.1103/PhysRevE.92.032807)
86. Seoane LF, Sole´ R. 2016 Multiobjective optimization
and phase transitions. In Proc. of ECCS 2014,
pp. 259–270. Cham, Switzerland: Springer.
87. Seoane LF, Sole´’ R. 2015 Systems poised to
criticality through Pareto selective forces. (http://
arxiv.org/abs/1510.08697).
88. Bak P. 1996 How nature works: the science of
self-organized criticality. New York, NY: Copernicus
Press.
89. Beggs JM, Plenz D. 2003 Neuronal avalanches in
neocortical circuits. J. Neurosci. 23, 11 167–11 177.
(doi:10.1523/JNEUROSCI.23-35-11167.2003)
90. Chialvo DR. 2010 Emergent complex neural
dynamics. Nat. Phys. 6, 744. (doi:10.1038/
nphys1803)
91. Mora T, Bialek W. 2011 Are biological systems
poised at criticality? J. Stat. Phys. 144, 268–302.
(doi:10.1007/s10955-011-0229-4)
92. Tagliazucchi E, Balenzuela P, Fraiman D, Chialvo DR.
2012 Criticality in large-scale brain fMRI dynamics
unveiled by a novel point process analysis. Front.
Physiol. 3, 15. (doi:10.3389/fphys.2012.00015)
93. Moretti P, Mun˜oz MA. 2013 Griffiths phases and the
stretching of criticality in brain networks. Nat. Com.
4, 2521. (doi:10.1038/ncomms3521)
94. Munoz MA. 2018 Colloquium: criticality and
dynamical scaling in living systems. Rev. Mod. Phys.
90, 031001. (doi:10.1103/RevModPhys.90.031001)
95. Wolfram S. 1984 Universality and complexity in
cellular automata. Physica D 10, 1–35. (doi:10.
1016/0167-2789(84)90245-8)
96. Langton CG. 1990 Computation at the edge of
chaos: phase transitions and emergent
computation. Physica D 42, 12–37. (doi:10.1016/
0167-2789(90)90064-V)
97. Mitchell M, Hraber P, Crutchfield JP. 1993 Revisiting
the edge of chaos: evolving cellular automata to
perform computations. (http://arxiv.org/abs/
9303003).
98. Bertschinger N, Natschla¨ger T. 2004 Real-time
computation at the edge of chaos in recurrent
neural networks. Neural Comput. 16, 1413–1436.
(doi:10.1162/089976604323057443)
99. Schrauwen B, Bu¨sing L, Legenstein RA. 2009
On computational power and the order–chaos
phase transition in reservoir computing. In Advances
in neural information processing systems (eds Y
Bengio, D Schuurmans, JD Lafferty, CKI Williams,
A Culotta), pp. 1425–1432. Vancouver, Canada:
Neural Information Processing Systems
Foundation, Inc.
100. Toyoizumi T, Abbott LF. 2011 Beyond the edge of
chaos: amplification and temporal integration by
recurrent networks in the chaotic regime. Phys. Rev.
E 84, 051908. (doi:10.1103/PhysRevE.84.051908)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
13
101. Boedecker J, Obst O, Lizier JT, Mayer NM, Asada M.
2012 Information processing in echo state networks
at the edge of chaos. Theory Biosci. 131, 205–213.
(doi:10.1007/s12064-011-0146-8)
102. Bianchi FM, Livi L, Alippi C. 2018 Investigating
echo-state networks dynamics by means of
recurrence analysis. IEEE Trans. Neural Netw.
Learning Sys. 29, 427–439. (doi:10.1109/TNNLS.
2016.2630802)
103. Bianchi FM, Livi L, Alippi C. 2018 On the
interpretation and characterization of echo state
networks dynamics: a complex systems perspective.
In Advances in data analysis with computational
intelligence methods (eds S Bengio, H Wallach, H
Larochelle, K Grauman, N Cesa-Bianchi, R Garnett),
pp. 143–167. Montreal, Canada: Neural
Information Processing Systems Foundation, Inc.
104. Livi L, Bianchi FM, Alippi C. 2018 Determination of
the edge of criticality in echo state networks
through Fisher information maximization. IEEE
Trans. Neural Netw. Learning Syst. 29, 706–717.
(doi:10.1109/TNNLS.2016.2644268)
105. Prokopenko M, Lizier JT, Obst O, Wang XR. 2011
Relating Fisher information to order parameters.
Phys. Rev. E 84, 041116. (doi:10.1103/PhysRevE.84.
041116)
106. Hawkins J, Blakeslee S. 2007 On intelligence: how a
new understanding of the brain will lead to the
creation of truly intelligent machines. New York, NY:
Times Books.
107. Oberlaender M, de Kock CP, Bruno RM, Ramirez A,
Meyer HS, Dercksen VJ, Helmstaedter M, Sakmann
B. 2011 Cell type-specific three-dimensional
structure of thalamocortical circuits in a column of
rat vibrissal cortex. Cereb. Cortex 22, 2375–2391.
(doi:10.1093/cercor/bhr317)
108. Habenschuss S, Jonke Z, Maass W. 2013 Stochastic
computations in cortical microcircuit models. PLoS
Comp. Biol. 9, e1003311. (doi:10.1371/journal.pcbi.
1003311)
109. Haeusler S, Maass W. 2006 A statistical analysis
of information-processing properties of
lamina-specific cortical microcircuit models.
Cereb. Cortex 17, 149–162. (doi:10.1093/
cercor/bhj132)
110. Maass W. 2016 Searching for principles of brain
computation. Curr. Opin. Behav. Sci. 11, 81–92.
(doi:10.1016/j.cobeha.2016.06.003)
111. Maass W, Natschla¨ger T, Markram H. 2004
Computational models for generic cortical
microcircuits. In Computational neuroscience: a
comprehensive approach (ed. J Feng), pp. 575–605.
(doi:10.1201/9780203494462.ch18)
112. Maass W, Natschla¨ger T, Markram H. 2004 Fading
memory and kernel properties of generic cortical
microcircuit models. J. Physiol.-Paris 98, 315–330.
(doi:10.1016/j.jphysparis.2005.09.020)
113. Maass W, Markram H. 2006 Theory of the
computational function of microcircuit dynamics. In
The interface between neurons and global brain
function, Dahlem Workshop Report, 25–30 April
2004, vol. 93 (eds S Grillner, AM Graybiel),
pp. 371–390. Cambridge, MA: MIT.
114. Thomson AM, West DC, Wang Y, Bannister AP. 2002
Synaptic connections and small circuits involving
excitatory and inhibitory neurons in layers 2–5 of
adult rat and cat neocortex: triple intracellular
recordings and biocytin labelling in vitro. Cereb.
Cortex 12, 936–953. (doi:10.1093/cercor/12.9.936)
115. Hubel DH, Wiesel TN. 1962 Receptive fields,
binocular interaction and functional architecture in
the cat’s visual cortex. J. Physiol. 160, 106–154.
(doi:10.1113/jphysiol.1962.sp006837)
116. Diamond ME, Von Heimendahl M, Knutsen PM,
Kleinfeld D, Ahissar E. 2008 ‘Where’ and ‘what’ in
the whisker sensorimotor system. Nat. Rev. Neurosci.
9, 601. (doi:10.1038/nrn2411)
117. Bernacchia A, Seo H, Lee D, Wang XJ. 2011 A
reservoir of time constants for memory traces in
cortical neurons. Nat. Neurosci. 14, 366. (doi:10.
1038/nn.2752)
118. Nikolic´ D, Ha¨usler S, Singer W, Maass W. 2009
Distributed fading memory for stimulus properties
in the primary visual cortex. PLoS Biol. 7, e1000260.
(doi:10.1371/journal.pbio.1000260)
119. Dranias MR, Ju H, Rajaram E, VanDongen AM. 2013
Short-term memory in networks of dissociated
cortical neurons. J. Neurosci. 33, 1940–1953.
(doi:10.1523/JNEUROSCI.2718-12.2013)
120. Ju H, Dranias MR, Banumurthy G, VanDongen AM.
2015 Spatiotemporal memory is an intrinsic
property of networks of dissociated cortical neurons.
J. Neurosci. 35, 4040–4051. (doi:10.1523/
JNEUROSCI.3793-14.2015)
121. Marre O, Botella-Soler V, Simmons KD, Mora T,
Tkacˇik G, Berry II MJ. 2015 High accuracy decoding
of dynamical motion from a large retinal
population. PLoS Comp. Biol. 11, e1004304. (doi:10.
1371/journal.pcbi.1004304)
122. Singer W. 2013 Cortical dynamics revisited. Trends
Cognit. Sci. 17, 616–626. (doi:10.1016/j.tics.2013.
09.006)
123. Klampfl S, Maass W. 2013 Emergence of dynamic
memory traces in cortical microcircuit models
through STDP. J. Neurosci. 33, 11 515–11 529.
(doi:10.1523/JNEUROSCI.5044-12.2013)
124. Chen JL, Carta S, Soldado-Magraner J, Schneider BL,
Helmchen F. 2013 Behaviour-dependent recruitment
of long-range projection neurons in somatosensory
cortex. Nature 499, 336. (doi:10.1038/nature12236)
125. Klampfl S, David SV, Yin P, Shamma SA, Maass W.
2012 A quantitative analysis of information about
past and present stimuli encoded by spikes of A1
neurons. J. Neurophysiol. 108, 1366–1380. (doi:10.
1152/jn.00935.2011)
126. Rigotti M, Barak O, Warden MR, Wang XJ, Daw ND,
Miller EK, Fusi S. 2013 The importance of mixed
selectivity in complex cognitive tasks. Nature 497,
585. (doi:10.1038/nature12160)
127. Fusi S, Miller EK, Rigotti M. 2016 Why neurons
mix: high dimensionality for higher cognition. Curr.
Opin. Neurobiol. 37, 66–74. (doi:10.1016/j.conb.
2016.01.010)
128. Nakajima K, Hauser H, Kang R, Guglielmino E,
Caldwell DG, Pfeifer R. 2013 Computing with a
muscular-hydrostat system. In 2013 IEEE Inter. Conf.
on Robotics and Automation (ICRA), Karlsruhe,
Germany, 6–10 May 2013, pp. 1504–1511. IEEE.
129. Godfrey-Smith P. 2016 Other minds: the octopus,
the sea, and the deep origins of consciousness.
New York, NY: Farrar, Straus and Giroux.
130. Sacks O. 2017 The river of consciousness, ch. 3.
New York, NY: Penguin Random House.
131. Pfeifer R, Bongard J. 2006 How the body shapes the
way we think: a new view of intelligence. New York,
NY: MIT press.
132. Pfeifer R, Lungarella M, Iida F. 2007 Self-
organization, embodiment, and biologically inspired
robotics. Science 318, 1088–1093. (doi:10.1126/
science.1145803)
133. Mu¨ller VC, Hoffmann M. 2017 What is
morphological computation? On how the body
contributes to cognition and control. Artif. Life 23,
1–24. (doi:10.1162/ARTL_a_00219)
134. McGeer T. 1990 Passive dynamic walking.
Int. J. Rob. Res. 9, 62–82. (doi:10.1177/
027836499000900206)
135. Wisse M, Van Frankenhuyzen J. 2006 Design and
construction of MIKE; a 2-D autonomous biped
based on passive dynamic walking. In Adaptive
motion of animals and machines, pp. 143–154.
Tokyo, Japan: Springer.
136. Fend M, Bovet S, Pfeifer R. 2006 On the influence of
morphology of tactile sensors for behavior and
control. Robot. Auton. Syst. 54, 686–695. (doi:10.
1016/j.robot.2006.02.014)
137. Murata S, Kurokawa H. 2007 Self-reconfigurable
robots. IEEE Robot. Autom. Mag. 14, 71–78.
(doi:10.1109/MRA.2007.339607)
138. Hauser H, Ijspeert AJ, Fu¨chslin RM, Pfeifer R,
Maass W. 2012 The role of feedback in
morphological computation with compliant
bodies. Biol. Cybern. 106, 595–613. (doi:10.
1007/s00422-012-0516-4)
139. Sumioka H, Hauser H, Pfeifer R. 2011 Computation
with mechanically coupled springs for compliant
robots. In 2011 IEEE/RSJ Inter. Conf. on Intelligent
Robots and Systems (IROS), San Francisco, CA,
25–30 September 2011, pp. 4168–4173. IEEE.
140. Nakajima K, Hauser H, Kang R, Guglielmino E,
Caldwell DG, Pfeifer R. 2013 Computing with a
muscular-hydrostat system. In 2013 IEEE Inter. Conf.
on Robotics and Automation (ICRA), Karlsruhe,
Germany, 6–10 May 2013, pp. 1504–1511. IEEE.
141. Caluwaerts K, Schrauwen B. 2011 The body as a
reservoir: locomotion and sensing with linear
feedback. In 2nd Int. Conf. on Morphological
Computation (ICMC 2011), Venice, Italy, 12–14
September 2011.
142. Caluwaerts K, D’Haene M, Verstraeten D, Schrauwen
B. 2013 Locomotion without a brain: physical
reservoir computing in tensegrity structures. Artif.
Life 19, 35–66. (doi:10.1162/ARTL_a_00080)
143. Gabalda-Sagarra M, Carey LB, Garcı´a-Ojalvo J. 2018
Recurrence-based information processing in gene
regulatory networks. Chaos 28, 106313. (doi:10.
1063/1.5039861)
144. Clark A. 1998 Being there: putting brain, body, and
world together again. Cambridge, MA: MIT Press.
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
14
145. Nakajima K, Li T, Hauser H, Pfeifer R. 2014
Exploiting short-term memory in soft body
dynamics as a computational resource. J. R. Soc.
Interface 11, 20140437. (doi:10.1098/rsif.2014.0437)
146. Nakajima K, Hauser H, Li T, Pfeifer R. 2015
Information processing via physical soft body. Sci.
Rep. 5, 10487. (doi:10.1038/srep10487)
147. Shim Y, Husbands P. 2007 Feathered flyer:
integrating morphological computation and sensory
reflexes into a physically simulated flapping-wing
robot for robust flight manoeuvre. In Advances in
Artificial Life. ECAL 2007. Lecture Notes in Computer
Science, vol. 4648 (eds F Almeida e Costa, LM
Rocha, E Costa, I Harvey, A Coutinho). (doi:10.1007/
978-3-540-74913-4_76)
148. Valero-Cuevas FJ, Yi JW, Brown D, McNamara RV,
Paul C, Lipson H. 2007 The tendon network of the
fingers performs anatomical computation at a
macroscopic scale. IEEE Trans. Bio-Med. Eng. 54,
1161–1166. (doi:10.1109/TBME.2006.889200)
149. Minsky M, Papert SA. 2017 Perceptrons: an
introduction to computational geometry. Cambridge,
MA: MIT Press.
150. Hopfield JJ. 1982 Neural networks and physical
systems with emergent collective computational
abilities. Proc. Natl Acad. Sci. USA 79, 2554–2558.
(doi:10.1073/pnas.79.8.2554)
151. Kohonen T. 1982 Self-organized formation of
topologically correct feature maps. Biol. Cybern. 43,
59–69. (doi:10.1007/BF00337288)
152. Hebb DO. 1963 The organizations of behavior: a
neuropsychological theory. London, NJ: Lawrence
Erlbaum.
153. Ritter H. 1990 Self-organizing maps for internal
representations. Psychol. Res. 52, 128–136. (doi:10.
1007/BF00877520)
154. Spitzer M, Bho¨ler P, Weisbrod M, Kischka U. 1995
A neural network model of phantom limbs.
Biol. Cybern. 72, 197–206. (doi:10.1007/
BF00201484)
155. Raup DM. 1966 Geometric analysis of shell
coiling: general problems. J. Paleontol. 40,
1178–1190.
156. Niklas KJ. 1997 The evolutionary biology of plants.
Chicago, IL: Chicago University Press.
157. McGhee GR. 1999 Theoretical morphology: the
concept and its applications. New York, NY:
Columbia University Press.
158. Niklas KJ. 2004 Computer models of early land
plant evolution. Annu. Rev. Earth Planet. Sci. 32,
47–66. (doi:10.1146/annurev.earth.32.092203.
122440)
159. Corominas-Murtra B, Gon˜i J, Sole´ RV, Rodrı´guez-
Caso C. 2013 On the origins of hierarchy in complex
networks. Proc. Natl Acad. Sci. USA 110, 13 316–
13 321. (doi:10.1073/pnas.1300832110)
160. Avena-Koenigsberger A, Gon˜i J, Sole´ R, Sporns O.
2015 Network morphospace. J. R. Soc. Interface
12, 20140881. (doi:10.1098/rsif.2014.0881)
161. Seoane LF, Sole´ R. 2018 The morphospace of
language networks. Sci. Rep. 8, 10465. (doi:10.
1038/s41598-018-28820-0)
162. Yuste R, MacLean JN, Smith J, Lansner A. 2005 The
cortex as a central pattern generator. Nat. Rev.
Neurosci. 6, 477–483. (doi:10.1038/nrn1686)
163. Marder E, Calabrese RL. 1996 Principles of rhythmic
motor pattern generation. Physiol. Rev. 76,
687–717. (doi:10.1152/physrev.1996.76.3.687)
164. Ijspeert AJ. 2008 Central pattern generators for
locomotion control in animals and robots: a review.
Neural Netw. 21, 642–653. (doi:10.1016/j.neunet.
2008.03.014)
165. Balusˇka F, Levin M. 2016 On having no head: cognition
throughout biological systems. Front. Psychol. 7, 902.
(doi:10.3389/fpsyg.2016.00902)
royalsocietypublishing.org/journal/rstb
Phil. Trans. R. Soc. B 374: 20180377
15
