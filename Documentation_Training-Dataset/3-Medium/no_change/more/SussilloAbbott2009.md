Neuron. Author manuscript; available in PMC 2010 Aug 27.
Published in ﬁnal edited form as:
Neuron. 2009 Aug 27; 63(4): 544–557.
doi: 10.1016/j.neuron.2009.07.018
PMCID: PMC2756108
NIHMSID: NIHMS137176
PMID: 19709635
Generating Coherent Patterns of Activity from Chaotic Neural Networks
David Sussillo and L. F. Abbott
Abstract
Neural circuits display complex activity patterns both spontaneously and when responding to a stimulus or generat‐
ing a motor output. How are these two forms of activity related? We develop a procedure called FORCE learning
for modifying synaptic strengths either external to or within a model neural network to change chaotic spontaneous
activity into a wide variety of desired activity patterns. FORCE learning works even though the networks we train
are spontaneously chaotic and we leave feedback loops intact and unclamped during learning. Using this approach,
we construct networks that produce a wide variety of complex output patterns, input-output transformations that re‐
quire memory, multiple outputs that can be switched by control inputs, and motor patterns matching human motion
capture data. Our results reproduce data on pre-movement activity in motor and premotor cortex, and suggest that
synaptic plasticity may be a more rapid and powerful modulator of network activity than generally appreciated.
Introduction
When we voluntarily move a limb or perform some other motor action, what is the source of the neural activity that
initiates and carries out this behavior? We explore the idea that such actions arise from the reorganization of spon‐
taneous neural activity. This hypothesis raises another question: How can apparently chaotic spontaneous activity
be reorganized into the coherent patterns required to generate controlled actions? Following, but modifying and ex‐
tending, earlier work (Jaeger and Haas, 2004; Maass et al., 2007), we show how external feedback loops or internal
synaptic modiﬁcations can be used to alter the chaotic activity of a recurrently connected neural network and gener‐
ate complex but controlled outputs.
Training a neural network is a process through which network parameters (typically synaptic strengths) are modi‐
ﬁed on the basis of output errors until a desired response is produced. Researchers in the machine learning and
computer vision communities have developed powerful methods for training artiﬁcial neural networks to perform
complex tasks (Rumelhart and McClelland, 1986; Hinton et al., 2007), but these apply predominantly to networks
with feedforward architectures. Biological networks tend to be connected in a highly recurrent manner. Training
procedures have also been developed for recurrently connected neural networks (Rumelhart et al., 1986; Williams
and Zipser, 1989; Pearlmutter, 1989; Atiya and Parlos, 2000), but these are more computationally demanding and
diﬃcult to use than feedforward learning algorithms, and there are fundamental limitations to their applicability
(Doya, 1992). In particular, these algorithms generally will not converge if applied to recurrent neural networks
with chaotic activity, that is, activity that is irregular and exponentially sensitive to initial conditions (Abarbanel et
al., 2008). This limitation is severe because models of spontaneously active neural circuits typically exhibit chaotic
dynamics. For example, spiking models of spontaneous activity in cortical circuits (van Vreeswijk and
Sompolinsky, 1996; Amit and Brunel, 1997; Brunel, 2000), which can generate realistic patterns of activity, and
the analogous spontaneously active ﬁring-rate model networks that we use here have chaotic dynamics
(Sompolinsky et al., 1988).
To develop a successful training procedure for recurrent neural networks, we must solve three problems. First, feed‐
ing erroneous output back into a network during training can cause its activity to deviate so far from what is needed
that learning fails to converge. In previous work (Jaeger and Haas, 2004), this problem was avoided by removing all
errors from the signal fed back into the network. In addition to the usual synaptic modiﬁcation, this scheme re‐
quired a mechanism for removing feedback errors, and it is diﬃcult to see how this latter requirement could be met
in a biological system. Furthermore, feeding back a signal that is identical to the desired network output prevents
the network from sampling ﬂuctuations during training, which can lead to stability problems in the ﬁnal network.
Here, we show how the synaptic modiﬁcation procedure itself can be used to control the feedback signal, without
any other mechanism being required, in a manner that allows ﬂuctuations to be sampled and stabilized. For reasons
given below, we call this procedure FORCE learning.
A second problem with training that is particularly severe in recurrent networks is credit assignment for output er‐
rors. Credit assignment amounts to ﬁguring out which neurons and synapses are most responsible for output errors
and therefore most in need of modiﬁcation. This problem is particularly challenging for network units that do not
produce the output directly. Jaeger and Haas (2004) dealt with this issue by restricting modiﬁcation solely to
synapses directly driving network output. Initially, we follow their lead in this, using the architecture of ﬁgure 1A.
However, the power of FORCE learning allows us to train networks with the architectures shown in
ﬁgures 1B and 1C, in which modiﬁcations are not restricted to network outputs. For reasons discussed below, these
architectures are more biologically plausible than the network in ﬁgure 1A.
Figure 1
Network architectures. In all three cases, a recurrent generator network with ﬁring rates r drives a linear readout unit with output z
through weights w (red) that are modiﬁed during training. Only connections shown in red are subject to modiﬁcation. A) Feedback to
the generator network (large network circle) is provided by the readout unit. B) Feedback to the generator network is provided by a
separate feedback network (smaller network circle). Neurons of the feedback network are recurrently connected and receive input
from the generator network through synapses of strength J
 (red), which are modiﬁed during training. C) A network with no exter‐
nal feedback. Instead, feedback is generated within the network and modiﬁed by applying FORCE learning to the synapses with
strengths J
 internal to the network (red).
The third problem we address is training in the face of chaotic spontaneous activity. Jaeger and Haas (2004)
avoided this problem by starting with networks that were inactive in the absence of input (which is the basis for
calling them echo-state networks). As we show in the Results, there are signiﬁcant advantages in using a network
that exhibits chaotic activity prior to training. To exploit these advantages, however, we must avoid chaotic network
activity during training. It turns out that the solution for learning in a recurrent network and for suppressing chaos
turn out to be one and the same: synaptic modiﬁcations must be strong and rapid during the initial phases of train‐
ing. This is precisely what the FORCE procedure achieves.
FORCE learning operates quite diﬀerently from traditional training in neural networks. Usually, training consists of
performing a sequence of modiﬁcations that slowly reduce initially large errors in network output. In FORCE learn‐
ing, errors are always small, even from the beginning of the training process. As a result, the goal of training is not
FG
GG
signiﬁcant error reduction, but rather reducing the amount of modiﬁcation needed to keep the errors small. By the
end of the training period, modiﬁcation is no longer needed, and the network can generate the desired output au‐
tonomously.
From a machine learning point of view, the FORCE procedure we propose provides a powerful algorithm for con‐
structing recurrent neural networks that generate complex and controllable patterns of activity either in the absence
of or in response to input. From a biological perspective, it can be viewed either as a model for training-induced
modiﬁcation or, more conservatively, as a method for building functioning circuit models for further study. Either
way, our approach introduces a novel way to think about learning in neural networks and to make contact with ex‐
perimental data.
Results
The recurrent network that forms the basis of our studies is a conventional model in which the outputs of individual
neurons are characterized by ﬁring rates and neurons are sparsely interconnected through excitatory and inhibitory
synapses of various strengths (Methods). Following ideas developed in the context of liquid-state (Maass et al.,
2002) and echo-state (Jaeger, 2003) models, we assume that this basic network is not designed for any speciﬁc task,
but is instead a general purpose dynamical system that will be co-opted for particular applications through subse‐
quent synaptic modiﬁcation. As a result, the connectivity and synaptic strengths of the network are chosen ran‐
domly (Methods). For the parameters we use, the initial state of the network is chaotic (ﬁgure 2A).
Figure 2
FORCE learning in the network of ﬁgure 1A. A-C) The FORCE training sequence. Network output, z, is in red, the ﬁring rates of 10
sample neurons from the network are in blue and the orange trace is the magnitude of the time derivative of the readout weight vector.
A) Before learning, network activity and output are chaotic. B) During learning, the output matches the target function, in this case a
triangle wave and the network activity is periodic because the readout weights ﬂuctuate rapidly. These ﬂuctuations subside as learning
progresses. C) After training, the network activity is periodic and the output matches the target without requiring any weight modiﬁ‐
cation. D-K) Examples of FORCE Learning. Red traces are network outputs after training with the network running autonomously.
Green traces, where not covered by the matching red traces, are target functions. D) Periodic function composed of 4 sinusoids. E)
Periodic function composed of 16 sinusoids. F) Periodic function of 4 sinusoids learned from a noisy target function. G) Square-
wave. H) The Lorenz attractor. Initial conditions of the network and the target were matched at the beginning of the traces. I) Sine
waves with periods of 60 ms and 8 s. J) A one-shot example using a network with two readout units (circuit insert). The red trace is
the output of unit 2. When unit 1 is activated, its feedback creates the ﬁxed point to the left of the left-most blue arrow, establishing
the appropriate initial condition. Feedback from unit 2 then produces the sequence between the two blue arrows. When the sequence
is concluded, the network output returns to being chaotic. K) A low amplitude sine wave (right of gray line) for which the FORCE
procedure does not control network chaos (blue traces) and learning fails.
To specify a task for the networks of ﬁgure 1, we must deﬁne their outputs. In a full model, this would involve sim‐
ulating activity all the way out to the periphery. In the absence of such a complete model, we need to have a way of
describing what the network is “doing”, and here we follow another suggestion from the liquid- and echo-state ap‐
proach (Maass et al., 2002; Jaeger, 2003; see also Buonomano and Merzenich, 1995). We deﬁne the network output
through a weighted sum of its activities. Denoting the activities of the network neurons at time t by assembling
them into a column vector r(t) and the weights connecting these neurons to the output by another column vector w,
we deﬁne the network output as
z(t) = w r(t).
(1)
Multiple readouts can be deﬁned in a similar manner, each with its own set of weights, but we restrict the discus‐
sion to one readout at this point. Although a linear readout is a useful way of deﬁning what we mean by the output
of a network, it should be kept in mind that it is a computational stand-in for complex transduction circuitry. For
this reason, we refer to the output-generating element as a unit rather than a neuron, and we call the components of
w weights rather than synaptic strengths.
Having speciﬁed the network output, we can now deﬁne the task we want the network to perform, which is to set
z(t) = f (t) for a pre-deﬁned target function f (t). In most of the examples we present, the goal is to make a network
produce the target function in the absence of any input. Later, we consider the more conventional network task of
generating outputs that depend on inputs to the network in a speciﬁed way. Due to stability issues, this is an easier
task than generating target functions without inputs, so we mainly focus on the no-input case.
In the initial instantiation of our model (ﬁgure 1A), we follow Jaeger and Haas (2004) and modify only the output
weight vector w. All other network connections are left unchanged from their initial, randomly chosen values. The
critical element that makes such a procedure possible is a feedback loop that carries the output z back into the net‐
work (ﬁgure 1A). Learning cannot be accomplished in a network receiving no external input without including
such a loop. The strengths of the synapses from this loop onto the neurons of the network are chosen randomly and
left unmodiﬁed. The strength of the feedback synapses is of order 1 whereas that of synapses between neurons of
the recurrent network is of order 1 over the square root of the number of recurrent synapses per neuron. The feed‐
back synapses are made stronger so that the feedback pathway has an appreciable eﬀect on the activity of the recur‐
rent network. Later, when we consider the architectures of ﬁgures 1B & 1C, we will no longer need such strong
synapses.
FORCE Learning
Training in the presence of the feedback loop connecting the output in ﬁgure 1A back to the network is challenging
because modifying the readout weights produces delayed eﬀects that can be diﬃcult to calculate. Modifying w has
a direct eﬀect on the output z given by equation 1, and it is easy to determine how to change w to make z closer to f
through this direct eﬀect. However, the feedback loop in ﬁgure 1A gives rise to a delayed eﬀect when the resulting
change in the output caused by modifying w propagates repeatedly along the feedback pathway and through the
network, changing network activities. Because of this delayed eﬀect, a weight modiﬁcation that at ﬁrst appears to
bring z closer to f may later cause it to deviate away. This problem of delayed eﬀects arises when attempting to
modify synapses in any recurrent architecture, including those of ﬁgure 1B and 1C.
As stated in the Introduction, Jaeger and Haas (2004) eliminated the problem of delayed eﬀects by clamping feed‐
back during learning. In other words, the output of the network, given by equation 1 was compared with f to deter‐
mine an error that controlled modiﬁcation of the readout weights, but this output was not fed back to the network
during training. Instead the feedback pathway was clamped to the target function f. The true output was only fed
back to the network after training was completed.
We take another approach, which does not require any clamping or manipulation of the feedback pathway, it relies
solely on error-based modiﬁcation of the readout weights. In this scheme, we allow output errors to be fed back into
the network, but we keep them small by making rapid and eﬀective weight modiﬁcations. As long as output errors
are small enough, they can be fed back without disrupting learning, i.e. without introducing signiﬁcant delayed, re‐
verberating eﬀects. Because the method requires tight control of a small (ﬁrst-order) error, we call it First-Order,
Reduced and Controlled Error or FORCE learning. Although the FORCE procedure holds the feedback signal close
to its desired value, it does not completely clamp it. This diﬀerence, although numerically small, has extremely sig‐
T
niﬁcant implications for network stability. Small diﬀerences between the actual and desired output of the network
during training allow the learning procedure to sample instabilities in the recurrent network and stabilize them.
A learning algorithm suitable for FORCE learning must rapidly reduce the magnitude of the diﬀerence between the
actual and desired output to a small value, and then keep it small while searching for and eventually ﬁnding a set of
ﬁxed readout weights that can maintain a small error without further modiﬁcation. A number of algorithms are ca‐
pable of doing this (Discussion). All of them involve updates to the values of the weights at times separated by an
interval Δt. Each update consists of evaluating the output of the network, determining how far this output deviates
from the target function, and modifying the readout weights accordingly. Note that Δt is the interval of time be‐
tween modiﬁcations of the readout weights, not the basic integration time step for the network simulation, which
can be smaller than Δt .
At time t, the training procedure starts by sampling the network output, which is given at this point by w  (t -
Δt)r(t). The reason that the weights appear here evaluated at time t - Δt is that they have not yet been updated by
the modiﬁcation procedure, so they take the same values that they had at the end of the previous update.
Comparing this output with the desired target output f (t), we deﬁne the error
e (t) = w (t − Δt)r(t) − f(t).
(2)
The minus subscript signiﬁes that this is the error prior to the weight update at time t . The next step in the training
process is to update the weights from w(t - Δt) to w(t) in a way that reduces the magnitude of e (t). Immediately af‐
ter the weight update, the output of the network is w  (t)r(t), assuming that the weights are modiﬁed rapidly on the
scale of network evolution (Discussion). Thus, the error after the weight update is
e (t) = w (t)r(t) − f(t), 
(3)
with the plus subscript signifying the error after the weights have been updated.
The goal of any weight modiﬁcation scheme is to reduce errors by making |e (t)| < |e (t)| and also to converge to a
solution in which the weight vector is no longer changing so that training can be terminated. This latter condition
corresponding to making e (t)/e (t)→1 by the end of training. In most training procedures, these two conditions are
accompanied by a steady reduction in the magnitude of both errors (e  and e ) over time, which are both quite large
during the early stages of training. FORCE learning is unusual in that the magnitudes of these errors are small
throughout the learning process, although they are similarly reduced over time. This is done by making a large re‐
duction in their size at the time of the ﬁrst weight update, and then maintaining small errors throughout the training
process that decrease with time.
If the training process is initialized at time t = 0, the ﬁrst weight update will occur at time Δt. A weight modiﬁca‐
tion rule useful for FORCE learning should make |e (Δt)|, the error after the ﬁrst weight update has been per‐
formed, small, and then keep |e (Δt)| small while slowly increasing e (t)/e (t)→1. Given a small magnitude of e (t -
Δt)), e (t), which is equal to e (t - Δt) plus a term of order Δt, is kept small by keeping the updating interval Δt
suﬃciently short. This means that learning can be performed with an error that starts and stays small.
As stated above, several modiﬁcation rules meet the requirements of FORCE learning, but the recursive least
squares (RLS) algorithm is particularly powerful (Haykin, 2002), and we use it here (see Discussion and
Supplementary Materials for another, simpler algorithm). In RLS modiﬁcation,
w(t) = w(t − Δt) − e (t)P(t)r(t), 
(4)
where P(t) is an N × N matrix that is updated at the same time as the weights according to the rule
T
−
T
-
T
+
T
+
-
+
-
+
-
+
-
+
-
+
-
+
−
(5)
The algorithm also requires an initial value for P , which is taken to be
(6)
where I is the identity matrix and α is a constant parameter. Equation 4 can be viewed as a standard delta-type rule
(that is, a rule involving the product of the error and the presynaptic ﬁring rate), but with multiple learning rates
given by the matrix P , rather than by a scalar quantity. In this algorithm, P is a running estimate of the inverse of
the correlation matrix of the network rates r plus a regularization term (Haykin, 2002), i.e.
.
It is straightforward to show that the RLS rule satisﬁes the conditions necessary for FORCE learning. First, if we
assume that the initial readout weights are zero for simplicity (this is not essential), the above equations imply that
the error after the ﬁrst weight update is
(7)
The quantity r r is of order N , the number of neurons in the network, so as long as α = N, this error is small, and
its size can be controlled by adjusting α (see below). Furthermore, at subsequent times, the above equations imply
that
e (t) = e (t)(1 − r (t)P(t)r(t)), 
(8)
The quantity r Pr varies over the course of learning from something close to 1 to a value that asymptotically ap‐
proaches 0, and it is always positive. This means that the size of the error is reduced by the weight update, as re‐
quired, and ultimately e (t)/e (t)→1.
The parameter α , which acts as a learning rate, should be adjusted depending on the particular target function be‐
ing learned. Small α values result in fast learning, but sometimes make weight changes so rapid that the algorithm
becomes unstable. In those cases, larger α should be used (subject to the constraint α = N), but if α is too large, the
FORCE algorithm may not keep the output close to the target function for a long enough time, causing learning to
fail. In practice, values from 1 to 100 are eﬀective, depending on the task.
In addition to dealing with feedback, FORCE learning must control the chaotic activity of the network during the
training process. In this regard, it is important to note that the network we are considering is being driven through
the feedback pathway by a signal approximately equal to the target function. Such an input can induce a transition
between chaotic and non-chaotic states (Molgedey et al., 1992; Bertchinger and Natschläger, 2004; Rajan et al.,
2008). This is how the problem of chaotic activity can be avoided. Provided that the feedback signal is of suﬃcient
amplitude and frequency to induce a transition to a non-chaotic state (the required properties are discussed in Rajan
et al., 2008), learning can take place in the absence of chaotic activity, even though the network is chaotic prior to
𝐏(𝑡) = 𝐏(𝑡−𝛥𝑡) −
.
𝐏(𝑡−𝛥𝑡) 𝐫(𝑡)
(𝑡) 𝐏(𝑡−𝛥𝑡)
𝐫𝑇
1 +
(𝑡) 𝐏(𝑡−𝛥𝑡) 𝐫(𝑡)
𝐫𝑇
𝐏(0) =
,
𝐈
𝛼
𝐏= (
(𝑡)
(𝑡) + 𝛼𝐈)
𝐫
𝑡
𝐫𝑇
−1
(𝛥𝑡) = −
.
𝑒−
𝛼𝑓(𝛥𝑡)
𝛼+
(𝛥𝑡) 𝐫(𝛥𝑡)
𝐫𝑇
T
+
−
T
T
+
-
learning.
Examples of FORCE Learning
Figure 2A-C illustrates how the activity of an initially chaotic network can be modiﬁed so that it ends up producing
a periodic, triangle-wave output autonomously. Initially, with the output weight vector w chosen randomly, the neu‐
rons in the network exhibit chaotic spontaneous activity, as does the network output (ﬁgure 2A). When we start
FORCE learning, the weights of the readout connections begin to ﬂuctuate rapidly, which immediately changes the
activity of the network so that it is periodic rather than chaotic and forces the output to match the target triangle
wave (ﬁgure 2B). The progression of learning can be tracked by monitoring the size of the ﬂuctuations in the read‐
out weights (orange trace in ﬁgure 2B), which diminish over time as the learning procedure establishes a set of
static weights that generate the target function without requiring modiﬁcation. At this point, learning can be turned
oﬀ, and the network continues to generate the triangle wave output on its own indeﬁnitely (ﬁgure 2C). The learning
process is rapid, converging in only four cycles of the triangle wave in this example.
FORCE learning can be used to modify networks that are initially in a chaotic state so that they autonomously pro‐
duce a wide variety of outputs (Figure 2D-K). In these examples, training typically converges in about 1000τ,
where τ is the basic time constant of the network, which we set to 10 ms. This means learning takes about 10 s of
simulated time. Networks can be trained to produce periodic functions of diﬀerent complexity and form (
ﬁgures 2D-G and I), even when the target function is corrupted by noise (ﬁgure 2F). The dynamic range of the out‐
puts that chaotic networks can be trained to generate by FORCE learning is impressive. For example, a 1000 neu‐
ron network with a time constant of 10 ms can produce sine wave outputs with periods ranging from 60 ms to 8 s (
ﬁgure 2I).
FORCE learning is not restricted to periodic functions. For example, a network can be trained to produce an output
matching one of the dynamic variables of the three-dimensional chaotic Lorenz attractor (Methods, see also Jaeger
and Haas, 2004), although in this case, because the target is itself a chaotic process, a precise match between output
and target can only last for a ﬁnite amount of time (ﬁgure 2H). After the two traces diverge, the network still pro‐
duces a trace that looks like it comes from the Lorenz model.
FORCE learning can also produce a segment matching a one-shot, non-repeating target function (ﬁgure 2J). To
produce such a one-shot sequence, the network must be initialized properly, and we do this by introducing a ﬁxed-
point attractor as well as the network conﬁguration that produces the one-shot sequence. This is done by adding a
second readout unit to the network that also provides feedback (Methods, network diagram in ﬁgure 2J). The ﬁrst
feedback unit induces the ﬁxed point corresponding to a constant z output (horizontal red line in ﬁgure 2J), and
then the second unit induces the target pattern (red trace between the arrows in ﬁgure 2J). As shown below, initial‐
ization can also be achieved through appropriate input.
As discussed above, FORCE learning must induce a transition in the network from chaotic to non-chaotic activity
during training. This requires an input to the network, through the feedback loop in our case, of suﬃcient ampli‐
tude. If we try to train a network to generate a target function with too small an amplitude, the activity of the net‐
work neurons remains chaotic even after FORCE learning is activated (ﬁgure 2K). In this case, learning does not
converge to a successful solution. There are a number of solutions to this problem. It is possible for the network to
generate low amplitude oscillatory and non-oscillatory functions if these are displaced from zero by a constant
shift. Alternatively, the networks shown in ﬁgure 1B and C can be trained to generate low amplitude signals cen‐
tered near zero.
PCA Analysis of FORCE Learning
The activity of a network that has been modiﬁed by the FORCE procedure to produce a particular output can be an‐
alyzed by principal component analysis (PCA). For a network producing the periodic pattern shown in ﬁgure 3A,
the distribution of PCA eigenvalues (ﬁgure 3C) indicates that the trajectory of network activity lies primarily in a
subspace that is of considerably lower dimension than the number of network neurons. The projections of the net‐
work activity vector r(t) onto the PC vectors form a set of orthogonal basis functions (ﬁgure 3B) from with the tar‐
get function is generated. An accurate approximation of the network output (brown trace in Figure 3A) can be gen‐
erated using the basis functions derived from only the ﬁrst 8 principal components (with components labeled in de‐
creasing order of the size of their eigenvalues). These 8 components are not the whole story, however, because,
along with generating the target function, the network must be stable. If we express the readout weight vector in
terms of its projections onto the PC vectors of the network activity, we ﬁnd that learning sets about the top 50 of
these projections to uniquely speciﬁed values (ﬁgure 3E). The remaining projections take diﬀerent values from one
learning trial to the next, depending on initial conditions (ﬁgure 3F). This multiplicity of solutions greatly simpli‐
ﬁes the task of ﬁnding successful readout weights.
Figure 3
Principal component analysis of network activity. A) Output after training a network to produce a sum of four sinusoids (red), and the
approximation (brown) obtained using activity projected onto the 8 leading principal components. B) Projections of network activity
onto the leading eight PC vectors. C) PCA eigenvalues for the network activity that generated the waveform in A. Only the largest 100
of 1000 eigenvalues are shown. D) Schematic showing the transition from control to learning phases of learning as a function of time
and of PC eigenvalue. E) Evolution of the projections of w onto the two leading PC vectors during learning starting from ﬁve diﬀer‐
ent initial conditions. These values converge to the same point on all trials. F) The same weight evolution but now including the pro‐
jection onto PC vector 80 as a third dimension. The ﬁnal values of this projection are diﬀerent on each of the 5 runs, resulting in the
vertical line at the center of the ﬁgure. Nevertheless, all of these networks generate the output in A.
The uneven distribution of eigenvalues shown in ﬁgure 3C illustrates why the RLS algorithm works so well for
FORCE learning. As mentioned previously, the matrix P acts as a set of learning rates for the RLS algorithm. This
is seen most clearly by shifting to a basis in which P is diagonal. Assuming learning has progressed long enough
for P to have converged to the inverse correlation matrix of r, the diagonal basis is achieved by projecting w and r
onto the PC vectors. Doing this, it is straightforward to show that the learning rate for the component of w aligned
with PC vector a after M weight updates is 1/(Mλ  +α), where λ  is the corresponding PC eigenvalue. This rate di‐
vides the RLS process into two stages, one when M <α/λ  in which the major role of weight modiﬁcation is to con‐
trol the output (set it close to f), and another when M >α/λ  in which the goal is learning, that is, ﬁnding a static
weight that accomplishes the task. Components of w with large eigenvalues quickly enter the learning phase,
whereas those with small eigenvalues spend more time in the control phase (ﬁgure 3D). Controlling components
with small eigenvalues allows weight projections in dimensions with large eigenvalues to be learned.
The learning rate for all components during the control phase is 1/α. During the learning phase, the rate for PC
component a is proportional to 1/λ . The average rate of change (as opposed to just the learning rate) of the projec‐
tion of the output weight vector onto principal component a is proportional to 
 because the factor
of r in equation 4 introduces a term proportional to 
, so the full rate of change for large M goes as 
.
This is exactly what it should be, because in the expression for z, this change is multiplied by the projection of r
onto PC vector a, which again has an amplitude proportional to 
. Thus, RLS, by having rates of change of w
proportional to 
 in the PC basis, allows all the projections to, potentially, contribute equally to the output of
the network.
Comparison of Echo-State and FORCE Feedback
In echo-state learning (Jaeger and Haas, 2004), the feedback signal during training was set equal to the target func‐
tion f (t). In FORCE learning, the feedback signal is z(t) during training. To compare these two methods, we intro‐
duce a mixed feedback signal, setting the feedback equal to γ f (t) + (1-γ)z(t) during training. Thus, γ = 0 corre‐
sponds to FORCE learning and γ =1 to echo-state learning, with intermediate values interpolating between these
two approaches.
Training to produce the output of ﬁgure 3A, we ﬁnd the network is only stable on the majority of trials when γ <
0.15, in other words close to the FORCE limit (ﬁgure 4A). Furthermore, in this γ range, the error in the output after
training increases as a function of γ, meaning γ = 0 performs best (ﬁgure 4B). For a typical instability of pure echo-
state learning, the output matches the target brieﬂy after learning is terminated, but then it deviates away (ﬁgure 4C
). Because this stability problem arises from the failure of the network to sample feedback ﬂuctuations, it can be al‐
leviated somewhat by introducing noise into the feedback loop during training (Jaeger and Haas, 2004, introduced
noise into the network, which is less eﬀective). Doing this, we ﬁnd that pure echo-state learning converges on about
50% of the trials, but the error on these is signiﬁcantly larger than for pure FORCE learning.
a
a
a
a
a
/ (𝑀
+ 𝛼)
𝜆𝑎
‾‾‾
√
𝜆𝑎
𝜆𝑎
‾‾‾
√
1 /
𝜆𝑎
‾‾‾
√
𝜆𝑎
‾‾‾
√
1 /
𝜆𝑎
‾‾‾
√
Figure 4
Comparison of diﬀerent mixtures of FORCE (γ = 0 and echo-state (γ =1) feedback. A) Percent of trials resulting in stable generation
of the target function. B) Mean absolute error (MAE) between the output and target function after learning over the γ range where
learning converged. C) Example run with output (red) and target function (green) for γ =1. The trajectory is unstable.
Advantages of Chaotic Spontaneous Activity
To study the eﬀect of spontaneous chaotic activity on network performance, we introduce a factor g that scales the
strengths of the recurrent connections within the network. Networks with g <1 are inactive prior to training,
whereas networks with g >1 exhibit chaotic spontaneous activity (Sompolinsky et al., 1988) that gets more irregu‐
lar and ﬂuctuates more rapidly as g is increased beyond 1 (we typically use g =1.5).
The number of cycles required to train a network to generate the periodic target function shown in ﬁgure 3A drops
dramatically as a function of g , continuing to fall as g gets larger than 1 (ﬁgure 5A). The average root-mean-square
(RMS) error, indicating the diﬀerence between the target function and the output of the network after FORCE
learning, also decreases with g (ﬁgure 5B). Another measure of training success is the magnitude of the readout
weight vector |w| (ﬁgure 5C). Large values of |w| indicate that the solution found by a learning process involves
cancellations between large positive and negative contributions. Such solutions tend to be unstable and sensitive to
noise. The magnitude of the weight vector falls as a function of g and takes its smallest values in the region g > 1
characterized by chaotic spontaneous activity.
Figure 5
Chaos improves training performance. Networks with diﬀerent g values (Methods) were trained to produce the output of ﬁgure 3A.
Results are plotted against g in the range 0.75 < g <1.56, where learning converged. A) Number of cycles of the periodic target func‐
tion required for training. B) The RMS error of the network output after training. C) The length of the readout weight vector |w| after
training.
These results indicate that networks that are initially in a chaotic state are quicker to train and produce more accu‐
rate and robust outputs than non-chaotic networks. Learning works best when g > 1 and, in fact, fails in this exam‐
ple for networks with g > 0.75. This might suggest that the larger the g value the better, but there is an upper limit.
Recall that FORCE learning does not work if the feedback from the readout unit to the network fails to suppress the
chaos in the network. For any given target function and set of feedback synaptic strengths, there is an upper limit
for g beyond which chaos cannot be suppressed by FORCE learning. Indeed, the range of g values in ﬁgure 5 ter‐
minates at g = 1.56 because learning did not converge for higher g values due to this problem. Thus, the best value
of g for a particular target function is at the “edge of chaos” (Bertchinger and Natschläger, 2004), that is the g value
just below the point where FORCE learning fails to suppress chaotic activity during training.
Distorted and Delayed Feedback
The linear readout unit was introduced into the network model as a stand-in for a more complex, un-modeled pe‐
ripheral system, in order to deﬁne the output of the network. The critical information provided by the readout unit
is the error signal needed to guide weight modiﬁcation, so its biological interpretation should be as a system that
computes or estimates the deviation between an action generated by a network and the desired action. However, in
the network conﬁguration presented to this point (ﬁgure 1A), the readout unit, in addition to generating the error
signal that guides learning, is also the source of feedback. Given that the output in a biological system is actually
the result of a large amount of nonlinear processing and that feedback, whether proprioceptive or a motor eﬀerence
copy, may have to travel a signiﬁcant distance before returning to the network, we begin this section by examining
the eﬀect of introducing delays and nonlinear distortions along the feedback pathway from the readout unit to the
network neurons.
The FORCE learning scheme is remarkably robust to distortions introduced along the feedback pathway (ﬁgure 6A
). Nonlinear distortions of the feedback signal can be introduced as long as they do not diminish the temporal ﬂuc‐
tuations of the output to the point where chaos cannot be suppressed. We have also introduced low-pass ﬁltering
into the feedback pathway, which can be quite extreme before the network fails to learn. Delays can be more prob‐
lematic if they are too long. The critical point is that FORCE learning works as long as the feedback is of an appro‐
priate form to suppress the initial chaos in the network. This means that the feedback really only has to match the
period or the duration of the target function and roughly have the same frequency content.
Figure 6
Feedback variants. A) Network trained to produce a periodic output (red trace) when its feedback (cyan trace) is 1.3tanh(sin(πz(t -100
ms)), a delayed and distorted function of the output z(t) (gray oval in circuit diagram). B) FORCE learning with a separate feedback
network (circuit diagram). Output is the red trace, and blue traces show activity traces from 5 neurons within the feedback network.
C) A network (circuit diagram) in which the internal synapses are trained to produce the output (red). Activities of 5 representative
network neurons are in blue. The thick cyan traces are overlays of the component of the input to each of these 5 neurons induced by
FORCE learning, 
 for i=1K 5.
FORCE Learning with Other Network Architectures
Even allowing for distortion and delay, the feedback pathway, originating as it does from the linear readout unit, is
a non-biological element of the network architecture of ﬁgure 1A. To address this problem, we consider two ways
of separating the feedback pathway from the linear readout of the network and modeling it more realistically. The
ﬁrst is to provide feedback to the network through a second neural network (ﬁgure 1B) rather than via the readout
unit. To distinguish the two networks, we call the original network, present in ﬁgure 1A, the generator network and
this new network the feedback network. The feedback network has nonlinear, dynamic neurons identical to those of
the generator network, and is recurrently connected. Each unit of the feedback network produces a distinct output
that is fed back to a subset of neurons in the generator network, so the task of carrying feedback is shared across
multiple neurons. This repairs two biologically implausible aspects of the architecture of ﬁgure 1A: the strong feed‐
back synapses mentioned above and the fact that every neuron in the network receives the same feedback signal.
When we include a feedback network (ﬁgure 1B), FORCE learning takes place both on the weights connecting the
generator network to the readout unit (as in the architecture of ﬁgure 1A) and on the synapses connecting the gener‐
ator network to the feedback network (red connections in ﬁgure 1B). Separating feedback from output introduces a
credit-assignment problem because changes to the synapses connecting the generator network to the feedback net‐
work do not have a direct eﬀect on the output. To solve this problem within the FORCE learning scheme, we treat
every neuron subject to synaptic modiﬁcation as if it were the readout unit, even when it is not. In other words, we
apply equations 4 & 5 to every synapse connecting the generator network to the feedback network (Supplementary
Materials), and we also apply them to the weights driving the readout unit. When we modify synapses onto a par‐
ticular neuron of the feedback network, the vector r in these equations is composed of the ﬁring rates of generator
network neurons presynaptic to that feedback neuron, and the weight vector w is replaced by the strengths of the
synapses it receives from these presynaptic neurons. However, the same error term that originates from the readout
(
(𝑡) −
(0))
(𝑡)
𝑗𝐽𝑖𝑗
𝐽𝑖𝑗
𝑟𝑗
(equation 2) is used in these equations whether they are applied to the weights of the readout unit or synapses onto
neurons of the feedback network (Methods). The form of FORCE learning we are using is cell autonomous, so no
communication of learning-related information between neurons is required to implement these modiﬁcations, ex‐
cept that they all use a global error signal.
FORCE learning with a feedback network and independent readout unit can generate complex outputs similar to
those in ﬁgure 4, although parameters such as α (equation 6) may require more careful adjustment. After training,
when the output of these networks matches the target function, the activities of neurons in the feedback network do
not, despite the fact that their synapses are modiﬁed by the same algorithm as the readout weights (ﬁgure 6B). This
diﬀerence is due to the fact that the feedback network neurons receive input from each other as well as from the
generator network, and these other inputs are not modiﬁed by the FORCE procedure. Diﬀerences between the ac‐
tivity of feedback network neurons and the output of the readout unit can also arise from diﬀerent values of the
synapses and the readout weights prior to learning.
With a separate feedback network, the feedback to an individual neuron of the generator network is a random com‐
bination of the activities of a subset of feedback neurons, summed through random synaptic weights. While these
sums bear a certain resemblance to the target function, they are not identical to it nor are they identical for diﬀerent
neurons of the generator network. Nevertheless, FORCE learning works. This extends the result of ﬁgure 6A,
showing not only that the feedback does not have to be identical to the network output, but that it does not even
have to be identical for each neuron of the generator network.
Why does this form of learning, in which every neuron with synapses being modiﬁed is treated as if it were produc‐
ing the output, work? In the example of ﬁgure 6B, the connections from the generator network to the readout unit
and to the feedback network are sparse and random (Methods), so that neurons in the feedback network do not re‐
ceive the same inputs from the generator network as the readout unit. However, suppose for a moment that each
neuron of the feedback network, as well as the readout unit, received synapses from all of the neurons of the gener‐
ator network. In this case, the changes to the synapses onto the feedback neurons would be identical to the changes
of the weights onto the readout unit and therefore would induce a signal identical to the output into each neuron of
the feedback network. This occurs, even though there is no direct connection between these two circuit elements,
because the same learning rule with the same global error is being applied in both cases.
The explanation of why FORCE learning works in the feedback network when the connections from the generator
network are sparse rather than all-to-all (as in ﬁgure 6B) relies on the accuracy of randomly sampling a large sys‐
tem (Sussillo, 2009). With sparse connectivity, each neuron samples a subset of the activities within the full genera‐
tor network, but if this sample is large enough, it can provide an accurate representation of the leading principal
components of the activity of the generator network that drive learning. This is enough information to allow learn‐
ing to proceed. For ﬁgure 6B, we used an extremely sparse connectivity (Methods) to illustrate that FORCE learn‐
ing can work even when the connections of the units being modiﬁed are highly non-overlapping.
The original generator network we used (ﬁgure 1A) is recurrent and can produce its own feedback. This means that
we should be able to apply FORCE learning to the synapses of the generator network itself, in the arrangement
shown in ﬁgure 1C. To implement FORCE learning within the generator network (Supplementary Materials), we
modify every synapse in that network using equations 4-5. To apply these equations, the vector w is replaced by the
set of synapses onto a particular neuron being modiﬁed, and r is replaced by the vector formed from the ﬁring rates
of all the neurons presynaptic to that network neuron. As in the example of learning in the feedback network,
FORCE learning is also applied to the readout weights, and the same error, given by equation 2, is used for every
synapse or weight being modiﬁed.
FORCE learning within the network can produce a complex target output (ﬁgure 6C). An argument similar to that
given for learning within the feedback network can be applied to FORCE learning for synapses within the generator
network. To illustrate how FORCE learning works, we express the total current into each neuron of the generator
network as the sum of two terms. One is the current produced by the original synaptic strengths prior to learning,
 for neuron i. The other is the extra current generated by the learning-induced changes in these synapses,
. The ﬁrst term, as well as the total current, is diﬀerent for each neuron of the generator net‐
work because of the random initial values of the synaptic strengths. The second, learning-induced current, however,
is virtually identical to the target function for each neuron of the network (ﬁgure 6C, cyan). Thus, FORCE learning
induces a signal representing the target function into the network, just as it does for the architecture of ﬁgure 1A,
but in a subtler and more biologically realistic manner.
Output patterns like those in ﬁgure 2 can be reproduced by FORCE learning applied within the generator or feed‐
back networks. In the following sections, we illustrate the capacity of these forms of FORCE learning while, at the
same time, introducing new tasks. All of the examples shown can be reproduced using all three of the architectures
in ﬁgure 1, but for compactness we show results from learning in the generator network in ﬁgure 7 and learning in
the feedback network in ﬁgure 8. For the interested reader, Matlab ﬁles that implement FORCE learning in the diﬀ‐
ﬀerent architectures are included with the Supplementary Materials.
Figure 7
Multiple pattern generation and 4-Bit memory through learning in the generator network. A) Network with control inputs used to pro‐
duce multiple output patterns (synapses and readout weights that are modiﬁable in red). B) Five outputs (1 cycle of each periodic
function made from 3 sinusoids is shown) generated by a single network and selected by static control inputs. C) A network with 4
outputs and 8 inputs used to produce a 4-bit memory (modiﬁable synapses and readout weights in red). D) Red traces are the 4 out‐
puts, with green traces showing their target values. Purple traces show the 8 inputs, divided into ON and OFF pairs associated with
the output trace above them. The upper input in each pair turns the corresponding output on (sets it to +1). The lower input of each
pair turns the output oﬀ (sets it to -1). After learning, the network has implemented a 4-bit memory, with each output responding only
to its two inputs while ignoring the other inputs.
(0)
(𝑡)
𝑗
𝐽𝑖𝑗
𝑟𝑗
(
(𝑡) −
(0))
(𝑡)
𝑗𝐽𝑖𝑗
𝐽𝑖𝑗
𝑟𝑗
Figure 8
Networks that generate both running and walking human motions. A) Either of these two network architectures can be used to gener‐
ate the running and walking motions (modiﬁable readout weights shown in red), but the upper network is shown. Constant inputs diﬀ‐
ﬀerentiate between running and walking (purple). Each of 95 joint angles is generated through time by one of the 95 readout units
(curved arrows). B) The running motion generated after training. Cyan frames show early and magenta frames late movement phases.
C) Ten sample network neuron activities during the walking motion. D) The walking motion, with colors as in B.
Switching Between Multiple Outputs and Input-Output Mapping with Memory
The examples to this point have involved a single target function. We can train networks with the architecture of
ﬁgure 1C in both sparse and fully connected conﬁgurations (we illustrate the sparse case) to produce multiple func‐
tions, with a set of inputs controlling which is generated at any particular time. We do this by introducing static
control inputs to the network neurons (ﬁgure 7A) and pairing each desired output function with a particular input
pattern (Methods). The constant values of the control inputs are chosen randomly. When a particular target function
is being either trained or generated, the control inputs to the network are set to the corresponding static pattern and
held constant until a diﬀerent output is desired. The control inputs do not supply any temporal information to the
network, they act solely as a switching signal to select a particular output function. The result is a single network
that can produce a number of diﬀerent outputs depending on the values of the control inputs (ﬁgure 7B).
Up to now, we have treated the network we are studying as a source of what are analogous to motor output patterns.
Networks can also generate complex input/output maps when inputs are present. Figure 7C shows a particularly
complex example of a network that functions as a 4-bit memory that is robust to input noise. This network has 8 in‐
puts that randomly connect to neurons in the network and are functionally divided into pairs (Methods). The input
values are held at zero except for short pulses to positive values that act as ON and OFF commands for the 4 read‐
out units. Input 1 is the ON command for output 1 and input 2 is its OFF command. Similarly, inputs 3 and 4 are
the ON and OFF commands for output 2, and so on. Turning on an output means inducing a transition to a state
with a ﬁxed positive value of 1, and turning it oﬀ means switching it to -1. After FORCE learning, the inputs cor‐
rectly turn the appropriate outputs on and oﬀ with little crosstalk between inputs and inappropriate outputs (
ﬁgure 7C). This occurs despite the random connectivity of the network, which means that the inputs do not segre‐
gate into diﬀerent channels. This example requires the network to have, after learning, 16 diﬀerent ﬁxed point at‐
tractors, one for each of the 4  possible combinations of the 4 outputs, and the correct transitions between these at‐
2
tractors induced by pulsing the 8 inputs.
A Motion Capture Example
Finally, we consider an example of running and walking based on data obtained from human subjects performing
these actions while wearing a suit that allows variables such as joint angles to be measured (also studied by Taylor
et al., 2008 using a diﬀerent type of network and learning procedure). These data, from the CMU Motion Capture
Library, consist of 95 joint angles measured over hundreds of time steps.
We implemented this example using all the architectures in ﬁgure 1 in both sparse and fully connected conﬁgura‐
tions with similar results (we show a sparse example using the architecture of ﬁgure 1B). Producing all 95 joint an‐
gle sequences in the data sets requires that these networks have 95 readout units. For internal learning, subsets of
neurons subjected to learning were assigned to each readout unit and trained using the error generated by that unit
(Methods). Although running and walking might appear to be periodic motions, in fact the joint angles in the real
data are non-periodic. For this reason, we introduced static control inputs to initialize the network prior to starting
the running or walking motion. Because we wanted a single network to generate both motions, we also used the
control inputs to switch between them, as in ﬁgure 7A. The successfully trained networks produced both motions (
ﬁgure 8; for an animated demo showing all the architectures of ﬁgure 1 see the avi ﬁles included with the
Supplementary Materials) demonstrating that a single chaotic recurrent network can generate multiple, high-
dimensional, non-periodic patterns that resemble complex human motions.
Discussion
In the Introduction, we mentioned that FORCE learning could be viewed either as a model for learning-induced
modiﬁcation of biological networks or, more simply, as a method for constructing models of these networks. Our
results should be evaluated in light of both of these interpretations.
FORCE learning solves some, but certainly not all, of the problems associated with applying ideas about learning
from mathematical neural networks to biological systems. Biological networks exhibit complex and irregular spon‐
taneous activity that probably has both chaotic and stochastic sources. FORCE learning provides an approach to
network training that can be applied under these conditions (see, in particular, ﬁgure 2F). Furthermore, training
does not require any reconﬁguration of the network or changes in its dynamics other than the introduction of synap‐
tic modiﬁcation. Finally, the networks constructed by FORCE learning are more stable than in previous approaches.
FORCE learning relies on an error signal that, in our examples, is based on a readout unit that is not intended to be
a realistic circuit element. It is not clear how the error is computed in biological systems. This is a problem for all
models of supervised learning. In motor learning, we imagine that the target function is generated by an internal
model of a desired movement and that circuitry exists for comparing this internal model with the motor signal gen‐
erated by the network and for producing a modulatory signal that guides synaptic plasticity. The cerebellum has
been proposed as a possible locus for such internal modeling (Miall et al., 1993). Examples like that of ﬁgure 8,
which involve multiple outputs, require multiple error signals. For ﬁgure 8, we subdivided the network being
trained into diﬀerent regions in which plasticity was controlled by a diﬀerent error signal. If the error is carried by a
neuromodulator, this would require multiple pathways (though not necessarily multiple modulators) with at least
some spatial targeting. If the error signal is transmitted as in the case of the climbing ﬁbers of the cerebellum, mul‐
tiple error signals are more straightforward to handle. Examples with a single output only require a single global er‐
ror signal.
It is also not known how the error signal, once generated, controls synaptic plasticity. Again, this is a problem asso‐
ciated with all models of error- or reward-based learning. FORCE learning adds the condition that this modiﬁcation
act rather quickly compared to the timescale of the action being learned, at least during the initial phases of learn‐
ing. Both because it is under the control of an error signal and because it acts rapidly, the plasticity required does
not match that of typical long-term potentiation experiments, and it is a challenge raised by this work to uncover
how such rapid plasticity can be realized biologically, or if it is realized at all. Whatever the plasticity mechanism, a
key component of FORCE learning is producing the roughly correct output even during the initial stages of train‐
ing. Analogously, people cannot learn ﬁne manual skills by randomly ﬂailing their arms about and having their
movement errors slowly diminish over time, which would be analogous to more conventional network learning
schemes. FORCE learning reminds us that motor learning works best when the desired motor action is duplicated
as accurately as possible during training.
The RLS algorithm we have used is neuron-speciﬁc but not synapse-speciﬁc. By this we mean that the algorithm
uses information about all the inputs to a given neuron to guide modiﬁcation of its individual synapses. The algo‐
rithm requires some fairly involved calculations, although not matrix inversion. It is possible to use a simpler,
synapse-speciﬁc weight modiﬁcation procedure in which the matrix P is replaced by a single learning rate
(Supplementary Material). Provided that this scalar rate is adapted over time, FORCE learning can work with such
a simpler plasticity mechanism. Nevertheless, RLS is clearly a more powerful algorithm because it adapts the
learning rate to the magnitude of diﬀerent principal components of the network activity. It is possible that a scheme
that is simpler and more biologically plausible than RLS can be devised that retains this desirable feature.
The architectures of ﬁgures 1B and 1C, where learning occurs within feedback or generator networks, match bio‐
logical circuits better than that of ﬁgure 1A, where feedback comes directly from the readout unit. A key feature of
learning in these cases is that network plasticity is accompanied by plasticity along the output or error-computing
pathway. Plasticity in multiple areas (at least two, in these examples) coupled by a common error signal is a basic
prediction of the model. It is a curious feature that performance is comparable for all three architectures in ﬁgure 1,
despite that fact that the case of ﬁgure 1C involves changing many more synaptic strengths. We do not currently
know whether changing synapses within a network oﬀers advantages for the function-generation task. It may, but
the modiﬁcation algorithms developed thus far are not powerful enough exploit these advantages.
We now come to an analysis of FORCE learning as a model-building scheme. We have studied how spontaneously
active neural networks can be modiﬁed to generate desired outputs, and how control inputs can be used to initiate
and select among those outputs. Although this has most direct application to motor systems, it can be generalized to
a broader picture of cognitive processing (Yuste et al., 2005; Buonomano and Maass, 2009), as our example of a 4-
bit, input-controlled memory suggests.
Ganguli et al. (2008) have discussed the advantages of using an eﬀective delay-line architecture in applications of
networks to memory retention. Provided that a feedback loop from the output, as in ﬁgure 1A, is in place, a delay
line structure within the generator network should be quite eﬀective for function generation as well. However, be‐
cause we were interested in networks that generate spontaneous activity even in the absence of the output feedback
loop, we did not consider such an arrangement in any detail.
The two-step process by which we induced a chaotic network to produce a non-periodic sequence (ﬁgures 2J and 8)
may have an analog in motor and premotor cortex. The brief ﬁxed point that we introduced to terminate chaotic ac‐
tivity results in a sharp drop in the ﬂuctuations of network neurons just before the learned sequence is generated.
Churchland et al. (2006) and Churchland and Shenoy (2007a) have reported just such a drop in variability in their
recordings from motor and premotor areas in monkeys immediately before they performed a reaching movement.
Except in the simplest of examples, the activity of the generator neurons bears little relationship to the output of the
network. Trying to link single-neuron responses to motor actions may thus be misguided. Instead, results from our
network models suggest that it may be more instructive to study network-wide modes or patterns of activity ex‐
tracted by principal components analysis of multi-unit recordings (Fetz, 1992; Robinson, 1992; Churchland and
Shenoy, 2007b).
There are some interesting and perhaps unsettling aspects of the networks we have studied. First, the connectivity
of the generator network in the architectures of ﬁgure 1A and 1B is completely random, even after the network has
been trained to perform a speciﬁc task. It would be extremely diﬃcult to understand how the generator network
“works” by analyzing its synaptic connectivity. Even when the synapses of the generator network are modiﬁed (as
in ﬁgure 1C), there is no obvious relationship between the task being performed and the connectivity, which is in
any case not unique. The lesson here is that the activity, response properties and function of locally connected neu‐
rons can be drastically modiﬁed by feedback loops passing through distal networks. Circuits may need to be studied
with an eye toward how they modulate each other, rather than how they function in isolation.
The architecture of ﬁgure 1B, involving a separate feedback network (basal ganglia or cerebellum), may be a way to
keep plasticity from disrupting the generator network (motor cortex), a disruption that would be disastrous for all
motor output, not merely the current task being learned. Modiﬁcation of synapses in a second network (as in
ﬁgure 1B) may dominate when a motor task is ﬁrst learned, whereas changes within motor cortex (analogous to
learning within the network of ﬁgure 1C) may be reserved for “virtuoso” highly trained motor actions. Our exam‐
ples show the power of adding feedback loops as a way of modifying network activity. Nervous systems often seem
to be composed of loops within loops within loops. Because adding a feedback loop leaves the original circuit un‐
changed, this is a non-destructive yet highly ﬂexible way of increasing a behavioral repertoire through learning, as
well as during development and evolution.
Methods
All the networks we use are based on ﬁring-rate descriptions of neural activity. To encompass all the models, we
write the network equations for the generator network as (note that, in the Results, we called the parameter labeled
here as g
 simply g)
for i =1,2,K ,N  with ﬁring rates r  = tanh(x ). For the feedback network,
for a =1,2,K,N  with ﬁring rates s  = tanh(y  ). Equation 1 determines z and τ =10 ms. Sometimes we assign a
sparseness p  to the readout unit, meaning that a random fraction 1- p of the components of w are set and held to
zero. The connection matrices are also assigned sparseness parameters, p, meaning that each element is set and
held to 0 with probability 1-p . Nonzero elements of J
, J
, J
, J
 are drawn independently from Gaussian
distributions with zero means and variances equal to the inverses of p
 N , p
 N , p
 N  and p
 N , respec‐
tively. Rows of J  and J  have a single non-zero element drawn from a Gaussian distribution with zero mean and
unit variance. Elements of J
 are drawn from a uniform distribution between -1 and 1. Nonzero elements of w are
set initially either to zero or to values generated by a Gaussian distribution with zero mean and variance 1/(p N).
For figures 2-5 and 6A
N  = 1000, p
 = 0.1, p  =1, g
 = 1, g
 = 0, α = 1.0, and N  = 0. For ﬁgure 5, g
 was varied, otherwise g
= 1.5 . For ﬁgure 2H, the standard Lorenz attractor model (see Strogatz, 1994) was used with σ = 10, β = 8/3, and
ρ = 28. The target function was what is conventionally labeled as x divided by 10, to ﬁt it roughly into the range of
-1 to 1. For ﬁgure 2J, the two readouts and feedback loops are similar except for diﬀerent random choices for the
strengths of the feedback synapses onto the network neurons. The additional readout unit takes two possible states,
one called active in which its output is determined by equation 1, and another called inactive in which its output is
GG
𝜏
= −
+
+
𝑍+
𝑑𝑥𝑖
𝑑𝑡
𝑥𝑖
𝑔𝐺𝐺
𝑁𝐺
𝑗=1𝐽𝐺𝐺
𝑖𝑗𝑟𝑗
𝑔𝐺𝑧𝐽𝐺𝑧
𝑖
𝑔𝐺𝐹
𝑁𝐹
𝑎=1𝐽𝐺𝐹
𝑖𝑎𝑆𝑎+𝑁𝐼
𝜇=1 𝐽𝐺𝐼
𝑖𝜇𝐼𝜇
G
i
i
𝜏
= −
+
+
𝑑𝑦𝑎
𝑑𝑡
𝑦𝑎
𝑔𝐹𝐹
𝑁𝐹
𝑏=1𝐽𝐹𝐹
𝑎𝑏𝑆𝑏
𝑔𝐹𝐺
𝑁𝐺
𝑖=1𝐽𝐹𝐺
𝑎𝑖𝑟𝑖+𝑁𝐼
𝜇=1 𝐽𝐹𝐼
𝑎𝜇𝐼𝜇
F
a
a
z
z
GG
GF
FG
FF
GG
G
GF
F
FG
G
FF
F
GI
FI
Gz
z
G
GG
z
Gz
GF
I
GG
GG
0 . For further discussion of training in this case, see the Supplementary Materials.
For figures 6B
N  = 20,000, N  = 95, p
 = 0.1, p
 = 0.25, p
 = 0.025, p
 = 0.25, p  = 0.025, g
 =1.5, g
 =1, g
 =1,
g
 =1.2, α =1.0, and N  = 0. RLS modiﬁcation was applied to w and J
 .
For figures 6C
N  = 750, p
 = 0.5, p  = 0.5, g
 = 1.5, g
 = 0, α = 1.0, and N  = 0. RLS modiﬁcation was applied to w and
J
 .
For figure 7B
N  =1200, p
 = 0.8, p  = 0.8, g
 = 1.5, g
 = 0, α = 80, and N  = 100. The inputs I  where chosen randomly
and uniformly over the range -2 to 2 for inputs generating initialization ﬁxed points, and -0.5 to 0.5 for inputs con‐
trol the choice of output function. RLS modiﬁcation was applied to w and J
, but of the 1200 network neurons,
800 were subject to synaptic modiﬁcation of their incoming synapses (due to memory considerations).
For figure 7D
N  =1200, p
 = 0.8, p  =1, g
 = 1, g
 = 0, α = 40, and N  = 8. The elements of the control input vector I  had
OFF values of 0.0 and ON values of 0.375. RLS modiﬁcation was applied to w and J
 , with 800 of the network
neurons subject to synaptic modiﬁcation of their incoming synapses.
For figure 8
Although all network variants in ﬁgure 1 were implemented successfully, the following parameters are for the
generator-feedback architecture: N  = 5000, N  = 285, p
 = 0.05, p
 = 0.5, p
 = 0.185, p
 = 0.5, p  = 0.185,
g
 = 1.5, g
 = 2.0, g
 = 1.0, g
 = 1.5, α = 2.0, and N  = 50. RLS modiﬁcation was applied to w and J
 .
Motion capture data were downloaded from the Carnegie Mellon University Motion Capture Library (MOCAP)
(http://mocap.cs.cmu.edu/). Data set 09_02.amc was used for the running example and data set 08_01.amc for the
walking case. The data were preprocessed by a simple moving average ﬁlter to remove discontinuities and then in‐
terpolated to ﬁll in to 10 times density, which works better for our continuous time models. The resulting joint an‐
gles were transformed into exponential form (see Taylor et al., 2006) and the means were removed. Movement
through space was ignored, so we modeled a runner or walker on a treadmill. Four sets of control inputs were used,
one each for running and walking and two for initial-value ﬁxed points for these motions. Fixed-point inputs were
chosen randomly and uniformly over the range -2 to 2 and control inputs over -0.25 to 0.25.
Supplementary Material
01
Click here to view.
G
F
GG
GF
FG
FF
z
GG
GF
FG
FF
I
FG
G
GG
z
GG
GF
I
GG
G
GG
z
GG
GF
I
μ
GG
G
GG
z
GG
GF
I
μ
GG
G
F
GG
GF
FG
FF
z
GG
GF
FG
FF
I
FG
(686K, pdf)
02
Click here to view.
03
Click here to view.
Acknowledgments
Research supported by an NIH Director’s Pioneer Award, part of the NIH Roadmap for Medical Research, through
grant number 5-DP1-OD114-02 and by National Institute of Mental Health grant MH-58754. We thank Taro
Toyoizumi, Surya Ganguli, Greg Wayne, and Graham Taylor for helpful comments and suggestions.
Footnotes
Publisher's Disclaimer: This is a PDF ﬁle of an unedited manuscript that has been accepted for publication. As a service to our
customers we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and re‐
view of the resulting proof before it is published in its ﬁnal citable form. Please note that during the production process errors
may be discovered which could aﬀect the content, and all legal disclaimers that apply to the journal pertain.
References
1. Abarbanel HD, Creveling DR, Jeanne JM. Estimation of parameters in nonlinear systems using balanced synchronization. Phys. Rev. E Stat.
Nonlin. Soft. Matter Phys. 2008;77:016208. [PubMed] [Google Scholar]
2. Atiya AF, Parlos AG. New results on recurrent network training: Unifying the algorithms and accelerating convergence. IEEE Transactions
on Neural Networks. 2000;11:697–709. [PubMed] [Google Scholar]
3. Amit DJ, Brunel N. Model of global spontaneous activity and local structured activity during delay periods in the cerebral cortex. Cereb.
Cortex. 1997;7:237–252. [PubMed] [Google Scholar]
4. Bertchinger N, Natschläger T. Real-time computation at the edge of chaos in recurrent neural networks. Neural Comput. 2004;16:1413–
1436. [PubMed] [Google Scholar]
5. Brunel N. Dynamics of networks of randomly connected excitatory and inhibitory spiking neurons. J. Physiol. Paris. 2000;94:445–463.
[PubMed] [Google Scholar]
6. Buonomano DV, Maass W. State-dependent computations: spatiotemporal processing in cortical networks. Nat. Rev. Neurosci. 2009;10:113–
125. [PubMed] [Google Scholar]
7. Buonomano DV, Merzenich MM. Temporal information transformed into a spatial code by a neural network with realistic properties.
Science. 1995;267:1028–1030. [PubMed] [Google Scholar]
8. Churchland MM, Shenoy KV. Delay of movement caused by disruption of cortical preparatory activity. J. Neurophysiol. 2007a;9:348–359.
[PubMed] [Google Scholar]
9. Churchland MM, Shenoy KV. Temporal Complexity and Heterogeneity of Single-Neuron Activity in Premotor and Motor Cortex. J.
Neurophysiol. 2007b;97:4235–4257. [PubMed] [Google Scholar]
(5.5K, zip)
(7.3M, avi)
10. Churchland MM, Yu BM, Ryu SI, Santhanam G, Shenoy KV. Neural variability in premotor cortex provides a signature of motor
preparation. J. Neurosci. 2006;26:3697–3712. [PMC free article] [PubMed] [Google Scholar]
11. Doya K. Bifurcations in the learning of recurrent neural networks; IEEE International Symposium on Circuits and Systems; 1992.pp. 2777–
2780. [Google Scholar]
12. Fetz E. Are movement parameters recognizably coded in the activity of single neurons? Behavioral and brain sciences. 1992;15:679–690.
[Google Scholar]
13. Ganguli S, Huh D, Sompolinsky H. Memory traces in dynamical systems. Proc . Natl. Acad. Sci. USA. 2008;105:18970–18975. [PMC free
article] [PubMed] [Google Scholar]
14. Haykin S. Adaptive Filter Theory. Prentice Hall; Upper Saddle River NJ: 2002. [Google Scholar]
15. Hinton GE, Osindero S, Teh YW. A fast learning algorithm for deep belief nets. Neural Comput. 2006;18:1527–1554. [PubMed] [Google
Scholar]
16. Jaeger H. Adaptive nonlinear system identiﬁcation with echo state networks. In: Becker S, Thrun S, Obermayer K, editors. Advances in
Neural Information Processing Systems 15. MIT Press; Cambridge MA: 2003. pp. 593–600. [Google Scholar]
17. Jaeger H, Haas H. Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication. Science. 2004;304:78–
80. [PubMed] [Google Scholar]
18. Maass W, Joshi P, Sontag ED. Computational aspects of feedback in neural circuits. PLoS Comput, Biol. 2007;3:e165. [PMC free article]
[PubMed] [Google Scholar]
19. Maass W, Matschlager T, Markram H. Real-time computing without stable states: a new framework for neural computation based on
perturbations. Neural Comput. 2002;14:2531–2560. [PubMed] [Google Scholar]
20. Miall RC, Weir DJ, Wolpert DM, Stein JF. Is the cerebellum a smith predictor? J. Motor Behav. 1993;25:203–216. [PubMed] [Google
Scholar]
21. Molgedey L, Schuchhardt J, Schuster HG. Suppressing chaos in neural networks by noise. Phys. Rev. Lett. 1992;69:3717–3719. [PubMed]
[Google Scholar]
22. Pearlmutter B. Learning state space trajectories in recurrent neural networks. Neural Comput. 1989;1:263–269. [Google Scholar]
23. Rajan K, Abbott LF, Sompolinsky H. Stimulus-Dependent Suppression of Intrinsic Variability in Recurrent Neural Networks. 2008.
submitted. [PMC free article] [PubMed] [Google Scholar]
24. Robinson D. Implications of neural networks for how we think about brain function. Behavioral and Brain Sciences. 1992;15:644–655.
[Google Scholar]
25. Rumelhart DE, Hinton GE, Williams RJ. Learning internal representations by error propagation. In: Rumelhart DE, McClelland JL, editors.
Parallel Distributed Processing: Explorations in the Microstructure of Cognition. Vol. 1, Foundations. MIT Press; Cambridge MA: 1986.
1986. chapter 8. [Google Scholar]
26. Rumelhart DE, McClelland JL, editors. Parallel Distributed Processing: Explorations in the Microstructure of Cognition. Vol. 1,
Foundations. MIT Press; Cambridge MA: 1986. [Google Scholar]
27. Sompolinsky H, Crisanti A, Sommers HJ. Chaos in Random Neural Networks. Phys. Rev. Lett. 1988;61:259–262. [PubMed] [Google
Scholar]
28. Sussillo D. Columbia University Ph.D. Thesis. 2009. Learning in chaotic recurrent networks; pp. 93–102. [Google Scholar]
29. Taylor GW, Hinton GE, Roweis S. Advances in Neural Information Processing Systems. MIT Press; Cambridge MA: 2006. Modeling
human motion using binary latent variables. [Google Scholar]
30. van Vreeswijk C, Sompolinsky H. Chaos in neuronal networks with balanced excitatory and inhibitory activity. Science. 1996;274:1724–
1726. [PubMed] [Google Scholar]
31. Williams RJ, Zipser D. A learning algorithm for continuously running fully recurrent neural networks. Neural Comput. 1989;1:270–280.
[Google Scholar]
32. Yuste R, MacLean JN, Smith J, Lansner A. The cortex as a central pattern generator. Nature Rev. Neurosci. 2005;6:477–483. [PubMed]
[Google Scholar]
