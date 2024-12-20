arXiv:1712.04323v4  [cs.LG]  25 Sep 2020
Deep Echo State Network (DeepESN):
A Brief Survey
Claudio Gallicchio and Alessio Micheli
Department of Computer Science, University of Pisa
gallicch@di.unipi.it, micheli@di.unipi.it
Abstract
The study of deep recurrent neural networks (RNNs) and, in particu-
lar, of deep Reservoir Computing (RC) is gaining an increasing research
attention in the neural networks community.
The recently introduced
Deep Echo State Network (DeepESN) model opened the way to an ex-
tremely eﬃcient approach for designing deep neural networks for tempo-
ral data. At the same time, the study of DeepESNs allowed to shed light
on the intrinsic properties of state dynamics developed by hierarchical
compositions of recurrent layers, i.e. on the bias of depth in RNNs ar-
chitectural design. In this paper, we summarize the advancements in the
development, analysis and applications of DeepESNs.
Keywords: Deep Echo State Network, DeepESN, Reservoir Computing,
Echo State Networks, Recurrent Neural Networks, Deep Learning, Deep
Neural Networks
1
Introduction
In the last decade, the Reservoir Computing (RC) paradigm [1, 2] has attested
as a state-of-the-art approach for the design of eﬃciently trained Recurrent Neu-
ral Networks (RNNs). Though diﬀerent instances of the RC methodology exist
in literature (see e.g. [3, 4]), the Echo State Network (ESN) [5, 6] certainly
represents the most widely known model, with a strong theoretical ground (e.g.
[7, 8, 9, 10, 11, 12]) and a plethora of successful applications reported in litera-
ture (see e.g. [13, 14] and references therein, as well as more recent works e.g.
in [15, 16, 17]). Essentially, ESNs are recurrent randomized neural networks
[18, 19] in which the state dynamics are implemented by an untrained recurrent
hidden layer, whose activation is used to feed a static output module that is
the only trained part of the network. In this paper we deal with the extension
of the ESN approach to the deep learning framework. This line of research can
be interestingly framed within the context of deep randomized neural networks
[20], in which the analysis is focused on the behavior of deep neural architectures
where most of the connections are untrained.
1
The study of deep neural network architectures for temporal data processing
is an attractive area of research in the neural networks community [21, 22].
Investigations in the ﬁeld of hierarchically organized Recurrent Neural Networks
(RNNs) showed that deep RNNs are able to develop in their internal states a
multiple time-scales representation of the temporal information, a much desired
feature e.g.
when approaching complex tasks in the area of speech or text
processing [23, 24].
Recently, the introduction of the Deep Echo State Network (DeepESN)
model [25, 26] allowed to study the properties of layered RNN architectures
separately from the learning aspects.
Remarkably, such studies pointed out
that the structured state space organization with multiple time-scales dynamics
in deep RNNs is intrinsic to the nature of compositionality of recurrent neural
modules. The interest in the study of the DeepESN model is hence twofold. On
the one hand, it allows to shed light on the intrinsic properties of state dynamics
of layered RNN architectures [27]. On the other hand it enables the design of
extremely eﬃciently trained deep neural networks for temporal data.
Previous to the explicit introduction of the DeepESN model in [25], works
on hierarchical RC models targeted ad-hoc constructed architectures, where
diﬀerent modules were trained for discovery of temporal features at diﬀerent
scales on synthetic data [28]. Ad-hoc constructed modular networks made up
of multiple ESN modules have also been investigated in the speech processing
area [29, 30]. More recently, the advantages of multi-layered RC networks have
been experimentally studied on time-series benchmarks in the RC area [31].
Diﬀerently from the above mentioned works, the studies on DeepESN considered
in the following aim to address some fundamental questions pertaining to the
true nature of layering as a factor of architectural RNN design [27]. Such basic
questions can be essentially summarized as follows: (i) Why stacking layers of
recurrent units?
(ii) What is the inherent architectural eﬀect of layering in
RNNs (independently from learning)? (iii) Can we extend the advantages of
depth in RNN design using eﬃciently trained RC approaches? (iv) Can we
exploit the insights from such analysis to address the automatic design of deep
recurrent models (including fundamental parameters such as the architectural
form, the number of layers, the number of units in each layer, etc.)?
This paper is intended both to draw a line of recent developments in response
to the above mentioned key research questions and to provide an up-to-date
overview on the progress and on the perspectives in the studies of DeepESNs,
which are presented in Section 3. Before that, in Section 2 we recall the major
characteristics of the DeepESN model.
2
The Deep Echo State Network Model
As for the standard shallow ESN model, a DeepESN is composed by a dynamical
reservoir component, which embeds the input history into a rich state repre-
sentation, and by a feed-forward readout part, wich exploits the state econding
provided by the reservoir to compute the output. Crucially, the reservoir of a
2
DeepESN is organized into a hierarchy of stacked recurrent layers, where the
output of each layer acts as input for the next one, as illustrated in Figure 1.
Figure 1: Reservoir architecture of a Deep Echo State Network.
In this case, at each time step t, the state computation proceeds by following
the pipeline of recurrent layers, from the ﬁrst one, which is directly fed by
the external input, up to the highest one in the reservoir architecture. In our
notation we use NU to denote the external input dimension, NL to indicate
the number of reservoir layers, and we assume, for the sake of simplicity, that
each reservoir layer has NR recurrent units.
Moreover, we use u(t) ∈RNU
to denote the external input at time step t, while x(i)(t) ∈RNR is the state
of the reservoir layer i at time step t. In general, we use the superscript (i)
to indicate that an item is related to the i-th reservoir in the stack. At each
time step t, the composition of the states in all the reservoir layers, i.e. x(t) =
 x(1)(t), . . . , x(NL)(t)

∈RNR NL, gives the global state of the network.
The computation carried out by the stacked reservoir of a DeepESN can be
understood under a dynamical system viewpoint as an input-driven discrete-
3
time non-linear dynamical system, where the evolution of the global state x(t)
is governed by a state transition function F =
 F (1), . . . , F (NL)
, with each
F (i) ruling the state dynamics at layer i. Assuming leaky integrator reservoir
units [32] in each layer and omitting the bias terms for the ease of notation, the
reservoir dynamics of a DeepESN are mathematically described as follows. For
the ﬁrst layer we have that:
x(1)(t)
= F(u(t), x(1)(t −1))
= (1 −a(1))x(1)(t −1) + a(1)f(W(1)u(t) + ˆ
W(1)x(1)(t −1)),
(1)
while for successive layers i > 1 the state update is given by:
x(i)(t)
= F(x(i−1)(t), x(i)(t −1))
= (1 −a(i))x(i)(t −1) + a(i)f(W(i)x(i−1)(t) + ˆ
W(i)x(i)(t −1)).
(2)
In the above equations 1 and 2, W(1) ∈RNR×NU is the input weight matrix,
W(i) ∈RNR×NR for i > 1 is the weight matrix for inter-layer connections from
layer (i −1) to layer i, ˆ
W(i) ∈RNR×NR is the recurrent weight matrix for layer
i, a(i) ∈[0, 1] is the leaking rate for layer i and f denotes the element-wise
applied activation function for the recurrent reservoir units (typically, the tanh
non-linearity is used).
Interestingly, as graphically illustrated in Figure 2, we can observe that
the reservoir architecture of a DeepESN can be characterized, with respect to
the shallow counterpart, by interpreting it as a constrained version of standard
shallow ESN/RNN with the same total number of recurrent units. In particular,
the following constraints are applied in order to obtain a layered architecture:
• all the connections from the input layer to reservoir layers at a level higher
than 1 are removed (inﬂuencing the way in which the external input in-
formation is seen by recurrent units progressively more distant from the
input layer);
• all the connections from higher layers to lower ones are removed (which
aﬀects the ﬂow of information and the dynamics of sub-parts of the net-
work’s state);
• all the connections from each layer to higher layers diﬀerent from the
immediately successive one in the pipeline are removed (which aﬀects the
ﬂow of information and the dynamics of sub-parts of the network’s state).
The above mentioned constraints, that graphically correspond to layering, have
been explicitly and extensively discussed in our previous work in [25]. Under this
point of view, the DeepESN architecture can be seen as a simpliﬁcation of the
corresponding single-layer ESN, leading to a reduction in the absolute number
of recurrent weights which, assuming full-connected reservoirs at each layer, is
quadratic in both the number of recurrent units per layer and total number of
layers [33]. As detailed in the above points, however, note that this peculiar
architectural organization inﬂuences the way in which the temporal information
4
is processed by the diﬀerent sub-parts of the hierarchical reservoir, composed
by recurrent units that are progressively more distant from the external input.
Furthermore, diﬀerently from the case of a standard ESN/RNN, the state
information transmission between consecutive layers in a DeepESN presents no
temporal delays. In this respect, we can make the following considerations:
• the aspect of sequentiality between layers operation is already present and
discussed in previous works in literature on deep RNN (see e.g. [24, 23,
34, 35]), which actually stimulated the investigation on the intrinsic role of
layering in such hierarchically organized recurrent network architectures;
• this choice allows the model to process the temporal information at each
time step in a “deep” temporal fashion, i.e. through a hierarchical com-
position of multiple levels of recurrent units;
• in particular, notice that the use of (hyperbolic tangent) non-linearities
applied individually to each layer during the state computation does not
allow to describe the DeepESN dynamics by means of an equivalent shal-
low system.
Based on the above observations, a major research question naturally arises
and drives the motivation to the studies reported in Section 3, i.e. how and to
what extent the described constraints that rule the layered construction and the
hierarchical representation in deep recurrent models have an inﬂuence on their
dynamics.
As in the standard RC framework, the reservoir parameters, i.e. the weights
in matrices W(i) and ˆ
W(i), are left untrained after initialization under stability
constraints, which are given through the analysis of the Echo State Property
for deep reservoirs provided in [36].
As regards the output computation, although diﬀerent choices are possible
for the pattern of connectivity between the reservoir layers and the output
module (see e.g. [24, 37]), a typical setting consists in feeding at each time
step t the state of all reservoir layers (i.e. the global state of the DeepESN)
to the output layer, as illustrated in Figure 3. Note that this choice enables
the readout component to give diﬀerent weights to the dynamics developed
at diﬀerent layers, thereby allowing to exploit the potential variety of state
representations in the stacked reservoir. Denoting by NY the size of the output
space, in the typical case of linear readout, the output at time step t is computed
as:
y(t) = Woutx(t) = Wout
 x(1)(t), . . . , x(NL)(t)

,
(3)
where Wout ∈RNL×NR NL is the readout weight matrix that is adapted on a
training set, typically in closed form through direct methods such as pseudo-
inversion or ridge-regression.
5
Figure 2: The layered reservoir architecture of DeepESN as a constrained version
of a shallow reservoir. Compared to the shallow case with the same total number
of recurrent units, in a stacked DeepESN architecture the following connections
are removed: from the input to reservoir levels at height > 1 (blue dashed
arrows), from higher to lower reservoir levels (green dash dotted arrows), from
each reservoir at level i to all reservoirs at levels > i+1 (orange dotted arrows).
3
Advances
Here we brieﬂy survey the recent advances in the study of the DeepESN model.
The works described in the following, by addressing the key questions summa-
rized in the Introduction, provide a general support to the signiﬁcance of the
DeepESN, also critically discussing advantages and drawbacks of its construc-
tion.
• The DeepESN model has been introduced in [25], which extends the pre-
liminary work in [26].
The analysis provided in these papers revealed,
through empirical investigations, the hierarchical structure of temporal
data representations developed by the layered reservoir architecture of
a DeepESN. Speciﬁcally, the stacked composition of recurrent reservoir
layers was shown to enable a multiple time-scales representation of the
temporal information, naturally ordered along the network’s hierarchy.
Besides, in [25] layering proved eﬀective also as a way to enhance the eﬀect
of known RC factors of network design, including unsupervised reservoir
adaptation by means of Intrinsic Plasticity [38]. The resulting eﬀects have
6
Figure 3: Readout organization for DeepESN in which at each time step the
reservoir states of all layers are used as input for the output layer.
been analyzed also in terms of state entropy and memory.
• The hierarchically structured state representation in DeepESNs has been
investigated by means of frequency analysis in [39], which speciﬁcally con-
sidered the case of recurrent units with linear activation functions. Results
pointed out the intrinsic multiple frequency representation in DeepESN
states, where, even in the simpliﬁed linear setting, progressively higher
layers focus on progressively lower frequencies. In [39] the potentiality of
the deep RC approach has also been exploited in predictive experiments,
showing that DeepESNs outperform state-of-the-art results on the class of
Multiple Superimposed Oscillator (MSO) tasks by several orders of mag-
nitude.
• The fundamental RC conditions related to the Echo State Property (ESP)
have been generalized to the case of deep RC networks in [36]. Speciﬁ-
cally, through the study of stability and contractivity of nested dynamical
systems, the theoretical analysis in [36] gives a suﬃcient condition and a
necessary condition for the Echo State Property to hold in case of deep
RNN architectures. Remarkably, the work in [36] provides a relevant con-
ceptual and practical tool for the deﬁnition, validity and usage of DeepESN
7
in an “autonomous” way with respect to the standard ESN model.
• The study of DeepESN dynamics under a dynamical system perspective
has been pursued in [33, 40], which provide a theoretical and practical
framework for the study of stability of layered recurrent dynamics in terms
of local Lyapunov exponents. This study also provided interesting insights
in terms of the quality of the developed system dynamics, showing that
(under simple initialization settings) layering has the eﬀect of naturally
pushing the global dynamical regime of the recurrent network closer to the
stable-unstable transition condition known as the edge of chaos [41, 42, 43].
• The study of the frequency spectrum of deep reservoirs enabled to ad-
dress one of the fundamental open issues in deep learning, namely how to
choose the number of layers in a deep RNN architecture. Starting from
the analysis of the intrinsic diﬀerentiation of the ﬁltering eﬀects of suc-
cessive levels in a stacked RNN architecture, the work in [44] proposed an
automatic method for the design of DeepESNs. Noticeably, the proposed
approach allows to tailor the DeepESN architecture to the characteristics
of the input signals, consistently relieving the cost of the model selection
process, and leading to new state-of-the-art results in speech and music
processing tasks.
• A ﬁrst extension of the deep RC framework for learning in structured do-
mains has been presented in [45, 46], which introduced the Deep Tree Echo
State Network (DeepTESN) model. The new model points out that it is
possible to combine the concepts of deep learning, learning for trees and
RC training eﬃciency, taking advantages from the layered architectural
organization and from the compositionality of the structured representa-
tions both in terms of eﬃciency and in terms of eﬀectiveness. On the
application side, the experimental results reported in [45, 46] concretely
showed that the deep RC approach for trees can be extremely advanta-
geous, beating previous state-of-the-art results in challenging tasks from
domains of document processing and computational biology. As regards
the mathematical description of the model, the reservoir operation is ex-
tended to implement a (non-linear) state transition system over discrete
tree structures, whose asymptotic stability analysis enables the deﬁnition
of a generalization of the ESP for the case of tree structured data [45].
Overall, DeepTESN provides a ﬁrst instance of an extremely eﬃcient ap-
proach for the design of deep neural networks for learning in cases where
the input is given by (hierarchically-)structured data. Moreover, from a
theoretical viewpoint, the work in [45] also gives an in-depth analysis of
asymptotic stability of untrained (and non-linear) state transition systems
operating on discrete trees. In this context, the analysis in [45] results into
a generalization of the ESP of conventional reservoirs, described under the
name of Tree Echo State Property.
The Deep RC approach has been proved very advantageous also in the case
of learning with graph data, enabling the development of Fast and Deep
8
Graph Neural Networks (FDGNNs) in [47]. The concept of reservoirs op-
erating on discrete graph structures has been ﬁrst introduced in [48], and
revolves around the computation of a state embedding for each vertex in
an input graph. In particular, the state for a vertex v is computed as
a function of the input information attached to the vertex v itself (i.e.,
a vector of features that takes the role of external input in the system),
and of the state computed for the neighbors of v (a concept that takes
the role of “previous time-step” in the case of conventional RC systems
for time-series). The stability of the resulting dynamics can be studied by
generalizing the mathematical means considered for conventional reser-
voirs, leading to the deﬁnition of Graph Embedding Stability (GES), a
stability property for neural embedding systems on graphs introduced in
[47], to which the interested reader is referred for further information. Be-
sides the introduction of GES, the work in [47] shows how to design a deep
RC system for graphs, where each layer builds its embedding on the ba-
sis of the state information produced in the previous layer. The FDGNN
approach was shown to reach (and even outperform) state-of-the-art accu-
racy on known benchmarks for graph classiﬁcation, comparing well with
many literature approaches, especially based on convolutional neural net-
works and kernel for graphs. Inheriting the easy of training algorithms
from the RC paradigm, the approach is also extremely faster than liter-
ature models, enabling a sensible speed-up in the required training times
(up to ≈3 orders of magnitude in the experiments reported in [47]).
• For what regards the experimental analysis in applications, DeepESNs
were shown to bring several advantages in both cases of synthetic and
real-world tasks. Speciﬁcally, DeepESNs outperformed shallow reservoir
architectures (under fair conditions on the number of total recurrent units
and, as such, on the number of trainable readout parameters) on the
Mackey-Glass next-step prediction task [27], on the short-term Memory
Capacity task [25, 49], on MSO tasks [39], as well as on a Frequency Based
Classiﬁcation task [44], purposely designed to assess multiple-frequency
representation abilities. As pertains to real-world problems, the DeepESN
approach recently proved eﬀective in a variety of domains, including Am-
bient Assisted Living (AAL) [50], medical diagnosis [51], speech and poly-
phonic music processing [44, 52], metereological forecasting [53, 54], solar
irradiance prediction [55], energy consumption and wind power generation
prediction [56], short-term traﬃc forecasting [57], destination prediction
[58] car parking and bike-sharing in urban computing [54], ﬁnancial mar-
ket predictions [54], and industrial applications (for blast furnace oﬀ-gas)
[59, 60].
• Software implementations of the DeepESN model have been recently made
publicly available in the following forms:
– DeepRC TensorFlow Library (DeepRC-TF)
https://github.com/gallicch/DeepRC-TF.
9
– DeepESN Python Library (DeepESNpy)
https://github.com/lucapedrelli/DeepESN.
– Deep Echo State Network (DeepESN) MATLAB Toolbox
https://it.mathworks.com/matlabcentral/fileexchange/69402-deepesn.
– Deep Echo State Network (DeepESN) Octave library
https://github.com/gallicch/DeepESN_octave.
Please note that references [25, 44] represent citation requests for the use
of the above mentioned libraries.
4
Conclusions
In this survey we have provided a brief overview of the extension of the RC
approach towards the deep learning framework, describing the salient features of
the DeepESN model. Noticeably, DeepESNs enable the analysis of the intrinsic
properties of state dynamics in deep RNN architectures, i.e. the study of the
bias due to layering in the design of RNNs.
At the same time, DeepESNs
allow to transfer the striking advantages of the ESN methodology to the case of
deep recurrent architectures, leading to an eﬃcient approach for designing deep
neural networks for temporal data.
The analysis of the distinctive characteristics and dynamical properties of
the DeepESN model has been carried out ﬁrst empirically, in terms of entropy of
state dynamics and system memory. Then, it has been conducted through more
abstract theoretical investigations that allowed the derivation of the fundamen-
tal conditions for the ESP of deep networks, as well as the characterization of
the developed dynamical regimes in terms of local Lyapunov exponents. Be-
sides, studies on the frequency analysis of DeepESN dynamics allowed us to
develop an algorithm for the automatic setup of (the number of layers of) a
DeepESN. Current developments already include model variants and applica-
tions to both synthetic and real-world tasks. Finally, a pioneering extension of
the deep RC approach to learning in structured domains has been introduced
with DeepTESN (for trees) and FDGNN (for graphs).
Overall, the ﬁnal aim of this paper is to summarize the successive advances
in the development, analysis and applications of the DeepESN model, providing
a document that is intended to contain a constantly updated view over this
research topic.
References
[1] D. Verstraeten, B. Schrauwen, M. d’Haene, D. Stroobandt, An experimen-
tal uniﬁcation of reservoir computing methods, Neural networks 20 (3)
(2007) 391–403.
[2] M. Lukoˇseviˇcius, H. Jaeger, Reservoir computing approaches to recurrent
neural network training, Computer Science Review 3 (3) (2009) 127–149.
10
[3] W. Maass, T. Natschl¨ager, H. Markram, Real-time computing without sta-
ble states: A new framework for neural computation based on perturba-
tions, Neural computation 14 (11) (2002) 2531–2560.
[4] P. Tiˇno, G. Dorﬀner, Predicting the future of discrete sequences from fractal
representations of the past, Machine Learning 45 (2) (2001) 187–217.
[5] H. Jaeger, H. Haas, Harnessing nonlinearity: Predicting chaotic systems
and saving energy in wireless communication, Science 304 (5667) (2004)
78–80.
[6] H. Jaeger, The ”echo state” approach to analysing and training recurrent
neural networks - with an erratum note, Tech. rep., GMD - German Na-
tional Research Institute for Computer Science, Tech. Rep. (2001).
[7] P. Tiˇno, B. Hammer, M. Bod´en, Markovian bias of neural-based archi-
tectures with feedback connections, in: Perspectives of neural-symbolic
integration, Springer, 2007, pp. 95–133.
[8] C. Gallicchio, A. Micheli, Architectural and markovian factors of echo state
networks, Neural Networks 24 (5) (2011) 440–456.
[9] I. B. Yildiz, H. Jaeger, S. J. Kiebel, Re-visiting the echo state property,
Neural networks 35 (2012) 1–9.
[10] G. Manjunath, H. Jaeger, Echo state property linked to an input: Ex-
ploring a fundamental characteristic of recurrent neural networks, Neural
computation 25 (3) (2013) 671–696.
[11] M. Massar, S. Massar, Mean-ﬁeld theory of echo state networks, Physical
Review E 87 (4) (2013) 042809.
[12] P. Tiˇno, Fisher memory of linear wigner echo state networks, in: Pro-
ceedings of the 25th European Symposium on Artiﬁcial Neural Networks
(ESANN), i6doc.com, 2017, pp. 87–92.
[13] M. Lukoˇseviˇcius, H. Jaeger, B. Schrauwen, Reservoir computing trends,
KI-K¨unstliche Intelligenz 26 (4) (2012) 365–371.
[14] B. Schrauwen, D. Verstraeten, J. Van Campenhout, An overview of reser-
voir computing: theory, applications and implementations, in: Proceedings
of the 15th European Symposium on Artiﬁcial Neural Networks. p. 471-482
2007, 2007, pp. 471–482.
[15] F. Palumbo, C. Gallicchio, R. Pucci, A. Micheli, Human activity recogni-
tion using multisensor data fusion based on reservoir computing, Journal
of Ambient Intelligence and Smart Environments 8 (2) (2016) 87–107.
[16] E. Crisostomi, C. Gallicchio, A. Micheli, M. Raugi, M. Tucci, Prediction
of the italian electricity price for smart grid applications, Neurocomputing
170 (2015) 286–295.
11
[17] D. Bacciu, P. Barsocchi, S. Chessa, C. Gallicchio, A. Micheli, An experi-
mental characterization of reservoir computing in ambient assisted living
applications, Neural Computing and Applications 24 (6) (2014) 1451–1464.
[18] C. Gallicchio, A. Micheli, P. Tiˇno, Randomized Recurrent Neural Networks,
in: Proceedings of the 26th European Symposium on Artiﬁcial Neural Net-
works (ESANN), i6doc.com, 2018, pp. 415–424.
[19] C. Gallicchio, J. D. Martin-Guerrero, A. Micheli, E. Soria-Olivas, Random-
ized machine learning approaches: Recent developments and challenges, in:
Proceedings of the 25th European Symposium on Artiﬁcial Neural Net-
works (ESANN), i6doc.com, 2017, pp. 77–86.
[20] C. Gallicchio, S. Scardapane, Deep randomized neural networks, in: Recent
Trends in Learning From Data, Springer, 2020, pp. 43–68.
[21] P. Angelov, A. Sperduti, Challenges in deep learning, in: Proceedings of
the 24th European Symposium on Artiﬁcial Neural Networks (ESANN),
i6doc.com, 2016, pp. 489–495.
[22] I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT press, 2016.
[23] A. Graves, A.-R. Mohamed, G. Hinton, Speech recognition with deep re-
current neural networks, in: IEEE International Conference on Acoustics,
speech and signal processing (ICASSP), IEEE, 2013, pp. 6645–6649.
[24] M. Hermans, B. Schrauwen, Training and analysing deep recurrent neural
networks, in: NIPS, 2013, pp. 190–198.
[25] C.
Gallicchio,
A.
Micheli,
L.
Pedrelli,
Deep
reservoir
computing:
A critical experimental analysis, Neurocomputing 268 (2017) 87–99.
doi:https://doi.org/10.1016/j.neucom.2016.12.089.
[26] C. Gallicchio, A. Micheli, Deep reservoir computing: A critical analysis,
in: Proceedings of the 24th European Symposium on Artiﬁcial Neural Net-
works (ESANN), i6doc.com, 2016, pp. 497–502.
[27] C. Gallicchio, A. Micheli, Why layering in Recurrent Neural Networks? a
DeepESN survey, in: Proceedings of the 2018 IEEE International Joint
Conference on Neural Networks (IJCNN), IEEE, 2018, pp. 1800–1807.
[28] H. Jaeger, Discovering multiscale dynamical features with hierarchical echo
state networks, Tech. rep., Jacobs University Bremen (2007).
[29] F. Triefenbach, A. Jalalvand, K. Demuynck, J.-P. Martens, Acoustic mod-
eling with hierarchical reservoirs, IEEE Transactions on Audio, Speech,
and Language Processing 21 (11) (2013) 2439–2450.
[30] F. Triefenbach, A. Jalalvand, B. Schrauwen, J.-P. Martens, Phoneme recog-
nition with large hierarchical reservoirs, in: Advances in neural information
processing systems, 2010, pp. 2307–2315.
12
[31] Z. K. Malik, A. Hussain, Q. J. Wu, Multilayered echo state machine: a
novel architecture and algorithm, IEEE Transactions on cybernetics 47 (4)
(2017) 946–959.
[32] H. Jaeger, M. Lukoˇseviˇcius, D. Popovici, U. Siewert, Optimization and
applications of echo state networks with leaky-integrator neurons, Neural
Networks 20 (3) (2007) 335–352.
[33] C. Gallicchio, A. Micheli, L. Silvestri, Local lyapunov exponents of deep
echo state networks, Neurocomputing 298 (2018) 34–45.
[34] S. El Hihi, Y. Bengio, Hierarchical recurrent neural networks for long-
term dependencies, in: Advances in neural information processing systems
(NIPS), 1996, pp. 493–499.
[35] J. Schmidhuber, Learning complex, extended sequences using the principle
of history compression, Neural Computation 4 (2) (1992) 234–242.
[36] C. Gallicchio, A. Micheli, Echo state property of deep reservoir computing
networks., Cognitive Computation 9 (3) (2017) 337–350.
[37] R. Pascanu, C. Gulcehre, K. Cho, Y. Bengio, How to construct deep recur-
rent neural networks, arXiv preprint arXiv:1312.6026v5.
[38] B. Schrauwen, M. Wardermann, D. Verstraeten, J. Steil, D. Stroobandt,
Improving reservoirs using intrinsic plasticity, Neurocomputing 71 (7)
(2008) 1159–1171.
[39] C. Gallicchio, A. Micheli, L. Pedrelli, Hierarchical temporal representation
in linear reservoir computing, in: A. Esposito, M. Faundez-Zanuy, F. C.
Morabito, E. Pasero (Eds.), Neural Advances in Processing Nonlinear Dy-
namic Signals, Springer International Publishing, Cham, 2019, pp. 119–129,
arXiv preprint arXiv:1705.05782. doi:10.1007/978-3-319-95098-3\_11.
[40] C. Gallicchio, A. Micheli, L. Silvestri, Local lyapunov exponents of deep
rnn, in: Proceedings of the 25th European Symposium on Artiﬁcial Neural
Networks (ESANN), i6doc.com, 2017, pp. 559–564.
[41] N. Bertschinger, T. Natschl¨ager, Real-time computation at the edge of
chaos in recurrent neural networks, Neural computation 16 (7) (2004) 1413–
1436.
[42] R. Legenstein, W. Maass, Edge of chaos and prediction of computational
performance for neural circuit models, Neural networks 20 (3) (2007) 323–
334.
[43] J. Boedecker, O. Obst, J. T. Lizier, N. M. Mayer, M. Asada, Information
processing in echo state networks at the edge of chaos, Theory in Bio-
sciences 131 (3) (2012) 205–213.
13
[44] C. Gallicchio, A. Micheli, L. Pedrelli, Design of Deep Echo State Networks,
Neural Networks 108 (2018) 33–47.
[45] C. Gallicchio, A. Micheli, Deep Reservoir Neural Networks for Trees, In-
formation Sciences 480 (2019) 174–193.
[46] C. Gallicchio, A. Micheli, Deep Tree Echo State Networks, in: Proceedings
of the 2018 International Joint Conference on Neural Networks (IJCNN),
IEEE, 2018, pp. 499–506.
[47] C. Gallicchio, A. Micheli, Fast and deep graph neural networks., in: Pro-
ceedings of the Thirty-Fourth AAAI Conference on Artiﬁcial Intelligence
(AAAI-20), 2020, pp. 3898–3905.
[48] C. Gallicchio, A. Micheli, Graph echo state networks, in: The 2010 Inter-
national Joint Conference on Neural Networks (IJCNN), IEEE, 2010, pp.
1–8.
[49] C. Gallicchio, Short-term Memory of Deep RNN, in: Proceedings of the
26th European Symposium on Artiﬁcial Neural Networks (ESANN), 2018,
pp. 633–638.
[50] C. Gallicchio, A. Micheli, Experimental analysis of deep echo state net-
works for ambient assisted living, in: Proceedings of the 3rd Workshop
on Artiﬁcial Intelligence for Ambient Assisted Living (AI*AAL 2017), co-
located with the 16th International Conference of the Italian Association
for Artiﬁcial Intelligence (AI*IA 2017), 2017.
[51] C. Gallicchio, A. Micheli, L.Pedrelli, Deep Echo State Networks for Diag-
nosis of Parkinson’s Disease, in: Proceedings of the 26th European Sym-
posium on Artiﬁcial Neural Networks (ESANN), 2018, pp. 397–402.
[52] C. Gallicchio, A. Micheli, L. Pedrelli, Comparison between DeepESNs and
gated RNNs on multivariate time-series prediction, in: 27th European Sym-
posium on Artiﬁcial Neural Networks, Computational Intelligence and Ma-
chine Learning (ESANN 2019), i6doc. com publication, 2019.
[53] M. Alizamir, S. Kim, O. Kisi, M. Zounemat-Kermani, Deep echo state net-
work: a novel machine learning approach to model dew point temperature
using meteorological variables, Hydrological Sciences Journal 65 (7) (2020)
1173–1190.
[54] T. Kim, B. R. King, Time series prediction using deep echo state networks,
Neural Computing and Applications (2020) 1–19.
[55] Q. Li, Z. Wu, R. Ling, L. Feng, K. Liu, Multi-reservoir echo state computing
for solar irradiance prediction: A fast yet eﬃcient deep learning approach,
Applied Soft Computing 95 (2020) 106481.
14
[56] H. Hu, L. Wang, S.-X. Lv, Forecasting energy consumption and wind power
generation using deep echo state network, Renewable Energy 154 (2020)
598–613.
[57] J. Del Ser, I. Lana, E. L. Manibardo, I. Oregi, E. Osaba, J. L. Lobo, M. N.
Bilbao, E. I. Vlahogianni, Deep echo state networks for short-term traf-
ﬁc forecasting: Performance comparison and statistical assessment, arXiv
preprint arXiv:2004.08170.
[58] Z. Song, K. Wu, J. Shao, Destination prediction using deep echo state
network, Neurocomputing 406 (2020) 343–353.
[59] S. Dettori, I. Matino, V. Colla, R. Speets, Deep echo state networks in
industrial applications, in: IFIP International Conference on Artiﬁcial In-
telligence Applications and Innovations, Springer, 2020, pp. 53–63.
[60] V. Colla, I. Matino, S. Dettori, S. Cateni, R. Matino, Reservoir computing
approaches applied to energy management in industry, in: International
Conference on Engineering Applications of Neural Networks, Springer,
2019, pp. 66–79.
15
