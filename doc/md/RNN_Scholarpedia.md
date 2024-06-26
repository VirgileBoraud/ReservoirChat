# Recurrent neural networks

Dr. Stephen Grossberg, Boston University, MA

A recurrent neural network (RNN) is any network whose neurons send feedback signals to each other. This concept includes a huge number of possibilities. A number of reviews already exist of some types of RNNs. These include [1], [2], [3], [4].

Typically, these reviews consider RNNs that are artificial neural networks (aRNN) useful in technological applications. To complement these contributions, the present summary focuses on biological recurrent neural networks (bRNN) that are found in the brain. Since feedback is ubiquitous in the brain, this task, in full generality, could include most of the brain's dynamics. The current review divides bRNNS into those in which feedback signals occur in neurons within a single processing layer,  which occurs in networks for such diverse functional roles as storing spatial patterns in short-term memory, winner-take-all decision making, contrast enhancement and normalization, hill climbing, oscillations of multiple types (synchronous, traveling waves, chaotic), storing temporal sequences of events in working memory, and serial learning of lists; and those in which feedback signals occur between multiple processing layers, such as occurs when bottom-up adaptive filters activate learned recognition categories and top-down learned expectations focus attention on expected patterns of critical features and thereby modulate both types of learning.

# Contents

# Types of Recurrent Neural Networks

There are at least three streams of bRNN research: binary, linear, and continuous-nonlinear (Grossberg, 1988):

# Binary

Binary systems were inspired in part by neurophysiological observations showing that signals between many neurons are carried by all-or-none spikes. The binary stream was initiated by the classical McCulloch and Pitts (1943) model of threshold logic systems that describes how the activities, or short-term memory (STM) traces, \(x_i\) of the \(i^{th}\) node in a network interact in discrete time according to the equation:

\[ \tag{1} x_i(t+1) = \text{sgn} \left[ \sum_j A_{ij} x_j(t) - B_j \right], \]

where \(\text{sgn}(w) = +1\) if \(w > 0\), \(0\) if \(w = 0\), and \(-1\) if \(w < 0\). The McCulloch-Pitts model had an influence far beyond the field of neural networks through its influence on von Neumann's development of the digital computer.

Caianiello (1961) used a binary STM equation that is influenced by activities at multiple times in the past:

\[ \tag{2} x_i(T + \tau) = 1 \left[ \sum_{j=1}^n \sum_{k=0}^{l(m)} A_{ij}^{(k)} x_j(t - k \tau) - B_i \right] ,\]

where \(l(w) = 1\) if \(w \ge 0\) and \(0\) if \(w < 0\).

Rosenblatt (1962) used an STM equation that evolves in continuous time, whose activities can spontaneously decay, and which can generate binary signals above a non-zero threshold:

\[ \tag{3} \frac{d}{dt} x_i = -Ax_i + \sum_{j=1}^n \phi (B_j + x_j) C_{ij}, \]

where \(\phi(w) = 1\) if \(w \ge \theta\) and \(0\) if \(w < \theta\). This equation was used in the classical Perceptron model.

Both Caianiello (1961) and Rosenblatt (1962) introduced equations to change the weights \(A_{ij}^{(k)}\) in (2) and \(C_{ij}\) in (3) through learning. Such adaptive weights are often called long-term memory (LTM) traces. In both these models, interactions between STM and LTM were uncoupled in order to simplify the analysis. These LTM equations also had a digital aspect. The Caianiello (1961) LTM equations increased or decreased at constant rates until they hit finite upper or lower bounds. The Rosenblatt (1962) LTM equations were used to classify patterns into two distinct classes, as in the Perceptron Learning Theorem.

# Linear

Widrow (1962) drew inspiration from the brain to introduce the gradient descent Adeline adaptive pattern recognition machine. Anderson (1968) initially described his intuitions about neural pattern recognition using a spatial cross-correlation function.  Concepts from linear system theory were adapted to represent some aspects of neural dynamics, including solutions of simultaneous linear equations \(Y = AX\) using matrix theory, and concepts about cross-correlation. Kohonen (1971) made a transition from linear algebra concepts such as the Moore-Penrose pseudoinverse to more biologically motivated studies that are summarized in his books (Kohonen, 1977, 1984). These ideas began with a mathematically familiar engineering framework before moving towards more biologically motivated nonlinear interactions.

# Continuous-Nonlinear

Continuous-nonlinear network laws typically arose from an analysis of behavioral or neural data. Neurophysiological experiments on the lateral eye of the Limulus, or horseshoe crab, led to the award of a Nobel prize to H.K. Hartline. These data inspired the steady state Hartline-Ratliff model (Hartline and Ratliff, 1957):

\[ \tag{4} r_i = e_i - \sum_{j=1}^n k_{ij} \left[ r_j - r_{ij} \right]^+ , \]

where \([w]^+ = \text{max}(w, 0)\). Equation (4) describes how cell activations \(e_i\) are transformed into smaller net responses \(r_i\) due to recurrent inhibitory threshold-linear signals \(-k_{ij} \left[r_i - r_{ij}\right]^+\). The Hartline-Ratliff model is thus a kind of continuous threshold-logic system. Ratliff et al. (1963) extended this steady-state model to a dynamical model:

\[ \tag{5} r_i(t) = e_i(t) - \sum_{j=1}^n k_{ij} \left[ \frac{1}{\tau} \int_0^t e^{- \frac{t-s}{\tau}} r_j(s) ds - r_{ij} \right]^+, \]

which also behaves linearly above threshold. This model is a precursor of the Additive Model that is described below.

Another classical tradition arose from the analysis of how the excitable membrane of a single neuron can generate electrical spikes capable of rapidly and non-decrementally traversing the axon, or pathway, from one neuron's cell body to a neuron to which it is sending signals. This experimental and modeling work on the squid giant axon by Hodgkin and Huxley (1952) also led to the award of a Nobel prize. Since this work focused on individual neurons rather than neural networks, it will not be further discussed herein except to note that it provides a foundation for the Shunting Model described below.

Another source of continuous-nonlinear RNNs arose through a study of adaptive behavior in real time, which led to the derivation of neural networks that form the foundation of most current biological neural network research (Grossberg, 1967, 1968b, 1968c). These laws were discovered in 1957-58 when Grossberg, then a college Freshman, introduced the paradigm of using nonlinear systems of differential equations to model how brain mechanisms can control behavioral functions. The laws were derived from an analysis of how psychological data about human and animal learning can arise in an individual learner adapting autonomously in real time. Apart from the Rockefeller Institute student monograph Grossberg (1964), it took a decade to get them published.

# Additive STM equation

The following equation is called the Additive Model because it adds the terms, possibly nonlinear, that determine the rate of change of neuronal activities, or potentials, \(x_i\):

\[ \tag{6} \frac{d}{dt} x_i = - A_i x_i + \sum_{j=1}^n f_j(x_j) B_{ji} z_{ji}^{(+)} - \sum_{j=1}^n g_j(x_j) C_{ji} z_{ji}^{(-)} + I_i . \]

Equation (6) includes a term for passive decay (\(-A_i x_i\)), positive feedback (\(\sum_{j=1}^n f_j(x_j) B_{ji} z_{ji}^{(+)}\)), negative feedback (\(-\sum_{j=1}^n g_j(x_j) C_{ji} z_{ji}^{(-)}\)) and input (\(I_i\)). Each feedback term includes an activity-dependent (possibly) nonlinear signal (\(f_j(x_j)\), \(g_j(x_j)\)); a connection, or path, strength (\(B_{ji}, C_{ji}\)), and an adaptive weight, or LTM trace (\(z_{ij}^{(+)}, z_{ij}^{(-)}\)). If the positive and negative feedback terms are lumped together and the connection strengths are lumped with the LTM traces, then the Additive Model may be written in the simpler form:

\[ \tag{7} \frac{d}{dt} x_i = -A_i x_i + \sum_{j=1}^n f_j(x_j)z_{ji} + I_i. \]

Early applications of the Additive Model included computational analyses of vision, learning, recognition, reinforcement learning, and learning of temporal order in speech, language, and sensory-motor control (Grossberg, 1969b, 1969c, 1969d, 1970a, 1970b, 1971a, 1971b, 1972a, 1972b, 1974, 1975; Grossberg and Pepe, 1970, 1971). The Additive Model has continued to be a cornerstone of neural network research to the present time; e.g., in decision-making (Usher and McClelland, 2001). Physicists and engineers unfamiliar with the classical status of the Additive Model in neural networks called it the Hopfield model after the first application of this equation in Hopfield (1984). Grossberg (1988) summarizes historical factors that contributed to their unfamiliarity with the neural network literature. The Additive Model in (7) may be generalized in many ways, including the effects of delays and other factors. In the limit of infinitely many cells, an abstraction which does not exist in the brain, the discrete sum in (7) may be replaced by an integral (see Neural fields).

# Shunting STM equation

Grossberg (1964, 1968b, 1969b) also derived an STM equation for neural networks that more closely model the shunting dynamics of individual neurons (Hodgkin, 1964). In such a shunting equation, each STM trace is bounded within an interval \([-D,B]\). Automatic gain control, instantiated by multiplicative shunting, or mass action, terms, interacts with balanced positive and negative signals and inputs to maintain the sensitivity of each STM trace within its interval (see The Noise-Saturation Dilemma):

\[ \tag{8} \frac{d}{dt} x_i = -A_i x_i + (B - x_i) \left[ \sum_{j=1}^n f_j(x_j) C_{ji} z_{ji}^{(+)} + I_i \right] - (D + x_i) \left[ \sum_{j=1}^n g_j(x_j) E_{ji} z_{ji}^{(-)} + J_i \right]. \]

The Shunting Model is approximated by the Additive Model in cases where the inputs are sufficiently small that the resulting activities \(x_i\) do not come close to their saturation values \(-D\) and \(B\).

The Wilson-Cowan model (Wilson and Cowan, 1972) also uses a combination of shunting and additive terms, as in (8). However, instead of using sums of sigmoid signals that are multiplied by shunting terms, as in the right hand side of (8), the Wilson-Cowan model uses a sigmoid of sums that is multiplied by a shunting term, as in the expression \((B - x_i) f_j \left( \sum_{j=1}^n C_{ji} x_j z_{ji}^{(+)} - x_j E_{ji} z_{ji}^{(-)} + I_i \right)\). This form can saturate activities when inputs or recurrent signals get large, unlike (8), as noted in Grossberg (1973).

# Generalized STM equation

Equations (6) and (8) are special cases of an STM equation, introduced in Grossberg (1968c), which includes LTM and medium-term memory (MTM) terms that changes at a rate intermediate between the faster STM and the slower LTM. The laws for STM, MTM, and LTM are specialized to deal with different evolutionary pressures in neural models of different brain systems, including additional factors such as transmitter mobilization (Grossberg, 1969c, 1969b). This generalized STM equation is:

\[ \tag{9} \frac{dx_i}{dt} = -A x_i + (B - Cx_i) \left[ \sum_{k=1}^n f_k(x_k) D_{ki} y_{ki} z_{ki} + I_i \right] - (E + Fx_i) \left[ \sum_{k=1}^n g_k (x_k) G_{ki} Y_{ki} Z_{ki} + J_i \right]. \]

In the shunting model, the parameters \(C \ne 0\) and \(F \ne 0\). The parameter \(E = 0\) when there is "silent" shunting inhibition, whereas \(E \ne 0\) describes the case of hyperpolarizing shunting inhibition. In the Additive Model, parameters \(C = F = 0\). The excitatory interaction term \(\left[ \sum_{k=1}^n f_k (x_k) D_{ki} y_{ki} z_{ki} + I_i \right]\)describes an external input \(I_i\) plus the total excitatory feedback signal \(\left[ \sum_{k=1}^n f_k (x_k) D_{ki} y_{ki} z_{ki} \right]\) that is a sum of signals from other populations via their output signals \(f_k (x_k)\). The term \(D_{ki}\) is a constant connection strength between cell populations \(v_k\) and \(v_i\), whereas terms \(y_{ki}\) and \(z_{ki}\) describe MTM and LTM variables, respectively. The inhibitory interaction term \(\left[ \sum_{k=1}^n g_k (x_k) G_{ki} Y_{ki} Z_{ki} + J_i \right]\) has a similar interpretation. Equation (9) assumes "fast inhibition"; that is, inhibitory interneurons respond instantaneously to their inputs. Slower inhibition with inhibitory interneuronal activities \(X_i\) uses an equation like (9) to describe the temporal evolution of the inhibitory activities. The output signals from these inhibitory interneurons provide inhibitory feedback signals to the excitatory activities. With slow inhibition, the inhibitory feedback signals in (9) would be \(g_k (X_k)\) instead of \(g_k (x_k)\).

Cohen and Grossberg (1983) derived a Liapunov function for a generalization of the Additive and Shunting Models in (9), with constant MTM and LTM variables and symmetric connections. This Liapunov function includes as a special case the Liapunov function that Hopfield (1984) stated for the Additive Model (see Cohen-Grossberg model, Liapunov function, and theorem).

# MTM: Habituative Transmitter Gates and Depressing Synapses

Medium-term memory (MTM), or activity-dependent habituation, often called habituative transmitter gates, has multiple roles. One role is to carry out intracellular adaptation that divides the response to a current input with a time-average of recent input intensity. A related role is to prevent recurrent activation from persistently choosing the same neuron, by reducing the net input to this neuron. MTM traces also enable reset events to occur. For example, in a gated dipole opponent processing network, they enable an antagonistic rebound in activation to occur in the network's OFF channel in response to either a rapidly decreasing input to the ON channel, or to an arousal burst to both channels that is triggered by an unexpected event (Grossberg, 1972b, 1980a). This property enables a resonance that reads out a predictive error to be quickly reset, thereby triggering a memory search, or hypothesis testing, to discover a recognition category capable of better representing an attended object or event (see Adaptive Resonance Theory; Grossberg, 2012; [5]). MTM reset dynamics also help to explain data about the dynamics of visual perception, cognitive-emotional interactions, decision-making under risk, and sensory-motor control (Francis and Grossberg, 1996; Francis et al., 1994; Gaudiano and Grossberg, 1991, 1992; Grossberg, 1972b, 1980a, 1984a, 1984b; Grossberg and Gutowski, 1987; Ogmen and Gagné, 1990).

In (9), the \(i^{th}\) MTM trace, or habituative transmitter gate, \(y_i\), typically obeys the equation:

\[ \tag{10} \frac{dy_i}{dt} = H(K - y_i) - Lf_k(x_k)y_k. \]

By (10), \(y_i\) accumulates at a fixed rate \(H\) to its maximum value \(K\) via term \(h(K - y_i)\) and is inactivated, habituated, or depressed via a mass action interaction between the feedback signal \(f_k(x_k)\) and the gate concentration \(y_k\) via term \(Lf_k(x_k)y_k\). Abbott et al. (1997) reported neurophysiological data from the visual cortex and rederived this MTM equation from it, calling it a depressing synapse. Tsodyks and Markram (1997) derived a related equation using their data from the somatosensory cortex, calling it a dynamic synapse. The mass action term may be more complex than it is in (10) in some situations; e.g., Gaudiano and Grossberg (1991, 1992) and Grossberg and Seitz (2003). The habituative transmitter gate \(Y_k\) in the inhibitory feedback term of (1) obeys a similar equation. By multiplying intercellular signals, transmitter gates can modulate their efficacy in an activity-dependent way. Not all signals need to be habituative.

# LTM: Gated steepest descent learning: Not Hebbian learning

An oft-used equation for the learning of adaptive weights, or long-term memory (LTM) traces, is called gated steepest descent learning. Gated steepest descent learning permits adaptive weights to increase or decrease (Grossberg, 1967, 1968b, 1968c). This is because the unit of LTM in the Additive and Shunting Models was proved to be a distributed pattern of LTM traces across a network, and the LTM traces learn to match the pattern of activities, or STM traces, of cells across the network (see Processing and STM of Spatial Patterns). If an STM activity is large (small), then the LTM trace can increase (decrease). These learning laws are thus not Hebbian, because the Hebb (1949) learning postulate says that: "When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some grown process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased". This postulate only allows LTM traces to increase. Thus, after sufficient learning took place, Hebbian traces would saturate at their maximum values. The Hebb postulate assumed the wrong processing unit: It is not the strength of an individual connection; rather it is a distributed pattern of LTM traces.

One variant of gated steepest descent learning, called Outstar Learning, was introduced in Grossberg (1968b) for spatial pattern learning (Figure 1 and Figure 2).

Another variant is called Instar Learning, which was used in Grossberg (1976a) for the learning of bottom-up adaptive filters (Figure 3) in Self-Organizing Map (SOM) models [6]. A SOM uses a recurrent on-center off-surround network (Figure 4) to choose one, or a small number, of cells for storage in STM (see Processing and STM of Spatial Patterns), before the stored activities trigger learning of LTM traces in abutting synapses (see Sparse Stable Category Learning Theorem). Kohonen (1984) also used Instar Learning in his applications of the SOM model.

Outstar and Instar Learning are dual networks in the sense that they are the same, except for reversing which cells are sampling and which are sampled (Figure 5).

Outstars and Instars were combined in Grossberg (1976a) to form a three-layer Instar-Outstar network for learning multi-dimensional maps from any m-dimensional input space to any n-dimensional output space (Figure 6). The Instars learn recognition categories that selectively respond to an m-dimensional input pattern (see Sparse Stable Category Learning Theorem), and an active category samples a simultaneously active n-dimensional input pattern (see Outstar Learning Theorem). Hecht-Nielsen (1987) called such a network a counterpropagation network.

In ART models, these concepts were used to define a bRNN. In the article Grossberg (1976b) that introduced ART, Instars define the learning in bottom-up adaptive filters, and Outstars define the learning in top-down expectations (Figure 7). The learning instabilities of competitive learning and SOM models that were described in Grossberg (1976a) led Grossberg (1976b) to show how matching of bottom-up feature patterns by top-down learned expectations, and the ensuing focusing of attention upon critical feature patterns, can dynamically stabilize the memories learned in SOM models, as well as the multi-dimensional maps learned by an Instar-Outstar network (see Adaptive Resonance Theory).

Outstar Learning equation:

\[ \tag{11} \frac{dz_{ij}}{dt} = Mf_i(x_i) \left[ h_j(x_j) - z_{ij} \right] \]

Instar Learning Equation:

\[ \tag{12} \frac{dz_{ij}}{dt} = Mf_j(x_j) \left[ h_i(x_i) - z_{ij} \right]. \]

Equation (11) describes the outstar learning equation, by which the \(i^{th}\) source, or sampling, cell can sample and learn a distributed spatial pattern of activation across a network of sampled cells (\(j \in J\)). When the gating signal \(f_i(x_i)\) is positive, the adaptive weights \(z_{ij}\) can sample the activity-dependent signals \(h_j(x_j)\) across the sampled network of cells. Equation (12) describes the instar learning equation, by which the \(j^{th}\) target cell can sample and learn the distributed pattern of signals (\(i \in I\)) that activated it. There are many variations of these gated steepest descent equations, including doubly-gated learning, spike-timing dependent learning, and self-normalizing learning (e.g., Gorchetchnikov et al., 2005; Grossberg and Seitz, 2003). Not all connections need to be adaptive.

As illustrated below, various combinations of these STM, MTM, and LTM equations have been used in scores of modeling studies since the 1960s. In particular, they were used by O'Reilly and Munakata (2000) in what they call the Leabra model.

# Processing and STM of Spatial Patterns

# Transformation and short-term storage of distributed input patterns by neural networks

The brain is designed to process patterned information that is distributed across networks of neurons. For example, a picture is meaningless as a collection of independent pixels. In order to understand 2D pictures and 3D scenes, the brain processes the spatial pattern of inputs that is received from them by the photosensitive retinas. Within the context of a spatial pattern, the information from each pixel can acquire meaning. The same is true during temporal processing. For example, individual speech sounds heard out of context may sound like meaningless chirps. They sound like speech and language when they are part of a characteristic temporal pattern of signals. The STM, MTM, and LTM equations enable the brain to effectively process and learn from both spatial and temporal patterns of information.

Both spatial and temporal patterns may be received at multiple intensities. Scenes can be seen in dim or bright light, and speech can be heard if it is whispered or shouted. In order to process either spatial or temporal patterns using neurons, brains have evolved network designs that can compensate for variable input intensities without a loss of pattern information.

# The Noise-Saturation Dilemma

Without suitable interactions between neurons, their input patterns can be lost in cellular noise if they are too small, or can saturate cell activities at their maximum values if they are too large. Input amplitudes can also vary from small to large through time, just as the intensity of light can vary from dim to bright.  During the processing of a visual input from a fixed object, the total intensity of an input can change while the relative intensity remains constant. The relative intensity is called the reflectance of the surface that reflects variable intensities of light to the eye. Many other examples exist wherein total intensity changes while relative intensity remains constant.

What sort of network interactions enable neurons to retain their sensitivities to the relative sizes of their inputs across the network, even while these inputs may vary in size through time over several orders of magnitude?  The answer is: an on-center off-surround network whose cells obey the membrane, or shunting, equations of neurophysiology (Grossberg, 1973, 1980a). This fact helps to explain why such networks are ubiquitous in the brain.

# A thought experiment to solve the noise-saturation dilemma

Suppose that a spatial pattern \(I_i = \theta_i I\) of inputs is processed by a network of cells \(v_i, i=1,2,...,n\). Each \(\theta_i\) is the constant relative size, or reflectance, of its input \(I_i\) and \(I = \sum_{k=1}^n I_k\) is the variable total input size. Thus \(\sum_{k=1}^n \theta_k = 1\). How can each cell \(v_i\) maintain its sensitivity to \(\theta_i\) when \(I\) is parametrically increased? How is saturation avoided?

To compute \(\theta_i = I_i \left( \sum_{k=1}^n I_k \right)^{-1}\), each cell \(v_i\) must have information about all the inputs \(I_k, k=1,2,...,n\). Rewriting the ratio \(\theta_i\) as \(\theta_i = I_i \left( I_i + \sum_{k \ne i} I_k \right)^{-1}\) calls attention to the fact that increasing \(I_i\) increases \(\theta_i\), whereas increasing any \(I_k, k \ne i\), decreases \(\theta_i\). When this property is translated into an anatomy for delivering feedforward inputs to the cells \(v_i\), it suggests that the input \(I_i\) excited \(v_i\) and that all the inputs \(I_k, k \ne i\), inhibit \(v_i\). In other words, all the inputs compete among themselves while trying to activate their own cell. This rule represents a feedforward on-center off-surround anatomy. It has been known that on-center off-surround anatomies are ubiquitous in the brain at least since they were reported in the cat retina by Kuffler (1953).

How does the on-center off-surround anatomy activate and inhibit the cells \(v_i\) through time? Suppose that each cell possesses \(B\) excitable sites of which \(x_i(t)\) are excited and \(B - x_i(t)\) are not excited at time \(t\). Thus, at cell \(v_i\), input \(I_i\) excites the \(B - x_i(t)\) unexcited sites, and the total inhibitory input \(\sum_{k \ne i} I_k\) from the off-surround inhibits the \(x_i(t)\) excited sites. Suppose, in addition, that excitation \(x_i(t)\) can spontaneously decay at a fixed rate \(A\), so that the cell can return to an equilibrium point, set to equal \(0\) for simplicity, after all inputs cease. Putting these properties together in one equation yields:

\[ \tag{13} \frac{d}{dt} x_i = -A x_i + (B - x_i) I_i - x_i \sum_{k \ne i} I_k . \]

Equation (13) defines a feedforward on-center (\(I_i\)) off-surround (\(\sum_{k \ne i} I_k\)) network whose cells obey a simple version of the Shunting Model in equation (8).

If a fixed spatial pattern \(I_i = \theta_i I\) is presented and the total input \(I\) is held constant for awhile, then each \(x_i(t)\) approaches an equilibrium value that is found by setting \(\frac{d}{dt} x_i = 0\) in equation (13). Then

\[ \tag{14} x_i = \theta_i \frac{BI}{A + I}. \]

Note that the relative activity \(X_i = x_i \left( \sum_{k=1}^n x_k \right)^{-1}\) equals \(\theta_i\) no matter how large \(I\) is chosen; there is no saturation. However, if the off-surround input is removed, then all the \(x_i\) saturate at \(B\) as the total input \(I\) becomes large.

# Automatic gain control by the off surround prevents saturation

Saturation is prevented in (13) due to automatic gain control by the inhibitory inputs from the off-surround. In other words, the off-surround \(\sum_{k \ne i} I_k\) multiplies \(x_i\). The total gain is found by rewriting (13) as:

\[ \tag{15} \frac{d}{dt} x_i = -(A + I) x_i + BI_i. \]

The gain is the coefficient of \(x_i\), namely \(-(A + I)\). Indeed, if \(x_i(0) = 0\), then (15) can be integrated to yield:

\[ \tag{16} x_i(t) = \theta_i \frac{BI}{A + I} \left( 1 - e^{-(A + I)t} \right) .\]

By (16), both the steady state and the rate of change of \(x_i\) depend upon input strength \(I\). This is characteristic of mass action, or shunting, networks but not of additive networks, in which the inputs do not multiply the activities \(x_i\).

# Contrast normalization and pattern processing by real-time probabilities.

Another property of (14) is that the total activity:

\[ \tag{17} x = \sum_{k=1}^n x_k = \frac{BI}{A + I} \]

is independent of the number of active cells and approaches \(B\) as \(I\) increases. This normalization rule is a conservation law which says, for example, that increasing one activity forces a decrease in other activities. This property helps to explain such properties of visual perception as brightness constancy and brightness contrast (Cornsweet, 1970; Grossberg and Todorovic, 1988). During brightness contrast, increasing the luminance of inputs to the off-surround makes the on-center look darker. The normalization property is called contrast normalization in applications to visual perception. More generally, normalization underlies many properties of limited capacity processing in the brain, notably in perception and cognition, with working memory capacity limits being a classical example.

# Weber Law and shift property

Writing equation (14) in logarithmic coordinates shows that increasing the off-surround input does not reduce the sensitivity of the network to inputs to the on-center; rather, it shifts network responses to larger input sizes without a loss of sensitivity. In particular, let \(K_i = \ln(I_i)\) and \(I_i = e^{K_i}\). Also write the total off-surround input as \(J_i = \sum_{k \ne i} I_k\). Then (14) can be written in logarithmic coordinates as:

\[ \tag{18} x_i(K_i, J_i) = \frac{B e^{K_i}}{A + e^{K_i} + J_i}. \]

How does the activity \(x_i\) change if the off-surround input \(J_i\) is parametrically set at increasingly high values? Equation (18) shows that the entire response curve of \(x_i\) to its on-center input \(K_i\) also shifts, and thus its dynamic range is not compressed. For example, suppose that the off-surround input is increased from \(J_i^{(1)}\) to \(J_i^{(2)} = J_i^{(1)} + S_i\) by an amount \(S_i\) . Then the amount of shift in the response curve is:

\[ \tag{19} S_i = \ln \frac{A + J_i^{(2)}}{A + J_i^{(1)}}. \]

Such a shift property is found, for example, in the retina of the mudpuppy Necturus (Werblin, 1971). Generalizations of the feedforward on-center off-surround shunting network equations generate many other useful properties, including Weber law processing, adaptation level processing, and edge and spatial frequency processing (Grossberg, 1983).

# Physiological interpretation of shunting dynamics: The membrane equation of neurophysiology

The Shunting equation (13) has the form of the membrane equation on which cellular neurophysiology is based. This membrane equation is the voltage equation that appears in the equations of Hodgkin and Huxley (1952). In other words, the gedanken experiment shows how the noise-saturation dilemma is solved by using the membrane, or shunting, equation of neurophysiology to describe cells interacting in on-center off-surround anatomies. Because on-center off-surround anatomy and shunting dynamics work together to solve the noise-saturation dilemma, it is reasonable to predict that they coevolved during evolution.

The membrane equation describes the voltage \(V(t)\) of a cell by the law:

\[ \tag{20} C \frac{\partial V}{\partial t} = (V^+ - V)g^+ + (V^- - V)g^- + (V^p - V) g^- . \]

In (20), \(C\) is a capacitance; \(V^+\), \(V^-\), and \(V^p\) are constant excitatory, inhibitory, and passive saturation voltages, respectively; and \(g^+\), \(g^-\), and \(g^p\) are excitatory, inhibitory, and passive conductances, respectively. When the saturation voltages are chosen to satisfy \(V^- \le V^p < V^+\), then the cell voltage satisfies \(V^- \le V(t) \le V^+\). Often \(V^+\) represents the saturation point of a \(Na^+\) channel, and \(V^-\) represents the saturation point of a \(K^+\) channel.

There is symmetry-breaking in (20) because \(V^+ - V^p\) is usually much larger than \(V^p - V^-\). Symmetry-breaking implies a noise suppression property when it is coupled to an on-center off-surround anatomy (Grossberg, 1988). Then the network suppresses uniformly active inputs and generates suprathreshold responses only to inputs that are larger than a baseline value, or adaptation level. This property illustrates that excitation and inhibition need to be properly balanced to achieve efficient neuronal dynamics: When excitation is too large, seizure activity can occur in a bRNN. When inhibition is too large, processing can never get started. Symmetry-breaking can be achieved during development by an opposites attract rule whereby the relative sizes of the intracellular excitatory and inhibitory saturation voltages \(V^+\) and \(V^-\) control the relative total strengths of the intercellular off-surround and on-center connections, respectively (Grossberg, 1978a, Section 45).

# Recurrent competitive fields

The activities \(x_i\) in (13) rapidly decay if their inputs \(I_i\) are shut off. Persistent storage in STM is achieved when feedback signals exist among the various populations, thereby creating a bRNN. The noise-saturation dilemma confronts all cellular tissues which process input patterns, whether the cells exist in a feedforward or feedback anatomy. To solve the noise-saturation dilemma in a RNN, excitatory feedback signals need to be balanced by inhibitory feedback signals. The simplest recurrent on-center off-surround shunting RNN, also called a recurrent competitive field (RCF), is defined by (Grossberg, 1973):

\[ \tag{21} \frac{d}{dt} x_i = -A x_i + (B - x_i) \left[ I_i + f(x_i) \right] - x_i \left[ J_i + \sum_{k \ne i} f(x_k) \right] . \]

# Winner-take-all, contrast enhancement, normalization, and quenching threshold

Grossberg (1973) proved theorems showing how the choice of feedback signal function \(f(w)\) transforms an input pattern before it is stored persistently in STM. Given the fundamental nature of these results for all bRNNs, they will be reviewed below.

Figure 8 summarizes the results. These theorems provided the first rigorous proofs of winner-take-all (WTA) properties, and of the use of sigmoid signal functions to generate a self-normalizing "bubble", or partial contrast-enhancement, above a quenching threshold. The theorems began the mathematical classification of cooperative-competitive recurrent nonlinear dynamical systems, whose properties are applicable to many fields, ranging from morphogenesis to economics (Grossberg, 1988).

To prove the theorems, (21) is transformed into total activity variables \(\sum_{k=1}^n x_k\) and pattern variables \(X_i = x_i x^{-1}\) under the assumption that the inputs \(I_i\) and \(J_i\) are set to zero during the STM storage process. Then (21) may be rewritten as:

\[ \tag{22} \frac{d}{dt} X_i = BX_i \sum_{k=1}^n X_k \left[ h(X_i x) - h(X_k x) \right] \]

and

\[ \tag{23} \frac{d}{dt} x = -Ax + (B - x) \sum_{k=1}^n f(X_k x), \]

where the function \(h(w) = f(w)w^{-1}\) is called the hill function because it exhibits a "hill" of activity for every transition between a faster-than-linear, linear, and slower-than-linear shape in the signal function, as shown for the sigmoid function in Figure 8 and Figure 9.

If \(f(w)\) is linear—that is, \(f(w) = Cw\), then \(h(w) = C\) and all \(\frac{d}{dt} X_i = 0\) in (22). Then (21) can preserve any pattern in STM! However, by (23), if \(A \ge B\), then \(x(t)\) approaches \(0\) as \(t \rightarrow \infty\), so that no pattern is stored in STM. A pattern is stored in STM only if \(B > A\). Then \(x(t) \rightarrow B - A\) as \(t \rightarrow \infty\), so that the total activity is normalized. This result implies that, if STM storage is possible and \(x(0) > 0\), then \(x(t) \rightarrow B - A\) even if no input occurs. In other words, noise will be amplified as vigorously as inputs. A linear signal function amplifies noise, and is therefore inadequate despite its perfect memory of any input pattern. That is why nonlinear signal functions are needed.

A slower-than-linear signal function—for example, \(f(x) = Cw(D + w)^{-1}\) or, more generally, any \(f(w)\) whose hill function \(h(w)\) is monotone decreasing—is even worse, because it amplifies noise and eliminates all differences in inputs within the stored pattern. This happens because, by (22), if \(X_i > X_k, k \ne i\), then \(\frac{d}{dt} X_i < 0\) and if \(X_i < X_k, k \ne i\) then \(\frac{d}{dt} X_i > 0\). Thus the maximum activity decreases and the minimum activity increases until all the activities become equal.

If both linear and slower-than-linear signal functions amplify noise, then one must turn to faster-than-linear functions in the hope that they suppress noise. If \(f(w)\) is faster-than-linear—--for example, \(f(x) = Cw^n, n > 1\), or, more generally, any \(f(w)\) whose hill function \(h(w)\) is monotone increasing—then noise is, indeed, suppressed. In this case, if \(X_i > X_k, k \ne i\), then \(\frac{d}{dt} X_i > 0\) and if \(X_i < X_k, k \ne i\), then \(\frac{d}{dt} X_i < 0\). As a result, the network chooses the population with the initial maximum of activity and totally inhibits activity in all other populations. This network behaves like a winner-take-all binary choice machine. The same is true for total activity, since as \(t \rightarrow \infty\), (23) becomes approximately:

\[ \tag{24} \frac{d}{dt} x \cong x [ -A + (B - x) h(x) ]. \]

Thus, the equilibrium points of \(x_i\) as \(t \rightarrow \infty\) are \(E_0 = 0\) and all the solutions of the equation

\[ \tag{25} h(x) = A(B - x)^{-1}; \]

see Figure 10. If \(h(0) < A / B\), then the smallest solution, \(E_1\), of (25) is unstable, so that activities \(x(t) < E_1\) are suppressed as \(t \rightarrow \infty\). This is noise suppression due to recurrent competition. Every other solution \(E_2, E_4, ...\) of (25) is a stable equilibrium point of \(x(t)\) as \(t \rightarrow \infty\) (total activity quantization) and all equilibria are smaller than \(B\) (normalization).

The faster-than-linear signal contrast-enhances the pattern so vigorously that the good property of noise suppression is joined to the extreme property of winner-take-all (WTA) choice. Although WTA is often a useful property in applications to choice behavior (e.g., Dev, 1975; Grossberg, 1976a; Grossberg and Pilly, 2008; Koch and Ullman, 1985; Wang, 2008), there are many cases where noise suppression is desired but more than one feature or category needs to be stored in STM. How can this be accomplished?

The results above show that any signal function that suppresses noise must be faster-than-linear at small activities. In addition, all signal functions in biology must be bounded. Such a combination is achieved most simply by using a sigmoid signal function, which is a hybrid of faster-than-linear at small activities, approximately linear at intermediate activities, and slower-than-linear at high activities (Figure 9). Then there exists a quenching threshold (QT) such that if initial activity falls below the QT, then its activity is quenched. All initial activities that exceed the QT are contrast-enhanced and stored in STM (Figure 8). The faster-than-linear part of the sigmoid suppresses noise and starts to contrast-enhance the activity pattern. As total activity normalizes, the approximately linear range is reached and tends to store the partially contrast-enhanced pattern. The QT converts the network into a tunable filter. For example, a burst of nonspecific arousal in response to an unexpected event that multiplicatively inhibits all the recurrent inhibitory interneurons will lower the QT and facilitate storage of inputs in STM until the cause of the unexpected event can be determined.

# Shunting dynamics in cortical models

Multiple generalizations of RCFs have been studied and used to explain data ranging from visual and speech perception and attentive category learning (see Unifying horizontal, bottom-up, and top-down STM and LTM interactions) to the selection of commands for arm movement control (e.g., Cisek, 2006) and for eye movement control in response to probabilistically defined visual motion signals (e.g., Grossberg and Pilly, 2008). As noted above, Usher and McClelland (2001) modeled probabilistic decision making using an Additive Model. This model does not exhibit the self-normalization properties that arise from RCF shunting dynamics.

A number of authors have applied shunting properties to simulate data about the properties of the cortical circuits that subserve visual perception; e.g., Douglas et al. (1995), Grossberg and Mingolla (1985), Grossberg and Todorovic (1988), Heeger (1992), and McLaughlin et al. (2000). Shunting dynamics also played a key role in the development of the Competitive Learning (CL), Self-Organizing Map (SOM), and Adaptive Resonance Theory (ART) models (Scholarpedia: Adaptive Resonance Theory; Grossberg, 1976a, 1976b, 1980a), but not in the CL and SOM versions of von der Malsburg (1973) and Kohonen (1984). An RCF with spiking neurons has also been shown to replicate key properties of the Grossberg (1973) theorems for rate-based neurons (Palma et al., 2012).

# Decision-making in Competitive Systems: Liapunov methods

The ubiquity of RCFs led to a search for the most general networks that could guarantee stable STM storage. The RCF in (21) is a special case of a competitive dynamical system. In general, a competitive dynamical system is defined by a system of differential equations such that:

\[ \tag{26} \frac{d}{dt} x_i = f_i(x_1, x_2, ..., x_n) \]

where

\[ \tag{27} \frac{\partial f_i}{\partial x_j} \le 0, i \ne j \]

and the \(f_i\) are chosen to generate bounded trajectories. By (27), increasing the activity \(x_j\) of a given population can only decrease the growth rates \(\frac{d}{dt}x_i\) of other populations, \(i \ne j\), or may not influence them at all. In such systems, cooperative interactions typically occur within a population while competitive interactions can occur between populations, as in the recurrent on-center off-surround network (21). Grossberg  (1978d, 1980b) developed a mathematical method to classify the dynamics of competitive dynamical systems by proving that any competitive system can be analyzed by keeping track of the population that is winning the competition through time. This method defines jump sets at the times when the winning population is replaced by—that is, jumps to—another population. Tracking trajectories through jump sets formalizes keeping track of the population that is winning the competition through time. Jump sets define a kind of decision hypersurface. If the jumps form a cycle, so that no globally consistent winner exists, then oscillations can occur. In particular, such a jump cycle occurs in the May and Leonard (1975) model of the voting paradox. If the jumps only form decision trees, without cycles, then all trajectories converge to limits. A global Liapunov functional was defined and provides the "energy" that moves system trajectories through these oscillatory or convergent decision hypersurfaces through time. See Grossberg (1988, Section 11) for a review.

# Competition, decision, and consensus

This method was applied to study a general problem that has intrigued philosophers and scientists for hundreds of years, and which includes many RCFs as special cases: How do arbitrarily many individuals, populations, or states, each obeying unique and personal laws, ever succeed in harmoniously interacting with each other to form some sort of stable society, or collective mode of behavior? If each individual obeys complex laws, and is ignorant of other individuals except via locally received signals, how is social chaos averted? How can local ignorance and global order, or consensus, be reconciled? Considerable interest has focused on the question: How simple can a system be and still generate "chaotic" behavior (e.g., Alligood et al., 1996)? The above issue considers the converse question: How complicated can a system be and still generate order?

Grossberg (1978c) posed these questions and introduced a class of bRNNs in which this type of global consensus arises, along with mathematical methods to prove it. Consensus arises in these systems because, despite essentially arbitrary irregularities and nonlinearities in local system design, there exists a powerful symmetry in the global rules that bind together the interacting populations. This symmetry is expressed by the existence of a shared, but state-dependent, inter-population competition function, also called an adaptation level. These results suggest that a breakdown of symmetry in competitive RNNs, say due to the existence of asymmetric biases in short-range inter-population interactions, is a basic cause of oscillations and chaos in these systems, as is illustrated by the voting paradox. There appears to exist a trade-off between how global the adaptation level ("communal understanding") is and how freely local signals ("individual differences") can be chosen without destroying global consensus.

# Adaptation level systems: Globally-consistent decision-making

System (21) is a special case of a competitive network with a broad inhibitory surround. A much more general class of systems, the adaptation level systems, also has this property:

\[ \tag{28} \frac{d}{dt} x_i = a_i(x) \left[ b_i(x_i) - c(x) \right] , \]

where \(x = (x_1, x_2, ..., x_n)\), \(a_i(x)\) is a state-dependent amplification function, \(b_i(x_i)\) is a self-signal function, and \(c(x)\) is the state-dependent adaptation level against which each \(b_i(x_i)\) is compared. For the special case of (21),

\[ \tag{29} a_i(x) = x_i, \]

\[ \tag{30} b_i(x_i) = x_i^{-1} [Bf(x_i) + I_i] - A - I_i - J_i, \]

and

\[ \tag{31} c(x) = \sum_{k=1}^n f(x_k). \]

The same equations hold with \(A\), \(B\), and \(f(x_i)\) in (21) replaced by \(A_i\), \(B_i\), and \(f_i(x_i)\); that is, different parameters and signal functions for each cell, for arbitrarily many cells.

Grossberg (1978c) proved that all trajectories in such systems are "stored in STM"; that is, converge to equilibrium values as \(t \rightarrow \infty\), even in systems which possess infinitely many equilibrium points. The proof shows how each \(x_i(t)\) gets trapped within a sequence of decision boundaries that get laid down through time at the abscissa values of the peaks in the graphs of the signal functions \(b_i(x_i)\), starting with the highest peaks and working down. These signal functions generalize the hill function in (22); see (30). Multiple peaks correspond to multiple cooperating subpopulations. These graphs may thus be very complex if each population contains multiple cooperating subpopulations. After all the decision boundaries get laid down, each \(x_i(t)\) is trapped within a single valley of its \(b_i\) graph. After this occurs for all the \(x_i\) variables, the function \(B(x(t)) = \max [ b_i (x(t)) : i = 1, 2, ..., n]\) is a Liapunov function, whose Liapunov property is then used to complete the proof of the theorem.

A special case of the theorem concerns a competitive market with an arbitrary number of competing firms (Grossberg, 1988, Section 12). Each firm can choose one of infinitely many production and savings strategies that is unknown to the other firms. The firms know each other's behaviors only through their effect on a competitive market price, and they produce more goods at any time only if application of their own firm's production and savings strategy will lead to a net profit with respect to that market price. The theorem proves that the price in such a market is stable and that each firm balances its books. The theorem does not, however, determine which firms become rich and which go broke.

# Cohen-Grossberg model, Liapunov function, and theorem

Due to the importance of symmetry in proving global approach to equilbria, as in the adaptation level systems (28), Cohen and Grossberg attempted to prove that all trajectories of systems of the Cohen-Grossberg form:

\[ \tag{32} \frac{d}{dt} x_i = a_i(x_i) [ b_i(x_i) - \sum_{k=1}^n c_{ij} d_j(x_j)], \]

with symmetric interaction coefficients \(c_{ij} = c_{ji}\) and weak assumptions on their defining functions, approach equilibria as \(t \rightarrow \infty\). Systems (32) include both Additive Model and Shunting Model networks (6) and (8) with distance-dependent, and thus symmetric, interaction coefficients, the Brain-State-in-a-Box model (Anderson et al., 1977), the continuous-time version of the McCulloch and Pitts (1943) model, the Boltzmann Machine equation (Ackley et al., 1985), the Ratliff et al. (1963) model, the Volterra-Lotka model (Lotka, 1956), the Gilpin and Ayala (1973) model, the Eigen and Schuster (1978) model, the Cohen and Grossberg (1986, 1997) Masking Field model, and so on.

Cohen and Grossberg first attempted to prove global equilibrium by showing that all Cohen-Grossberg systems generate jump trees, and thus no jump cycles, which would immediately prove the desired result. This hypothesis still stands as an unproved conjecture. While doing this, inspired by the use of Liapunov methods for more general competitive systems, Cohen and Grossberg (1983; see also Grossberg (1982)) discovered the Cohen-Grossberg Liapunov function that they used to prove that global equilibria exist:

\[ \tag{33} V = - \sum_{i=1}^n \int^{x_i} b_i (\xi_i) d_i^{\prime} (\xi_i) d \xi_i + \frac{1}{2} \sum_{j, k=1}^n c_{jk} d_j(x_j) d_k (x_k). \]

Equation (33) defines a Liapunov function because integrating \(V\) along trajectories implies that:

\[ \tag{34} \frac{d}{dt} V = - \sum_{i=1}^n a_i d_i^{\prime} \left[ b_i - \sum_{j=1}^n c_{ij} d_j \right]^2 . \]

If \(a_i d_i^{\prime} \ge 0\), then (34) implies that \(\frac{d}{dt} V \le 0\) along trajectories. Once this basic property of a Liapunov function is in place, it is a technical matter to rigorously prove that every trajectory approaches one of a possibly large, or infinite, number of equilibrium points.

As noted above, the Liapunov function (33) proposed in Cohen and Grossberg (1983) includes both the Additive Model and Shunting Model, among others. A year later, Hopfield (1984) published the special case of the Additive Model and Liapunov function and asserted, without proof, that trajectories approach equilibria. Based on this 1984 article, the Additive Model has been erroneously called the Hopfield network by a number of investigators, despite the fact that it was published in multiple articles since the 1960s and its Liapunov function was also published in 1982-83. A historically more correct name, if indeed names must be given, is the Cohen-Grossberg-Hopfield model, which is the name already used in articles such as Burton (1993), Burwick (2006), Guo et al. (2004), Hoppensteadt and Izhikevich (2001), Menon et al. (1996), and Wu and Zou (1995).

# Symmetry does not imply convergence: Synchronized oscillations

Cohen (1988) showed that symmetric coefficients are not sufficient to ensure global convergence by constructing distance-dependent (hence symmetric) on-center off-surround networks that support persistent oscillations. These networks can send excitatory feedback signals to other populations than themselves. They are a special case of (8) with fast-acting inhibitory interneurons. It has long been known that shunting networks with slow inhibitory interneurons can persistently oscillate (e.g., Ellias and Grossberg, 1975). This observation led to the prediction that neural networks can undergo synchronized oscillations, first called order-preserving limit cycles (Grossberg, 1976b), during attentive resonant states. The early articles concerning synchronized oscillations during attentive brain dynamics (e.g., Gray et al., 1989; Grossberg and Somers, 1991; Grossberg and Grunewald, 1997; Eckhorn et al., 1990; Somers and Kopell, 1993) have been followed by hundreds more. Persistent oscillations can also occur in RCFs in which, instead of slow inhibitory interneurons, habituative gates (10) multiply the recurrent signals in (9) (e.g., Carpenter and Grossberg, 1983)

# Unifying horizontal, bottom-up, and top-down STM and LTM interactions

Most of the RNNs considered above characterize their interaction terms in abstract mathematical terms; e.g., symmetry of connection strengths. In contrast, the bRNNs in the brain are embodied in architectures with highly differentiated anatomical circuits. For example, models of how the cerebral cortex works are defined by bRNNs that integrate bottom-up, horizontal, and top-down interactions in laminar circuits with identified cells. These models illustrate the computational paradigm of Laminar Computing (Grossberg, 1999, 2003) which has begun to classify how different behavioral functions, such as vision, cognition, speech, and behavioral choice, may be realized by architectures that are all variations on a shared laminar design. These architectures include the:

LAMINART family of models of how the visual cortex, notably cortical areas V1, V2, and V4, interact together to see (Figure 11; e.g., Cao and Grossberg, 2005, 2012; Grossberg and Raizada, 2000; Grossberg and Versace, 2008; Raizada and Grossberg, 2001),

the LIST PARSE model of how prefrontal cortical working memory and list chunk learning in the ventrolateral and dorsolateral prefrontal cortices interact with adaptively-timed volitional processes in the basal ganglia to generate variable-speed motor trajectory commands in the motor cortex and cerebellum (Figure 12; Grossberg and Pearson, 2008),

the cARTWORD model of contextual interactions during speech perception by the auditory cortex, including backwards effects in time (Figure 13; Grossberg and Kazerounian, 2011),

the TELOS model of learning and choice of saccadic eye movement commands by interactions between prefrontal cortex (PFC), frontal eye fields (FEF), posterior parietal cortex (PPC) , anterior and posterior inferotemporal cortex (ITa, ITp), nigro-thalamic and nigro-collicular circuits of the basal ganglia (BG), superior colliculus (SC), and related brain regions (Figure 14; Brown et al., 1999, 2004).

and the lisTELOS model of learning and choice of sequences of saccadic eye movements, wherein an Item-Order-Rank spatial working memory in the prefrontal cortex (PFC) stores sequences of saccadic eye movement commands that can include repeats, and which are selected in the supplementary eye fields (SEF) as these regions interact with posterior parietal cortex (PPC), frontal eye fields (FEF), basal ganglia (BG), and superior colliculus (SC) to carry out operations such as loading the sequences in working memory, opening gates to enable the various processing stages to selectively generate their outputs, and releasing saccadic commands (Figure 15; Silver et al., 2011).

There are also bRNNs of cognitive-emotional interactions during reinforcement learning and motivated attention, such as the MOTIVATOR model, which can focus on valued goals and block learning of irrelevant events (Grossberg and Levine, 1987; Kamin blocking) by interactions of the object categories in the inferotemporal (IT) and rhinal (RHIN) cortices, the object-value categories in the lateral and medial orbitofrontal cortices (ORBl, ORBm), the value categories in the amygdala (AMYGD) and lateral hypothalamus (LH), and the reward expectation filter in several parts of the basal ganglia (Figure 16; Dranias et al., 2008),

the ARTSCAN model of view-invariant object learning and visual search during unconstrained saccadic eye movements by interactions between visual cortices V1, V2, V3A, and V4, prefrontal cortex (PFC), posterior parietal cortex (PPC) and lateral intraparietal area (LIP), posterior and anterior inferotemporal cortex (pIT, aIT), and superior colliculus (SC) (Figure 17; Fazl et al., 2009; Grossberg, 2009),

the ARTSCENE Search model of object and spatial contextual cueing of visual search for desired objects in a scene by interactions between visual cortices V1, V2, and V4, ventral and dorsolateral prefrontal cortex (VPFC, DLPFC), perirhinal cortex (PRC), parahippocampal cortex (PHC), anterior and posterior inferotemporal cortex (ITa, ITp), posterior parietal cortex (PPC), and superior colliculus (SC) (Figure 18; Huang and Grossberg, 2010),

and the GridPlaceMap model of entorhinal grid cell learning of hexagonal receptive fields, and hippocampal place cell learning of (primarily) unimodal receptive fields, in a hierarchy of Self-Organizing Maps that obey the same laws, driven by path integration signals that are generated as the model navigates realistic trajectories (Figure 19; Pilly and Grossberg, 2012).

# Interactions of STM and LTM during Neuronal Learning

# Unbiased spatial pattern learning by Generalized Additive RNNs

Various of the architectures above include interactions between STM and LTM that allow them to learn from their environments. The fact that these architectures "work" is based on a foundation of mathematical theorems which demonstrate how STM and LTM laws can be joined to design the most general networks capable of learning spatial patterns in an unbiased way, even when the cells in the network sample each other's activities through recurrent interactions.  These theorems demonstrate how unbiased learning can be achieved in networks with an arbitrarily large number of neurons, or neuron populations, that interact in suitable anatomies under general neurophysiological constraints. Once such spatial pattern learning is assured, then the results can be extended to demonstrate learning of any number of arbitrarily complicated space-time patterns, and to build from there in a series of steps towards the type of complexity that is found in primate brains. Some of these steps are reviewed below.

The theorem in this section shows that two types of anatomy and variants thereof are particularly well suited to spatial pattern learning: Let any finite number of cells \(v_i, i \in I\), send axons to any finite number of cells \(v_j, j \in J\). The cases \(I = J\) (complete recurrence) and \(I \cap J = \varnothing\) (complete non-recurrence) permit perfect pattern learning even if the strengths of axon connections from \(I\) to \(J\) are arbitrary positive numbers. In these anatomies, axon diameters can be chosen with complete freedom, and one can grow axons between cells separated by arbitrary distances without concern about their diameters. Grossberg (1969a, 1971b) summarizes how to extend these results to more general anatomies.

Only certain types of signal transmission between cells can compensate for differences in connection strengths, and thereby yield unbiased pattern learning (Grossberg, 1974, Section VI). The simplest possibility is to let action potentials propagate along the circumference of a cylindrical axon to the axon's synaptic knob (Hodgkin and Huxley, 1952). Let the signal disperse throughout the cross-sectional area of the synaptic knob as ionic fluxes, and let local chemical transmitter production/release in the knob be proportional to the local signal density.  Finally, let the effect of the signal on the postsynaptic cell be proportional to the product of local signal density, available transmitter density, and the cross-sectional area of the knob (Katz, 1969). By contrast, signals that propagate throughout the cross-sectional area of the axon, such as electrotonic signals, do not produce unbiased learning given arbitrary axon connection strengths.

Another constraint is that the time lag for signals to propagate from any cell to all the cells that it activates should be (almost) the same. How can different axons from a given source cell have the same time lag if they have different lengths? This property is achieved if signal velocity is proportional to axon length. But signal velocity is a local property of signal transmission, whereas axon length is a global property of the anatomy. How can this global property be controlled by a local one? This is possible if axon length is proportional to axon diameter, and signal velocity is proportional to axon diameter. The latter is often the case during spike transmission down an axon (Ruch et al., 1961, p. 73) and the former is qualitatively true: longer axons are usually thicker; see Cohen and Grossberg (1986) for developmental laws whereby this can happen. Systems with self-similar connections of this kind can be converted, through a coordinate change, into systems whose connections depend only on the source, or sampling, cells. The following Generalized Additive system is of this type. Its activities, or STM traces, \(x_i\) and adaptive weights, or LTM traces, \(z_{ij}\), obey:

\[ \tag{35} \frac{d}{dt} x_i = Ax_i + \sum_{k \in J} B_k z_{ki} + \theta_i C(t) \]

and

\[ \tag{36} \frac{d}{dt} z_{ji} = D_j z_{ji} + E_j x_i, \]

where the number of sampled cells (\(i \in I\)) and sampling cells (\(j \in J\)) can be arbitrarily large, and \(A\), \(B_j\), \(D_j\), and \(E_j\) can be continuous functionals, possibly highly nonlinear, of the entire past of the system. The signal functional \(B_j\) and the sampling functional \(E_j\) are non-negative, since they represent spike-based signaling terms. The decay functional \(D_j\) also includes a wide range of possibilities, including passive decay of associative learning, and gated steepest descent learning (Figure 1 and Figure 3). The terms \(\theta_i\) represent an arbitrary spatial pattern (\(\sum_{i \in I} \theta_i = 1\)), and different spatial patterns can be presented (\(C(t) > 0\)) as different combinations of sampling cells are active. Of particular note is the stimulus sampling operation, which means that learning only occurs if the sampling functional \(E_j > 0\). If both the decay and learning functionals equal zero (\(D_j = E_j = 0\)), then neither learning nor forgetting occurs. The stimulus sampling property enables arbitrary subsets of sampling cells to learn different spatial patterns through time; see Serial learning.

The Unbiased Spatial Pattern Learning Theorem proves how unbiased learning may occur in response to sampling signals, or conditioned stimuli (CS), that are correlated with particular spatial patterns, or unconditioned stimuli (US). This simple form of associative learning is also called classical, or Pavlovian, conditioning. The theorems prove that, if the system is bounded, and each CS and US are practiced together sufficiently often, then perfect pattern learning occurs (Grossberg, 1969a, 1971b). That is, the relative activities \(X_i = x_i \left( \sum_k x_k \right)^{-1}\) and \(Z_{ji} = z_{ji} \left( \sum_k z_{jk} \right)^{-1}\) approach the training pattern \(\theta_i\) without bias as time goes on, no matter how many sampling cells \(j \in J\) are simultaneously active, each with its own signaling, sampling, and decay functionals, even in a fully recurrent anatomy.

If the delays from a given cell to all of its target cells are not identical, properly designed networks can rapidly resynchronize the activities of the target cells using recurrent interactions (Grossberg and Somers, 1991; Grossberg and Grunewald, 1997; Somers and Kopell, 1993, 1995), even in laminar cortical circuits with realistic synaptic and axonal delays (Yazdanbakhsh and Grossberg, 2004).

# Outstar Learning Theorem

The simplest case of the Generalized Additive model in (35) and (36) occurs for the Outstar Learning Theorem (Grossberg, 1968b), in which the network has a single sampling cell (population) in \(J\) and a non-recurrent anatomy (Figure 1 and Figure 2). Given this theorem, the stimulus sampling operation suggests how a series of Outstars can learn an arbitrary spatiotemporal pattern as a series of spatial patterns, ordered in time; see Avalanches.

# Sparse Stable Category Learning Theorem

Another version of spatial pattern learning occurs using the dual network to the Outstar, namely the Instar (Figure 3 and Figure 5). When multiple Instars compete with each other via a RCF, they form a Competitive Learning or Self-Organizing Map network (Figure 4; Grossberg, 1976a; Kohonen, 1984; von der Malsburg, 1973). Grossberg (1976a) proved that, if there are not too many input spatial patterns presented sequentially to the network, relative to the number of available category learning cells, then category learning occurs with adaptive weights that track the input statistics, self-normalize, and lead to stable LTM, and the network has Bayesian decision properties. However, in response to a sequence of sufficiently dense non-stationary input patterns, the system can experience catastrophic forgetting in which previously learned categories are recoded by intervening input patterns (Carpenter and Grossberg, 1987, Grossberg, 1976a). Adaptive Resonance Theory, or ART, was introduced in Grossberg (1976b) to propose how top-down learned expectations and attentional focusing could dynamically stabilize learning in a Competitive Learning or Self-Organizing Map model in response to an arbitrary series of input patterns.

# Adaptive Bidirectional Associative Memory

Kosko (1987, 1988) adapted the Cohen-Grossberg model and Liapunov function (Cohen and Grossberg, 1983), which proved global convergence of STM, to define a system that combines STM and LTM and which also globally converges to a limit. The main trick was to observe how the symmetric connections in the Cohen-Grossberg equation (32) could be used to define symmetric LTM traces interacting reciprocally between two processing levels. An Additive Model BAM system is, accordingly, defined by:

\[ \tag{37} \frac{d}{dt} x_i = -x_i + \sum_k f(y_k) z_{ki} + I_i \]

and

\[ \tag{38} \frac{d}{dt} y_j = -y_j + \sum_m f(x_m) z_{mj} + J_i. \]

A Shunting Model BAM can also be analogously defined. One type of learning law to which BAM methods apply is the passive decay associative law that was introduced in Grossberg (1967, 1968b, 1968c); see Figure 1 and Figure 3:

\[ \tag{39} \frac{d}{dt} z_{ij} = -z_{ij} + f(x_i) f(x_j). \]

Kosko calls the equation in (39) the signal Hebb law, although it does not obey the property of monotonely increasing learned weights that Hebb (1949) ascribed to his law. Kosko (1988) wrote that: "When the BAM neurons are activated, the network quickly evolves to a stable state of two-pattern reverberation, or resonance". Indeed, another inspiration for BAM was Adaptive Resonance Theory, or ART.

# Adaptive Resonance Theory

Adaptive Resonance Theory, or ART, is currently the most advanced cognitive and bRNN theory of how the brain autonomously learns to categorize, recognize, and predict objects and events in a changing world. To a remarkable degree, humans can rapidly learn new facts without being forced to just as rapidly forget what they already know. Grossberg (1980) called this problem the stability-plasticity dilemma. It is also sometimes called the problem of catastrophic forgetting (Carpenter, 2001; French, 1999; Page, 2000). ART proposes a solution of this problem by demonstrating how top-down expectations (Figure 7) can learn to focus attention on salient combinations of cues ("critical feature patterns"), and characterizing how attention may operate via a form of self-normalizing "biased competition" (Desimone, 1998). In particular, when a good enough match between a bottom-up input pattern and a top-down expectation occurs, a synchronous resonant state emerges that embodies an attentional focus and is capable of driving fast learning of bottom-up recognition categories and top-down expectations; hence the name adaptive resonance. For a review of ART, see (Scholarpedia: Adaptive Resonance Theory). For a more comprehensive review, see Grossberg (2012; [7]).

# Working Memory: Processing and STM of Temporal Sequences

Intelligent behavior depends upon the capacity to think about, plan, execute, and evaluate sequences of events. Whether we learn to understand and speak a language, solve a mathematics problem, cook an elaborate meal, or merely dial a phone number, multiple events in a specific temporal order must somehow be stored temporarily in working memory. A working memory (WM) is thus a network that is capable of temporarily storing a sequence of events in STM (e.g., Baddeley, 1986; Baddeley and Hitch, 1974; Bradski et al., 1994; Cooper and Shallice, 2000); see Working memory). As event sequences are temporarily stored, they are grouped, or chunked, through learning into unitized plans, or list chunks, and can later be performed at variable rates under volitional control, either via imitation or from a previously learned plan. How these processes work remains one of the most important problems confronting cognitive scientists and neuroscientists.

# Relative activity codes temporal order in working memory

Grossberg (1978a, 1978b) introduced an Item-and-Order WM to explain how, as successive items in a list are presented through time, they may be stored in WM as a temporally evolving spatial pattern of activity across working memory cells (Figure 20). The "relative activity" of different cell populations codes the temporal order in which the items will be rehearsed. Items with the largest activities are rehearsed earliest. Hence, the name Item-and-Order working memory for this class of models. This representation represented a radical break from the popular model of Atkinson and Shiffrin (1971), which proposed binary activations of a series of linearly ordered "slots" wherein each item moves to the next slot as additional items are stored.

# Working memory design enables stable learning of list chunks

How is an Item-and-Order WM in the brain designed? In particular, is a WM a bRNN and, if it is, how could evolution discover a bRNN network to embody a function as seemingly sophisticated as a WM? Grossberg (1978a, 1978b) noted that WMs would be useless unless the item sequences that they temporarily stored could be unitized through learning into list categories, or chunks, for recognition and recall of familiar lists, much as words and motor skills are recognized and recalled.  He predicted that all WMs are designed to solve the temporal chunking problem; namely, WMs are designed to be able to learn a novel list chunk, under unsupervised learning conditions, from a sequence of stored items some of whose subsequences may have already learned to code their own list chunks, without forcing the previously learned list chunks to be forgotten. For example, a list chunk for the novel word MYSELF can be learned even when there may already be strong learned representations for the familiar words MY, SELF, and/or ELF. Why does not storage of MYSELF in WM distort the storage of its subwords MY, SELF, and ELF in a way that leads to catastrophic forgetting of their already learned list chunks?

# LTM Invariance and Normalization Rule are realized by specialized RCFs

Grossberg (1978a, 1978b) predicted that Item-and-Order models embody two constraints to ensure that stable learning and memory of list chunks can occur: the LTM Invariance Principle and the Normalization Rule. The LTM Invariance Principle is the main postulate to ensure that a new superset list chunk, such as MYSELF, can be learned without forcing catastrophic forgetting of familiar subset list chunks, such as MY, SELF, and ELF. As a result, subset list chunks can continue to activate their familiar list chunks until they are inhibited by contextually more predictive superset list chunks; e.g., until MY is supplanted by MYSELF through time. Mathematically, this postulate implies the following property: activities of items in working memory preserve their relative activations, or ratios, throughout the time that they are stored in working memory.

The Normalization Rule assumes that the total activity of the working memory network has a maximum that is (approximately) independent of the total number of actively stored items. In other words, working memory has a limited capacity and activity is redistributed, not just added, when new items are stored.

It was proved in Grossberg (1978a, 1978b) that these simple rules generate working memories that can support stable learning and long-term memory of list chunks. This analysis also showed that Item-and-Order WMs could be embodied by specialized recurrent on-center off-surround shunting networks, or RCFs, which are ubiquitous in the brain, thereby clarifying how WMs could arise through evolution. The recurrent connections in an RCF help to store inputs in short-term memory after the inputs shut off. An RCF obeys the LTM Invariance Principle due to the way that shunting, or multiplicative, interactions compute ratios of cell activities across the network; e.g., equation (14). The Normalization Rule follows from the tendency of RCFs to normalize total network activity; e.g., equation (24). As explained below, an RCF behaves like an Item-and-Order working memory model when it is equipped with a volitionally-activated nonspecific rehearsal wave to initiate read-out of stored activity patterns, and output-contingent self-inhibitory feedback interneurons to prevent perseverative performance of the most activity stored item (Figure 20).

The prediction that all WMs are specialized RCFs that obey the LTM Invariance Principle and Normalization Rule implies the additional prediction that all verbal, spatial, and motor WMs have a similar network design. For example, the LIST PARSE model predicts how such a WM can be realized in the deeper layers of ventrolateral prefrontal cortex and how list chunks of the stored sequences can be learned in the superficial layers of the same cortex (see Figure 12, Cognitive Working Memory). Item-and-Order WMs have also been generalized to Item-Order-Rank working memories in which rank, or positional, information is also included, thereby permitting the temporary storage of repeated items in a list, as in the list ABACBDE (Figure 15; Grossberg and Pearson, 2008; Silver et al., 2011).

# Primacy, recency, and bowed activation gradients

Free recall data were one source of inspiration for the discovery of Item-and-Order WMs. During free recall, a human tries to recall a once-heard list in any order. Typically, the beginning and end of the list are recalled earlier and with higher probability. If the brain is adaptively designed, then why are listed not recalled always in the correct temporal order?

It was mathematically proved that, under constant attentional conditions, the pattern of activation that evolves in an Item-and-Order working memory is one of following types (Bradski et al., 1992, 1994; Grossberg, 1978a, 1978b):

These hypotheses have found their way into many variants of the Item-and-order WM design (e.g., Boardman and Bullock, 1991; Houghton and Hartley, 1996; Page and Norris, 1998; Rhodes et al., 2004). Houghton (1990) called Item-and-Order models Competitive Queuing models.

# Experimental support

Item-and-Order WM properties have received support from many subsequent psychological and neurobiological experiments. Farrell and Lewandowsky (2004) wrote: “Several competing theories of short-term memory can explain serial recall performance at a quantitative level. However, most theories to date have not been applied to the accompanying pattern of response latencies…these data rule out three of the four representational mechanisms. The data support the notion that serial order is represented by a primacy gradient that is accompanied by suppression of recalled items”.

Averbeck et al. (2002, 2003a, 2003b) reported the first neurophysiological evidence in monkeys that a primacy gradient, together with inhibition of the most active cell after its command is read out, governs the sequential performance of sequential copying movements.

Jones et al. (1995) reported similar performance characteristics to those of verbal WM for a spatial serial recall task, in which visual locations were remembered. Agam et al. (2005) reported psychophysical evidence of Item-and-Order WM properties in humans as they perform sequential copying movements. Silver et al. (2011) used Item-and-Order WMs to simulate neurophysiological data about spatial WMs. The fact that verbal, spatial, and motor sequences, in both humans and monkeys, seem to obey the same WM laws provides accumulating evidence for the Grossberg (1978a, 1978b) prediction that all working memories have a similar design to enable stable list chunks to be learned

Agam et al. (2007) reported data consistent with the formation of list chunks as movement sequences are practiced, thereby supporting the Grossberg (1978a) prediction that WM networks are designed to interact closely with list chunking networks.

# Stable chunk learning implies the Magical Numbers Four and Seven

The Grossberg (1978a, 1978b) prediction that primacy gradients become bows for longer lists provides a conceptually satisfying explanation of the well-known immediate memory span of 7 +/- 2 items (Miller, 1956). Because relative activity translates into both relative order and probability of recall (bigger activities can provide more reliable recall in a noisy brain), such a model helps to explain why items from the beginning and end of a list in free recall may be recalled earlier and with larger probability (Murdock, 1962). Transposition errors also have a natural explanation in such a working memory, since stored items with similar activity levels will transpose their relative activities, and thus their rehearsal order, more easily than items with very different activity levels if noise perturbs these levels through time. Grossberg (1978a, 1978b) also proved that, if attention varies across items, then multi-modal bows, or Von Restorff (1933) effects, also called isolation effects (Hunt and Lamb, 2001), can be obtained by altering the relative sizes of stored activities. Von Restorff effects can also be caused by rate and feature similarity differences across items, factors that also influence bowing in the present modeling framework. Associative and competitive mechanisms can also cause Von Restorff effects during serial verbal learning (see Serial learning and Grossberg, 1969, 1974).

Grossberg (1978a) distinguished between the classical immediate memory span (IMS) of Miller (1956) and the then new concept of the transient memory span(TMS). The TMS was predicted to be the result of purely STM WM storage and recall, without a significant top-down long-term memory (LTM) component.  The TMS is, accordingly, the longest list length for which a WM can store a primacy gradient. The IMS was predicted to be the result of combining bottom-up inputs and top-down read-out of list chunk learned expectations on the relative activities stored in WM, and thus the temporal order that is recalled. Grossberg (1978a) proved that the TMS is smaller than the IMS. Estimating the IMS at the famous Magical Number Seven, it was predicted that the TMS would be around four. Cowan (2001) has reviewed experimental data that support the existence of a four plus-or-minus one WM capacity limit when LTM and grouping influences are minimized, consistent with this prediction. Indeed, long-term memory (LTM) does bias working memory toward more primacy dominance (e.g. Knoedler, 1999), and its influence can be difficult to limit. Also see Visual short term memory.

# Equations for some Item-and-Order RNNs

An Item-and-Order RCF with mathematically provable primacy, recency, and bowed gradient properties is defined by the family of STORE (Sustained Temporal Order REcurrent) models (Bradski et al., 1992, 1994). The STORE 1 model is defined by the following RNN:

Input: Let the input \(I_i(t)\) to the \(i^{th}\) neuronal population satisfy:

\[ \tag{40} I_i(t) = 1 \text{ if } \alpha_i - t_i < t < t_i \text{ and } 0 \text{ otherwise.} \]

Let the inter-input delays \(\beta_i\) be long enough for the system variables \(x_i\) and \(y_i\) to approach equilibrium.

Layer \(F_1\): Let the activity \(x_i\) of the \(i^{th}\) population in layer \(F_1\) satisfy:

\[ \tag{41} \frac{d}{dt} x_i = (AI_i + y_i - x_i x)I, \]

where \(x = \sum_{k=1}^n x_k\) and \(I = \sum_{k=1}^n I_k\).

Layer \(F_2\): Let the activity \(y_i\) of the \(i^{th}\) population in layer \(F_2\) satisfy:

\[ \tag{42} \frac{d}{dt} y_i = (x_i - y_i) I^c , \]

where \(I^c = 1 - I\). Initially, \(x_i(0) = y_i(0) = 0\). Equation (41) is a RCF with a broad off-surround \(x\) that can update its STM pattern only when some input is on; that is, when \(I > 0\). Equations (41) and (42) together define recurrent feedback loops whereby \(y_i\) can store a new value of \(x_i\) when all inputs shut off; that is, when \(I = 0\). While some inputs are on, previously stored values of \(y_i\) influence the STM pattern of \(x_i\)'s that is stored, without themselves being changed by them.

# Serial Learning: From Command Cells to Values, Decisions, and Plans

When sequences of items from a list are stored in STM, they can trigger learning and LTM that enable them to be fluently recalled at a future time. Such serial learning has been studied experimentally and theoretically for a long time in experimental psychology; e.g., Dixon and Horton (1968); Hovland (1938a, 1938); Hull et al. (1940); Jung (1968); McGeogh and Irion (1952); Osgood (1953), and Underwood (1952). In fact, the Additive Model and Shunting Model were first derived in order to explain associative learning of temporal order information in serial learning and related paradigms (Grossberg, 1969c; Grossberg and Pepe, 1970, 1971).  The step-wise historical development of models for learning of temporal order, leading to sophisticated bRNNs that can respond with increasing flexibility to different types of environmental feedback, can be summarized as follows:

# Avalanches

The properties of stimulus sampling and of encoding LTM in spatial pattern units show how to learn an arbitrary act, such as a piano recital, a dance, or a sequence of sensory images, in a minimal way. The simplest example, called an Avalanche (Grossberg, 1969d, 1970b, 1974), describes a ritualistic encoding wherein performance is insensitive to environmental feedback. In this case, only one cell is needed to encode the memory of an arbitrary space-time pattern. This fact shows that encoding complexity per se is relatively easy. Indeed, nervous systems with few cells can activate complicated behaviors, as is well known in invertebrates. The ritualistic construction is also universal, because such a cell can encode any act.

Intuitively, an Avalanche samples and learns a spatiotemporal pattern as a sequence of spatial patterns, much as we experience the continuous flow of scenes in a movie from a rapidly presented sequence of still pictures (Figure 21). In the Avalanche, a series of Outstars (black circuits in Figure 21) are sequentially activated (red series of connections in Figure 20). The Outstars sample a spatiotemporal pattern (green region in Figure 21) as they are sequentially activated by the sampling pulse (blue pulse in Figure 21). A related concept is the synfire chain (Abeles, 1991).

Despite their simplicity, Avalanche-type circuits occur in vivo. Figure 22 illustrates that an Avalanche-type circuit occurs in the HVC-RA network that controls songbird singing (Hahnloser et al., 2002). As illustrated by the bRNNs in Figure 12, Figure 13, and Figure 15, in addition to a primary circuit for temporally ordered recall, many other circuits, such as those in frontal cortex and the basal ganglia, are also needed to ensure flexible performance, at least in higher species. Even in the songbird, frontal and basal ganglia circuits modulate song performance (Andalman and Fee, 2009).

# Command cells and nonspecific arousal

Once a pulse activates the Avalanche in Figure 21, there is no way to stop it. If, for example, the Avalanche controlled the performance of a long dance, and the stage on which the dance was being performed began to burn, there would be no way to stop the dance in mid-course to escape the flames. Sensitivity to environmental feedback is possible only if the pulse can be abruptly terminated as it travels along the Avalanche axon, and replaced by a more appropriate escape behavior. Grossberg (1969d, 1970b) proposed that the minimal circuit for including such sensitivity to environmental feedback would include command cells (Figure 23). Command cells are, in fact, found even in invertebrates, where they control such stereotyped behaviors as the rhythmic beating of crayfish swimmerets (Stein, 1971).

Suppose that activation of a command cell is necessary to fire the chain of Avalanche cells (Figure 23). The command cell simultaneously sends signals to all of the Outstars within the Avalanche, which can now fire only if they receive a signal from the previous Outstar source cell and from the command cell ("polyvalence"). Thus, the command cell provides nonspecific arousal to the avalanche. Withdrawal of the command cell arousal can abruptly terminate output from the next link in the Avalanche. In addition, changing the size of the command cell signal can vary the speed of performance, with larger command signals causing faster performance speeds. Command cells are also familiar in the control of other behavioral acts in invertebrates (Carlson, 1968; Dethier, 1968). Competition between command cells can then determine which ritualistic behavior the system will activate.

Grossberg (1978a) describes a series of increasingly sophisticated mechanisms that modulate Avalanche performance, leading to ever-greater sensitivity to environmental feedback, including recurrent interactions. These include issues such as sensitivity to the value of actions for achieving desired goals, and the ability to volitionally decide what actions to perform and at what speed.

Concerning the former issue: There is a difference between aborting your dance on stage if the theater is being consumed by flames, and risking your career because a mosquito is hovering above. Only more important events should be able to shut off the arousal that supports a given act. Knowing what is important to an organism requires that the network can evaluate what events are rewarding and punishing. This issue historically led to the Cognitive-Emotional-Motor (CogEM) theory of reinforcement learning in which incentive motivation plays the role of a conditionable form of nonspecific arousal, and competition between different drive representations that control the incentive motivation can determine switching between different valued actions (Grossberg, 1971a, 1972a, 1972b, 1975; see also Armony et al. (1995) and Computational models of classical conditioning). These ideas and their generalizations and extensions led eventually to the MOTIVATOR model (Figure 16).

Concerning the latter issue: How does an organism decide what act to perform? This question involves issues about volitional control of behavioral choice by prefrontal-basal ganglia circuits that led eventually to circuits such as TELOS and lisTELOS (Figure 14 and Figure 15) and related models (e.g., Frank and Claus, 2006; Schultz et al., 1997). The ritualistic performance in Avalanches hereby focused attention in Grossberg (1978a) and thereafter on multiple issues concerning the global organization of brain mechanisms that are sensitive to different kinds of environmental feedback. Articulating these mechanisms led to the types of high-dimensional bRNNs that are illustrated in Figure 11-Figure 19 and that are familiar in advanced brains.

# Self-organizing avalanches: Instar-outstar maps and serial learning of temporal order

The Outstar source cells and the links between them are pre-wired in an Avalanche (Figure 21). These limitations led Grossberg (1972, 1976a, 1976b) to interact through the published literature with von der Malsburg (1973; see also Willshaw and Malsburg, 1976)) to introduce Competitive Learning and Self-Organizing Maps (Figure 4) so that the source, or sampling, cells could self-organize as learned category cells.

After the Outstar source cells self-organize, they need to learn the spatial patterns that they will perform (Figure 24), as occurs in pre-wired Outstars as well. Taken together, these learned Instars and Outstars define the Instar-Outstar associative map learning circuits (Figure 6) that were introduced in Grossberg (1976a).

If the source cells self-organize, then the links between them must also be learned. This is the problem of serial learning.  The simplest network capable of learning an arbitrary temporal order among its constituent cells is a fully-recurrent RNN (Figure 25) whose sampling cells can sequentially learn to embed a temporal order of performance in the network, by building on the guarantee of the Unbiased Spatial Pattern Learning Theorem; see equations (35) and (36). Grossberg (1969c) and Grossberg and Pepe (1970, 1971) provide mathematical analyses of how serial learning can proceed through time, and thereby explain classical data properties such as the bowed serial position curve. The net result of all these learning processes is a Self-Organizing Avalanche that can learn its sampling cells, its temporal order links, and its output spatial patterns (Figure 24).

# Context-Sensitive Self-Organizing Avalanches: What categories control temporal order?

Once a mechanism is in place for learning categories that act as sampling cells, the question arises: What do these categories code? In Avalanches, each link in the associative chain is sensitive only to the previous link. However, in many types of tasks, information about more than one previous event or action is needed to choose the correct subsequent action. This issue led to the introduction of Item-and-Order working memories (Figure 20) so that list chunks could be learned which are sensitive to whole sequences of previous events. In such a network, list chunks are the sampling cells that are linked through serial learning into a temporally ordered circuit. List chunks also play the role of planning nodes through read-out of their learned top-down spatial patterns and serial links. Such a network is called a Context-Sensitive Self-Organizing Avalanche, or a Context-Sensitive Avalanche, for short.

# Serial learning

Issues about what previous events control subsequent responses were articulated in the classical literature about serial verbal learning. New verbal units are continually being synthesized as a result of practice, and need not be the obvious units that the experimentalist is directly manipulating. Indeed, entire sequences of previous events can create the context that determines the next response. The same problem arises in verbal, spatial, and motor learning. The concept of list chunks was introduced to explain such learned sequence-sensitive contextual control.

The severity of such difficulties led the serial learning expert Young (1968, p. 146) to write: "If an investigator is interested in studying verbal learning processes ... he would do well to choose some method other than serial learning". Underwood (1966, p. 491) went even further by writing: "The person who originates a theory that works out to almost everyone's satisfaction will be in line for an award in psychology equivalent to the Nobel prize". The mechanisms summarized in this review enable many of the classical serial learning data that inspired these statements to be explained and simulated. However, a full discussion of these data and their explanations goes beyond the scope of the current review. See Grossberg (1969c) and Grossberg and Pepe (1970, 1971) for explanations and simulations of classical serial learning data, and Grossberg (1978a, 1993) for reviews.
