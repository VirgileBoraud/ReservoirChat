**Beginner** 
 

- What is an echo state network? 

  

A. A place where anything is kept in store. 

B. A recurrent neural network in which usually only the output neurons are trained. 

C.  A black box component that receives an input signal to be read out and mapped by another process. 

D. A large natural or artificial lake used as a source of water supply. 

  

 ---

 

- What is reservoir computing? 

  

A. A machine learning paradigm for timeseries. 

B. The construction of a computer model of a petroleum reservoir. 

C. A branch of fluid dynamics that studies the design of tank trucks to reduce rollover accidents. 

D. It is only a computational framework in which all the computations are done by a pool of water. 

  

 ---


 

- What is call the readout? (in the context of reservoir computing) 

  

A. A linear regression that models the markovian process of a support vector machine. 

B. An efficent way for input processing in order to read only part of the inputs. 

C. A black box component that receives an input signal to be read out and mapped back to the input for predictive coding. 

D. A trainable layer that extracts the dynamics of a reservoir to produce some desired outputs. 

  

 ---

 

- On which task is reservoir computing is known to compete with the state-of-the-art?  

  

A. Video segmentation. 

B. Natural language processing. 

C. Image recognition. 

D. Chaotic timeseries prediction. 

   

 ---

 

- On which task is it challenging to apply reservoir computing compared to other state-of-the-art methods? 

  

A. A task with small univariate dataset. 

B. Video segmentation. 

C. Chaotic timeseries prediction. 

D. Classification of timeseries with few classes. 

  

 ---

 

- Approximately how many neurons are used inside an echo state network? 

  

A. around 1 thousand neurons 

B. around 100 thousand neurons 

C. around 10 neurons 

D. around 1 million neurons 

 

  ---

 

- What is the purpose of using ridge regression instead of linear regression for the readout? 

  

A. Reduce the computational cost. 

B. Improve numerical stability. 

C. Improve explainability of the model. 

D. Avoid the exploding/vanishing gradient problem. 

   

 ---

 

- How are the weights most often initialized in an echo state network? 

  

A. They can be randomly initialized and then scaled to have a specific spectral radius. 

B. They are tuned according to an autocorrelation Hebbian learning rule. 

C. Trained using a linear regression. 

D. Each neuron is connected to only one input. 

   

 ---

**Intermediate** 

 

- What is the difference between ‘echo state network’ and ‘reservoir computing’? 

  

A. Reservoir computing is a type of recurrent neural network model based on the philosophy of the echo state network. 

B. There is no difference, we can use the terms reservoir computing and echo state network indistinctly. 

C. An echo state network is a type of recurrent neural network model based on the reservoir computing paradigm. 

D. In reservoir computing, the reservoir part is a physical system, in echo state networks, the reservoir is a recurrent neural network. 

   

 ---

 

- Are there other forms of reservoir computing than echo state networks? 

  

A. No, an echo state network is not even a form of reservoir computing. 

B. Yes, any random kernel method, even those who don't apply to timeseries, are considered to be a form of reservoir computing. 

C. No, reservoir computing only refers to the philosophy behind echo state networks. 

D. Yes, there are for instance physical reservoirs or liquid state machines, in which the reservoir is a spiking neural network. 

   

 ---

 

- What is a liquid state machine? 

  

A. An architecture in which the reservoir part is a pool of spiking neurons. 

B. A physical reservoir in which the reservoir is a reservoir of liquid, usually water. 

C. Liquid state machine and reservoir computing designates the same concept. 

D. A computational model of the hippocampus using the reservoir computing paradigm. 

   

 ---

 

- Why is it called ‘computing at the edge of chaos’? 

  

A. Because it is common to add noise (and thus chaos) to the reservoir in order to stabilize its activity. 

B. Because reservoir computing works best for chaotic timeseries forecasting. 

C. Because reservoir computing often works best when the dynamics of the reservoir are approaching chaotic dynamics, but are not chaotic. 

D. It requires computers to be physically placed at the edge of a chaotic environment, such as a volcano or a kindergarten. 

   

 ---

 

- What is the ‘echo state property’? 

  

A. The echo state property allows the reservoir to operate with chaotic behavior to enhance computational power. 

B. The ability to memorize past activity and cycle over it (the echo). 

C. The ability to perfectly reconstruct any input signal from the reservoir activity. 

D. The influence of initial states on the reservoir dynamics decays over time, ensuring that the current state primarily reflects recent inputs. 

   

 ---

 

- What are some of the most important hyper-parameters? 

  

A. The spectral radius of the recurrent weight matrix, the leak rate, the input scaling. 

B. The spectral radius of the recurrent weight matrix, the reservoir connectivity, the weight distribution. 

C. The spectral radius of the recurrent weight matrix, the reservoir connectivity, the incoding of the input. 

D. The activation function, the reservoir connectivity, the regularization parameter. 

   

 ---

**Advanced** 

 

- How explainable are reservoir computing models? 

  

A. Reservoir computing models are generally less explainable due to their reliance on complex, nonlinear dynamics within the reservoir, making it difficult to trace the exact path of information processing. 

B. Reservoir computing models are fully explainable because they rely on predefined, static patterns that do not change over time. 

C. Reservoir computing models are highly explainable because they use simple linear transformations that can be easily interpreted. 

D. Reservoir computing models are completely explainable due to their deterministic nature, which allows for perfect traceability of every computation. 

   

 ---

 

- To what extent do the results vary between two differently initialized reservoirs? 

  

A. The results between two differently initialized reservoirs are completely unpredictable and random, regardless of the input data. 

B. The results between two differently initialized reservoirs are always identical because the initialization has no impact on the reservoir's dynamics. 

C. The results between two differently initialized reservoirs can vary somewhat, but the overall performance and behavior of the reservoir are generally robust to different initialisations, provided the reservoir is sufficiently large and well-designed. 

D. The results between two differently initialized reservoirs vary significantly because the initialization determines the exact sequence of outputs. 

   

 ---

 

- What influence does the sparsity of the weight matrix have on performance? 

  

A. Increasing the connectivity of the reservoir always decreases performance by reducing the richness of the interactions between the neurons. 

B. Given a minimum number of connections, the sparsity of the weight matrix only affects the speed of computation, not so much the accuracy of the results. 

C. The sparsity of the weight matrix can significantly influence performance; a moderate level of sparsity can enhance the reservoir's ability to capture complex dynamics, while too much or too little sparsity can degrade performance. 

D. Increasing the sparsity of the weight matrix always decreases performance by reducing possible non-linear combinations. 

   

 ---

**NEW Knowledge Based** 

 

- What is the effective spectral radius? 

  

A. The real spectral radius of the matrix W, that is always a bit different from the specified spectral radius. 

B. A weighted sum of all eigenvalues norms, that takes into account the distribution of the spectrum. 

C. A value that has similar properties to the spectral radius of a matrix, taking into account the full reservoir equation 

D. The norm of the singular values for each neuron. It is a way to evaluate the impact of each neuron on the reservoir 

   

 ---
 


- What is a deep reservoir? 

  

A. A deep reservoir is a reservoir computing architecture that consists of multiple layers of interconnected reservoirs, allowing for hierarchical processing and the capture of more complex temporal dynamics. 

B. An underground gas or petroleum reservoir that cannot be reached using traditional tools and infrastructure. 

C. A deep reservoir is a reservoir that employs deep learning techniques, such as backpropagation, to train the weights within the reservoir, enhancing its ability to learn from data. 

D. A deep reservoir is a reservoir that uses extremely large and dense weight matrices to store vast amounts of data. 

   

 ---

 

- What is the use of an orthogonal matrix in the reservoir equation ? (rep : augmenter la mémoire) 

  

A. An orthogonal matrix can be represented in a condensed form, improving matrix multiplication computation time 

B. An orthogonal matrix in the reservoir equation is used to prevent any interaction between neurons, maintaining independence. 

C. This is a trick question, there is no point in using an orthogonal matrix 

D. It augments the memory capacity of the reservoir 

   

 ---

 

- What is a Conceptor? 

  

A. A conceptor is a mathematical function used to compress the data within a reservoir, reducing its dimensionality for faster processing. 

B. A conceptor is a hardware component that accelerates the computation of reservoir dynamics by offloading calculations to a dedicated processor. 

C. A derivation of reservoir computing which can store and recall patterns. 

D. A conceptor is a specialized type of neuron within a reservoir that is designed to store and retrieve specific concepts or patterns. 