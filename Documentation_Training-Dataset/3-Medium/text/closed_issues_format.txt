## Issue 159: How can I save the trained model?

**Question 1:** I saw that the save function is available v0.2 [https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.compat.ESN.html]. But when I tried it for the 0.3.11, it failed.





Could you help me on how to save the trained NN?

Thank you.

**Response 1:**
> Hello,

Please take a look at #137 .

In short, we don't have any save function, but pickling should work just fine. Please open a new issue if this doesn't work or if it doesn't suit your needs.


---

---


## Issue 157: ESN Parameter Effects

**Question 1:** I am trying to perform system identification using reservoir computing. I have .mat data consisting of the time and values of the input and output to a system. I have attached this data (named "train_trial_60s.xlsx"), as well as a script (named "rpy_RCP.txt") that reads in this data, separates it into training and testing (trial) sets, trains an ESN on the training set, and evaluates it on the testing set. The original data was a .MAT file (train_trial_60s.mat) while the original code was a Python script (rpy_RCP.py)



Originally, the reservoir of my ESN contained 1000 units, a leaking rate 'lr' = 1.0, and a spectral radius 'sr' = 1.0. This resulted in an RMSE of 45.719 and R^2 of -272.420. However, when I repeat the experiment with the leaking rate and spectral radius omitted (keeping the number of units equivalent, the regression improves to an RMSE of 0.738 and R^2 of 0.918. 



What does this mean? Why does removing the leaking rate and spectral radius affect the performance to this degree (especially when the documentation states that the default value for the leaking rate = 1, so why does explicitly stating this lead to a worse performance)?



I also noticed that, for the case where the ESN is trained with just the number of units defined, that despite the better performance, the regression result appears very 'noisy'. I am curious as to why this is as well.



**Response 1:**
> Hello,

Indeed, specifying the leaking rate to 1.0 shouldn't change anything, as this is already the default in ReservoirPy.

The difference in performance comes from the spectral radius : 

- if not specified, a random sparse matrix W is generated, and its non-zero values are normally distributed (you can also change the distribution).

- if specified, the same matrix is created, and is then scaled to have a spectral radius of `sr`



In your case, the spectral radius you have if you don't specify its value is around ~10. You can get its value by using the reservoirpy.observables.spectral_radius method:

```python

from reservoirpy.nodes import Reservoir

from reservoirpy.observables import spectral_radius

import numpy as np



my_reservoir = Reservoir(

    units=1000,

    # sr=1.0

)



my_reservoir.initialize(np.random.normal(size=(12, 1)))



spectral_radius(my_reservoir.W)

```



The spectral radius of the recurrent connection weights has a significant impact on the performances of the task, so it's not a surprise that you have such bad performances if you don't specify it. You can read more about its impact [in the documentation](https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html#Spectral-radius).

**Question 2:**
> Hello Paul,



Thank you for your response. 



I actually achieved my best response when I left the spectral radius unspecified. I even attempted a hyperparameter optimization using Hyperopt, but my ESN with just the number of units specified outperformed the ESN with the optimized hyperparameters. I've attached some figures to illustrate what I mean. 



Despite the better result, the unspecified predictions appear very noisy. I am guessing this has to do with the chaotic nature of the reservoir due to the high spectral radius. I was wondering if there was a way to smooth this out. I thought about using a filter, but my initial attempt proved unsuccessful.



*All the "Zoom In" figures are from the ESN with unspecified leaking rate and spectral radius.



**Response 3:**
> Hello,

Its difficult to tell why you don't have better results with an hyper-parameter exploration. Can you provide more details about it ? What kind of performances does the hyper-parameter exploration gives you around the default parameters ?



It seems you have more units in the default reservoir (500) than in the optimized version (150). That could explain the performance decrease.



For the smoothness of your output, many parameters comes into play, with high inter-dependencies.

**Question 4:**
> Hello Paul,



I attached the hyperparameter optimization script, as well as the data I used to train the ESN. When I ran it, I got the values stated in the rctest_hyperopt figure that I shared previously.



As far as the number of units in my reservoir, I started with 150 and found that the performance increased as I increased the number of units.



**Question 5:**
> Hello,



To illustrate my question further, in a previous experiment, I was able to create a ESN model (rpy_PK4GA7P1_ESN.py) that performed well in predicting future dynamics of a piezoelectric actuator. 



When I rerun the same experiment with a new set of data from an electric linear actuator (rpy_RCP.py), keeping the structure of the model the same, I get very different performance results (RMSE: 14.22177, R^2: -16.21228). Furthermore, when I rerun the experiment again on the electric linear actuator data, but leave the leaking rate and spectral radius unspecified, the performance improves (RMSE: 0.75309, R^2: 0.95170), but still not to the level that was achieved with the piezoelectric data. I am wondering what adjustments I should make in order to achieve similar regression results to the piezoelectric data with the electric linear data.


**Response 6:**
> Hello,



Thank you for testing new tasks with ReservoirPy.

Reservoir Computing rarely works just out-of-the-box, as it is a machine learning tool with few trained parameters (only the output layer), one need to find optimal hyperparameters (parameters not trained) for each kind of task.



Here there are several factors that could influence your performances. 

1. Normalize your data if not done. Between -1 and 1 by default, or maybe between 0 and 1 if you have only positive values.

2. Start with the simplest model, i.e. an ESN that do not have a feedback from the read-out layer and the reservoir. This makes the training more complicated and more unstable, in particular if you use offline learning (ridge) and not online learning (RLS, FORCE, ...).

3. Make an extensive hyperparameter search, including changing the ridge parameter (regularization parameter). It is also important to keep fixed the number of units inside the reservoir while doing this search, because results will be less interpretable. It is better to make several searches with fixed number of units for each search, but increase the number of units for different searches instead.

4. Look at the results of hyperparameter search to understand what are the most robust set of hyperparameters, and do not just take the "best" result. If you just want to take the best, then be careful to take also the same seed, to be sure to obtain the same results. You can find several exemples of the influence of hyperparameters and how to look at the hyperparameter search results:

https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/4-Understand_and_optimize_hyperparameters.ipynb



This kind of plot will help you to understand what are the hyperparameters that give the most robust results.

<img width="658" alt="Capture d’écran 2024-04-24 à 14 06 36" src="https://github.com/reservoirpy/reservoirpy/assets/9768731/d92d4477-accf-4eb1-a13a-5c584eeaf168">



I hope this helps. If you show us this kind of plot for all the hyperparameters, we could help you interpret them.



**Response 7:**
> Additionally, we provide some hints of how to optimise hyperparemeters in this paper:



Which Hype for My New Task? Hints and Random Search for Echo State Networks Hyperparameters. ICANN 2021 [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_7) [HAL](https://hal.inria.fr/hal-03203318) [PDF](https://hal.inria.fr/hal-03203318)

---

---

## Issue 156: LMS doesn't work for single node readout

**Question 1:** I'm trying to replicate the results of this paper.

https://www.sciencedirect.com/science/article/abs/pii/S0957417421004632



It predicts a stock price in the next time step, so I'm hoping ESN outputs to be a single output.

I set LMS's output_dim = 1. But the ESN model returns 1 to 5 predictions in a single inference.

Any idea what I'm doing wrong ? 



```

  1 import reservoirpy as rpy

  2 import matplotlib.pyplot as plt

  3 import numpy as np

  4 import csv

  5 

  6 from reservoirpy.mat_gen import uniform, bernoulli

  7 from reservoirpy.nodes import IPReservoir, LMS

  8 from scipy.stats import expon 

  9 

 10 # Config

 11 rpy.verbosity(0)

 12 rpy.set_seed(123456789)

 13 

 14 filename = "stockPred_input.txt"

 15 

 16 window_size = 5

 17 train_ratio = 0.8

 18 

 19 reservoir = IPReservoir(

 20     units = 100,

 21     sr = 0.95,

 22     mu = 0.3,

 23     learning_rate = 5e-4,

 24     #input_scaling = 0.1,

 25     W = uniform( high = 1.0, low = -1.0),

 26     Win = bernoulli,

 27     rc_connectivity = 0.1,

 28     input_connectivity = 0.1,

 29     activation = "tanh", 

 30     epochs = 100 

 31 )   

 32 

 33 readout = LMS( output_dim = 1, Wout = bernoulli )

 34 esn_model = reservoir >> readout 

 35 

 36 # Data Load

 37 with open( filename, "r" ) as f:

 38   price_data = np.loadtxt( f, dtype = int )

 39   

 40 price_data = np.reshape( price_data, ( len( price_data ), 1 ) )

 41 train_len = round( len( price_data ) * train_ratio )

 42 eval_len = len( price_data ) - train_len

 43 

 44 

 45 # Train ESN

 46 esn_model = esn_model.fit( price_data[:len(price_data)-1], price_data[1:len(price_data)] )

 47 

 48 # Evaluate ESN

 49 pred = []

 50 eval_data = price_data[ train_len : ]

 51 for i in range ( 0, eval_len ):

 52   pred_next = esn_model.run( eval_data[ i : i + window_size ], shift_fb = False )

 53   print( pred_next )

 54   pred.append( pred_next )

 55   

 56 #plt.plot( pred )

 57 #plt.plot( eval_data )

 58 plt.savefig( "stockPred.png" )

 ```

**Question 1:**
> Found workaround here

https://reservoirpy.readthedocs.io/en/latest/user_guide/advanced_demo.html#Generation-and-long-term-forecasting



```

Y_pred = np.empty((100, 1))

x = warmup_y[-1].reshape(1, -1)



for i in range(100):

    x = esn_model(x)

    Y_pred[i] = x

```

---

---

## Issue 155: Creating a reservoir of custom nodes

**Question 1:** Hello, I have been working with `reservoirpy` with the goal of performing time-series classification using a reservoir of custom nodes, subclassed from the `Node` class .



I have followed the `Node` documentation to create a subclassed `Node` which redefines the `forward` function to compute states through oscillatory behaviour, however I now need to create a reservoir of multiple of these nodes with a connectivity rate and weights, which is slightly more confusing as there is no documentation on this and it doesn't seem accessible.



The `Reservoir` class seems to be statically bound to use a pool of leaky-integrator neurons, however it looks possible to create custom reservoirs using the `nodes/reservoirs/base.py` methods.



Do you have any guidance here on how this could be implemented, or could any documentation be added for creating custom Reservoirs?

**Response 1:**
> Hello, sorry for the delay. 



Connecting `Node`s together to create a reservoir should be feasible, but it's not the way we think about `Node`s in ReservoirPy.

They are parts of a reservoir computing architecture. In the standard Echo State Network model, we have a `Reservoir` node connected to a `Ridge` node.

But you can create and connect more nodes to create more complex architectures ([here are some examples](https://arxiv.org/pdf/1712.04323.pdf)).



The `Reservoir` node is a pool of **neurons**. What you probably want to do is to subclass `Node` for your specific reservoir, that will hold your custom-defined neurons.



Let me know if you still have questions, feel free to close the issue if not

**Question 2:**
> Thank you for your response, I apologise for my confusion, I understand now what a `Node` actually is, I had assumed a reservoir was built up from `Node`s, but that is not the case, a reservoir is a single Node.



That being said, I think it could be useful to include an example of creating a custom Reservoir in the tutorials / documentation on how the Reservoir Node works, to promote research into the capabilities of reservoir computing for people who may not be as familiar with software engineering.



But thank you for your help, and for developing this library :)

---

---

## Issue 154: cant do long term forecasting on yahoo stock market data

**Question 1:** I have used the ReservoirPy long term forecasting example to successfully forecast Lorenz, Mackey Glass, Atrial Fibrillation, and Channel State Information datasets. All chaotic datasets. When I try to forecast stock market datasets the forecast produced is nonsense (more than half of the predictions are the same data value). Is ReservoirPy designed to only predict on chaotic structured data? Any examples of Stock Market data forecasting?

**Response 1:**
> Hello,

If the predictions of your model is not good, you can try to tune the hyper-parameters ( see https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html ).



Otherwise, I would say that predicting stock market is a very difficult task, so I'm not surprised that you have worse results than on Mackey-Glass or Lorenz. 



> Is ReservoirPy designed to only predict on chaotic structured data?



Reservoir computing in general is a good framework for chaotic data, but it can be used for other tasks that are not really chaotic.



> Any examples of Stock Market data forecasting?



As far as I know, not with ReservoirPy, but there are a few articles here and there about using Echo State Network for this task (but I haven't read them, so I can't point specific ones).

**Question 2:**
> Thank you.



On Wed, Apr 3, 2024 at 8:11 AM PAUL BERNARD ***@***.***>

wrote:



> Hello,

> If the predictions of your model is not good, you can try to tune the

> hyper-parameters ( see

> https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html ).

>

> Otherwise, I would say that predicting stock market is a very difficult

> task, so I'm not surprised that you have worse results than on Mackey-Glass

> or Lorenz.

>

> Is ReservoirPy designed to only predict on chaotic structured data?

>

> Reservoir computing in general is a good framework for chaotic data, but

> it can be used for other tasks that are not really chaotic.

>

> Any examples of Stock Market data forecasting?

>

> As far as I know, not with ReservoirPy, but there are a few articles here

> and there about using Echo State Network for this task (but I haven't read

> them, so I can't point specific ones).

>

> —

> Reply to this email directly, view it on GitHub

> <https://github.com/reservoirpy/reservoirpy/issues/154#issuecomment-2034410365>,

> or unsubscribe

> <https://github.com/notifications/unsubscribe-auth/AAPTOI4MKOIKOBSMS3SQTWLY3PWYDAVCNFSM6AAAAABFFW3O62VHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDAMZUGQYTAMZWGU>

> .

> You are receiving this because you authored the thread.Message ID:

> ***@***.***>

>



**Response 3:**
> There are papers doing stock market prediction. One was done by Herbert Jaeger himself with students for a competition:

Iles et al (2007) Stepping forward through echoes of the past: forecasting with echo state networks

https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=368e4c95c69d3499ff1a1e46cf80b855c3b97fb5

**Question 4:**
> Thanks so much.



On Thu, Apr 4, 2024, 10:54 AM Xavier Hinaut ***@***.***>

wrote:



> There are papers doing stock market prediction. One was done by Herbert

> Jaeger himself with students for a competition:

> Iles et al (2007) Stepping forward through echoes of the past: forecasting

> with echo state networks

>

> https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=368e4c95c69d3499ff1a1e46cf80b855c3b97fb5

>

> —

> Reply to this email directly, view it on GitHub

> <https://github.com/reservoirpy/reservoirpy/issues/154#issuecomment-2037435562>,

> or unsubscribe

> <https://github.com/notifications/unsubscribe-auth/AAPTOI5PDA5L654JZSTYT3LY3VST3AVCNFSM6AAAAABFFW3O62VHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDAMZXGQZTKNJWGI>

> .

> You are receiving this because you authored the thread.Message ID:

> ***@***.***>

>



---

---

## Issue 153: Understand and optimize ESN hyperparameters errors

**Question 1:** Hello,



The code on your webpage [Understand and optimize ESN hyperparameters](https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html#Understand-and-optimize-ESN-hyperparameters) returns errors. When I run it as is, I receive: 



TypeError: objective() got an unexpected keyword argument 'leak'



I then tried to change "leak" in the hyper config dictionary definition and received this instead:



KeyError: 'leak'



Also, I am not able to load the [Understand and Optimize Hyperparameters Jupyter notebook](https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/4-Understand_and_optimize_hyperparameters.ipynb) in your github tutorials page. 

**Response 1:**
> Just changing the key ```"leak"``` in the dict ```hyperopt_config``` to ```"ls"```  fixes the problem.

**Question 2:**
> Hello,



I wasn't clear in my original message, however, I tried to change `leak` in the `hyper config` dictionary definition to `ls` and received this instead:



`KeyError: 'leak'`

**Response 3:**
> Hello, sorry for the delay,



Indeed, there was an error in the documentation (which was not in the `tutorials/`). This should be fixed now, thanks to @jodemaey :)



> Also, I am not able to load the [Understand and Optimize Hyperparameters Jupyter notebook](https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/4-Understand_and_optimize_hyperparameters.ipynb) in your github tutorials page.



I am able to access the tutorial you mention, can you try again or on a different computer ?

---

---

## Issue 152: Is the long term forecasting example opertion explanation correct

**Question 1:** I don't understand explanation of the long term forecasting example predicts 100 steps ahead. Below according to the tutorial 100 next steps will be predicted. Training is on the last 10 steps which sets the current state. 



What I don't understanding is why it is stated that "Based on this state, we will now predict the next step in the timeseries. Then, this predicted step will be fed to the ESN again, and so on 50 times, to generate the 50 following timesteps. In other words, the ESN is running over its own predictions."  It's the "to generate the 50 flowing timesteps" statement that I don't understand as the prediction loop is running over 100 timesteps.



Y_pred = np.empty((100, 1))

x = warmup_y[-1].reshape(1, -1)



for i in range(100):

    x = esn_model(x)

    Y_pred[i] = x



Generation and long term forecasting example

In this section, we will see how to use ReservoirPy nodes and models to perform long term forecasting or timeseries generation.



We will take a simple ESN as an example:



from reservoirpy.nodes import Reservoir, Ridge



reservoir = Reservoir(100, lr=0.5, sr=0.9)

ridge = Ridge(ridge=1e-7)



esn_model = reservoir >> ridge

Imagine that we now desire to predict the 100 next steps of the timeseries, given its 10 last steps.



In order to achieve this kind of prediction with an ESN, we first train the model on the simple one-timestep-ahead prediction task defined in the sections above:



esn_model = esn_model.fit(X_train, Y_train, warmup=10)

Now that our ESN is trained on that simple task, we reset its internal state and feed it with the 10 last steps of the training timeseries.



warmup_y = esn_model.run(X_train[:-10], reset=True)

This updated the state of the reservoir inside the ESN to some value. We assume that this value is representative of the dynamics of these 10 timesteps.



Based on this state, we will now predict the next step in the timeseries. Then, this predicted step will be fed to the ESN again, and so on 50 times, to generate the 50 following timesteps. In other words, the ESN is running over its own predictions.



Y_pred = np.empty((100, 1))

x = warmup_y[-1].reshape(1, -1)



for i in range(100):

    x = esn_model(x)

    Y_pred[i] = x

plt.figure(figsize=(10, 3))

plt.title("100 timesteps of a sine wave.")

plt.xlabel("$t$")

plt.plot(Y_pred, label="Generated sin(t)")

plt.legend()

plt.show()



**Response 1:**
> Thank you for your issue ! This definitely isn't right.

"50" should be "100"

**Question 2:**
> Thanks for your quick response. Also note that if the intention is to truly

get the X_train last 10 steps I believe the current code is wrong. It is

currently warmup_y = esn_model.run(X_train[:-10], reset=True). This would

get all the X_train values up to the start of the last 10 values. warmup_y

= esn_model.run(X_train[-10:] would get the last 10 values.



As I tested:

print (X_train) [[ 0.00000000e+00]

 [ 1.89251244e-01]

 [ 3.71662456e-01]

 [ 5.40640817e-01]

 [ 6.90079011e-01]

 [ 8.14575952e-01]

 [ 9.09631995e-01]

 [ 9.71811568e-01]

 [ 9.98867339e-01]

 [ 9.89821442e-01]

 [ 9.45000819e-01]

 [ 8.66025404e-01]

 [ 7.55749574e-01]

 [ 6.18158986e-01]

 [ 4.58226522e-01]

 [ 2.81732557e-01]

 [ 9.50560433e-02]

 [-9.50560433e-02]

 [-2.81732557e-01]

 [-4.58226522e-01]

 [-6.18158986e-01]

 [-7.55749574e-01]

 [-8.66025404e-01]

 [-9.45000819e-01]

 [-9.89821442e-01]

 [-9.98867339e-01]

 [-9.71811568e-01]

 [-9.09631995e-01]

 [-8.14575952e-01]

 [-6.90079011e-01]

 [-5.40640817e-01]

 [-3.71662456e-01]

 [-1.89251244e-01]

 [-2.44929360e-16]

 [ 1.89251244e-01]

 [ 3.71662456e-01]

 [ 5.40640817e-01]

 [ 6.90079011e-01]

 [ 8.14575952e-01]

 [ 9.09631995e-01]

 [ 9.71811568e-01]

 [ 9.98867339e-01]

 [ 9.89821442e-01]

 [ 9.45000819e-01]

 [ 8.66025404e-01]

 [ 7.55749574e-01]

 [ 6.18158986e-01]

 [ 4.58226522e-01]

 [ 2.81732557e-01]

 [ 9.50560433e-02]]



print("last 10 training steps", X_train[-10:]) [[0.97181157] # return last

10 values

 [0.99886734]

 [0.98982144]

 [0.94500082]

 [0.8660254 ]

 [0.75574957]

 [0.61815899]

 [0.45822652]

 [0.28173256]

 [0.09505604]]



Code currently

print( X_train[:-10]) [[ 0.00000000e+00] # returns all the values up to

where the last 10 values start

 [ 1.89251244e-01]

 [ 3.71662456e-01]

 [ 5.40640817e-01]

 [ 6.90079011e-01]

 [ 8.14575952e-01]

 [ 9.09631995e-01]

 [ 9.71811568e-01]

 [ 9.98867339e-01]

 [ 9.89821442e-01]

 [ 9.45000819e-01]

 [ 8.66025404e-01]

 [ 7.55749574e-01]

 [ 6.18158986e-01]

 [ 4.58226522e-01]

 [ 2.81732557e-01]

 [ 9.50560433e-02]

 [-9.50560433e-02]

 [-2.81732557e-01]

 [-4.58226522e-01]

 [-6.18158986e-01]

 [-7.55749574e-01]

 [-8.66025404e-01]

 [-9.45000819e-01]

 [-9.89821442e-01]

 [-9.98867339e-01]

 [-9.71811568e-01]

 [-9.09631995e-01]

 [-8.14575952e-01]

 [-6.90079011e-01]

 [-5.40640817e-01]

 [-3.71662456e-01]

 [-1.89251244e-01]

 [-2.44929360e-16]

 [ 1.89251244e-01]

 [ 3.71662456e-01]

 [ 5.40640817e-01]

 [ 6.90079011e-01]

 [ 8.14575952e-01]

 [ 9.09631995e-01]]



On Sun, Mar 17, 2024 at 9:15 AM PAUL BERNARD ***@***.***>

wrote:



> Thank you for your issue ! This definitely isn't right.

> "50" should be "100"

>

> —

> Reply to this email directly, view it on GitHub

> <https://github.com/reservoirpy/reservoirpy/issues/152#issuecomment-2002461734>,

> or unsubscribe

> <https://github.com/notifications/unsubscribe-auth/AAPTOI2L6TENZCH3M4TBDUDYYWJN3AVCNFSM6AAAAABEZMDF2SVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDAMBSGQ3DCNZTGQ>

> .

> You are receiving this because you authored the thread.Message ID:

> ***@***.***>

>



**Response 3:**
> Fixed in 14b7e6795ef1a193f27b9b5fd5fa18f41a45f0dd

Thanks again :)

---

---

## Issue 151: how to save and load a prediction model

**Question 1:** How can I save and load a prediction model?

**Response 1:**
> You can simply pickle your model. If there are any issues with this method, please open another issue, but there shouldn't be.



Duplicate of #137 

**Question 2:**
> Thanks



On Fri, Mar 15, 2024 at 2:33 PM PAUL BERNARD ***@***.***>

wrote:



> You can simply pickle your model. If there are any issues with this

> method, please open another issue, but there shouldn't be.

>

> Duplicate of #137 <https://github.com/reservoirpy/reservoirpy/issues/137>

>

> —

> Reply to this email directly, view it on GitHub

> <https://github.com/reservoirpy/reservoirpy/issues/151#issuecomment-2000233779>,

> or unsubscribe

> <https://github.com/notifications/unsubscribe-auth/AAPTOIZFZHOKAANLGJGJLE3YYM5G5AVCNFSM6AAAAABEYN33KKVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDAMBQGIZTGNZXHE>

> .

> You are receiving this because you authored the thread.Message ID:

> ***@***.***>

>



---

---

## Issue 150: I trying to forecast using reservoirpy

**Question 1:** I have used to_forecasting ,,,how to forecast using  reservoirpy right way

**Response 1:**
> Hi @hunnder, [as documented](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.datasets.to_forecasting.html), the to_forecasting function transforms a timeseries dataset into two time-shifted timeseries of the same length.



To forecast a timeseries, you can read the [quickstart guide to ReservoirPy](https://reservoirpy.readthedocs.io/en/latest/user_guide/quickstart.html).

Let me know if there is something unclear in the guide or the documentation.

---

---

## Issue 149: Fixing variables scope and leak parameter in documentation hyper notebook

**Question 1:** Hi there,



While trying to reuse the code of the documentation for hyperparameters optimization, I found that the variables scopes for the datasets are not correct, with the `objective` function using variables defined in the global scope.



Also the notebook was crashing because in this function the leak rate is defined as `lr` but in the `hyperopt_config` dictionary it is defined as `leak`.



I thus propose some change so that this code works.

I have checked that it runs correctly, but please double check that before any merging.



Cheers,



Jonathan

**Response 1:**
> Thank you for your contribution !

It looks good to me, and it works fine, I'm merging this



This fixes #153 

**Question 2:**
> You are welcome, have a nice day.

---

---

## Issue 148: Rank list of degree of influence of input variables

**Question 1:** Sorry I had an issue opened but I accidentally closed it. So with many other AI techniques like transformers you have the ability to see which variables contribute the most to the ai model results. I was wondering if reservoirpy had a way of finding out how influential each input variable is.



this website below explains it more.



[Feature importance](https://builtin.com/data-science/feature-importance)

**Response 1:**
> Duplicate of #146

---

---

## Issue 146: Feature Importance

**Question 1:** What’s the best way of determining the most important variables used? 

**Response 1:**
> Hello :wave: 

We are missing information here. What's your task, what kind of variable are you talking about ? What do you mean by "best way" ? What do you mean by important ?



If you want to optimize the input scaling of each feature, you can start a hyper-parameter search on the scaling of all your input features.

You can specify the input scaling of each feature like so:

`input_scaling = [feature1_scaling, feature2_scaling, ...]`

And then you can use an hyper-parameter optimization algorithm on the `(feature1_scaling, feature2_scaling, ...)` tuple.



Let us know if that helps, or if you have a different problem.

**Question 2:**
> What I mean is that how do I get a rank list of the most and least influential variables of a model? Say you input 10 variables I wanted to know which ones were more important than others.

**Question 3:**
> Sorry accidentally closed the issue.

**Response 4:**
> > Sorry I had an issue opened but I accidentally closed it. So with many other AI techniques like transformers you have the ability to see which variables contribute the most to the ai model results. I was wondering if reservoirpy had a way of finding out how influential each input variable is.

> 

> this website below explains it more.

> 

> [Feature importance](https://builtin.com/data-science/feature-importance)



The feature permutation method proposed in the webpage you linked should work with an echo state network, as with any model.



Beyond this and optimizing your input scaling for each feature as hyper-parameters, there have been some research on the echo state network specifically, such as [ESNigma](https://www.esann.org/sites/default/files/proceedings/legacy/es2015-104.pdf), which are things I would like to have in ReservoirPy, but for now, you would have to re-implement the algorithm by hand.

---

---

## Issue 145: Fitting a model on non-temporal data

**Question 1:** ### Auto-translated with DeepL



Hello,



In your guide, you mention that "an accepted shape may be (series, timesteps, features)".

I have a somewhat chaotic series of time series in the format (N x T x D), where N is the number of samples and T and D respectively the length of the series and its number of features.



How can I pass such an input to a reservoir model, knowing that I expect an output y = (N x D) for my model?



But in most cases I get an error, for example: 



> ValueError: shapes (1,1331) and (29,101) not aligned: 1331 (dim 1) != 29 (dim 0)



Should I loop over the number N? 

Do I need a different output architecture?



---

### Original issue



Bonjour,



Dans votre guide, vous mentionnez que "an accepted shape may be (series, timesteps, features)".

J'ai une suite de série de série temporelle un peu chaotique sous le format (N x T x D), où N le nombre d'échantillons et T et D respectivement la longueur de la série et son nombre de features.



Comment puis-je passer une telle entrée a modèle de reservoir sachant que j'attends une sortie y = (N x D) pour mon modèle ?



Mais dans la plupart des cas j'obtiens par exemple une erreur : 



> ValueError: shapes (1,1331) and (29,101) not aligned: 1331 (dim 1) != 29 (dim 0)



Faut-il que je fasse une boucle sur le nombre N ? 

Que j'un une architecture au niveau de l'output différentes ?



:)

**Response 1:**
> Hello :wave: 



All ReservoirPy models take one (a 2D numpy array) or multiple (a list of 2D numpy arrays or a 3D numpy array) timeseries.



If you want your model to predict a single value (or vector) for each timeseries, you mainly have 2 solutions:

- you can convert your output to a timeseries consisting of your feature vector repeated T times, fit your model with this timeseries, and then convert your output timeseries back to a single vector through a method of your choice

- run your reservoir on each timeseries, and fit your readout with the last state of the reservoir

Those methods are illustrated in the [tutorial n°5](https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/5-Classification-with-RC.ipynb) (sequence-to-sequence and sequence-to-vector sections)



The best choice depends on your task. 



NB : Please post your issues in english (even if autotranslated) so that most users can benefit from the discussion

---

---


## Issue 142: datasets.narma doesn't return input series

**Question 1:** I think the ```datasets.narma``` function should return a tuple ```(u, y)```, where ```u``` and ```y``` are the input and output series, respectively, both of shape ```(n_timesteps, 1)```; rather than just return the output series since the echo state network needs ```u``` as input so as to be trained to predict ```y```.

**Response 1:**
> Hello @DS-Liu :wave: 

Thank you for your issue.

Timeseries in the `datasets` module of ReservoirPy all have the shape `(n_timesteps, n_features)`, as they are timeseries. Echo State Networks can be used on those timeseries to predict the following timesteps, considering the previous ones.



If you want to convert a timeseries into an `(input, output)` tuple for a prediction task, you can use the [`to_forecasting`](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.datasets.to_forecasting.html#reservoirpy.datasets.to_forecasting) method.

**Question 2:**
> I'm new to reservoir computing. But I think the input of the narma task should be the ```u(t)``` sampled uniformly from [0, 0.5], am I wrong?

**Response 3:**
> I think the confusion comes from the naming convention between the reservoir computing litterature where u(t) usually represents the input timeseries, and `u(t)` in the NARMA recurrent relation which stands for "uniform", and that is simply a sample from a uniform distribution and that is not meant to be used elsewhere.



In the end, the NARMA timeseries is a single timeseries of shape (n_timesteps, 1), and you can use reservoir computing to make predictions.



Let me know if the misunderstanding persists 

**Question 4:**
> The $n$-th order narma task is defined as

$$y_{k+1} = \alpha y_k + \beta y_k\left(\sum_{j=0}^{n-1}y_{k-j}\right) + \gamma u_{k-n+1}u_k + \delta,$$

where $(\alpha,\beta,\gamma,\delta)$ are set to $(0.3,0.05,1.5,0.1)$, respectively.



In the literatures of quantum reservoir computing, such as 

1. [Harnessing Disordered-Ensemble Quantum Dynamics for Machine Learning.](https://link.aps.org/doi/10.1103/PhysRevApplied.8.024030)

2. [Boosting Computational Power through Spatial Multiplexing in Quantum Reservoir Computing.](https://link.aps.org/doi/10.1103/PhysRevApplied.11.034021)

3. [Learning nonlinear input–output maps with dissipative quantum systems.](http://link.springer.com/10.1007/s11128-019-2311-9)

4. [Higher-Order Quantum Reservoir Computing.](https://arxiv.org/abs/2006.08999)

5. [Dynamical Phase Transitions in Quantum Reservoir Computing.](https://link.aps.org/doi/10.1103/PhysRevLett.127.100502)

6. [Unifying framework for information processing in stochastically driven dynamical systems.](https://link.aps.org/doi/10.1103/PhysRevResearch.3.043135)



the input to the reservoir at time step $k$ is the $u_k$ which is sampled uniformly from [0, 0.2] to assure the stability of the narma task.



I think this is reasonable since the narma system is driven by $u_k$. 



However, you mentioned that



> If you want to convert a timeseries into an (input, output) tuple for a prediction task, you can use the [to_forecasting](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.datasets.to_forecasting.html#reservoirpy.datasets.to_forecasting) method.



which means that for the narma task, the echo state at time $k$ is driven by its previous output $y_{k-1}$. This is equivalent to the case that the echo state has no input, but has an output feedback connection.



Are these literatures wrongly performed the narma task?

**Response 5:**
> Hello again, and thank you for your well-placed perseverance :)

Indeed, many papers use the uniformly distributed timeseries as an input for benchmarking. This will be fixed in the ReservoirPy v0.3.11.



In the meantime, if you want to use the NARMA task for your benchmarking, you can re-implement the function that returns `u(t)`:



```python

def narma(

    n_timesteps: int,

    order: int = 30,

    a1: float = 0.2,

    a2: float = 0.04,

    b: float = 1.5,

    c: float = 0.001,

    x0: Union[list, np.ndarray] = [0.0],

    seed: Union[int, RandomState] = None,

) -> np.ndarray:

    if seed is None:

        seed = get_seed()

    rs = rand_generator(seed)



    y = np.zeros((n_timesteps + order, 1))



    x0 = check_vector(np.atleast_2d(np.asarray(x0)))

    y[: x0.shape[0], :] = x0



    u = rs.uniform(0, 0.5, size=(n_timesteps + order, 1))

    for t in range(order, n_timesteps + order - 1):

        y[t + 1] = (

            a1 * y[t]

            + a2 * y[t] * np.sum(y[t - order : t])

            + b * u[t - order] * u[t]

            + c

        )

    return u, y[order:, :]

```



Sorry for my misunderstanding, and thank you again for your issue

---

---

## Issue 137: Save/Load to/from disk

**Question 1:** I am having trouble finding how to save or load a trained model to disk. I see some stuff for previous versions of the library (`*.compat.*` methods), but I am not sure what to do about newly created models. Anywhere I should look?

**Question 1:**
> Pickling seems to work. Any issue I should watch out for?

**Response 2:**
> Hello @benureau :wave: 

Indeed, there were dedicated methods in the library to save and load a model before v0.3, but they are now deprecated and cannot be used for newly created models. You should definitely use `pickle`, which should work just fine.

You just have to be careful if you have user-defined methods as parameters of one of your nodes, as they won't get pickled. Apart from that, it should work just fine :)

---

---


## Issue 135: Segfault in classification notebook

**Question 1:** If I run the code below, copied from the classification notebook in `tutorial/`:

```python

from reservoirpy.datasets import japanese_vowels

from reservoirpy import set_seed



set_seed(42)



X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)



from reservoirpy.nodes import Reservoir, Ridge, Input



source = Input()

reservoir = Reservoir(500, sr=0.9, lr=0.1)

readout = Ridge(ridge=1e-6)



model = [source >> reservoir, source] >> readout



Y_pred = model.fit(X_train, Y_train, stateful=False, warmup=2).run(X_test, stateful=False)

```



I get this output: 

```shell

§ python tests/classification.py

Running Model-1: 20it [00:00, 5418.65it/s]

Running Model-1: 26it [00:00, 2019.74it/s]

etc...

Running Model-1: 9it [00:00, 1270.45it/s]

Running Model-1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 270/270 [00:02<00:00, 104.37it/s]

Fitting node Ridge-0...

Bus error: 10

(echoenv) λ:reservoirpy § /Users/fcyb/.pyenv/versions/3.10.11/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown

  warnings.warn('resource_tracker: There appear to be %d 



```

Sometimes "Bus error: 10" is replaced by a "Segmentation fault: 11". 



I am running the code on macOS 13.4 (22F66), Apple Silicon (Arm) with python 3.10.11.

**Question 1:**
> Code is freshly cloned from bb00cb09

**Question 2:**
> Ok, this is solved for me by upgrading scipy and numpy to the latest versions (1.11.2 and 1.26.0, resp.), even though reservoirpy wants scipy<=1.7.3. Consider updating the requirements or adding a warning in the installation section.

**Question 3:**
> The segfault was triggered by this line btw:

https://github.com/reservoirpy/reservoirpy/blob/bb00cb0937828f851d2e87fc0d4f49c7366f8fa3/reservoirpy/nodes/readouts/ridge.py#L17



Any ideas of issues I might encounter while using a newer scipy?

**Response 4:**
> Hi @benureau, thanks for your interest in reservoirpy and for finding this issue!



Usually this kind of error, related to the regression when training the readout, could come from a bad conditioning of the matrix to be pseudo-inversed or if the size of the training data (nr of timesteps / number of reservoir dimensions) is too big.

Changing the ridge parameter could solve the first problem and reducing the data solve the second kind of problem.

But in your case this seems very related to the version of scipy in which the internal computations of linalg.solve seems to have changed.

No clear idea yet.

**Response 5:**
> Hello @benureau , thank you for your extensive report.



Even if I wasn't able to reproduce on my machine, which makes it difficult to debug, it seems that this is an issue with older scipy versions (<1.9.0), specifically on the ARM architecture.



As you mentionned, updating your scipy version should fix this. Feel free to reopen if your problem persists on recent scipy versions.

 

---

---

## Issue 134: Potential Error in Documentation

**Question 1:** I noticed that there may be a potential (and honestly harmless error) in [this](https://reservoirpy.readthedocs.io/en/latest/user_guide/getting_started.html#Train-the-ESN) section of the documentation: https://reservoirpy.readthedocs.io/en/latest/user_guide/getting_started.html#Train-the-ESN



Here you use readout.is_initialized and readout.fitted, but in previous statements, you use 'ridge' as the keyword for the readout layer. It was probably an oversight since in the section prior when the documentation talks about training the readout alone, you refer to the ridge as readout, but I think it would be worth correcting that for those skimming the documentation for a quick implementation! I apologize if I am incorrect in my assumption with this 'error'!

**Response 1:**
> Hello @dsheena17,

Thank you for reporting this! We will fix the error in the next release.

---

---

## Issue 133: Where does the randomness come from

**Question 1:** Hi,



I think I fixed the input and reservoir weights but I still observed the difference in the internal state after running the following code multiple times. Where does the randomness come from? Thanks in advance.



from reservoirpy.nodes import Reservoir

import numpy as np

import matplotlib.pyplot as plt



def normal_w(n, m, **kwargs):

    np.random.seed(42)

    W_res = np.random.normal(0, 1, size=(n, m))

    rhoW = max(abs(np.linalg.eig(W_res)[0]))

    return rhoW*W_res/rhoW



def normal_win(n, m, **kwargs):

    np.random.seed(43)

    W_in = np.random.normal(0, 1, size=(n, m))

    return W_in



X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

reservoir = Reservoir(50, lr=1, Win=normal_win, W=normal_w)

states = reservoir.run(X)

plt.plot(states[:,1])

reservoir = reservoir.reset()

**Response 1:**
> Hi @zonglunli7515 ,



The randomness in your reservoir state comes from the input bias of the reservoir. If you want a deterministic behavior, you can set a value for `bias`, disable input bias by setting `input_bias=False`, or set a seed when creating your Reservoir.



Let me know if that helps

**Question 2:**
> > 



Thanks a lot! I overconcentrated on noise parameters and overlooked this! 

---

---

## Issue 129: Suppress output in train and run functions

**Question 1:** Hi,

I am running multiple reservoir in a row and I want to suppress the outputs.

Running Reservoir-3: 100%|██████████| 1502/1502 [00:00<00:00, 5634.48it/s]

Running Reservoir-3: 100%|██████████| 1778/1778 [00:00<00:00, 4366.85it/s]

Running Reservoir-3: 100%|██████████| 2403/2403 [00:00<00:00, 5612.39it/s]



Is there an equivalent of verbose=False for the train and run methods ?



Thank you

**Response 1:**
> Hello,

afaik, there is no way to change the verbosity for one specific method.

However, you can suppress those outputs by setting ReservoirPy's verbosity to `0` on top of your script :

```python

import reservoirpy

reservoirpy.verbosity(0)

```

**Question 2:**
> Exactly what I was looking for. Thank you !

---

---

## Issue 128: Scipy Dependency Version

**Question 1:** Due to Reservoirpy requiring building scipy from source if installed via pip on Python 3.11, I had a poke around and it's evident that there's a problem with the scipy dependency which isn't set consistently, as follows:



1. requirements.txt and pipfile give the version requirement as ">=1.4.1".

2. setup.py gives the version requirement as "scipy>=1.0.0,<=1.7.3".



I have run reservoirpy with scipy 1.11.1 and it all seems to work, so I think setup.py has the incorrect value. However, regardless of what is 'correct', it does seem worth making all the requirements have the same constraints.

**Response 1:**
> Same issue here, removing the upper limit in setup.py allows to successfully install reservoirpy.

**Response 2:**
> Please add support for scipy>=1.7.3, since some of the libraries I use in my project need newer version of scipy.

**Response 3:**
> Hello everyone,



Thank you for your reports. Indeed, configuration files were inconsistent, and the dependencies needed a bit of refreshment. This should be fixed for ReservoirPy >= 0.3.10.


---

---