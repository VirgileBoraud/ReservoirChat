**Code (not debug) questions**


I want to train my echo state network on multiple timeseries that have different lengths. I have seen in the documentation that you can put a 3D numpy array with shape (timeseries, timesteps, dimensions), but it wouldn’t work in my case as the timeseries have different lengths 

A. There is no way to do that in ReservoirPy as it is most probably not a good idea to train a model with different lengths and it would induce unexpected results. 
 B. You can pass a list of 2D numpy arrays that represents timeseries. As lists can contain numpy arrays of different shapes, you can specify timeseries with different lengths 
 C. You would have to pad every timeseries with zeros and then concatenate them. 
 D. As NumPy doesn’t support sparse arrays, it is best to use xarray for this use-case.

 

--- 

Make me a reservoir, with 1000 neurons, and with a uniform distribution of weights, and a sparsity of 95%. 

A. 

from reservoirpy as rpy 
reservoir = rpy.nodes.Reservoir(neurons=1_000, connectivity=0.05, weights="uniform") 
 

B. 

from reservoirpy as rpy 
reservoir = rpy.nodes.Reservoir(units=1_000, sparsity=0.95, W=rpy.mat_gen.uniform) 
 

C. 

from reservoirpy as rpy 
reservoir = rpy.Reservoir(units=1_000, rc_connectivity=0.05, distr="uniform") 
 

D. 

from reservoirpy as rpy 
reservoir = rpy.nodes.Reservoir(units=1_000, rc_connectivity=0.05, W=rpy.mat_gen.uniform) 

--- 

Create a model in which there are several reservoirs connected in a chain, and a readout at the end. 

A. 

from reservoirpy.nodes import Reservoir, Ridge 
model = [Reservoir(100, name="1"), Reservoir(100, name="2"), Reservoir(100, name="3"), Reservoir(100, name="4"), Reservoir(100, name="5"), Ridge(ridge=1e-5)] 
 

B. 

from reservoirpy.nodes import Reservoir, Ridge 
model = Reservoir(100, name="1") >> Reservoir(100, name="2") >> Reservoir(100, name="3") >> Reservoir(100, name="4") >> Reservoir(100, name="5") >> Ridge(ridge=1e-5) 
 

C. 

from reservoirpy.nodes import Reservoir, Ridge 
 
model = Reservoir(100) > Reservoir(100) > Reservoir(100) > Reservoir(100) > Reservoir(100) > Ridge(ridge=1e-5) 
 

D. 

from reservoirpy.nodes import Reservoir, Ridge 
model = Ridge(ridge=1e-5, previous=Reservoir(100, name="5", previous=Reservoir(100, name="4", previous=Reservoir(100, name="3", previous=Reservoir(100, name="2", previous=Reservoir(100, name="1")))))) 
 

--- 

Write me an echo state network that can efficiently use the many CPU cores my machine has 

A. 

import reservoirpy as rpy 
rpy.set_param("backend", "parallel") 
 
from reservoirpy.nodes import ESN 
model = ESN(units=100) 
model.fit(train_data, train_data) 
 

B. 

from reservoirpy.utils import parallel 
from reservoirpy.nodes import ESN 
model = ESN(units=100) 
with parallel(n_jobs=-1): 
	model.fit(train_data, train_data) 
 

C. 

from reservoirpy.nodes import ESN 
model = ESN(units=100, workers=-1) 
model.fit(train_data, train_data) 
 

D. ReservoirPy already parallelize computation by default to ensure the best performances. 

 

--- 


I have a model with several trainable readouts inside as such: 

from reservoirpy.nodes import Reservoir, Ridge 
 
model = Reservoir(100, name="R1") >> Ridge(name="readout1") 
model >>= Reservoir(100, name="R2") >> Ridge(name="readout2") 
model >>= Reservoir(100, name="R3") >> Ridge(name="readout3") 
 

How can I fit the model, by specifying the Y values to each Ridge nodes ? 

A. It is not possible to do such thing in ReservoirPy as it wouldn’t make sense 
B. You can pass a dict as a y parameter: model.fit(X, {"readout1": Y1, "readout2": Y2, "readout3": Y3, }) 
C. You would have to fit each part separately before concatenating them 
D. You can specify the node names as parameters and ReservoirPy will dispatch them correctly: model.fit(X, readout1=Y1, readout2=Y2, readout3=Y3) 

--- 


I have a NumPy array X of shape (timeseries, timesteps, dimensions) and I want to classify them according to my Y array of shape (timeseries, ) which contains number from 0 to 9 according to the class the timeseries belongs to. How can I do that in ReservoirPy ? 

A. 

from reservoirpy.nodes import Reservoir, ScikitLearnNode, Ridge 
from sklearn.linear_model import RidgeClassifier 
 
Y_ = Y.reshape(-1, 1, 1).repeat(X.shape[1], 1) 
 
model = Reservoir(1000, lr=0.9, sr=1.0) >> ScikitLearnNode(RidgeClassifier, model_hypers=dict(alpha: 1e-8)) 
model.fit(X, Y_) 
 

B. Reservoir computing is only a framework for timeseries forecasting, it is not suited for classification. 
 C. 

from reservoirpy.nodes import Reservoir, ScikitLearnNode, Ridge 
from sklearn.linear_model import RidgeClassifier 
 
model = Reservoir(1000, lr=0.9, sr=1.0) >> RidgeClassifier(alpha=1e-8) 
model.fit(X, Y) 
 

D. 

from reservoirpy.nodes import Reservoir, ScikitLearnNode, Ridge 
from sklearn.linear_model import RidgeClassifier 
 
Y_ = Y.reshape(-1, 1, 1).repeat(X.shape[1], 1) 
 
model = Reservoir(1000, lr=0.9, sr=1.0) >> ScikitLearnNode(RidgeClassifier) 
model.fit(X, Y_) 
 

--- 

*Code Debug questions*


Here is my code: 

from reservoirpy.nodes import Reservoir, Ridge 
 
model = Reservoir(units=200, lr=0.2, sr=1.0) >> Ridge(ridge=1e-4) 
 
for x_series, y_series in zip(X_train, Y_train): 
	model.fit(x_series, y_series, warmup=10) 
 
y_pred = model.run(X_test[0]) 
 

A. Calling .fit on a model erases the previous trained results. You can instead call .fit once by passing the lists X_train and Y_train as parameters. 
B. Everything is correct ! 
C. .fit method is not suited for online training. Use .train instead. 
D. Reservoir parameters should be written in full form: leak_rate, spectral_radius. 

--- 

Here is my code: 

from reservoirpy.nodes import Reservoir, Ridge 
 
model = Reservoir(units=200, lr=0.2, sr=1.0, iss=0.2) >> Ridge(ridge=1e-4) 
 
model.fit(X_train, Y_train, warmup=200) 
Y_pred = model.run(X_test) 
 

A. iss is not a parameter. For scaling the input, the correct parameter is scale_factor. 
B. Reservoir parameters should be written in full form: leak_rate, spectral_radius, input_scaling. 
C. You must first create the reservoir and readout nodes, and then connect them, in three separate lines. 
D. iss is not a parameter. For scaling the input, the correct parameter is input_scaling. 

--- 

Here is my code: 

from reservoirpy.nodes import Reservoir, RLS 
 
model = Reservoir(units=200, lr=0.2, sr=1.0) >> RLS(alpha=1e-4) 
 
for x_series, y_series in zip(X_train, Y_train): 
	model.fit(x_series, y_series, warmup=10) 
 
y_pred = model.run(X_test[0]) 
 

I have an error. What’s wrong ? 

A. The RLS node can only be trained online, but the .fit method is for offline training. You should use .train instead. 
B. The model has been trained on a list of timeseries but is run on a single timeseries. 
C. There is not enough units inside the reservoir. For this task, having at least 1000 neurons is recommended 
D. Wrong import path: to import the Reservoir node, use from reservoirpy.nodes.reservoirs import Reservoir. 

--- 

Here’s my code: 

from reservoirpy.nodes import Input, Output, Reservoir, Ridge 
R1 = Reservoir(100, lr=0.01, sr=1.) 
R2 = Reservoir(100, lr=0.03, sr=1.) 
R3 = Reservoir(100, lr=0.09, sr=1.) 
R4 = Reservoir(100, lr=0.3, sr=1.) 
R5 = Reservoir(100, lr=0.9, sr=1.) 
R6 = Reservoir(100, lr=0.01, sr=1.) 
readout = Ridge(ridge=1e-5, name="readout") 
 
path1, path2 = R1 >> R6, R2 >> R5 
path3 = Input(name="input") >> [R1, R2, R3] 
path4 = R1 >> R2 >> R3 >> R4 >> R5 >> R6 >> readout >> Output() 
model = path1 & path2 & path3 & path4 
 
model.fit({"input": X_train}, {"readout":Y_train}, warmup=10) 
model.run({"input": X_test}) 
 

Is that correct ? 

A. The .fit and .run methods only takes numpy arrays or list of numpy arrays, not dictionaries. 
B. Yes, everything is correct ! 
C. There is a circular connection in the model. 
D. path2 is not defined. 

--- 

Is this the correct usage of the method partial_fit ? 

 
reservoir, readout = Reservoir(100, sr=1), Ridge(ridge=1e-8) 
 
for x, y in zip(X, Y): 
    states = reservoir.run(x) 
    readout.partial_fit(states, y) 
 
readout.fit() 
model = reservoir >> readout 
 

A. By calling the method .fit, the readout forgets its previous training 
B. The created model won’t be fitted 
C. While it works, it can be simplified by creating the model and calling partial_fit on it 
D. Yes, everything is correctly coded 

--- 

Here’s my code: 

from reservoirpy.nodes import Input, Output, Reservoir, Ridge 
R1 = Reservoir(100, lr=0.01, sr=1.) 
R2 = Reservoir(100, lr=0.03, sr=1.) 
R3 = Reservoir(100, lr=0.09, sr=1.) 
R4 = Reservoir(100, lr=0.3, sr=1.) 
R5 = Reservoir(100, lr=0.9, sr=1.) 
R6 = Reservoir(100, lr=0.01, sr=1.) 
readout = Ridge(ridge=1e-5, name="readout") 
 
path1, path2 = R1 >> R6, R2 >> R5 
path3 = Input(name="input") >> [R1, R2, R3] 
path4 = R1 >> R2 >> R3 >> R4 >> R5 >> R6 >> readout >> Output() 
model = path1 & path2 & path3 & path4 
 
model.fit(X_train, Y_train, warmup=10) 
model.run(X_test) 
 

Is that correct ? 

A. 
B. 
C. 
D. 

--- 


Here is my code: 

reservoir, readout = Reservoir(100, sr=1), Ridge(ridge=1e-8) 
   
model = reservoir >> readout 
   
model.fit(X[:800], Y[:800], warmup=10) 
   
steps = 1000 
results = np.zeros((steps, 1)) 
 
last_output = X[800] 
for i in range(steps): 
	last_output = model(last_output) 
	results[i] = last_output 
 

Is that the best way to have a model that generates new values by looping on itself ? 

A. No, you can connect the readout to the reservoir in order to loop the results back as an input after training: readout >> reservoir 
B. No, it won’t work as the reservoir has an input dimension of 100 and the results array containing the results only has its feature dimension set to 
C. You can call the .autoregress(n=1000) Model method. 
D. Yes, this is probably the best solution. 

--- 


? 

--- 


weights = np.random.choice([1, -1], p=[0.6, 1 - 0.6], replace=True, size=(200, 200)) 
reservoir = Reservoir(W=weights, sr=0.9, lr=0.6) 
 

I created my reservoir this way, but it seems the reservoir has a very chaotic behavior, even though the spectral radius is below 1. 

A. The rule of the spectral radius <1 holds for matrices with a normal distribution, not a Bernoulli one, which explains why you have a chaotic behavior with a spectral radius of only 1. 
B. The rule of the spectral radius <1 only holds when there is no leak rate, so that explains why you have a chaotic behavior with a spectral radius of only 1. 
C. The parameter sr is only valid when no weight matrix has been specified. If a matrix is already specified, this argument is ignored. 
D. the reservoir argument names are incorrect. You should use spectral_radius and leak_rate. 

--- 


Here’s my code: 

from reservoirpy.nodes import Input, Output, Reservoir, Ridge 
R1 = Reservoir(100, lr=0.01, sr=1.) 
R2 = Reservoir(100, lr=0.03, sr=1.) 
R3 = Reservoir(100, lr=0.09, sr=1.) 
R4 = Reservoir(100, lr=0.3, sr=1.) 
R5 = Reservoir(100, lr=0.9, sr=1.) 
R6 = Reservoir(100, lr=0.01, sr=1.) 
readout = Ridge(ridge=1e-5, name="readout") 
 
path1, path2 = R1 >> R6, R2 >> R5 
path3 = Input(name="input") >> [R1, R2, R3] 
path4 = R1 >> R2 >> R4 >> R3 >> R5 >> R6 >> readout >> Output() 
model = path1 & path2 & path3 & path4 
 
model.fit({"input": X_train}, {"readout":Y_train}, warmup=10) 
model.run({"input": X_test}) 
 

A. The .fit and .run methods only takes numpy arrays or list of numpy arrays, not dictionaries. 
B. Yes, everything is correct ! 
C. There is a circular connection in the model. 
D. path2 is not defined. 