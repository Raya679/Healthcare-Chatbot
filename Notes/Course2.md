## Week 1

### Setting up your machine learning applications -

**Train/Dev/Test sets**

Applied ML is a highly iterative process.

Data is split into three parts
- traing set
- dev set (Hold out cross validation set / development set)
- test set

One must make sure that the dev and test set come from the same distribution 

**Bias/Variance**

- High bias : Model is underfitting
- High variance : Model is overfitting
- Proper balance between bias and variance : Model is just right

**Basic Recipe For Machine Learning**

If the algoritm has high bias:
- Use bigger NN
- Try a different NN architechture
- Try to train it longer

If the algoritm has high variance:
- More data
- Regularization
- Try a different NN architechture

### Regularizing your neural network -

**Regularization**

Regularization helps in avoiding Overfitting (High variance).

L2 regularization -
`||w||^2 ie sum of all w squared`

L1 regularization - 
`||w|| ie sum of absolute values of w`

In Neural Networks -
`J(w[1],b[1].......w[l],b[l]) = (1/m) * sum(L(y_hat(i),y(i)) + (lamd/2m) * sum(||w[l]||^2 ))`

For back propogation - 
`dw[l] = (back propogation) + (lamd/m) * w[l]`

Thus,
```
w[l] = w[l] - learning_rate * dw[l]
w[l] = w[l] - learning_rate * ((back propogation) + (lamd/2m) * w[l])
w[l] = (1 - learning_rate * (lamd/2m))*w[l] - learning_rate * (back propogation)
```

The above is known as weight decay.

After plotting the cost function, we can see that J decreases monotonically after every iteration of gradient descent with regularization.

**Why regularization reduces overfitting?**

- If lamd --> very large then a lot of weights --> 0, thus the NN will become simpler and prevent overfitting
- If we consider the `tanh function` if lamd --> very large then a lot of weights --> 0, then it will use the linear part of the tanh function which would make the NN a roughly linear classifier

**Dropout Regularization**

In dropout regularization , node are eleminated based on some probability and thus a smaller neural network is trained.
This helps in reducing overfitting.

- inverted dropout
```python
l = 3
keep_prob = 0.8
d3 = np.random.rand(a3.shape[0],a3.shape[1])<keep_prob
a3 = np.multiply(a3,d3)
a3 =/keep_prob  #inverted dropout technique
```
Inverted dropout technique is used so as to not reduce the expected value of a3.

At test time don't use dropout. If you implement dropout at test time - it would add noise to predictions.

**Understanding dropout**

- By applying dropout, it cannot rely on any one feature so the weights are spread out 
- It forces each node to learn features independently from other nodes.
- The shrinking weights reduces overfitting. (similar to L2 regularization)
- Different keep_prob can be applied to different layers

- Downside of Dropout -
  - More hyperparameters
  - Cost function J is not well defined thus difficult to debug as J cannot be plotted by iteration

**Other regularization methods**

- Data augmentation -
  - Make more (fake) training examples with the same training example 

- Early stopping -
  - Point at which the training set error and dev set error are best (lowest training cost with lowest dev cost) is choosen
  - Advantage - Does not require hyperparameters
  - Downside - No orthogonalisation

### Setting up your optimization problem -

**Normalizing inputs**

Normalizing training sets -

`mean = (1/m) * sum(x(i))`
<br>
`X = X - mean`
<br>
`variance = (1/m) * sum(x(i)^2)`
<br>
`X /= variance`

- It is important to normalize if input features have diiferent range (eg : 0 - 1 and 1 - 1000)
- If we don't normalize then cost function will be deep and its shape will be inconsistent ,thus a smaller learning rate has to be used which will take a long time
- On normalizing the shape of the cost function will be consistent and the optimization will be faster as a larger learning rate can be used 


**Vanishing / Exploding gradients**

The Vanishing / Exploding gradients occurs when your derivatives become very small or very big.
- If w[l] > I (Identity matrix) the activation and gradients will explode.
- If w[l] < I (Identity matrix) the activation and gradients will vanish.

**Weight Initialization for Deep Networks**

This helps with vanishing and exploding gradients
`Z = w1x1 + w2x2 + ... + wnxn`
If n-->large then w-->small

`np.random.rand(shape) * np.sqrt(1/n[l-1])

In case of ReLU
`np.random.rand(shape) * np.sqrt(2/n[l-1])`

He initialization 
`np.random.rand(shape) * np.sqrt(2/(n[l-1] + n[l]))`

**Gradient checking**

- Steps -
  - First take `J(W[1],b[1],...,W[L],b[L])` and reshape into one big vector `J(theta)`.
  - Take `dW[1],db[1],...,dW[L],db[L]`nto one big vector `(d_theta)`

```python
eps = 10^-7   # small number
for i in len(theta):
  d_theta_approx[i] = (J(theta1,...,theta[i] + eps) -  J(theta1,...,theta[i] - eps)) / 2*eps
```

Then check :
`(d_theta - approx_d_theta)/approx_d_theta <= epsilon` 
 - if it is < 10^-7 - great
 - if around 10^-5 - can be OK
 - if it is >= 10^-3

 **Gradient checking implementation notes**

 - Don't use gradient checking while training.
 - Use only to debug
 - If algorithm fails grad check, look at components to try to identify the bug
 - Grad check does not work with dropout regularization because J is not consistent.


## Week 2 -

### Optimization algorithms

**Mini-batch gradient descent**

- Training NN with a large data is slow.
- Thus we split the m training set into mini-batches
- We split X and Y such that `t: X{t}, Y{t}`
- Then compute cost function on each batch using forward propogation and backpropogation separately.

```
for t = 1 to no. of mini-batches
    forward prop on X{t}
    compute cost J
    Backprop to compute grads
    Update parameters
```

**Understanding mini-batch gradient descent**

![LCO](https://i.imgur.com/FbSYeEy.png)

- Conditions -
  - mini batch size = m 
    - Batch gradient descent
    - Takes too long per iteration
  - mini batch size = 1 
    - Stochastic gradient descent (Every example is its own mini-batch)
    - lose speedup from vectorization 
  - mini batch size = between 1 and m 
    - Mini-batch gradient descent
    - faster learning due to vectorization and can make progress without processing the entire training set

- Choosing your mini-batch size -
  - If m < 2000 examples - use batch gradient descent
  - Typical mini-batch sizes - 6,128,256,512
  - Make sure that the mini-batch size fits in the CPU/GPU memory

**Exponentially weighted averages**

- General equation : 
`V(t) = beta * v(t-1) + (1-beta) * theta(t)`
<br>
This represents averages over ` ~ (1 / (1 - beta))` entries

**Understanding exponentially weighted averages**

- Implementation :
```
v = 0
Repeat{
  Get next theta
  v = beta * v + (1-beta) * theta(t)
}
```

- Bias Correction :
Helps in making the exponential weighted average more accurate especially in the initial stage.

Here we divide v(t) by `(1 - beta^t)`
```
v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)
```

**Gradient descent with momentum**

- Up and down oscillations slows down the gradient descent and prevents us from using a larger learning rate. Thus a smaller learning rate has to be used.

- The momentum algorithm almost always works faster than standard gradient descent.
- Momentum takes into account previous gradients while updating weights of a neural network. 

Momentum -
```
vdW = 0, vdb = 0
On iteration t:
  compute dw, db on current mini-batch
  vdW = beta * vdW + (1 - beta) * dW
	vdb = beta * vdb + (1 - beta) * db
	W = W - learning_rate * vdW
	b = b - learning_rate * vdb    
```

- beta is a hyperparameter
- beta = 0.9 (common choice)

**RMSprop**
- Stands for Root mean square prop
- RMSProp is an extension to the basic idea of momentum and it helps us avoid getting stuck at local minima or saddle points.

RMSprop -
```
sdW = 0, sdb = 0
on iteration t:
	compute dw, db on current mini-batch
	sdW = (beta * sdW) + (1 - beta) * dW^2  # squaring is element-wise
	sdb = (beta * sdb) + (1 - beta) * db^2  # squaring is element-wise
	W = W - learning_rate * dW / sqrt(sdW)
	b = B - learning_rate * db / sqrt(sdb)
```

- With RMSprop you can increase your learning rate.

**Adam optimization algorithm**
- Adaptive Moment Estimation.
- Adam optimization simply combines RMSprop and momentum together.

Adam optimization algorithm -
```
vdW = 0, vdW = 0
sdW = 0, sdb = 0
on iteration t:
	compute dw, db on current mini-batch                

  # momentum	
	vdW = (beta1 * vdW) + (1 - beta1) * dW     
	vdb = (beta1 * vdb) + (1 - beta1) * db     
	
  # RMSprop
	sdW = (beta2 * sdW) + (1 - beta2) * dW^2  
	sdb = (beta2 * sdb) + (1 - beta2) * db^2   
			
  #bias correction
	vdW = vdW / (1 - beta1^t)      
	vdb = vdb / (1 - beta1^t)      
			
	sdW = sdW / (1 - beta2^t)      
	sdb = sdb / (1 - beta2^t)      

  #updating the parameters				
	W = W - learning_rate * vdW / (sqrt(sdW) + epsilon)
	b = B - learning_rate * vdb / (sqrt(sdb) + epsilon)
  ```
Here - beta1 = 0.9 , beta2 = 0.999 and epsilon = 10^-8 (recommended)

**Learning rate decay**

- During the initial steps of learning one can afford to take much bigger steps (i.e have larger learning rate) but as the learning approaches convergence then having a smaller learning rate helps.

- learning rate decay methods -
1.learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0
2.learning_rate = (0.95 ^ epoch_num) * learning_rate_0
3.learning_rate = (k / sqrt(epoch_num)) * learning_rate_0

Here decay_rate is a hyperparameter

**The problem of local optima**

- For higher dimensions it is more likely to see the saddle point and the local optima
- It's unlikely to get stuck in a bad local optima
- Problem of plateus - region where the derivative is close to zero for a long time.
- Plateaus can make learning slow











