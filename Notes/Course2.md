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

***Regularization***

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

***Why regularization reduces overfitting?***

- If lamd --> very large then a lot of weights --> 0, thus the NN will become simpler and prevent overfitting
- If we consider the `tanh function` if lamd --> very large then a lot of weights --> 0, then it will use the linear part of the tanh function which would make the NN a roughly linear classifier

***Dropout Regularization***

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

***Understanding dropout***

- By applying dropout, it cannot rely on any one feature so the weights are spread out 
- It forces each node to learn features independently from other nodes.
- The shrinking weights reduces overfitting. (similar to L2 regularization)
- Different keep_prob can be applied to different layers

- Downside of Dropout -
  - More hyperparameters
  - Cost function J is not well defined thus difficult to debug as J cannot be plotted by iteration

***Other regularization methods***

- Data augmentation -
  - Make more (fake) training examples with the same training example 

- Early stopping -
  - Point at which the training set error and dev set error are best (lowest training cost with lowest dev cost) is choosen
  - Advantage - Does not require hyperparameters
  - Downside - No orthogonalisation

### Setting up your optimization problem -

***Normalizing inputs***

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


***Vanishing / Exploding gradients***

The Vanishing / Exploding gradients occurs when your derivatives become very small or very big.
- If w[l] > I (Identity matrix) the activation and gradients will explode.
- If w[l] < I (Identity matrix) the activation and gradients will vanish.

***Weight Initialization for Deep Networks***

This helps with vanishing and exploding gradients
`Z = w1x1 + w2x2 + ... + wnxn`
If n-->large then w-->small

`np.random.rand(shape) * np.sqrt(1/n[l-1])

In case of ReLU
`np.random.rand(shape) * np.sqrt(2/n[l-1])`

He initialization 
`np.random.rand(shape) * np.sqrt(2/(n[l-1] + n[l]))`

***Gradient checking***

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

 ***Gradient checking implementation notes***

 - Don't use gradient checking while training.
 - Use only to debug
 - If algorithm fails grad check, look at components to try to identify the bug
 - Grad check does not work with dropout regularization because J is not consistent.





