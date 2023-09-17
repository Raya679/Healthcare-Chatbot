# Batch Normalization

- This is done to normalize activations in a network
- By normalizing inputs we can speed up learning

```PYTHON
Given Z[l] = [z(1), ..., z(m)]
mean = 1/m * sum(z[i])
variance = 1/m * sum((z[i] - mean)^2)
Z_norm[i] = (z[i] - mean) / np.sqrt(variance + epsilon)
```

#### To get different value for mean and variance except 0 and 1:

```python
Z_tilde[i] = gamma * Z_norm[i] + beta
```

- We generally use Z_tilde instead of Z_norm to get different values for mean and variance

### Batch Norm to Neural Networks:

- In neural networks, instead of computing and using Z[i] in the activation function, we compute Z_tilde[i] by batch norm and using $\beta$ and $\gamma$
- Since in the process of bacth norm we subtract mean value from Z[i], the parameter b[i] is ignored

```python
for i in range(num_of_mini-batches):
    Compute forward prop on X{t}
    In each hidden layer, use batch norm to
    replace Z[l] with Z_tilde[l]
    Use back prop to compute dw[l], dbeta[l] and
    dgamma[l]
    Update w[l]:=w- learning_rate*dw[l]
    Update beta[l]:=beta- learning_rate*dbeta[l]
    Update gamma[l]:=gamma- learning_rate*
    dgamma[l]
```

![LCO](https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Notes/Images/bn.png)

### How does Batch Norm work?

#### COVARIATE SHIFT: Reduces the shift by keeping the mean and variance same even though the actual value changes. It thus keeps the earlier values stable.

# Batch Norm at test time

#### mu , sigma^2 are estimated exponentially weighted averages across mini-batches

# Multi Class Classification

#### The number of hidden units in the upper layer = number of classes i.e (0,C-1), where C is the number of classes

# Softmax Regression

### Softmax activation function:

```python
t = e^(Z[L])
A[L] = e^(Z[L]) / sum(t)
```

- The Softmax layer / function gives many outputs since its a multi classifier
- Softmax is referred to as gentle mapping
- Hardmax converts the highest value among the outputs to 1 and the rest outputs to 0
- Softmax regression generalises LR to 'C' number of classes

# Training a Softmax classifier

```python
L(y, y_hat) = - sum(y[j] * log(y_hat[j])) # Loss function
J(w[1], b[1], ...) = - 1 / m * (sum(L(y[i], y_hat[i]))) # cost function
dZ[L] = Y_hat - Y   # back propogation
Y_hat * (1 - Y_hat)  # derivative of softmax
```

- The probability of the desired output must be larger for making the loss function as small as possible

# Deep Learning Frameworks

### We use frameworks because of the following reasons:

- Ease of progarmming (developement and deployment)
- Running speed
- Truly open (open source with good governance)

### The different types of frameworks used are:

- Tensorflow
- Pytorch
- Lasagne
- Keras
- mxnet
- Caffe/Caffe2
- PaddlePaddle, etc.
