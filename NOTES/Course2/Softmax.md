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
