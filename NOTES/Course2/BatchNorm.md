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
