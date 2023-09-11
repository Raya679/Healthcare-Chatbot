# Mini-Btch Gradient Descent

#### Let's say we split mini-batches in groups of 1000 each in a training set of 5 million examples

#### X<sup>{t}</sup> is a t<sup>th</sup> mini batch containing 1000 examples. Therefore number of such batches will be 5000

#### Applying formal prop, back prop and computing cost fuction over 5000 mini-batches:

```python
for t = 1 to no. of mini-batches
    forward prop on X{t}
    compute cost J
    Backprop to compute grads
    Update parameters
```

### Mini-batch size:

#### It should be somewhere between 1 and m.This ensures fastest learning and it makes progress without iterating the entire set

### For small training set(<=2000):

#### Use Batch-Gradient Descent

### For huge training set:

#### Use mini-batch gradient descent with size in the powers of 2

# Exponentially weighted averages

#### General equation is :

```python
V(t) = beta * v(t-1) + (1-beta) * theta(t)
```

#### V(t) is the approx avg over 1/(1-beta) days of temp

#### for e.g : beta= 0.9(common value) for approx 10 days of temp

#### V(theta)=0

#### V(theta):=beta*v +(1-beta)*theta1

#### V(theta):=beta*v +(1-beta)*theta2

```python
V(theta)=0
Repeat{
    get next theta(t)
    V(theta):=beta*v +(1-beta)*theta(t)
}
```

# Bias Correction

#### If 't' is large, it makes no difference.

#### But in initial stages it helps a lot (when 't' is small)

#### Here we divide v(t) by (1 - beta^t)

```python
v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)
```

# Gradient Descent with Momentum

#### Gradient Descent Mom-

```python
vdW = 0, vdb = 0
On iteration t:
  compute dw, db on current mini-batch
  vdW = beta * vdW + (1 - beta) * dW
	vdb = beta * vdb + (1 - beta) * db
	W = W - learning_rate * vdW
	b = b - learning_rate * vdb
```

#### If you neglect (1-beta) term,

#### Remember that vdw is scaled by 1/(1-beta) and learning_rate also needs to be updated.

#### beta remains the same

# RMS Prop

#### It is the 'Root Mean Square' prop

#### RMS prop-

```python
sdW = 0, sdb = 0
on iteration t:
	compute dw, db on current mini-batch
	sdW = (beta * sdW) + (1 - beta) * dW^2  # squaring is element-wise
	sdb = (beta * sdb) + (1 - beta) * db^2  # squaring is element-wise
	W = W - learning_rate * dW / sqrt(sdW)
	b = B - learning_rate * db / sqrt(sdb)
```

#### Since the vertical oscillations are reduced in this algorithm, learning_rate can be larger

# Adam optimization Algo

#### It is termed as "adaptive moment estimation"

#### It is a combination of the plus points of both Gradient Descent Algo and RMS prop

```python
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

- learning_rate needs to be tuned
- beta1 = 0.9
- beta2 = 0.999
- epsilon = 10^-8

# Learning rate decay

- It helps algorithm to slowly reduce the learning rate over time.
- Formula : learning_rate= (1/(1+decay_rate*epoch_num)) * learning_rate0

#### 1 epoch = 1 pass through data

#### For e.g : for learning_rate0=0.2 and deacy_rate=1

| epoch | learning_rate |
| ----- | ------------- |
| 1     | 0.1           |
| 2     | 0.067         |
| 3     | 0.05          |
| 4     | 0.04          |

### Other decay methods :

- expoenetial decay: learning_rate = (0.95)^epoch_num \* learning_rate0
- (k/sqrt(epoch_num)) \* learning_rate0 or (k/sqrt(t)) \* learning_rate0
- Discrete staircase
- Manual decay

# Problem of local optima

- In more number of dimensions, we are more likely to find a saddle point than a local optima, where the curve bends upwards as well as downwards
- Since, these saddle points have lots of flat surfaces, they have low learning rates
