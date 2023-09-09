## Week 1 -

### Introduction to deep learning

**What is Neural network?**

- In neural networks, given enough data about x and y, given enough training examples with both x and y, neural networks are remarkably good at figuring out functions that accurately map from x to y.
- Supervised learning means we have the (X,Y) and we need to get the function that maps X to Y.

**Supervised learning with neural network**

- Different types of neural networks for supervised learning which includes:
   - CNN or convolutional neural networks 
   - RNN or Recurrent neural networks 
   - Standard Neural network
   - Hybrid/custom neural networks

- Structured data --> Databases and Tables
- Unstructured data --> video, audio, and text.

**Why is deep learning taking off?**

![Self explanatory](https://global-uploads.webflow.com/6434d6a6071644318551fd72/64c0e510cbed575decaad573_19%20what%20is%20deep%20learning_datahunt.webp)

## Week 2 -

### Logistic Regression as a neural network 

**Binary Classification**

- In Binary Classification, the goal is to learn a classifier that can input data represented by feature vector X and predict the output (whether y = 0 or y = 1)

**Logistic regression**

- Logistic regression estimates the probability of an event ocurring based on dataset of independent variables
- performs well when the dataset is linearly separable

- Simple equation: `y = wx + b`
- If x is a vector: `y = w(transpose)x + b`
- If range is between 0 and 1 we can use the sigmoid function : `y = sigmoid(w(transpose)x + b)`

**Logistic regression cost function**

- Loss Function :
  - how good our output y_hat is when true output is y
  - Loss function is measured for a single training example

- It is given by , `L(y_hat,y) = - (y*log(y_hat) + (1-y)*log(1-y_hat))`
   - if `y = 1` ==> `L(y',1) = -log(y')` ==> we want y' to be the largest ==> y' biggest value is 1
   - if `y = 0` ==> `L(y',0) = -log(1-y')` ==> we want 1-y' to be the largest ==> y' to be smaller as possible because it can only has 1 value.

- Cost function -
  - It is the average of the individual over the m training 
  - It is given by, `J(w,b) = (1/m) * Sum(L(y'[i],y[i]))`

  **Gradient Descent**

 - Algorithm Gradient descent for 1 training example :
 ```
 Repeat{
     w := w - learning_rate * d(J(w,b)/dw)
     b := b - alpha * d(J(w,b) / db)
 }
 ```

**Logistic Regression Gradient Descent**

```
	d(a)  = d(l)/d(a) = -(y/a) + ((1-y)/(1-a))
	d(z)  = d(l)/d(z) = a - y
	d(W1) = X1 * d(z)
	d(W2) = X2 * d(z)
	d(B)  = d(z)
```

**Gradient Descent on m Examples**

```
	J = 0; dw1 = 0; dw2 =0; db = 0;                 
	w1 = 0; w2 = 0; b=0;							
	for i = 1 to m

		# Forward propogation
		z(i) = W1*x1(i) + W2*x2(i) + b
		a(i) = Sigmoid(z(i))
		J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))

		# Backward propogation
		dz(i) = a(i) - Y(i)
		dw1 += dz(i) * x1(i)
		dw2 += dz(i) * x2(i)
		db  += dz(i)

	J /= m
	dw1/= m
	dw2/= m
	db/= m

	# Gradient descent
	w1 = w1 - alpha * dw1
	w2 = w2 - alpha * dw2
	b = b - alpha * db
```
This is done to minimize errors.

### Python and Vectorization

**Vectorization**

- Allows to get rid of explicit for-loops in your code
- It speeds up the code.
- NumPy library (dot) function is using vectorization by default.

**Vectorizing Logistic Regression**

For `[z1,z2...zm] = W' * X + [b,b,...b]`

```
  	Z = np.dot(W.T,X) + b   
  	A = 1 / 1 + np.exp(-Z)   
```

```
  	dz = A - Y                  
  	dw = np.dot(X, dz.T) / m    
  	db = dz.sum() / m           
```


## Week 3

### Shallow Neural Networks -

**Neural Network Representation**

![Neural Network](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*buxOnswsinejx2FVZDuF8w.png)

**Vectorizing across multiple examples**

```
for i = 1 to m
  z[1, i] = W1*x[i] + b1      
  a[1, i] = sigmoid(z[1, i])  
  z[2, i] = W2*a[1, i] + b2   
  a[2, i] = sigmoid(z[2, i])  
  
Z1 = W1X + b1    
A1 = sigmoid(Z1) 
Z2 = W2A1 + b2   
A2 = sigmoid(Z2) 
```

**Activation Functions**

1. Sigmoid activation function -
   - `g(z) = 1/(1 + np.exp(-z))`
   - Range - [0,1]
   - Used mainly for Binary  Classification
   - `g'(z) = g(z)(1 - g(z))`

2. tanh activation function -
   - `g(z) = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))`
   - Range - [-1,1]
   - Used mainly for Binary  Classification
   - `g'(z) = (1 - (g(z))^2)`

3. ReLU - Rectified linear unit
   - `g(z) = max(0,z)`
   - `g'(z) = { 0  if z < 0 , 1  if z >= 0 }`
   - undefined if z = 0

4. Leaky ReLu -
    - `g(z)  = max(0.01 * z, z)`
    - `g'(z) = { 0.01  if z < 0 , 1 if z >= 0   }`
	- undefined if z = 0

**Gradient descent for Neural Networks**

- Parameters - w1 , b1, w2, b2
- Cost Function - `J(w1 , b1, w2, b2)`

- Gradient descent 
```
Repeat{
		Compute : (y'[i], i = 0,...m)
		Compute : dW1, db1, dW2, db2
	    W1 = W1 - LearningRate * dW1
		b1 = b1 - LearningRate * db1
		W2 = W2 - LearningRate * dW2
		b2 = b2 - LearningRate * db2
      }    
```

- Forward Propogation -
```
Z1 = W1X + b1   
A1 = g1(Z1)
Z2 = W2A1 + b2
A2 = g2(Z2)     
```

- Back propogation -
```
dZ2 = A2 - Y    
dW2 = (dZ2 * A1.T) / m
db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True) 
dZ1 = (W2.T * dZ2) * g'1(Z1)  
dW1 = (dZ1 * X.T) / m   
db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True) 
```

**Random Initialization**

```
W1 = np.random.randn((2,2)) * 0.01    
b1 = np.zeros((2,1))                  
```


## Week 4

### Deep Neural Networks -

**Forward Propagation in a Deep Network**

- For 1 training example -
```
z[l] = W[l]a[l-1] + b[l]
a[l] = g[l](a[l])
```

- Vectorized (For m training example) -
```
Z[l] = W[l]A[l-1] + B[l]
A[l] = g[l](A[l])
```

**Matrix dimensions**

- Vectorized implementation - 
  - `W[l] --> (n[l],n[l-1]) `
  - `b --> (n[l],1)`
  - `Z[l], A[l], dZ[l], and dA[l] --> (n[l],m)`

**Building blocks of deep neural networks**

- Forward and backward function - 

![Forward and backward function](https://pylessons.com/media/Tutorials/Deep-neural-networks-main/Deep-neural-networks-part3/backprop.jpg)

**Forward and Backward Propagation**

- forward propagation for layer l
```
Input  A[l-1]
Output A[l], cache(Z[l])

Z[l] = W[l]A[l-1] + b[l]
A[l] = g[l](Z[l])
```

- back propagation for layer l
```
Input da[l], Caches
Output dA[l-1], dW[l], db[l]

dZ[l] = dA[l] * g'[l](Z[l])
dW[l] = (dZ[l]A[l-1].T) / m
db[l] = (1/m)*np.sum(dZ[l],axis=1,keepdims=True) 
dA[l-1] = w[l].T * dZ[l]            
```

**Parameters vs Hyperparameters**

- Parameters - `w[1],b[1].........w[l],b[l]`
- Hyperparameter 
    - learning rate
    - #Number of iteration
    - #number of hidden units in each layer 
	- #number of hidden units
	- Choice of activation functions.