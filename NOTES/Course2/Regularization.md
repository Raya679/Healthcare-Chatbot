# Regularization

### In terms of LR:

#### J(w,b)=1/m \* np.sum(L(yhat,y)) + lambda/2m \* ||w||<sup>2</sup><sub>2</sub>

#### Where lambda is Regularization parameter. This a form of L2 regularization, where:

#### ||w||<sup>2</sup><sub>2</sub>=np.sum(w<sup>2</sup><sub>j</sub>)=w<sup>T</sup>.w

#### L1 regularization: lamda/2m _ np.sum(|w|)=lambda/2m _ ||w<sub>1</sub>||

#### w consists of a lot of 0s in the L2 regularization due to which less memory space is required

### In terms of NN:

#### J(w<sup>[1]</sup>,b<sup>[1]</sup>....w<sup>[l]</sup>,b<sup>[l]</sup>)=1/m \* np.sum(L(yhat,y)) + lambda/2m \* np.sum(||w<sup>[l]</sup>||<sup>2</sup>)

#### ||w<sup>[l]</sup>||<sup>2</sup><sub>F</sub>

#### This is called Frobenius norm

### From backprop:

#### dw<sup>[l]</sup>=(from backprop)+lambda/2m \* w<sup>[l]</sup>

#### Partial derivative of J wrt w =dw<sup>[l]</sup>

#### On updation of values:

#### w<sup>[l]</sup>:=w<sup>[l]</sup>-alpha \* dw<sup>[l]</sup>

#### w<sup>[l]</sup>:=w<sup>[l]</sup>-alpha((from backprop)+lambda/2m \* w<sup>[l]</sup>)

#### w<sup>[l]</sup>:=w<sup>[l]</sup>(1-lambda/2m \* w<sup>[l]</sup>)-alpha(from backprop)

#### Because of (1-lambda/2m \* w<sup>[l]</sup>) term L2 norm regularization is also called Weight Decay

#### As lambda increases, w<sup>[l]</sup> decreases

#### z<sup>[l]</sup>=w<sup>[l]</sup> a<sup>[l-1]</sup>+b<sup>[l]</sup>

#### let g(z)=tanh(z)

#### As z decreases both from positive as well as negative side, the g(z) function becomes more and more linear, having low variance(decrease in overfitting)

# Inverted Dropout

#### Consider layer 3, l=3

#### d3(dropout vector)=np.random.randn(a3.shape[0],a3.shape[1])<b keep_prob (keep_prob amy be 0.8)

#### It means eliminating 20% of the nodes

#### a3=np.multiply(a3,d3)

#### a3/=keep_prob (returns the value of a3 unchanged)

### Intuition of Dropout method:

#### Can't rely on one feature, therefore we need to spread out weights

#### If a particular hidden layer has high number of parameters, we keep the keep_prob high to avoid overfitting, as compared to the other layers

#### <b>Limitation</b>: One big downside of drop out is that the cost function J is no longer well defined on every iteration. You're randomly, calling off a bunch of notes. And so if you are double checking the performance of gradient descent is actually harder to double check that. You have a well defined cost function J. That is going downhill on every elevation because the cost function J that you're optimizing is actually less. Less well defined or it's certainly hard to calculate.

# Data Augmentation:

#### Instead of supplying a large amount of data for training for example the model of a cat and non-cat classifier, we can just flip the images or zoom or cause subtle distortions in them to create numerous fake examples. This is called data augmentation.

# Early Stopping

#### There's one other technique that is often used called early stopping. So what you're going to do is as you run gradient descent you're going to plot your, either the training error, you'll use 01 classification error on the training set. Or just plot the cost function J optimizing, and that should decrease monotonically. Because as you trade, hopefully, you're trading around your cost function J should decrease. So with early stopping, what you do is you plot this, and you also plot your dev set error.

#### This could be a classification error in a development sense, or something like the cost function, like the logistic loss or the log loss of the dev set. Now what you find is that your dev set error will usually go down for a while, and then it will increase from there. So what early stopping does is, you will say well, it looks like your neural network was doing best around that iteration, so we just want to stop trading on your neural network halfway and take whatever value achieved this dev set error. Well when you've haven't run many iterations for your neural network yet your parameters w will be close to zero. Because with random initialization you probably initialize w to small random values so before you train for a long time, w is still quite small. And as you iterate, as you train, w will get bigger and bigger and bigger until here maybe you have a much larger value of the parameters w for your neural network. So what early stopping does is by stopping halfway you have only a mid-size rate w. And so similar to L2 regularization by picking a neural network with smaller norm for your parameters w, hopefully your neural network is over fitting less. And the term early stopping refers to the fact that you're just stopping the training of your neural network earlier.

#### <b>Limitation</b>: The main downside of early stopping is that this couples these two tasks of optimizing the cost function and performing regularization on it. So you no longer can work on these two problems independently, because by stopping gradient decent early, you're sort of breaking whatever you're doing to optimize cost function J, because now you're not doing a great job reducing the cost function J. You've sort of not done that that well. And then you also simultaneously trying to not over fit. So instead of using different tools to solve the two problems, you're using one that kind of mixes the two. And this makes it more complicated to think about.
