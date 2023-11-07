# Course 2 :    
# Week 1: 

## Setting up your Machine Learning Application

- Properly setting up the machine learning problem is crucial for success.

### Train/Dev/Test Sets

- The importance of splitting data into training, development (validation), and test sets.
- Guidelines for choosing the proportions of data for each set.

### Bias and Variance

- Understanding the concepts of bias and variance in machine learning.
- How to diagnose whether your model has a bias or variance problem.

## Regularization

- Regularization techniques help prevent overfitting and improve the generalization of deep neural networks.

### L2 Regularization (Weight Decay)

- L2 regularization adds a penalty term to the cost function that discourages large weight values.

\[
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l=1}^{L} ||W^{[l]}||_F^2
\]

Where \(\lambda\) is the regularization parameter.

### Dropout Regularization

- Dropout randomly drops neurons during training to prevent reliance on any single neuron.

\[
\text{During training, for each neuron: } d^{[l]} \sim \text{Bernoulli}(p)
\]
\[
A^{[l]} = A^{[l]} \ast d^{[l]}
\]
\[
A^{[l]} = \frac{A^{[l]}}{p}
\]

Where:
- \(d^{[l]}\) is a dropout vector for layer \(l\).
- \(p\) is the dropout probability.

## Summary

- Week 1 introduced practical aspects of setting up machine learning applications.
- The importance of proper data splitting and understanding bias vs. variance was discussed.
- Regularization techniques like L2 regularization and dropout were explained.

### Key Concepts
- Train/Dev/Test sets
- Bias and variance
- L2 regularization (weight decay)
- Dropout regularization

## Week 2: Optimization Algorithms

## Mini-batch Gradient Descent

- Mini-batch gradient descent divides the training set into smaller batches for faster convergence.

### Mini-batch Notation

- Notation used in mini-batch gradient descent:

\[
\text{for } t = 1, 2, \ldots, \left(\frac{m}{\text{mini-batch size}}\right)
\]
\[
Z^{[t]} = W^{[l]}A^{[t-1]} + b^{[l]}
\]
\[
A^{[t]} = g^{[l]}(Z^{[t]})
\]

Where:
- \(t\) represents the mini-batch number.
- \(m\) is the number of training examples.

## Exponentially Weighted Averages

- Exponentially weighted averages are used in optimization algorithms.

### Moving Average

- A moving average at time \(t\) is defined as:

\[
v_t = \beta v_{t-1} + (1 - \beta)\theta_t
\]

Where:
- \(\beta\) is the smoothing parameter.
- \(\theta_t\) is the value at time \(t\).

## Bias Correction in Exponentially Weighted Averages

- Bias correction is used to obtain accurate moving averages at the beginning of training.

\[
v_t^{corrected} = \frac{v_t}{1 - \beta^t}
\]

## Gradient Descent with Momentum

- Momentum helps accelerate gradient descent by adding a fraction of the previous velocity to the current update.

### Momentum Update Rule

\[
V_{dW} = \beta V_{dW} + (1 - \beta) dW
\]
\[
V_{db} = \beta V_{db} + (1 - \beta) db
\]
\[
W = W - \alpha V_{dW}
\]
\[
b = b - \alpha V_{db}
\]

Where:
- \(\beta\) is the momentum parameter.
- \(V_{dW}\) and \(V_{db}\) are the velocity terms.

## RMSprop

- RMSprop adjusts the learning rates adaptively for each parameter.

### RMSprop Update Rule

\[
S_{dW} = \beta S_{dW} + (1 - \beta) (dW^2)
\]
\[
S_{db} = \beta S_{db} + (1 - \beta) (db^2)
\]
\[
W = W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon}
\]
\[
b = b - \alpha \frac{db}{\sqrt{S_{db}} + \epsilon}
\]

Where:
- \(\beta\) is the decay parameter.
- \(S_{dW}\) and \(S_{db}\) are the squared gradients.
- \(\epsilon\) is a small constant to avoid division by zero.

## Adam Optimization Algorithm

- Adam combines ideas from RMSprop and momentum for efficient optimization.

### Adam Update Rule

\[
V_{dW} = \beta_1 V_{dW} + (1 - \beta_1) dW
\]
\[
V_{db} = \beta_1 V_{db} + (1 - \beta_1) db
\]
\[
S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) (dW^2)
\]
\[
S_{db} = \beta_2 S_{db} + (1 - \beta_2) (db^2)
\]
\[
V_{dW}^{corrected} = \frac{V_{dW}}{1 - (\beta_1^t)}
\]
\[
V_{db}^{corrected} = \frac{V_{db}}{1 - (\beta_1^t)}
\]
\[
S_{dW}^{corrected} = \frac{S_{dW}}{1 - (\beta_2^t)}
\]
\[
S_{db}^{corrected} = \frac{S_{db}}{1 - (\beta_2^t)}
\]
\[
W = W - \alpha \frac{V_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}} + \epsilon}
\]
\[
b = b - \alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}} + \epsilon}
\]

Where:
- \(\beta_1\) and \(\beta_2\) are the decay parameters.
- \(V_{dW}\), \(V_{db}\), \(S_{dW}\), and \(S_{db}\) are the velocity and squared gradient terms.
- \(\epsilon\) is a small constant to avoid division by zero.
- \(t\) represents the iteration number.

## Learning Rate Decay

- Learning rate decay reduces the learning rate over time to fine-tune training.

### Exponential Decay

- One common approach is to use exponential decay:

\[
\alpha = \frac{1}{1 + \text{decay rate} \cdot \text{epoch number}} \cdot \alpha_0
\]

Where:
- \(\alpha\) is the learning rate at each epoch.
- \(\alpha_0\) is the initial learning rate.

## Summary

- Week 2 covered optimization algorithms for deep learning.
- Mini-batch gradient descent, momentum, RMSprop, and Adam were explained.
- Exponentially weighted averages and learning rate decay were introduced.

### Key Concepts
- Mini-batch gradient descent
- Exponentially weighted averages
- Bias correction
- Gradient descent with momentum
- RMSprop
- Adam optimization algorithm
- Learning rate decay

# Week 3: Structuring Machine Learning Projects

## Introduction to ML Strategy

- Structuring machine learning projects is crucial for success.

## Setting up your Goal

- Clearly define the goal of your machine learning project.
- Decide whether it's worth pursuing.

## Comparing to Human-Level Performance

- Understanding human-level performance helps set a benchmark for your model.

## Avoidable Bias

- Avoidable bias is the gap between your model's performance and human-level performance.

## Understanding Human-Level Error

- Analyzing the different types of errors humans make helps in improving machine learning models.

## Surpassing Human-Level Performance

- Once your model surpasses human-level performance, focus on reducing bias and variance.

## Machine Learning Flight Simulator

- A machine learning flight simulator is a process to iterate and improve your model.

## AI in Industry

- Practical insights into applying machine learning in real-world industries.

## Summary

- Week 3 focused on structuring machine learning projects.
- Setting clear goals and benchmarking against human-level performance is essential.
- Avoidable bias, understanding human-level error, and surpassing human-level performance were discussed.
- The machine learning flight simulator concept and AI in industry were introduced.

### Key Concepts
- Setting up goals
- Comparing to human-level performance
- Avoidable bias
- Understanding human-level error
- Surpassing human-level performance
- Machine learning flight simulator
- 