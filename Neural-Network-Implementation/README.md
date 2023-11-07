### What is Neural network?

- In neural networks, given enough data about x and y, given enough training examples with both x and y, neural networks are remarkably good at figuring out functions that accurately map from x to y.
- Supervised learning means we have the (X,Y) and we need to get the function that maps X to Y.

We trained a shallow neural network for cat non-cat classification . As this implementation
was from scratch, we made functions for all our tasks. These functions are listed below:
  1. sigmoid() , relu() - activation functions
  2. sigmoid_backward() , relu_backward() - to find the derivatives during backpropogation
  3. initialize_parameters() - to randomly initialize the weights, and set biases to 0
  4. linear_forward() - forward propogation of a layer
  5. linear_activation_forward() - applies activation function to the output of linear_forward()
  6. L_model_forward() - forward propgation for the entire neural network
  7. compute_cost() - for cost calculation
  8. linear_activation_backward() -  backpropogation of a layer
  9. L_model_backward() - uses derivatives from the previous functions to find changes to be made to
                          the weights and biases.
  10. update_parameters() - update the weights and biases to reduce the cost

  This was a 5 layer neural network.


