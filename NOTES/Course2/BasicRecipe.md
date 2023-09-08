# Basic Recipe for ML

### If the model has high bias:

#### if it does have high bias, does not even fitting in the training set that well, some things you could try would be to try pick a network, such as more hidden layers or more hidden units, or you could train it longer, you know, maybe run trains longer or try some more advanced optimization algorithms

### If the model has high variance:

#### if you have high variance, the best way to solve a high variance problem is to get more data, if you can get it. But sometimes you can't get more data. Or, you could try regularization, to try to reduce overfitting.But if you can find a more appropriate neural network architecture, sometimes that can reduce your variance problem as well, as well as reduce your bias problem

### To get a low bias and low variance model:

#### In the modern deep learning, big data era, so long as you can keep training a bigger network, and so long as you can keep getting more data, which isn't always the case for either of these, but if that's the case, then getting a bigger network almost always just reduces your bias, without necessarily hurting your variance, so long as you regularize appropriately. And getting more data, pretty much always reduces your variance and doesn't hurt your bias much.
