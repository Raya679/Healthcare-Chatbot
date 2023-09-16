## Week 1

### Recurrent Neural Networks

**Recurrent Neural Network Model**

- Why RNN and not Standard Neural network?
    - Inputs, outputs can be different lengths in different examples.
    - Doesn't share features learned across different positions of text/sequence.

![Image](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/15.png)

- Limitation of Unidirectional RNNs
   - prediction at a certain time uses info of inputs earlier in the sequence but not info later in the sequence

- Forward propogation equations -

![Forward Prop](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/04.png)

- Backward propogation equations -

![Backward prop](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/08.png)

**Different types of RNNs**

![Types of RNN](https://i.stack.imgur.com/6VAOt.jpg)

**Language model and sequence generation**

- Training set : large corpus of English text
1. Tokenize
2. Map to one-hot indies in the vocabulary
3. RNN - Predicts one word at a time going from left to right

![RNN](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/13.png)

- Loss function is given by -

![Loss function](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/14.png)

**Sampling a sequence from trained RNN**

- Character-level language model -
    - In the character-level language model, the vocabulary will contain [a...z,A...Z,0...9], punctuation, special characters and possibly token.

Disadvantages - 
   - much longer sequence
   - computationally expensive
   - not good at capturing long range dependencies

**Vanishing gradients with RNNs**

- RNNs are not good at capturing long term dependencies 
- Difficult to get the neural network to realise that it needs to memorize
- Vanishing gradients problem tends to be the bigger problem with RNNs than the exploding gradients problem.
- Exploding gradients can be easily seen when your weight values become NaN. So one of the ways solve exploding gradient is to apply gradient clipping means if your gradient is more than some threshold - re-scale some of your gradient vector so that is not too big. So there are cliped according to some maximum value.

**Gated Recurrent Unit (GRU)**

![GRU](https://miro.medium.com/v2/resize:fit:723/1*O8k5NP_FF3zKCwDZUYoiaA.png)

- Equations

![GRU equation](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/20.png)

**Long Short Term Memory (LSTM)**

![](https://franklinwu19.github.io/2018/08/27/rnn-lstm/rnn1.png)

![LSTMs](https://franklinwu19.github.io/2018/08/27/rnn-lstm/LSTM.png)

**Bidirectional RNN**

![BRNN](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*6QnPUSv_t9BY9Fv8_aLb-Q.png)

- The disadvantage of BRNNs that you need the entire sequence before you can process it.

**Deep RNNs**

![Deep RNNs](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/25.png)