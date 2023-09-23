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


## Week 2

### Introduction to Word Embeddings

**Word Representation**

-Word embeddings is a way of representing words.

- one-hot representation

![one-hot](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/27.png)

-  Featurized representation
   - Better than one-hot representation as it gives us the relationship between the words
   - It will allow the algorithm to quickly figure out the relationship between the words

![featurized representation](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/28.png)

- Visualizing word embeddings using t-SNE algorithm

![t-SNE](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/29.png)


**Using word embeddings**

- We use these representations in many different NLP tasks such as sentiment analysis, named entity recognition and machine translation etc..
- can examine very large text corpus

- Transfer Learning -
  - takes information learned from huge amount of unlabeled text and transfers that knowledge to a task for which you might have a small labeled training set

- Transfer Learning and word embeddings
1. Learn word embeddings from large text corpus 
2. Transfer embedding to new task with the smaller training set
3. Continue to finetune the word embeddings with new data (only if labelled data set is large enough)

**Properties of word embeddings**

- Helps with analogy reasoning

![analogy](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/32.png)

- Here eMan - eWoman ≈ eKing - e??

- Cosine similarity

```
CosineSimilarity(u, v) = u . v / ||u|| ||v|| = cos(θ)
```
- Thus we can use `u = ew and v = eking - eman + ewoman`

**Embedding matrix**

![matrix](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/36.png)


### Learning Word Embeddings: Word2vec & GloVe

**Word2Vec**

- Skip-grams
  - Vocabulary size = 10,000 words
  - Let's say that the context word are c and the target word is t
  - We want to learn c to t
  - We get ec by E.oc
  - We then use a softmax layer to get P(t|c) which is ŷ

![Formula](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/39.png)

- Problem with softmax classification
  - computational speed
- Solution - Hierarichal softmax classifier

![Hierarichal softmax classifier](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/40.png)

- How to sample context c?
  - One way is to choose the context by random from your corpus
  - In practice, we don't take the context uniformly random, instead there are some heuristics to balance the common words and the non-common words

**Negative Sampling**

- Negative sampling allows you to do something similar to the skip-gram model, but with a much more efficient learning algorithm.

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/41.png)

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/42.png)

- Selecting negative samples

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/43.png)

**GloVe word vectors**

- GloVe - global vectors for word representations

- Xct = # times t appears in context of c

- Model -

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/44.png)


### Applications using Word Embeddings

**Sentiment Classification**

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/46.png)

- RNN for sentiment classification

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/47.png)

**Debiasing word embeddings**

- We want to make sure that our word embeddings are free from undesirable forms of bias, such as gender bias, ethnicity bias and so on.

- Steps
1. Train a model on original data (without debias) and get its predictions
2. Identify the direction.

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/49.png)

3. Neutralize: For every word that is not definitional, project to get rid of bias

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/50.png)

4. Equalize pairs - We want each pair to have difference only in gender.

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/51.png)


# Week 3 -

## Sequence models & Attention mechanism

### Various sequence to sequence architectures

**Basic Models**

- Examples 
  - Machine translation
  - Image Captioning

- Basic model architecture will include encoder and decoder

![Encoder-Decoder](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/53.png)

**Picking the most likely sentence**

- Machine Translation is building a conditional Language model

![Model](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/55.png)

- The most common algorithm to find the next likely words is beam search

- Why not greedy search?
  - The method of picking the first word that is most likely, then picking the second word and so on doesn't really work

- Therefore Approximate Search Algorithm - tries to pick the sentence that maximizes the conditional probability 

**Beam Search**

- Approximate search Algorithm

- Step 1:
   - Suppose B = 3 where B is beam Width i.e number of possibilities that will be considered
   - So 3 most likely first words will be considered
- Step 2:
   - For each of the choices it will consider what should be the next word
   - i.e there will be `B*(no.of words in vocab)` possibilities
   - Then 3 most likely (as B=3) choices will be considered out of these (May reject choices which were candidates for the first word)
- Step 3:
   - Then the third most likely words will be considered 
   - Again top 3 words will be considered and so on.
- Outcome:
  - The most likely sentence

**Refinements to Beam Search**

- Length Normalization -

 ![eq1](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/56.png)

  - And to do that we multiply: `P(y<1> | x) * P(y<2> | x, y<1>) * ... * P(y<t> | x, y<y(t-1)>)`
  - Each probability is a fraction, most of the time a small fraction.
  - Multiplying small fractions will cause a numerical overflow. Meaning that it's too small for the floating part representation in your computer to store accurately.

  - summing logs of probabilities instead of multiplying directly.

![eq2](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/57.png)

   - The two optimization functions we have mentioned are preferring small sequences rather than long ones. Because multiplying more fractions gives a smaller value, so fewer fractions - bigger result.
   
   - dividing by the number of elements in the sequence

![eq3](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/58.png)

- How to choose B?
  - large B - better results but slower
  - small B - worse results but faster

**Error analysis in beam search**

- To do that, we calculate P(y* | X) and P(ŷ | X). There are two cases:
  - Case 1 `(P(y* | X) > P(y_hat | X))`: 
    - Conclusion: Beam search is at fault.
  - Case 2 `(P(y* | X) <= P(y_hat | X))`:
    - Conclusion: RNN model is at fault.

**BLEU Score**

- BLEU stands for bilingual evaluation understudy.
- Given a machine generated translation it allows you to automatically compute a score that measures how good is that machine translation

![eq](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/60.png)

- helps in measuring the degree to which the machine translation output overlaps with the references

- Pn = Bleu score on one type of n-gram
- Combined BLEU score = `BP * exp(1/n * sum(Pn))`
- BP stands for brevity penalty
  - adjustment factor that penalizes translation systems that outputs translations that are too short

![eq](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/62.png)

**Attention Model Intuition**

- allows a neural network to pay attention to only part of an input sentence while it generating translation much like human translation

![intuition](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/66.png)

**Attention Model**

![intuition](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/67.png)

![eq](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/70.png)

![eq](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/71.png)

- Computing attention
  - alpha<t, t'> = amount of attention y<t> should pay to a<t'>

![eq](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/72.png)

![eq](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/73.png)

- Visualizing attention weights

![eq](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/74.png)

- Dowside - Takes quadratic time and quadratic cost to run this algorithm

**Speech Recognition**

- can use attention model

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/77.png)

- One method - CTC cost which stands for "Connectionist temporal classification"

**Trigger Word Detection**

![img](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/81.png)

- One disadvantage of this creates a very imbalanced training set. There will be a lot of zeros and few ones.
