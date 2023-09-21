# Sequence Models
#### Mainly used for speech recognition, Music generation, sentiment classification, DNA sequence analysis, machine translation, video activity recognition, name entity recognition, etc.
### Notations:
- X<sup>(i)< t ></sup> denotes the t<sup>th</sup> element of the i<sup>th</sup> training example in the input sequence
- y<sup>(i)< t ></sup> denotes the t<sup>th</sup> element of the i<sup>th</sup> training example in the output sequence
- T<sub>x</sub><sup>(i)</sup> denotes the length of the input sequence for the i<sup>th</sup> training example
- T<sub>y</sub><sup>(i)</sup> denotes the length of the output sequence for the i<sup>th</sup> training example

### One-Hot representation:
#### Representing the given word / desired word as 1 and keeping others 0, in a vector(one hot vector) of size = size of the vocabulary dictionary

# RNNs
- The problems of using standard NN :
#### Inputs and outputs have different lengths
#### Doesn't share features learned across different positions of text
- Whereas RNN uses the info from inputs earlier in the sequence and not from those later in the sequence (for unidirectional RNN)
- For a bidirectional RNN, it uses the info which occurs later in the sequence

![LCO](https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C5%20-%20Sequence%20Models/Notes/Images/15.png)
# Forward Propagation:
![lco](https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C5%20-%20Sequence%20Models/Notes/Images/04.png)
# Backward Propagation:
![lco](https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C5%20-%20Sequence%20Models/Notes/Images/08.png)

- for Backprop compute loss for each time step and sum up
- Backprop proceeds in exactly opposite direction
### Types of RNN Architectures:
- Many to many architectures (e.g : encoder and decoder)
- Many to one (e.g : Sentiment Analysis)
- One to one
- One to many (e.g : Music generation)

# Language model and Sequence generation
- Training set: comprises of large corpus of english text. <EOS> is taken at the end of the sentence
- For unknown words, replace the word with a token <unk>

![lco](https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C5%20-%20Sequence%20Models/Notes/Images/13.png)

- X<sup> < t > </sup> = y<sup>< t-1 ></sup>
### Loss function:

![lco](https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C5%20-%20Sequence%20Models/Notes/Images/14.png)

# Sampling novel sequence
- Vocabulary level language model -> each y<sup>< t ></sup> points to a word
- Character level language model -> each y<sup>< t ></sup> points to a character
### Character level language model
- No issue regarding unknown words
- Ends up with long sequences 
- Not good at long term dependencies
- Computation is costly

# Vanishing Gradients
- It occurs in RNNs since they are weak at long term dependencies 
- For exploding gradients, perform gradient clipping ("NaN" is shown for exploding gradients)

# Gated Recurrent Unit
![lco](https://camo.githubusercontent.com/79e3f02becc35d0996a4cc3583dc5773a1fd6962ddc2a9ab8b63f6550e86c913/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f76322f726573697a653a6669743a3732332f312a4f386b354e505f4646337a4b4377445a55596f6961412e706e67)

### Equations:
![lco](https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C5%20-%20Sequence%20Models/Notes/Images/20.png)

# LSTM (Long Short Term Memory)
![loc](https://camo.githubusercontent.com/549db8785e6e81d1f3817786a5470523fca2f2141fa0d29a7ccaeee99d7f341c/68747470733a2f2f6672616e6b6c696e777531392e6769746875622e696f2f323031382f30382f32372f726e6e2d6c73746d2f726e6e312e706e67)

### Equations:
![lco](https://camo.githubusercontent.com/e2ad3f941889557802826babe9ab790b4c1a59309609a064eafd5a3353a9fcca/68747470733a2f2f6672616e6b6c696e777531392e6769746875622e696f2f323031382f30382f32372f726e6e2d6c73746d2f4c53544d2e706e67)

# Bidirectional RNNs (BRNNs)
- Backward recurrent layer is added
- Acyclic graph
- Forward prop occurs partially from left to right and partially from right to left
- BRNNs along with LSTM are used for NLP
- In case of BRNNs, you need to process the entire sequence before predicting anything 
#### y<sup>^< t ></sup> = g( Wy [a<sup>->< t ></sup> , a<sup><-< t ></sup>] + by)

![lco](https://camo.githubusercontent.com/a990eec1cfa296dc4c0f8de12aaa87944c6fef22de1b80fc020acabcd72f51e0/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f76322f726573697a653a6669743a313130302f666f726d61743a776562702f312a36516e505553765f74394259394676385f614c622d512e706e67)
# Deep RNNs
