## Recurrent Neural Networks

> Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

### Why sequence models
- Sequence Models like RNN and LSTMs have greatly transformed learning on sequences in the past few years.
- Examples of sequence data in applications:
  - Speech recognition (**sequence to sequence**):
    - Music generation (**one to sequence**):
    - Machine translation (**sequence to sequence**):

 **Representing words**:
    - We will now work in this course with **NLP** which stands for natural language processing. One of the challenges of NLP is how can we represent a word?

    1. We need a **vocabulary** list that contains all the words in our target sets.
        - Example:
            - [a ... And   ... Harry ... Potter ... Zulu]
            - Each word will have a unique index that it can be represented with.
            - The sorting here is in alphabetical order.
        - Vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the bigger companies use even a million.
        - To build vocabulary list, you can read all the texts you have and get m words with the most occurrence, or search online for m most occurrent words.
    2. Create a **one-hot encoding** sequence for each word in your dataset given the vocabulary you have created.
        - While converting, what if we meet a word thats not in your dictionary?
        - We can add a token in the vocabulary with name `<UNK>` which stands for unknown text and use its index for your one-hot vector.
### Recurrent Neural Network Model
- Why not to use a standard network for sequence tasks? There are two problems:
  - Inputs, outputs can be different lengths in different examples.
    - This can be solved for normal NNs by paddings with the maximum lengths but it's not a good solution.
  - Doesn't share features learned across different positions of text/sequence.
    - Using a feature sharing like in CNNs can significantly reduce the number of parameters in your model. That's what we will do in RNNs.
- Recurrent neural network doesn't have either of the two mentioned problems.

### Backpropagation through time
- Let's see how backpropagation works with the RNN architecture.
- Usually deep learning frameworks do backpropagation automatically

- So far we have seen only one RNN architecture in which T<sub>x</sub> equals T<sub>Y</sub>. In some other problems, they may not equal so we need different architectures.

- The architecture we have described before is called **Many to Many**.
- In sentiment analysis problem, X is a text while Y is an integer that rangers from 1 to 5. The RNN architecture for that is **Many to One**
- A **One to Many** architecture application would be music generation.  
 
  - Note that starting the second layer we are feeding the generated output back to the network.
- There are another interesting architecture in **Many To Many**. Applications like machine translation inputs and outputs sequences have different lengths in most of the cases. So an alternative _Many To Many_ architecture that fits the translation would be as follows:   
  
  - There are an encoder and a decoder parts in this architecture. The encoder encodes the input sequence into one matrix and feed it to the decoder to generate the outputs. Encoder and decoder have different weight matrices.
- Summary of RNN types:   
   ![](Images/12_different_types_of_rnn.jpg)
- There is another architecture which is the **attention** architecture which we will talk about in chapter

### Vanishing gradients with RNNs
- One of the problems with naive RNNs that they run into **vanishing gradient** problem.

- An RNN that process a sequence data with the size of 10,000 time steps, has 10,000 deep layers which is very hard to optimize.

### Gated Recurrent Unit (GRU)
- GRU is an RNN type that can help solve the vanishing gradient problem and can remember the long-term dependencies.
### Long Short Term Memory (LSTM)
- LSTM - the other type of RNN that can enable you to account for long-term dependencies. It's more powerful and general than GRU.
#### Word Representation
- NLP has been revolutionized by deep learning and especially by RNNs and deep RNNs.
- Word embeddings is a way of representing words. It lets your algorithm automatically understand the analogies between words like "king" and "queen".

- Word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively smaller training set.
- Also, one of the advantages of using word embeddings is that it reduces the size of the input!
  - 10,000 one hot compared to 300 features vector.


## Sequence models & Attention mechanism

> Sequence models can be augmented using an attention mechanism. This algorithm will help your model understand where it should focus its attention given a sequence of inputs. This week, you will also learn about speech recognition and how to deal with audio data.

#### Beam Search
- Beam search is the most widely used algorithm to get the best output sequence. It's a heuristic search algorithm.
#### BLEU Score
- One of the challenges of machine translation, is that given a sentence in a language there are one or more possible good translation in another language. So how do we evaluate our results?
- The way we do this is by using **BLEU score**. BLEU stands for _bilingual evaluation understudy_.
- The intuition is: as long as the machine-generated translation is pretty close to any of the references provided by humans, then it will get a high BLEU score.