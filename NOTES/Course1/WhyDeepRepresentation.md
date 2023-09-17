# Why Deep Representation?

#### Neural Networks work in a similar fashion as that of the human brain.

## Face Recognition

#### For e.g : In case of image or face recognition, we generally use CNNs. These Neural Networks initially recognise the small components like edges of the image by grouping together pixels. These edges are then grouped together to form parts of faces. And then, finally, by putting together different parts of faces, like an eye or a nose or an ear or a chin, it can then try to recognize or detect different types of faces.

## Speech Recognition

#### For e.g : In case of speech recognition, the first level of a neural network might learn to detect low level audio wave form features, And then by composing low level wave forms, maybe you'll learn to detect basic units of sound. In linguistics they call phonemes and then composing that together maybe learn to recognize words in the audio. And then maybe compose those together, in order to recognize entire phrases or sentences.

## Circuit Theory and Deep Learning:

#### There are functions you can compute with a small L-layer deep neural networks that shallower neural networks require exponentially more hidden units to compute

#### Let's say for example we have to find XOR of the given inputs.

#### In case of a deep neural network containing small number of hidden units in a particular layer, our time complexity will be O(log n).

#### Whereas in case of a shallow neural network let's say containing only one hidden layer containing a large number of hidden units, our time complexity will be larger i.e. O(2<sup>n</sup>).
