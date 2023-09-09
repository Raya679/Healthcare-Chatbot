# Course 1 week 
```
tanhz activation function makes mean closer to zero as cntralizatio of data

only output layer of binary classification will use sigmoid


if z is very large or small the derivative becomes very small so gradient descent hampered in both.

so Relu  used have slope k

 only for output 0 1 ie binary classification use sigmoid
all other places if req tanh

most preferred for other layers is ReLU(default)


leaky relu : slope -ve when z -ve

slope of non zero very different zero so faster learning.


testing various activation for the diff test val to find the most appropriate

why do we need a non linear activation fu ction??

g(z)=z  linear activation function

the hidden layers becomes insignificant if only linear activation(so linear activaion not used for hidden layers)

regressin model uses linear activation at output layer...


keep dims does ie it avoids generation of rank 1 matrix (n,)
instead forms (n,1)vector

if all weights initialised to zero.. all hidden layers continue to compute the same function

weights initialised to smaller values so at activation small slope of sigmoid and tanh avoided ..(mainly for binary classification)

for deep neural networks choice of const diff.

X.shape[0] counts no.ofv rows

```