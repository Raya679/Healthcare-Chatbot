# Building Blocks of Deep Neural Networks

## Layer l:

## Forward Propagation:

#### Input: a<sup>[l-1]</sup>

#### Output: a<sup>[l]</sup>, z<sup>[l]</sup>(cache), W<sup>[l]</sup>, b<sup>[l]</sup>

#### Computation: z<sup>[l]</sup>=W<sup>[l]</sup> a<sup>[l-1]</sup> + b<sup>[l]</sup> and a<sup>[l]</sup>=g<sup>[l]</sup>(z<sup>[l]</sup>)

## Backward Propagation:

#### Input: da<sup>[l]</sup> and z<sup>[l]</sup>(cache)

#### Output: da<sup>[l-1]</sup>, dW<sup>[l]</sup> and db<sup>[l]</sup>

---

#### Each hidden layer has a corresponding forward and backward propagation step
