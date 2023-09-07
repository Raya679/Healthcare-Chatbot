# Forward and Backward Propagation: For layer 'l'

### Forward propagation:

#### Input: a<sup>[l-1]</sup>

#### Output: a<sup>[l]</sup>, z<sup>[l]</sup>(cache), W<sup>[l]</sup>, b<sup>[l]</sup>

#### Computation: z<sup>[l]</sup>=W<sup>[l]</sup> a<sup>[l-1]</sup> + b<sup>[l]</sup> and a<sup>[l]</sup>=g<sup>[l]</sup>(z<sup>[l]</sup>)

#### Vectorization: Z<sup>[l]</sup>=W<sup>[l]</sup> A<sup>[l-1]</sup> + b<sup>[l]</sup> and A<sup>[l]</sup>=g<sup>[l]</sup>(Z<sup>[l]</sup>)

### Backward Propagation:

#### Input: da<sup>[l]</sup> and z<sup>[l]</sup>(cache)

#### Output: da<sup>[l-1]</sup>, dW<sup>[l]</sup> and db<sup>[l]</sup>

#### Computation:

#### 1. dz<sup>[l]</sup>=da<sup>[l]</sup> \* g<sup>[l]</sup>'(z<sup>[l]</sup>)

#### 2. dW<sup>[l]</sup>=dz<sup>[l]</sup> da<sup>[l-1]<sup>T</sup></sup>

#### 3. db<sup>[l]</sup>=dz<sup>[l]</sup>

#### 4. da<sup>[l-1]</sup>=W<sup>[l]<sup>T</sup></sup> dz<sup>[l]</sup>

#### 4. dz<sup>[l]</sup>=W<sup>[l+1]<sup>T</sup></sup> dz<sup>[l+1]</sup> \* g<sup>[l]</sup>'(z<sup>[l]</sup>)

#### Vectorization:

#### 1. dZ<sup>[l]</sup>=dA<sup>[l]</sup> \* g<sup>[l]</sup>'(Z<sup>[l]</sup>)

#### 2. dW<sup>[l]</sup>=1/m (dZ<sup>[l]</sup> dA<sup>[l-1]<sup>T</sup></sup>)

#### 3. db<sup>[l]</sup>=1/m \* np.sum(dz<sup>[l]</sup>,axis=1,keepdims=True)

#### 4. dA<sup>[l-1]</sup>=W<sup>[l]<sup>T</sup></sup> dZ<sup>[l]</sup>
