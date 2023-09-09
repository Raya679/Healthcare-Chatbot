# Normalizing Inputs

### Two steps:

#### 1) Subtract mean:

#### mu=(1/m) \* np.sum(x<sup>(i)</sup>)

#### x:= x-mu

#### 2) Normalize variance:

#### sigma<sup>2</sup>=(1/m) \* np.sum((x<sup>(i)</sup>)<sup>2</sup>) and x/=sigma

#### Use same mu and sigma<sup>2</sup> for both train and test sets

# Vanishing / Exploding Gradients

### For very deep NNs

#### w<sup>[l]</sup> > I : it means that the gradient is exploding as z also becomes larger

#### w<sup>[l]</sup> <> I : it means that the gradient is vanishing as z also becomes smaller

#### Two sided difference calculation is more accurate and has very less error approx. than one sided difference calculation

# Gradient Checking

#### Consider parameters w<sup>[1]</sup>,b<sup>[1]</sup>,....w<sup>[l]</sup>,b<sup>[l]</sup> and reshpe them into a matrix $\theta$.

#### Therefore cost function will be a function of $\theta$ ,J($\theta$)

#### During bacpropagation consider dw<sup>[1]</sup>,db<sup>[1]</sup>,....,dw<sup>[l]</sup>,db<sup>[l]</sup> and reshpe them into matrix dtheta

#### Therefore J($\theta$)=J($\theta$<sub>1</sub>,$\theta$<sub>2</sub>,$\theta$<sub>3</sub>,....)

#### for each i:

#### d $\theta$<sub>app</sub>[i]=J($\theta$<sub>1</sub>,$\theta$<sub>2</sub>,$\theta$<sub>3</sub>,...,$\theta$<sub>i</sub>+$\epsilon$,..)-J($\theta$<sub>1</sub>,$\theta$<sub>2</sub>,$\theta$<sub>3</sub>,...,$\theta$<sub>i</sub>-$\epsilon$,..)/2 $\epsilon$

#### This sholud be approx equal to d $\theta$[i]=partial derivative of J wrt $\theta$<sub>i</sub>

#### d $\theta$<sub>app</sub> must be approx equal to d $\theta$

#### Gradient check=||d $\theta$<sub>app</sub>-d $\theta$||<sub>2</sub>/||d $\theta$<sub>app</sub>||<sub>2</sub>+||d $\theta$||<sub>2</sub>

# Gradient Checking Implementation

#### Don,t use in training- only use for debugging

#### If algorithm fails during grad check, look at the components to find the bug

#### Remember regularization

#### Doesn't work with dropout

#### Run at random initialization, perhaps again after some time
