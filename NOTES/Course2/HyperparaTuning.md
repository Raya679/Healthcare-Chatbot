# Hyperparameter Tuning

### The various hyperparameters are :

#### 1.$\alpha$

#### 2.$\beta$ =0.9

#### 3.$\beta$ 1 =0.9

#### 4.$\beta$ 2 =0.999

#### 5.$\epsilon$ = 10^-8

#### 6. No. of layers

#### 7. Hidden units

#### 7. Learning rate decay

#### 7. Mini-batch-size : between 1 and m(num of examples)

### Using Grid:

#### Initially we displayed the hyperparameters in grid and then picked out the best out of these hyperparameters (applicable for small number of values)

### Random picking:

#### But for large amount of data, we pick the hyperparameters randomly, and then try out (experiment)

### Coarse to fine:

#### If a point works the best and maybe a few other points around it tended to work really well, then in the course of the final scheme what you might do is zoom in to a smaller region of the hyperparameters, and then sample more density within this space.

### Appropriate scale for hyperparameters:

#### For searching in the range [0.0001 , 1]

#### We use : r= -4 \* np.random.rand() and $\alpha$ = 10^r

#### Babysitting model: (Panda approach)

#### Usually you do this if you have maybe a huge data set but not a lot of computational resources, not a lot of CPUs and GPUs, so you can basically afford to train only one model or a very small number of models at a time. In that case you might gradually babysit that model even as it's training.

#### Caviar approach:

#### You have some setting of the hyperparameters and just let it run by itself ,either for a day or even for multiple days, and then at the same time you might start up a different model with a different setting of the hyperparameters. So this way you can try a lot of different hyperparameter settings and then just maybe quickly at the end pick the one that works best.
