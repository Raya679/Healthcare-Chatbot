# Bias and Variance

#### If we take the example of cat classification:

| Error           | case1         | case2     | case3                       | case4                     |
| --------------- | ------------- | --------- | --------------------------- | ------------------------- |
| Train set error | 1%            | 15%       | 15%                         | 0.5%                      |
| Dev set error   | 11%           | 16%       | 30%                         | 1%                        |
| Result          | High variance | High bias | High bias and High variance | Low bias and Low variance |

#### The above classification gives a brief idea regarding the what behaviour is shown by the two sets in different cases

#### If the train set has high % of error, it has a high bias and as we go from train to dev set if there is huge variation of error, it suggests that there is high variance

#### The human error is approx 0% when it comes to this classification.

#### The values of bias and variance also depend upon optimal (Bayes) error which is generally 0% similar to that of a human but can change if there are blurry images, etc.
