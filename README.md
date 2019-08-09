
### Questions/Insights
- PCA is really useful in image processing
    - b/c images take up space if we can reduce dimensions -> speed up process
- Pixels in images are features
- When fitting the transform if you want to apply it to the test data you can't use fit_transform again
    - because it uses mean normalization, so the mean changes from train to test data
- The way that PCA reduces dimensionality is by removing the least important axis...keeps one with higher variance
- PCAing data is supposed to be fast on grid searches.

### Objectives
- Define Eigenvalues and Eigenvectors
- Describe how these are used in PCA
- Apply PCA to reduce dimensions of data

### Outline
- Questions
- Explain eigenvalues and eigenvectors and why they're awesome
- Apply eigen decomposition to the correlation matrix and discuss how it's used in PCA
- Apply PCA to some dataset that we create using sklearn


```python
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
```

### Eigenvalues and Eigenvectors are only applied on square matrices


```python
A = np.random.randint(10, 20, size=(3, 3))
A 
```




    array([[16, 13, 17],
           [14, 16, 19],
           [12, 16, 17]])



### Eigen Properties
- Eigen Values multiply to the determinent of A
- Eigen Values add to the trace of matrix A (trace = sum of diagonals)
- Eigen Vectors are orthonormal *always normal not always orthogonal*
    - what is 'orthonormality'?
        - orthogonal -> perpendicular, the angle is 90 degrees
        - normal -> vector length = 1


```python
evals, evecs = np.linalg.eig(A)
evals, evecs
```




    (array([46.59404279,  3.08723726, -0.68128005]),
     array([[-0.56733781, -0.85134788, -0.28192077],
            [-0.60593555,  0.22744997, -0.60839181],
            [-0.55764677,  0.47272963,  0.74187605]]))




```python
evals_diag = np.diag(evals)
evals_diag
```




    array([[46.59404279,  0.        ,  0.        ],
           [ 0.        ,  3.08723726,  0.        ],
           [ 0.        ,  0.        , -0.68128005]])




```python
evals, evecs
```




    (array([46.59404279,  3.08723726, -0.68128005]),
     array([[-0.56733781, -0.85134788, -0.28192077],
            [-0.60593555,  0.22744997, -0.60839181],
            [-0.55764677,  0.47272963,  0.74187605]]))



### Eigenvalues


```python
np.prod(evals), np.linalg.det(A)
```




    (-98.00000000000033, -98.00000000000004)




```python
np.sum(evals), np.trace(A)
```




    (49.000000000000014, 49)




```python
evecs_inv = np.linalg.inv(evecs)
```


```python
evals, evecs, evecs_inv
```




    (array([46.59404279,  3.08723726, -0.68128005]),
     array([[-0.56733781, -0.85134788, -0.28192077],
            [-0.60593555,  0.22744997, -0.60839181],
            [-0.55764677,  0.47272963,  0.74187605]]),
     array([[-0.5153841 , -0.56279276, -0.65738204],
            [-0.89084741,  0.65289909,  0.19689283],
            [ 0.1802561 , -0.8390678 ,  0.72833724]]))




```python
v1 = evecs.T[0]
v2 = evecs.T[1]
```


```python
v1.dot(v2) # not orthogonal
```




    0.08156567299094766



### Eigenvectors if A is symmetrical


```python
### Let's make a symmetrical matrix
# A Matrix M is symmetrical iff M.T == M
A  = np.random.randint(10, 100, size=(5000, 3))
A_sym = np.cov(A.T) 
A_sym
```




    array([[685.36053115,  15.74046305,  19.44167121],
           [ 15.74046305, 682.13611326,   8.34016315],
           [ 19.44167121,   8.34016315, 681.98656867]])




```python
evals, evecs = np.linalg.eig(A_sym)
evals, evecs
```




    (array([712.95964765, 662.52013564, 674.0034298 ]),
     array([[ 0.67116995,  0.73428336,  0.10177836],
            [ 0.49262623, -0.33920074, -0.80141266],
            [ 0.55394069, -0.58802279,  0.58938859]]))




```python
### normal -> all eigenvectors are normal, always
# A vector v is normal iff length of v is 1
np.sqrt(np.sum(evecs[:, 0]**2))
```




    1.0




```python
### because A_sym is symmetrical the vectors are also orthogonal
# vectors a and b are orthogonal iff the angle between a and b is 90 degree (dot product = 0)

np.dot(evecs[:, 0], evecs[:, 1])
```




    -2.7755575615628914e-17




```python
## if A*B = I -> A, B are inverses
## evecs.T = evecs.inv()
np.round(evecs.dot(evecs.T), 2)
```




    array([[ 1., -0.,  0.],
           [-0.,  1.,  0.],
           [ 0.,  0.,  1.]])




```python
A_sym
```




    array([[685.36053115,  15.74046305,  19.44167121],
           [ 15.74046305, 682.13611326,   8.34016315],
           [ 19.44167121,   8.34016315, 681.98656867]])




```python
# Q L Q^T
evecs.dot(np.diag(evals)).dot(evecs.T)
```




    array([[685.36053115,  15.74046305,  19.44167121],
           [ 15.74046305, 682.13611326,   8.34016315],
           [ 19.44167121,   8.34016315, 681.98656867]])




```python
### eigenvecs and vals redescribe your space 
### if your space is symmetrical, then the eigenvecs are a basis
### so...why is this important for dimensionality reduction?
```

## PCA
1. Calculate the covariance matrix of your data, C -> Symmetrical
2. Calculate the eigenvecs and eigenvalues of covariance matrix
    - eigenvectors are normal and orthogonal
3. Project our data onto the vectors that most describe the correlation

* feature engineering (kinda) on the variance

## Pros
- Reducing dimensionality gives fast processing time
- Get more accurate results, at times
- Don't have to drop features
- Visualize your data



## Cons
- lose ALL interpretability
- To reduce could be computationally expensive
- lose relationships

### Decomposition for non square matrix
- SVD (Singular Value Decomposition)
- Linear Discriminant Analysis
- Reduces any matrix

### Assessment
- use PCA for predicting with distance mls or if feature interpretability are unimportant
- learned that the PCA features are combinations of other features

