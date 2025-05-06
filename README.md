This is my implementation of the algorithms described and proposed in the paper "Learning Fast Approximations of Sparse Coding" by Karol Gregor and Yann LeCun. 

## Introduction
Sparse Coding is an unsupervised learning paradigm in which, for each input x(t) 
we want to find a latent representation h(t) such that h(t) is SPARSE, and it allows us, by 
means of a linear transformation (known as a **dictionary matrix**, to reconstruct the input vector as well as possible.

The justifications for sparsity can be provided by interpreting the columns of the learned
dictionary matrix as elements of a dictionary, which combine together to reconstruct
a good approximation of the input vector.
## Results
#### Convergence times:
All the algorithms were run on a subsample of 600 images from the MNIST dataset.\
\
Average time taken in ISTA (in s) = 1.0653418576717377 \
Average time taken in FISTA (in s) = 0.34760572989781696 

Unfortunately, I have not yet been able to get the CoordinateDescent algorithm to truly converge, so the finishing time has not been reported.
