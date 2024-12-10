import numpy as np
import torch
import itertools

def shrink(a, b):
    return torch.sign(a) * torch.maximum(torch.abs(a) - b, torch.tensor(0.0))

def change(a, b):
    return torch.sum(torch.abs(a - b))

def ISTA(x : torch.Tensor,
         h_dim : int,
         D : torch.Tensor,
         regularization=0.5,
         lr=0.01,
         frequency=100):
    """
    Implements the ISTA algorithm to find the optimal
    sparse codes for the given input vector x and
    dictionary matrix D.

    :param x: The input vector
    :param h_dim: Dimension of sparse code required. In
    the overcomplete case, should be greater than or equal to
    dim(x).
    :param D: the dictionary matrix
    :param regularization: parameter which controls the relative
    weight given to sparsity vs reconstruction loss
    :param lr: the learning rate
    :param frequency: the number of iterations after which an update is printed
    :return: An approximation to the optimal sparse code of the
    given input h_opt.
    """
    n = x.shape[0]
    m = h_dim
    assert D.shape == (n, m), (f"D should be matrix of shape (x_sim, h_dim), where "
                               f"x_dim={n} and h_dim={m}, but dimensions of D are {D.shape}")
    h = torch.zeros(h_dim)
    h_old = None

    counter = itertools.count(start=0, step=1)
    while True:
        h_old = h
        u = torch.matmul(D, h_old) - x
        h = h_old - lr * torch.matmul(torch.transpose(D, 0, 1) , u)
        h = shrink(h, lr * regularization)
        if change(h, h_old) < 0.001:
            break
        iter_num = next(counter)
        if iter_num % frequency == 0:
            print(f"Iteration {iter_num}")
    return h
