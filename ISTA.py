import numpy as np
import torch
from torch.utils.data import DataLoader
import itertools

def shrink(a, b):
    return torch.sign(a) * torch.maximum(torch.abs(a) - b, torch.tensor(0.0))

def change(a, b):
    return torch.sum(torch.abs(a - b))

def ISTA(x : torch.Tensor,
         h_dim : int,
         D : torch.Tensor,
         regularization=0.5,
         lr=None,
         frequency=None) -> torch.Tensor:
    """
    Implements the ISTA algorithm to find the optimal
    sparse codes for the given input vector x and a given
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
    assert D.shape == (n, m), (f"D should be matrix of shape (x_dim, h_dim), where "
                               f"x_dim={n} and h_dim={m}, but dimensions of D are {D.shape}")
    h = torch.zeros(h_dim)
    h_old = None

    #--------------------------- SETTING CORRECT LEARNING RATE -----------------
    """
    A very crucial piece of information is that the ISTA algorithm converges
    if and only if 1/lr > the largest eigenvalue of matmul(D^T, D) where D is
    the dictionary matrix provided.
    """
    max_eig = torch.linalg.eigvals(torch.matmul(torch.transpose(D, 0, 1), D)).abs().max()
    if lr is None:
        lr = 1/(max_eig + 10)
    else:
        assert 1/lr > max_eig, (f"For convergence, the learning rate must satisfy 1/lr > the maximum eigenvalue of matmul(D^T, D)."
                                f"The maximum eigenvalue is {max_eig}, set the learning rate appropriately")

    counter = itertools.count(start=0, step=1)
    while True:
        h_old = h
        u = torch.matmul(D, h_old) - x
        h = h_old - lr * torch.matmul(torch.transpose(D, 0, 1) , u)
        h = shrink(h, lr * regularization)
        if change(h, h_old) < 0.001:
            break
        iter_num = next(counter)
        if frequency is not None and iter_num % frequency == 0:
            print(f"ISTA: Iteration {iter_num}")
    return h

