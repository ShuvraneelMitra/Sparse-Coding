import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import itertools
from typing import Tuple


def shrink(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a - b, min=0) - torch.clamp(-a - b, min=0)

def change(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(a - b))

def t_gen(init):
    """
    Implements the momentum recurrence for FISTA.
    :param init: Initial value of t
    :return: sequence values of t
    """
    t = init
    while True:
        yield t
        t = 0.5 * (1 + np.sqrt(1 + 4 * (t**2)))

class LISTABase(nn.Module):
    """
    Implements one step of the LISTA algorithm for estimating
    the sparse codes using supervised learning.
    """
    def __init__(self, x_dim: int, hidden_dim: int):
        super(LISTABase, self).__init__()
        self.We = nn.Parameter(torch.randn(hidden_dim, x_dim))
        self.S = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.theta = nn.Parameter(torch.randn([1]))
        self.B = None

    def init_matrices(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        self.B = torch.matmul(self.We, self.x)
        return torch.zeros_like(self.B), shrink(self.B, self.theta)

    def forward(self, x: torch.Tensor, C: torch.Tensor, Z: torch.Tensor) \
            -> Tuple[Tensor, Tensor]:
        C = self.B + torch.matmul(self.S, Z)
        Z = shrink(C, self.theta)
        return C, Z

class LISTA(LISTABase):
    """
    Implements the LISTA algorithm using the
    LISTABase class
    """
    def __init__(self, T: int, x_dim: int, hidden_dim: int):
        super(LISTA, self).__init__(x_dim, hidden_dim)
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, Z = self.init_matrices(x)
        for t in range(self.T):
            C, Z = super().forward(x, C, Z)
        return Z

class LCoDBase(nn.Module):
    """
    Implements one iteration of the Learned Coordinate
    Descent algorithm.
    """
    def __init__(self, x_dim: int, hidden_dim: int):
        super(LCoDBase, self).__init__()
        self.We = nn.Parameter(torch.randn(hidden_dim, x_dim))
        self.S = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.theta = nn.Parameter(torch.randn([1]))
        self.B = None
    
    def init_matrices(self, x: torch.Tensor) -> torch.Tensor:
        self.B = torch.matmul(self.We, self.x)
        return torch.zeros((self.hidden_dim, 1))

    def forward(self, x: torch.Tensor, Z: torch.Tensor) \
            -> torch.Tensor:
        Z_bar = shrink(self.B, self.theta)
        k = torch.argmax(torch.abs(Z - Z_bar))
        for j in range(self.hidden_dim):
            self.B[j] = self.B[j] + self.S[j][k] * (Z_bar - Z)
        Z[k] = Z_bar[k]
        return Z 
    
class LCoD(LCoDBase):
    """
    Implements the LCoD::fprop algorithm using the 
    LCoDBase class
    """
    def __init__(self, T: int, x_dim: int, hidden_dim: int):
        super(LCoDBase, self).__init__(x_dim, hidden_dim)
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.init_matrices(x)
        for t in range(self.T):
            Z = super().forward(x, Z)
        return Z
    
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
        lr = 1/(max_eig + 1)
    else:
        assert 1/lr > max_eig, (f"For convergence, the learning rate must satisfy 1/lr > the maximum eigenvalue of matmul(D^T, D)."
                                f"The maximum eigenvalue is {max_eig}, set the learning rate appropriately")
    #----------------------------------------------------------------------------

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

def FISTA(x : torch.Tensor,
         h_dim : int,
         D : torch.Tensor,
         regularization=0.5,
         lr=None,
         frequency=None) -> torch.Tensor:
    """
    Implements the Fast ISTA algorithm to find the optimal
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
    v = torch.zeros(h_dim)
    h_old = None
    v_old = None
    generator = t_gen(0)
    t_old = next(generator)
    t = 0

    #--------------------------- SETTING CORRECT LEARNING RATE -----------------
    """
    A very crucial piece of information is that the ISTA algorithm converges
    if and only if 1/lr > the largest eigenvalue of matmul(D^T, D) where D is
    the dictionary matrix provided.
    """
    max_eig = torch.linalg.eigvals(torch.matmul(torch.transpose(D, 0, 1), D)).abs().max()
    if lr is None:
        lr = 1/(max_eig + 1)
    else:
        assert 1/lr > max_eig, (f"For convergence, the learning rate must satisfy 1/lr > the maximum eigenvalue of matmul(D^T, D)."
                                f"The maximum eigenvalue is {max_eig}, set the learning rate appropriately or"
                                f"do not pass the learning rate which will set it to an appropriate value automatically.")
    #----------------------------------------------------------------------------

    counter = itertools.count(start=0, step=1)
    while True:
        h_old = h.clone()
        v_old = v.clone()

        u = torch.matmul(D, h_old) - x
        v = h_old - lr * torch.matmul(torch.transpose(D, 0, 1) , u)
        v = shrink(h, lr * regularization)

        t = next(generator)
        h = v + ((t_old - 1)/t) * (v - v_old)
        t_old = t

        if change(h, h_old) < 0.001:
            break
        iter_num = next(counter)
        if frequency is not None and iter_num % frequency == 0:
            print(f"FISTA: Iteration {iter_num}")
    return h

def CoD(x : torch.Tensor,
         h_dim : int,
         D : torch.Tensor,
         regularization=0.5,
         frequency=None) -> torch.Tensor:
    """
    Implements the Coordinate Descent algorithm to find the optimal
    sparse codes for the given input vector x and a given
    dictionary matrix D. This function follows notation corresponding to
    the paper "Learning Fast Approximations of Sparse Coding"

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
    print("Hi")
    B = torch.matmul(D.T, x)  
    Z = torch.zeros_like(B) 

    counter = itertools.count(start=0, step=1)
    while True:
        print("Entered loop")
        
        Z1 = shrink(B, regularization)
        Zd = Z1 - Z
        
        k = torch.argmax(torch.abs(Zd)).item()
        S = torch.eye(h_dim) - torch.matmul(D.T, D)
        
        B += S[:, k] * (Z1[k] - Z[k]).item()
        
        if torch.abs(Z[k] - Z1[k]).item() <= 0.00001:
            break
        
        Z[k] = Z1[k]
        
        iter_num = next(counter)
        if frequency is not None and iter_num % frequency == 0:
            print(f"Coordinate Descent: Iteration {iter_num}")
            
    Z = shrink(B, regularization)

    return Z
