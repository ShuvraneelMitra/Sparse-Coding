import torch
import itertools

def shrink(a, b):
    return torch.sign(a) * torch.maximum(torch.abs(a) - b, torch.tensor(0.0))

def change(a, b):
    return torch.sum(torch.abs(a - b))

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
    n = x.shape[0]
    m = h_dim
    assert D.shape == (n, m), (f"D should be matrix of shape (x_dim, h_dim), where "
                               f"x_dim={n} and h_dim={m}, but dimensions of D are {D.shape}")
    z = torch.zeros(h_dim)
    z_old = z.clone()

    S = torch.eye(h_dim) - torch.matmul(torch.transpose(D, 0, 1), D)
    B = torch.matmul(D.T, x)

    counter = itertools.count(start=0, step=1)
    while True:
        z = shrink(B, regularization)
        k = torch.argmax(torch.abs(z - z_old))
        for j in range(m):
            B[j] = B[j] + S[j][k] * (z[k] - z_old[k])
        if change(z, z_old) < 0.01:
            break
        z_old = z.clone()
        iter_num = next(counter)
        if frequency is not None and iter_num % frequency == 0:
            print(f"Coordinate Descent: Iteration {iter_num}")
    return shrink(B, regularization)

print(CoD(torch.randn(10), 10, torch.randn(10, 10), frequency=100))