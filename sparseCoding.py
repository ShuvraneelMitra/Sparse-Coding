import numpy as np
import torch
from fontTools.misc.plistlib import start_dict
from torch.utils.data import DataLoader, Dataset
import itertools
import ISTA

class Together(Dataset):
    def __init__(self, X, H):
        self.X = X
        self.H = H

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.H[idx]

def change(a, b):
    return torch.sum(torch.abs(a - b))

def BlockCoD(dataloader: DataLoader,
             D: torch.Tensor,
             frequency=None) -> torch.Tensor:
    """
    Computes the dictionary matrix given a set of vectors and
    the corresponding optimal sparse codes using Block Coordinate
    Descent
    :param dataloader: A dataloader of a torch Dataset containing X and H matrices together
    :param D: the dictionary matrix with some initialization
    :param frequency: the number of iterations after which an update is printed
    :return: the learned dictionary matrix D
    """
    iterator = iter(dataloader)
    x, h = next(iterator)
    n = x.shape[1]
    m = h.shape[1]
    A, B = torch.zeros((m, m)), torch.zeros((n, m))

    for X, H in dataloader:
        A += torch.einsum("ti, tj -> ij", H, H)
        B += torch.einsum("ti, tj -> ij", X, H)

    D_old = None
    counter = itertools.count(start=0, step=1)
    while True:
        D_old = D
        for j in range(D_old.shape[1]):
            D[:][j] = (B[:][j] - torch.matmul(D_old, A[:][j]) + A[j][j] * D_old[:][j]) / A[j][j]
            D[:][j] = D[:][j] / torch.linalg.vector_norm(D[:][j])
        if change(D, D_old) < 0.001:
            break
        iter_num = next(counter)
        if frequency is not None and iter_num % frequency == 0:
            print(f"BlockCOD: Iteration {iter_num}")
    return D

def learn_representations(dataloader: DataLoader,
                          hidden_dim: int,
                          sparse_code_inference=ISTA.ISTA,
                          frequency=None):
    """
    Implements the Sparse Coding representation learning
    algorithm using the specified sparse code inference algorithm
    along with Block Projected Coordinate Descent to update the
    Dictionary matrix.
    :param dataloader: Dataloader instance containing the examples
    :param hidden_dim: should be more than dim(x) for the overcomplete case
    :param sparse_code_inference: the inference method to find the optimal sparse codes; Default = ISTA
    :param frequency: the number of iterations after which an update is printed
    :return: the set of learned representations, H, along with the dictionary matrix, D.
    """
    iterator = iter(dataloader)
    sample = next(iterator)
    m = sample[0].squeeze().flatten(start_dim=1).shape[1]
    n = hidden_dim

    D = torch.randn((n, m))
    H = []
    counter = itertools.count(start=0, step=1)

    while True:
        D_old = D.clone()
        H = []

        print("Entering ISTA")
        for batch in dataloader:
            X_batch = batch[0].squeeze().flatten(start_dim=1)
            H_batch = [sparse_code_inference(x, hidden_dim, D, frequency=frequency) for x in X_batch]
            H.extend(H_batch)
        print("Leaving ISTA")
        ds = Together(torch.vstack([batch[0] for batch in dataloader]).squeeze().flatten(start_dim=1), H)
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        D = BlockCoD(dl, D_old, frequency=frequency)

        if change(D, D_old) < 0.001:
            break
        iter_num = next(counter)
        if frequency is not None and iter_num % frequency == 0:
            print(f"Learning: Iteration {iter_num}")

    return H, D

