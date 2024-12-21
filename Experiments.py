from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

import time
from utils import show_img, extract_patches, display_patches
import ISTA
from sparseCoding import learn_representations

MEAN = 0.1307
STD = 0.3081
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
mnist = MNIST(root='./data', train=True, download=True, transform=transform)
train = Subset(mnist, indices=range(len(mnist) // 100))
train_loader = DataLoader(train, batch_size=1, shuffle=False)
####################################### SANITY CHECK! ###############################################

dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
show_img(img_grid, MEAN, STD)
patches = extract_patches(train)
display_patches(patches)

############################### COMPARING SPEED OF CONVERGENCE #######################################

times = defaultdict(lambda: 0)

for data in train_loader:
    data = data[0].squeeze().flatten()
    D = torch.randn((784, 784))

    start = time.time()
    h1 = ISTA.ISTA(data, 784, D)
    times['ista'] += time.time() - start

    start = time.time()
    h2 = ISTA.FISTA(data, 784, D)
    times['fista'] += time.time() - start

    start = time.time()
    h3 = ISTA.CoD(data, 784, D)
    times['coordinate_descent'] += time.time() - start

print(f"Average time taken in ISTA = {times['ista']/len(train_loader)}")
print(f"Average time taken in FISTA = {times['fista']/len(train_loader)}")
print(f"Average time taken in Coordinate Descent = {times['coordinate_descent']/len(train_loader)}")
