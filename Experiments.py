import numpy as np
import torch
import torchvision
from numpy.ma.core import shape
from sympy.physics.units import frequency
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms

import time
from utils import show_img, extract_patches, display_patches
import ISTA
from sparseCoding import learn_representations

MEAN = 0.1307
STD = 0.3081
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
train = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train, batch_size=32, shuffle=False)

####################################### SANITY CHECK! ###############################################

dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
# show_img(img_grid, MEAN, STD)
patches = extract_patches(train)
# display_patches(patches)

################################################################################################

start = time.time()
dataset = TensorDataset(patches)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
H, D = learn_representations(train_loader, 784)
print(H.shape, D.shape)
print(f"Time taken = {time.time() - start}")