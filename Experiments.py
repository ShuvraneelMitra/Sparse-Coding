import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import time
from utils import show_img
import ISTA
from sparseCoding import learn_representations

MEAN = 0.1307
STD = 0.3081
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
train = MNIST(root='./data', train=True, download=True, transform=transform)
test = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train, batch_size=32, shuffle=False)

####################################### SANITY CHECK! ###############################################

dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
# show_img(img_grid, MEAN, STD)

################################################################################################

start = time.time()
H, D = learn_representations(train_loader, 784, frequency=50)
print(f"Time taken = {time.time() - start}")