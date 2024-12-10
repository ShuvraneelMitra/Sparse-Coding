import numpy as np
import matplotlib.pyplot as plt

def show_img(img, mean, std):
    img = img.mean(dim=0)
    img = img * std + mean
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.title("Sanity Check: Are images correctly downloaded?")
    plt.show()