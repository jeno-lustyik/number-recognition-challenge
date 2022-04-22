### Import required libraries 

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

### Load the datasets from torchvision

transform = transforms.compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])

trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = T.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset    = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = T.utils.data.DataLoader(testset, batch_size=64, shuffle=True)



