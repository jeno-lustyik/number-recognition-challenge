import sys

import processing
from model import Classifier
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from helper_func import view_classify

model = Classifier()
model.load_state_dict(T.load('checkpoint_10.pth'))

impath = sys.argv[1]

cont = processing.get_cont(impath)
cont, box = processing.sort_cont(cont)
processing.write_images(cont)
