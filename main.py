import os
import sys
import cv2 as cv
import processing
import torchvision.transforms
from model import Classifier
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from helper_func import view_classify
from torchvision.io import read_image
from PIL import Image
model = Classifier()
model.load_state_dict(T.load('checkpoint_10.pth'))

impath = sys.argv[1]

cont = processing.get_cont(impath)
cont, box = processing.sort_cont(cont)
processing.write_images(cont)

for img in os.listdir('img'):
    img = Image.open(f'img/{img}')
    transform1 = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5), (0.5))])
    img = transform1(img)
    img = img[None]
    img = img.type('torch.FloatTensor')
    with T.no_grad():
        logits = model.forward(img)
    ps = F.softmax(logits, dim=1)

    view_classify(img, ps, version='MNIST')
