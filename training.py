### Import libraries
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

from model import Classifier
from data_handler import trainloader, testloader

model = Classifier()

## Loss Function
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

## Training loop

epochs = 2
print_every = 40
final_trl = []
final_tel =[]
final_accuracy = []


for e in range(epochs):
    running_loss = 0
    total=0
    correct=0
    train_losses = []
    test_losses = []
    epoch_accurracy =[]
    print(f"Epoch: {e+1}/{epochs}")

    for i, (images, labels) in enumerate(iter(trainloader)):


        
        optimizer.zero_grad()
        
        logits = model.forward(images)   # 1) Forward pass
        pred = F.log_softmax(logits, dim=1)
        loss = criterion(pred, labels) # 2) Compute loss
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
        running_loss += loss.item()

        train_losses.append(loss.item())
        
        
        if i % print_every == 0:
            print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
            running_loss = 0

    
    model.eval()
    with T.no_grad():
        for i, (images, labels) in enumerate(iter(testloader)):

            logits = model.forward(images)
            test_pred = F.log_softmax(logits, dim=1)

            test_loss = criterion(test_pred, labels)

            test_losses.append(test_loss.item())

            _,predicted = test_pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100*correct/total


            epoch_accurracy.append(accuracy)



    model.train()
    
    mean_trl = mean(train_losses)
    mean_tel = mean(test_losses)
    mean_acc = mean(epoch_accurracy)
    final_trl.append(mean_trl)
    final_tel.append(mean_tel)
    final_accuracy.append(mean_acc)
    
    if (e+1) % 1 == 0:
        T.save(model.state_dict(), f'checkpoint_{e+1}.pth')

    #accuracies.append(acc)
    print(f'Epoch: {e + 1} | loss: {mean_trl} | test loss: {mean_tel} | accuracy:{mean_acc} ')