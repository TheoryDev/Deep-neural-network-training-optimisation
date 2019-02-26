from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import itertools


import Antisymmetric as anti

import time
start_time = time.time()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)


dtype = torch.float
download = True
input = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype)
target = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype) + 1


num_classes = 10
num_features = 784
weights = torch.tensor([[2, 0],[-2, 2]], dtype = dtype)
bias = torch.tensor([0,0], dtype = dtype)

begin = 0
end = 1000
reg = True

#-----------hyper parameters
learn_rate = 0.1
step = 0.02
alpha = 0.001
epochs = 10#200#000
gamma = 0.01

gpu = False

N = 16#50
batch_size = 250#000#60000
error_func=nn.CrossEntropyLoss()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
                #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=download, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                          shuffle=True, pin_memory = True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=download, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                         shuffle=False, pin_memory = True, num_workers=0)


anti_res = anti.AntiSymResNet(device, N, num_features, num_classes, torch.tanh, F.softmax, gamma = gamma, gpu = gpu)
#if gpu flag set to true then send model parameters to gpu
if gpu == True:
    anti_res.to(device)

#train model
anti_res.train(trainloader, error_func, learn_rate, epochs, begin, end, step, reg, alpha, graph = True)


results = anti_res.test(testloader, begin, end, step)


print("results: ", results)
print("--- %s seconds ---" % (time.time() - start_time))
