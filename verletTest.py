from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import itertools
import time
import h5py

import Verlet as vl


def main():
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    print(device)
    torch.set_printoptions(precision=10)

    dtype = torch.float
    download = True#False
    input = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype)
    target = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype) + 1


    num_classes = 10
    num_features = 784
    weights = torch.tensor([[2, 0],[-2, 2]], dtype = dtype)
    bias = torch.tensor([0,0], dtype = dtype)

    begin = 0
    end = 50000
    reg_f = False#True# False
    reg_c = False#True#False



    gpu = True
    graph = False
#    -----------hyper parameters
    learn_rate = 0.05#5
    step = 0.05
    alpha_f = 0.0001
    alpha_c = 0.0001

    epochs = 5#00#0#200#000

    N = 4#50
    batch_size = 256#000#60000
    end = end/batch_size
    end = round(end)
    error_func=nn.CrossEntropyLoss()
   # print(end)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    #trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            #download=download, transform=transform)
    
    myFile = h5py.File('./data/MNIST.h5', 'r')#, driver='stdio')
    coords = myFile.get('MNIST_train_inputs')
    label = myFile.get('MNIST_train_labels')
    
    coords = torch.from_numpy(np.array(coords))
    label = torch.from_numpy(np.array(label))
    label = label.long()
    coords = coords.float()
    #print("coords:", coords)
    #print("labels:", label)
    myFile.close()
    trainset = torch.utils.data.TensorDataset(coords, label)    
    
    end = 100000
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                              shuffle=False, num_workers=0, pin_memory = False)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                             shuffle=False, num_workers=2, pin_memory = True)


    myV = vl.Verlet(device,N, num_features, num_classes, torch.tanh, F.softmax , gpu = gpu)
    
    if gpu == True:
        myV.to(device)
    t = time.perf_counter()
    myV.train(trainloader, error_func, learn_rate, epochs, begin, end, step, reg_f, alpha_f, reg_c, alpha_c, graph)
    print("training time", time.perf_counter() - t)# - start_time))

    train_error = myV.test(trainloader, begin, end, step)
    valid_error = myV.test(trainloader, end, 60000, step)
    test_error = myV.test(testloader, begin, end, step)

    print("training error", train_error, "validation error: ", valid_error, "test error", test_error)
    



if __name__ == '__main__':
    main()
