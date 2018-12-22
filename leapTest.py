from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
import time

import leapfrog as lp


def main():
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    torch.set_printoptions(precision=10)

    dtype = torch.float
    download = False
    input = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype)
    target = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype) + 1

    num_classes = 10
    num_features = 784

    begin = 0
    end = 50000
    reg_f = False
    reg_c = False
    
    gpu = True
    graph = True
#    -----------hyper parameters
    learn_rate = 0.005
    step = 0.001
    alpha_f = 0.001
    alpha_c = 0.001
    
    epochs = 150

    N = 4
    batch_size = 150
    end = end/batch_size
    end = round(end)
    error_func=nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                   

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                              shuffle=True, num_workers=8, pin_memory = True)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                             shuffle=False, num_workers=8, pin_memory = True)


    leap = lp.Leapfrog(device,N, num_features, num_classes, torch.tanh, F.softmax, gpu = gpu)
    leap.to(device)
    
    leap.train(trainloader, error_func, learn_rate, epochs, begin, end, step, reg_f, alpha_f, reg_c, alpha_c, graph)
  
    train_error = leap.test(trainloader, begin, end, step)
    valid_error = leap.test(trainloader, end, 60000, step)
    test_error = leap.test(testloader, begin, end, step)

    print("training error", train_error, "validation error: ", valid_error, "test error", test_error)   


  
    print("--- %s seconds ---" % (time.time() - start_time))
    

if __name__ == '__main__':
    main()
