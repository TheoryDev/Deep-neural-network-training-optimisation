from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
import numpy as np
import h5py
import ResNet as rn
import dataloader as dl

import time

def main():

    np.random.seed(11)
    torch.manual_seed(11)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = "ELLIPSE"  
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    
    print(device)
    #torch.set_printoptions(precision=10)
    
    dtype = torch.float
    download = True#False
    input = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype)
    target = torch.tensor([[.1,.1],[-.1,-.1],[0,0.5]], dtype = dtype) + 1
    
    num_features, num_classes = dl.getDims(dataset)
    
    weights = None# torch.tensor([[2, 0],[-2, 2]], dtype = dtype)
    bias = None#torch.tensor([0,0], dtype = dtype)
    
    begin = 0
    end = 10000
    reg = False
    
    #-----------hyper parameters
    learn_rate = 0.1
    step = 1.0
    alpha = 0.0001
    epochs = 50
    gpu = False
    
    N = 4
    batch_size = 256
    error_func=nn.CrossEntropyLoss()
    
    t_data = time.perf_counter()
    
    #transform = transforms.Compose([transforms.ToTensor()])  
                                
    dataloader = dl.InMemDataLoader(dataset)
                                      
    loader = dataloader.getDataLoader(batch_size, shuffle = False, num_workers = 0, pin_memory = True, train = True)               
                                                                                            
    print("data time", time.perf_counter() - t_data)  
  
    res = rn.ResNet(device, N, num_features, num_classes, torch.tanh, F.softmax, gpu = gpu)
    if gpu == True:
        res.to(device)
    
    train_time = time.perf_counter()
    res.train(loader, error_func, learn_rate, epochs, begin, end, step, reg, alpha, graph = False)
    train_time = time.perf_counter() - train_time
    
    loader = dataloader.getDataLoader(batch_size, shuffle = False, num_workers = 0, pin_memory = True, train = False)  
        
    results = res.test(loader, begin, end, step)
        
    print("results: ", results)
    print("--- %s seconds ---" % (train_time))
    #print("in_net time", net_t)
    #myFile.close()
if __name__ == '__main__':
    main()

