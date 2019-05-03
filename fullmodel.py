from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
import itertools

#dnn codebase
import ResNet as res
import Verlet as ver
import Antisymmetric as anti
import leapfrog as lp
import ellipse as el
import swiss as sw
#synthetic gradient module
import synthetic as syn
import parallelNetworks as pa
import time
import h5py
import dataloader as dl
import ResNet as res
import Verlet as ver
import dataloader
import sys, getopt


def main(argv):
    #torch.set_num_threads(2)
    #preliminaires
    np.random.seed(11)
    torch.manual_seed(11)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    dataset_name = "MNIST" # choose from MNIST, CIFAR10, CIFAR100, ELLIPSE, SWISS
    #choose model
    choice = "r" # "v"
    gpu = True
    conv = True
    
    batch_size = 64 #1024
    #hyper parameters
    N = 4
    learn_rate = 0.5#0.05
    step = .15
    epochs = 10#20#0#50
    begin = 0
    end = 10000
    reg_f = False
    reg_c = False
    graph = True
    #0.059..
    
    alpha_f = 0.01
    alpha_c = 0.01
    
    error_func=nn.CrossEntropyLoss()    
    func_f = torch.tanh


    func_c = F.softmax
    #load trainset
    
    if len(argv) > 0:
        #print(argv)
        N = int(argv[0])
        epochs = int(argv[1])
        learn_rate = float(argv[2])
        step = float(argv[3])        
        choice = argv[4]
        graph = argv[5]
        print("N", N, "epochs", epochs, "lr", learn_rate, "step", step, "choice", choice, "graph", graph)  
     
    model = chooseModel(dataset_name, device, N, func_f, func_c, gpu, choice, conv=conv, first=True)    
    
    dataloader = dl.InMemDataLoader(dataset_name)
                                      
    loader = dataloader.getDataLoader(batch_size, shuffle = True, num_workers = 0, pin_memory = True, train = True)         
    #train
    if gpu == True:
        model.to(device)
        torch.cuda.synchronize()
    train_time = time.perf_counter()
    model.train(loader, error_func, learn_rate, epochs, begin, end, step, reg_f, alpha_f, reg_c, alpha_c, graph)
    torch.cuda.synchronize()
    train_time = time.perf_counter() - train_time
    
    result_train = model.test(loader, begin = 0, end = 10000, f_step = step)
    
    #load testset
    loader = dataloader.getDataLoader(batch_size, shuffle = False, num_workers = 0, pin_memory = False, train = False)
    #test
    result_test = model.test(loader, begin = 0, end = 10000, f_step = step)

    print("\nfine train result", result_train)
    print("fine test result", result_test, "\n")

    print("--- %s seconds ---" % (train_time))

def chooseModel(dataset, device, N, func_f, func_c, gpu, choice, last=True,  
                conv=False, first = True, in_chns=1, n_filters = 6):
    
    last = True
    
    num_features, num_classes, in_channels = dataloader.getDims(dataset)
    
    weights, bias = None, None
    
    if choice == 'v':
                print("v")
                model = ver.Verlet(device, N, num_features, num_classes, func_f, func_c, weights, bias, gpu, last
                                   , conv, first, in_chns, n_filters)    
    else:
                print("r")
                model = res.ResNet(device, N, num_features, num_classes, func_f, func_c, weights, bias, gpu, last
                                   , conv, first, in_chns, n_filters)    
    return model
    

if __name__ == '__main__':
    main(sys.argv[1:])


