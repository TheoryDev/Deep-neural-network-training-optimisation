# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:40:55 2018

@author: koryz
"""

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

def main():

    #complex net parameters
    M = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #---------------training data--------------------
    #ellipse dimensions
    a = np.array([[0,0.1],[0.1,0.2]])
    #    b = [[0,0.3],[0.6,1.2]]
    a = a*10
    b = a*0.5    
     
    torch.manual_seed(11)    
    np.random.seed(11)
    gpu = False
    batch_size = 1
    
    E = False
    S = True
    MNIST = False
    
    if E:
        num_classes = 2
        num_features = 2
        myE = el.ellipse(device, 500, 100, a, b)
        dataset = myE.create_dataset(myE.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)
        datasetT = myE.create_dataset(myE.valid)
        testloader = torch.utils.data.DataLoader(datasetT, batch_size = 10)
    
    if S:
        num_classes = 2
        num_features = 4
        myS = sw.SwissRoll(device, 500, 0.2)
        dataset = myS.create_dataset(myS.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)
        datasetT = myS.create_dataset(myS.valid)
        testloader = torch.utils.data.DataLoader(datasetT, batch_size = 10)   
    
    if MNIST:
        num_features = 784
        num_classes = 10
        batch_size = 256
        begin = 0
        end = 100
        gpu = True
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                  shuffle=False, num_workers=2, pin_memory = True)
    
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                                 shuffle=False, num_workers=2, pin_memory = True)   
    
        
    multilevel = True
    
    #neural net parameters---------------------------------------------------------
    
    weights = None
    bias = None  
    reg_f = False
    reg_c = False
    
    graph = True
    #-----------hyper parameters
    learn_rate = .2
    epochs = 250#round(150/2)
    f_step = .25
    N = 128 #-note coarse model will be 2* this, fine model with be 4* this
    
    # 0.00005
    alpha_f = 0.001
    alpha_c = 0.00025
    
    gamma = 0.05
    choice = 'r'
    begin = 0
    end = 10000#50#000
    
    #batch_size = 64
    func_f = torch.tanh
    func_c = F.softmax    
    error_func = nn.CrossEntropyLoss()
    #------------------------------------------------------------------------------
    #-------------------sg parameters--------------------------
    sg_func = syn.sgLoss
    sg_loss = nn.MSELoss
    #initial optimisation parameters for sg modules
    #sg_args = [torch.rand(size=(1,num_features)), torch.rand(size=(3,1)), torch.rand((1))]
    
    #init complex network
    complexNet = pa.complexNeuralNetwork(device, M, gpu)
    
    #init sub-neural networks
    complexNet.init_nets(N, num_features, num_classes, func_f, func_c, weights, bias,
                         gpu, choice, gamma, multilevel)
        
    #init SG modules
    complexNet.init_sgs(sg_func, sg_loss, num_features=num_features, batch_size=batch_size)    
        
    #train coarse model
    coarse_time = time.time()
    complexNet.train_multi_level(trainloader, error_func, learn_rate, epochs, begin, end
                     ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph = False)
    coarse_time = time.time() - coarse_time
    
    #print("after coarse train")
    coarse_result = complexNet.test(trainloader, begin = 0, end = 10000, f_step = f_step)
    #print("after coarse test")
    complexNet.double_complex_net()
    #print("double it")
    
    
        
    #train fine model    
    epochs = 250
    learn_rate = .1
    f_step = .1#f_step#*1.33
     #learn_rate#*1.33
    
    #train_network
    start_time = time.perf_counter()
    train_time = complexNet.train(trainloader, error_func, learn_rate, epochs, begin, end
                     ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph)
    end_time = time.perf_counter() - start_time
       
    print ("coarse test results", coarse_result, "\n")
    print ("coarse time", coarse_time)
    
    print("total time in series:" , end_time)
    #During training, each epoch we see the loss and mse for synthetic gradient
    
    result_train = complexNet.test(trainloader, begin = 0, end = 10000, f_step = f_step)
    result_test = complexNet.test(testloader, begin = 0, end = 10000, f_step = f_step)
    
    print ("fine train result", result_train, "\n")
    print("fine test result", result_test, "\n")
        
    print("Total time:", train_time[0], "\ntheoretical time:", train_time[1], "\nspeed up:", train_time[2])  
    print("Batch time adjusted speed up", train_time[3])
    
    print ("\n--------------------- Total time: ", coarse_time + train_time[1],"--------------------------")
    
    
    
    
if __name__ == '__main__':
    main()
