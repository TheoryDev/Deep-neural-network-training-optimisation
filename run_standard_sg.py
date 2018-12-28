# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:38:25 2018

@author: Corey
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
import h5py


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
    
    np.random.seed(11)
    torch.manual_seed(11)
    gpu = True
    batch_size = 1
    E = False
    S = False
    MNIST = True
    
    if E:
        num_classes = 2
        num_features = 2
        myE = el.ellipse(device, 500, 100, a, b)
        dataset = myE.create_dataset(myE.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)
        datasetT = myE.create_dataset(myE.valid)
        testloader = torch.utils.data.DataLoader(datasetT, batch_size = 10)
        gpu = False
    
    if S:
        num_classes = 2
        num_features = 4
        myS = sw.SwissRoll(device, 500, 0.2)
        dataset = myS.create_dataset(myS.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)
        datasetT = myS.create_dataset(myS.valid)
        testloader = torch.utils.data.DataLoader(datasetT, batch_size = 10)  
        gpu = False
    
    if MNIST:
        num_features = 784
        num_classes = 10
        batch_size = 256
        begin = 0
        end = 100
        gpu = True
        
        transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])
                        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
        #trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                #download=True, transform=transform)
                                                
        myFile = h5py.File('./data/MNIST.h5', 'r')#, driver='core')
        coords = myFile.get('MNIST_train_inputs')
        label = myFile.get('MNIST_train_labels')
        coords = torch.from_numpy(np.array(coords))
        label = torch.from_numpy(np.array(label))
        label = label.long()
        coords = coords.float()
        #print("coords:", coords)
        #print("labels:", label)
    
        trainset = torch.utils.data.TensorDataset(coords, label)                                            
                                                
                                                
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                  shuffle=True, num_workers=0, pin_memory = False)
    
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                                 shuffle=False, num_workers=2, pin_memory = True)
    
     
    
    multilevel = False
    
    #neural net parameters---------------------------------------------------------
    
    weights = None
    bias = None  
    reg_f = False
    reg_c = False
    
    graph = True
    #-----------hyper parameters
    learn_rate = 0.2
    epochs = 5#round(150/2)
   
    f_step = .1
    N = 64#50 #-note  model will be 2* this
    
    # 0.00005
    alpha_f = 0.001
    alpha_c = 0.00025
    
    gamma = 0.02
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
  
    #train_network
    start_time = time.perf_counter()
    train_time = complexNet.train(trainloader, error_func, learn_rate, epochs, begin, end
                     ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph=False)
    end_time = time.perf_counter() - start_time    
     
    print("total time in series:" , end_time)
    #During training, each epoch we see the loss and mse for synthetic gradient
    
    result_train = complexNet.test(trainloader, begin = 0, end = 10000, f_step = f_step)
    result_test = complexNet.test(testloader, begin = 0, end = 10000, f_step = f_step)
    
    print("fine train result", result_train, "\n")
    print("fine test result", result_test, "\n")
    
    #theoretical time is the training time using the lowest on each batch of training
    print("Total time:", train_time[0], "\ntheoretical time:", train_time[1])#, "\nspeed up:", train_time[2])  
    #print("Batch time adjusted speed up", train_time[3])
    
if __name__ == '__main__':
    main()
