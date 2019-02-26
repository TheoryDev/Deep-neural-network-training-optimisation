#from __future__ import print_function
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
import multiprocessing as mp
import os


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
import dataloader as dl
import sys, getopt

def main(argv):
    
    #complex net parameters
    M = 2        
    #---------------training data--------------------
              
    dataset_name = "MNIST" # choose from MNIST, CIFAR10, CIFAR100, ELLIPSE, SWISS
    choice = 'r'
    conv= False
    gpu = False
    
     #neural net parameters---------------------------------------------------------
    
    weights = None
    bias = None  
    reg_f = False
    reg_c = False   
    alpha_f = 0.001
    alpha_c = 0.00025
    graph = True
    
    #-----------hyper parameters
    batch_size = 1024
    N = 32#128#56#-note  model will be 2* this 
    learn_rate = 0.25
    f_step = .025
    epochs = 10#10#000  
      
    gamma = 0.02    
    begin = 0
    end = 10000#50#000
    
    #batch_size = 64
    func_f = torch.tanh
    func_c = F.softmax    
    error_func = nn.CrossEntropyLoss()        
            
    multilevel = False       
    #------------------------------------------------------------------------------
    #-------------------sg parameters--------------------------
    sg_func = syn.sgLoss
    sg_loss = nn.MSELoss       
    
    if len(argv) > 0:
        #print(argv)
        N = int(argv[0])
        epochs = int(argv[1])
        learn_rate = float(argv[2])
        step = float(argv[3])        
        choice = argv[4]
        graph = argv[5]
        print("N", N, "epochs", epochs, "lr", learn_rate, "step", step, "choice", choice, "graph", graph)
         
    np.random.seed(11)
    torch.manual_seed(11)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False        
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = dl.InMemDataLoader(dataset_name, conv_sg=conv)
        
    num_features, num_classes, in_channels = dl.getDims(dataset_name)
     
    #load training dataset                         
    loader = dataloader.getDataLoader(batch_size, shuffle = True, num_workers = 0, pin_memory = True, train = True)     
    
    multilevel = False       
    #------------------------------------------------------------------------------
    #-------------------sg parameters--------------------------
    sg_func = syn.sgLoss
    sg_loss = nn.MSELoss
       
    #init complex network
    complexNet = pa.complexNeuralNetwork(device, M, gpu, conv, in_channels)
    
    #init sub-neural networks
    complexNet.init_nets(N, num_features, num_classes, func_f, func_c, weights, bias,
                         choice, gamma, multilevel)
        
    #init SG modules
    complexNet.init_sgs(sg_func, sg_loss, num_features=num_features, batch_size=batch_size)  
      
    #accBefore = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step)
    
    #train model with distributed algorithm
    train_time = complexNet.distTrain(loader, error_func, learn_rate, epochs, begin, end
                    ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph, False, M)   
    
    
    result_train = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step)
    
    #load test dataset
    loader = dataloader.getDataLoader(batch_size, shuffle = False, num_workers = 0, pin_memory = False, train = False)
    
    result_test = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step)
    
    print("fine train result", result_train, "\n")
    print("fine test result", result_test, "\n")
          
    print("train time", train_time)  
   
if __name__ == '__main__':
    main(sys.argv[1:])