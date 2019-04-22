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
    M =2   
    
    #---------------training data--------------------          
    dataset_name = "MNIST" # choose from MNIST, CIFAR10, CIFAR100, ELLIPSE, SWISS
    choice = 'v'
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
    N = 16 #-note coarse model will be 2* this, fine model with be 4* this  
    learn_rate_c = .25
    f_step_c = .1
    learn_rate_f = .25
    f_step_f = .025 #coarse Verlet 64 could use .075
    epochs = 5#0#000  
      
    gamma = 0.02    
    begin = 0
    end = 10000#50#000
    
    #batch_size = 64
    func_f = torch.tanh
    func_c = F.softmax    
    error_func = nn.CrossEntropyLoss()         
    
    multilevel = False       
    #------------------------------------------------------------------------------
       
    if len(argv) > 0:
        
        N = int(argv[0])
        epochs = int(argv[1])
        learn_rate_c = float(argv[2])
        f_step_c = float(argv[3])        
        learn_rate_f = float(argv[4])
        f_step_f = float(argv[5])
        choice = argv[6]
        graph = argv[7]
        print("N", N, "epochs", epochs, "lr_c", learn_rate_c, "step_c", f_step_c,
              "lr_f", learn_rate_f,  "step_f", f_step_f, "choice", choice, "graph", graph)
    
  
    np.random.seed(11)
    torch.manual_seed(11)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = dl.InMemDataLoader(dataset_name, conv_sg=conv)
        
    num_features, num_classes, in_channels = dl.getDims(dataset_name)
    
    #load training data                              
    loader = dataloader.getDataLoader(batch_size, shuffle = True, num_workers = 0, pin_memory = True, train = True)     
    
    multilevel = True     
    #------------------------------------------------------------------------------
    
    #init complex network
    complexNet = pa.complexNeuralNetwork(device, M, gpu, conv, in_channels)
    
    #init sub-neural networks
    complexNet.init_nets(N, num_features, num_classes, func_f, func_c, weights, bias,
                         choice, gamma, multilevel)
        
    #init SG modules
    complexNet.init_sgs(num_features=num_features, batch_size=batch_size)  
      
    #train coarse model 
    torch.cuda.synchronize()
    coarse_time = time.time()
    complexNet.train_multi_level(loader, error_func, learn_rate_c, epochs, begin, end
                     ,f_step_c, reg_f, alpha_f, reg_c, alpha_c, graph = False)
    torch.cuda.synchronize()
    coarse_time = time.time() - coarse_time
    
    print("after coarse train")
    coarse_result = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step_c)
    #print("after coarse test")
    complexNet.double_complex_net()    
    
    #train model with distributed algorithm
    train_time = complexNet.distTrain(loader, error_func, learn_rate_f, epochs, begin, end
                    ,f_step_f, reg_f, alpha_f, reg_c, alpha_c, graph, False, M) 
        
    result_train = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step_f)
    
    #load test dataset
    loader = dataloader.getDataLoader(batch_size, shuffle = False, num_workers = 0, pin_memory = False, train = False)
    
    
    result_test = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step_f)
    
    print ("coarse train results", coarse_result, "\n")
    print ("coarse time", coarse_time, "\n")    
    
    print("fine train result", result_train, "\n")
    print("fine test result", result_test, "\n")
    
           
    print("total time", train_time + coarse_time)        
   



if __name__ == '__main__':
    main(sys.argv[1:])
    
    
 
    
    
  