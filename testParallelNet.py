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
import dataloader as dl

def main():

    #complex net parameters
    M = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     
    torch.manual_seed(11)    
    np.random.seed(11)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    gpu = False
    conv = False
    dataset_name = "MNIST"
    choice = 'r'
    
    multilevel = True
    
    #neural net parameters---------------------------------------------------------    
    weights = None
    bias = None  
    reg_f = False
    reg_c = False    
    graph = True
    
    #-----------hyper parameters
    batch_size = 256
    #-note coarse model will be 2* this, fine model with be 4* this    
    N = 1
    learn_rate_c = .5
    f_step_c = .4
    learn_rate_f = .25
    f_step_f = .15
    epochs = 2#10#0
      
    # 0.00005
    alpha_f = 0.001
    alpha_c = 0.00025    
    gamma = 0.05
   
    begin = 0
    end = 10#000  
           
    # choose from MNIST, CIFAR10, CIFAR100, ELLIPSE, SWISS
    
    dataloader = dl.InMemDataLoader(dataset_name)
        
    num_features, num_classes, in_channels = dl.getDims(dataset_name)
                                  
    loader = dataloader.getDataLoader(batch_size, shuffle = True, num_workers = 0, pin_memory = True, train = True)     
    
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
    complexNet = pa.complexNeuralNetwork(device, M, gpu, conv, in_channels)
    #init sub-neural networks
    complexNet.init_nets(N, num_features, num_classes, func_f, func_c, weights, bias
                         ,choice, gamma, multilevel)
        
    #init SG modules
    complexNet.init_sgs(sg_func, sg_loss, num_features=num_features, batch_size=batch_size)    
        
    #train coarse model
    torch.cuda.synchronize()
    coarse_time = time.time()
    complexNet.train_multi_level(loader, error_func, learn_rate_c, epochs, begin, end
                     ,f_step_c, reg_f, alpha_f, reg_c, alpha_c, graph = False)
    torch.cuda.synchronize()
    coarse_time = time.time() - coarse_time
    
   
    coarse_result = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step_c)
    
    complexNet.double_complex_net()
    #print("double it")      
        
    #train fine model            
    
    #train_network
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    train_time = complexNet.train(loader, error_func, learn_rate_f, epochs, begin, end
                     ,f_step_f, reg_f, alpha_f, reg_c, alpha_c, graph)
    torch.cuda.synchronize()
    end_time = time.perf_counter() - start_time       
      
    print ("coarse train results", coarse_result, "\n")
    print ("coarse time", coarse_time, "\n")    
    
    print("\ntotal time in series:" , end_time)
    #During training, each epoch we see the loss and mse for synthetic gradient
    
    result_train = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step_f)
    
    loader = dataloader.getDataLoader(batch_size, shuffle = False, num_workers = 0, pin_memory = False, train = False)
    
    result_test = complexNet.test(loader, begin = 0, end = 10000, f_step = f_step_f)
    
    print ("fine train result", result_train, "\n")
    print("fine test result", result_test, "\n")
        
    print("Total time:", train_time[0], "\ntheoretical time:", train_time[1], "\nspeed up:", train_time[2])  
    print("Batch time adjusted speed up", train_time[3])
    
    print ("\n--------------------- Total time: ", coarse_time + train_time[1],"--------------------------")  
    
if __name__ == '__main__':
    main()
