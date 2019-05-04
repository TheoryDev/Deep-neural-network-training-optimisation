from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
import itertools
from sklearn import linear_model
import copy
import time

#own codebase
import ResNet as res
   
class LinearRegressionSG(nn.Module):
    """
    Class used to build single layer neural network for SG module's gradient
    prediction. It uses the tanh activation function.
    """
    
    def __init__(self, num_features, conv=False, in_channels=1, n_filters=6):
        super(LinearRegressionSG, self).__init__()
        #replace 10 with num classes
        self.conv = conv
        if self.conv == True:
            self.layer = nn.Conv2d(n_filters+10, n_filters, 3, padding=1, groups=1)    
        else:
            self.layer = nn.Linear(num_features*in_channels+1, num_features*in_channels)
               
        
    def forward(self, x):
        x = self.layer(x)
        return torch.tanh(x) 
    
    def set_params(self, params):
        weight, bias = params
        self.layer.weight.data = weight.data            
        self.layer.bias.data = bias.data       
        
    def get_params(self):
        return self.layer.weight.detach().cpu().clone(), self.layer.bias.detach().cpu().clone()
    
class synthetic_module:
    """
    class to build synthetic gradient modules
    """    
    def __init__(self, device, gpu=True, num_features=2, conv=False, in_channels=16, n_filters=6):
        """
        The initialisation takes a function and its arguments and and
        an optimiser and its arguments.
        num_features -  the number of input features
        function - activation function for neural network that approximates gradients
        loss - loss function        
        gpu - boolean flag for gpu       
        """       
            
        self.__num_features = num_features
        #self.__lin = linear_model.LinearRegression(normalize=False)
        self.__device = device
        self.__gpu = gpu
        #__linSG is the neural network used to predict the synthetic gradients
        self.__linSG = LinearRegressionSG(num_features, conv, in_channels, n_filters)
        self.__conv = conv
        if self.__gpu == True:
            self.__linSG.to(device)
        self.__h_before = None
        self.__h_after = None
            
    def set_device(self, device):
        self.__device = device
        self.__linSG.to(device)
        if self.__h_before is not None:
            self.__h_before = self.__h_before.to(device)
        if self.__h_after is not None:
            self.__h_after = self.__h_after.to(device)
      
    
    def get_params(self):
        return self.__linSG.get_params()
    
    def set_params(self, params):
        self.__linSG.set_params(params)
    
    def init_optimiser(self, learn_rate=0.01):
        self.__opt = optim.Adam(self.__linSG.parameters(), lr = learn_rate)
        
    def zero_optimiser(self):
        self.__opt.zero_grad()
        
    def get_net(self):
        return self.__linSG
     
    def get_coefs(self):
        return self.__lin.coef_
                
    def get_input_before(self):        
        return self.__h_before
    
    def get_input_after(self):
        return self.__h_after
        
    def optimise_SG(self, y, grad, h=None):
        """
        The optimiser function 
        """     
        if h is None:
            h = self.__h_before          
           
    def calculate_sg(self, h, y):
       #get labels
       if self.__conv == True:
           #get one hot labels (extra channels)
           tmp_y = self.condition(h.shape, y.detach())                         
       else:
           tmp_y = y.reshape((len(y),1)).float()  
       
       tmp_h = h.detach()
       #cat inputs and labels
       x =  torch.cat((tmp_h, tmp_y), dim=1)   
       
       #get prediction
       pred = self.__linSG(x)        
         
       return pred    
      
           
    def propagate(self, h, para=False):
        self.__h_before = h
        if para:
            return h
        self.__h_after = h.clone().detach()
        self.__h_after.requires_grad = True        
        return self.__h_after
    
    def backward(self, y, multi=True):
        """
        The SG modules version of PyTorches backward() function.
        It calls backward() on the tensor input from the forward pass and
        this propagates all the gradients backwards into the weights of the 
        net before it. It propagates the synthetic gradient which approximates
        the graidents that h would normally recieved from further on in the network               
        """
        #calculate synthetic gradient        
        self.syn_grad = self.calculate_sg(self.__h_before, y)          
        
        #if training coarse model backpropagate real gradient 
        if multi == True:           
            self.__h_before.backward(self.__h_after.grad)
            return
                     
        #backpropagate synthetic gradient into sub-nn             
        self.__h_before.backward(self.syn_grad.detach())
        
    def optimise(self, y, multi=False, para=False, gradient=None):
        """
        The function optimises the synthetic gradient. 
        It obtains the gradient dL/dh^n from the h_after instance attribute. 
        It then calls the optimise function.        
        """       
                 
        self.zero_optimiser()               
        #get real gradient
        if para == False:
            grad = self.__h_after.grad.data
        else:
            grad = gradient
        #optimise gradient      
               
        error_func = nn.MSELoss()
        loss = error_func(self.syn_grad, grad)
        
        #no longer need self.syn_grad
        self.syn_grad = None
               
        #calculate loss
        loss.backward()        
        #update weights
        self.__opt.step()
       
        return loss.data.float()       
          
    def condition(self, inputs_shape, labels):
        """
        Takes input with the shape (N, C, H, W) and returns one hot labels mask channels
        with the shape (N, N_classes, H, W) 
        """
        num_classes = 10
        #replace 10 with num classes
        one_hot_channels = torch.zeros((inputs_shape[0],) + (num_classes,) + inputs_shape[-2:])
        one_hot = torch.zeros((inputs_shape[0], num_classes, 1,))
        
        if self.__gpu == True:
            one_hot_channels = one_hot_channels.to(self.__device)
            one_hot = one_hot.to(self.__device)
        
        #rank = len(labels.shape)
        #for each example set ith channel value to 1
        one_hot.scatter_(dim=1, index=labels.unsqueeze(1).unsqueeze(1), value= 1)
        #update values of channels to one hot labels
        one_hot_channels += one_hot.unsqueeze(2)
        #concatenate onto inputs        
        return one_hot_channels
    