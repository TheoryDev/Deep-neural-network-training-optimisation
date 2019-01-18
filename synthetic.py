# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:19:59 2018

@author: koryz
"""
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
   

 

def sgLoss(args, h, y, grad, loss=mse):
    #sg = h*args[0] + y*args[1] + args[2]
    x = sg(args, h, y)
    return loss(x, grad)
#def loss(h, y, func, args, loss=linear_SG):
def sg(args, h, y):
    print("h" , h)
    print("args[0]", args[0].shape)
    print("y", y)
    print("args[1]", args[1].shape)
    return h*args[0]  + y*args[1] + args[2]    


class LinearRegressionSG(nn.Module):
    """
    Class used to build single layer neural network for SG module's gradient
    prediction. It uses the tanh activation function.
    """
    
    def __init__(self, num_features, conv=False, in_channels=7, n_filters=6):
        super(LinearRegressionSG, self).__init__()
        
        self.conv = conv
        if self.conv == True:
            self.linear = nn.Conv2d(n_filters+1, n_filters, 3, padding=1, groups=1)    
        else:
            self.linear = nn.Linear(num_features*in_channels+1, num_features*in_channels)
                
        
    def forward(self, x):
        x = self.linear(x)
        return torch.tanh(x) 
    
class synthetic_module:
    """
    class to build synthetic gradient modules
    """    
    def __init__(self, device, function=sgLoss, loss=mse, args=np.array([0.5,0.5,0.5]), gpu=True, num_features=2, conv=False, in_channels=16, n_filters=6):
        """
        The initialisation takes a function and its arguments and and
        an optimiser and its arguments.
        """
        #if self.gpu == True:
            #num_features *= 6
            
        self.__num_features = num_features
        self.__function = function
        self.__loss = loss
        self.__args = args  
        self.__lin = linear_model.LinearRegression(normalize=False)
        self.__device = device
        self.__gpu = gpu
        self.__linSG = LinearRegressionSG(num_features, conv, in_channels, n_filters)
        self.__conv = conv
        if self.__gpu == True:
            self.__linSG.to(device)
        
    def init_optimiser(self, learn_rate=0.01):
        self.__opt = optim.SGD(self.__linSG.parameters(), lr = learn_rate)
        
    def zero_optimiser(self):
        self.__opt.zero_grad()
        
    def get_net(self):
        return self.__linSG
     
    def get_coefs(self):
        return self.__lin.coef_
        
    def store_labels(self, labels):
        self.__labels = labels
        
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
        
        #result = minimize(fun=sgLoss, x0=self.__args, args=(h, y, grad, self.__loss,))
        #self.__args = result.x
        #return #result.x       
        
    
    def calculate_sg(self, h, y,func=sg):
        
       if self.__conv == True:
           tmp_y = y.reshape(y.shape[0],1,28,28)
           #print("y",tmp_y[0])
           #print("h", h[0])
           #tmp_y = y self.convLabels(y, h[0][0].shape, 10)#.to(self.__device)
           #if self.__gpu == True:
               #tmp_y = tmp_y.to(self.__device)
       else:
           tmp_y = y.reshape((len(y),1)).float()  
       #reshape labels y
       tmp_h = h.detach()
       #print(tmp_h, tmp_y)
       x =  torch.cat((tmp_h, tmp_y), dim=1)   
       #print("x",x[0][0], x[0][6])
       #get prediction
       pred = self.__linSG(x)        
       #if self.__gpu == True:
           #pred = pred.reshape(h_shape)       
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
        #if training coarse model backpropagate real gradient 
        if multi == True:           
            self.__h_before.backward(self.__h_after.grad)
            return
        
        #ecalculate synthetic gradient        
        self.syn_grad = self.calculate_sg(self.__h_before, y)
        #print("sg", self.syn_grad)
        #backpropagate synthetic gradient into sub-nn
        #print("hs", self.__h_before.shape)
        #print("sg", syn_grad)        
        self.__h_before.backward(self.syn_grad.detach())
        
    def optimise(self, y, multi=False, para=False, gradient=None):
        """
        The function optimises the synthetic gradient. 
        It obtains the gradient dL/dh^n from the h_after instance attribute. 
        It then calls the optimise function.        
        """       
        #self.zero_optimiser() 
        #pred = self.calculate_sg(self.__h_before.detach(), y)          
        self.zero_optimiser()               
        #get real gradient
        if para == False:
            grad = self.__h_after.grad.data
        else:
            grad = gradient
        #optimise gradient
       
        #print("syn_grad", self.syn_grad, "real grad", grad)
        
        error_func = nn.MSELoss()#(pred ,grad)
        loss = error_func(self.syn_grad, grad)
        
        #regularisation - may use in future
        # = 0.0
        #for param in self.__linSG.parameters():
            #print("param", param)
            #t = time.perf_counter()
            #loss += torch.sum(torch.pow(param,2))
            #print(time.perf_counter()-t)
        
        #calculate loss
        loss.backward()        
        #update weights
        self.__opt.step()
       
        return loss.data.float()       

    def first_optimise(self, num_features = 2, num_classes = 2, batch_size=10):
        """
        used for old sg- will remove
        """
        #randomly initialise sg modules
        #print("num_classes", num_classes, "num_feats", num_features, "batch", batch_size)
        
        h = np.random.normal(size=(batch_size, num_features))
        y = np.random.normal(size=(batch_size, 1))
        grad = np.random.normal(size=(batch_size, num_features))
        self.__lin.fit(np.concatenate((h,y), axis=1), grad)
        
       # print("coefficients",self.__lin.coef_.shape)       
    
    def convLabels(self, labels, y_shape = (28, 28), num_classes=10):
        batch_size = labels.shape[0]
        convLabels = torch.zeros(batch_size, 10, *y_shape)#.to(self.__device)
        for i in range(batch_size):
            convLabels[i][labels[i]] += 1
        
        return convLabels#.to(self.__device)
        

    
def main():
    y = np.array([[0.,1.],[0.6,0.8]])
    h = np.array([[1.,1.],[0.1,0.9]]) 
    grad = np.array([[0.8, 0.5]])
    s = synthetic_module()
    
    result = s.optimise_SG(y, grad, h)
    print(result)

    est = s.calculate_sg(h,y)
    print("sg: ", est, "real grad: ", grad)

if __name__ == '__main__':
    main()
