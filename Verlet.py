from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import itertools

import ResNet as res


class Verlet(res.ResNet):

    def __init__(self, device, N, num_features, num_classes, func_f, func_c, weights = None, bias = None, 
                     gpu=False, last=True, conv=False, first = True, in_chns=1, n_filters = 6):
        """
         It is initialised with
            N - number of layers
            num_features - number of inputs
            num_classes - number of outpurs
            func_f - activation function
            func_c - classifier's activation function i.e softmax
            weights - weight matrix if none then a random initlisation will be used
            bias = - bias vector if none then a random initilisation with be used
            gpu - if True, gpu will be used if available
            last - set to true if network is the last in the series of networks and sg modules
            first - set to true if network is the first in the series of networks and sg modules
            in_chns - number of channels in input data for images 
            n_filters - number of filters in convolutional layers  
        """
        #use the ResNet superclass initialisation
        super(Verlet, self).__init__(device, N, num_features, num_classes, func_f, func_c, weights = weights,
             bias = bias, gpu = gpu, last=last, conv=conv, first=first, in_chns=in_chns, n_filters = n_filters)
        

    def forward(self, x, step=0.1, plot=False):
        i = 0
        #get first input form with n_channels
        if self.conv == True and self.first == True:
            x = self.func_f(self.firstMask(x))             
   
        #init z_minus 
        z_minus = torch.zeros(x.shape)
        if self.gpu == True:
            z_minus = z_minus.to(self.device)

        for layer in self.layers:
            direction = None
            if layer.weight.requires_grad == False and self.directions is not None:
                    #PVD
                    direction = self.directions[i]
                    i += 1
            #first equation
            z_plus = z_minus - step*self.func_f(self.z_sum(x, layer, direction))
            
            #second equation
            if direction is None:
                x = x + step*self.func_f(layer(z_plus))
            else:
                #PVD
                w, b = direction
                x = x + step*self.func_f(layer(z_plus) + z_plus.matmul(w) + b)
            z_minus = z_plus
        #save output of final layer    
        self.final = x.detach().to(torch.device("cpu")).numpy()     
        
        if self.last == True:
            #only the last network has a classifier
            if self.conv:
                #must flatten output before passing into classifier
                x = x.view(-1, self.num_features*self.n_filters)
            x = self.func_c(self.classifier(x), dim = 1)
    
        return x

    def z_sum(self, x, layer, direction = None):
        
        if self.conv == True:
            #performs a convolution using the transpose of the filters
            x = torch.nn.functional.conv2d(x, layer.weight.transpose(2,3), layer.bias, padding = 1)
            return x
        
        K = layer.weight.transpose(1,0)
        b = layer.bias

        if direction is not None:
            delW, delb = direction
            K = K + delW.transpose(1,0)
            b = b + delb

        return torch.matmul(x,K) + b
        
