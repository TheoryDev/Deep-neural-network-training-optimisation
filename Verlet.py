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
        The Verlet method uses a new
        """
        #use the ResNet superclass initialisation
        super(Verlet, self).__init__(device, N, num_features, num_classes, func_f, func_c, weights = weights,
             bias = bias, gpu = gpu, last=last, conv=conv, first=first, in_chns=in_chns, n_filters = n_filters)
        #elf.first_layer = 

    def forward(self, x, step=0.1, plot=False):
        i = 0
        #get first input form with n_channels
        if self.conv == True and self.first == True:
            x = self.func_f(self.firstMask(x))             
   
        z_minus = torch.zeros(x.shape)
        if self.gpu == True:
            z_minus = z_minus.to(self.device)

        for layer in self.layers:
            direction = None
            if layer.weight.requires_grad == False and self.directions is not None:
                    direction = self.directions[i]
                    i += 1

            z_plus = z_minus - step*self.func_f(self.z_sum(x, layer, direction))

            if direction is None:
                x = x + step*self.func_f(layer(z_plus))
            else:
                w, b = direction
                x = x + step*self.func_f(layer(z_plus) + z_plus.matmul(w) + b)
            z_minus = z_plus

        self.final = x.detach().to(torch.device("cpu")).numpy()     
        
        if self.last == True:
            if self.conv:
                x = x.view(-1, self.num_features*self.n_filters)
            x = self.func_c(self.classifier(x), dim = 1)
    
        return x

    def z_sum(self, x, layer, direction = None):
        
        if self.conv == True:
            print("l", layer.weight.shape)
           # layer.weight = layer.weight.transpose(2,3)
            #transConv = nn.Conv2d(6,6,3, padding=1)#.to(self.device)
            #print("shape", transConv.weight.shape)
            #print("w", transConv.weight)
            #print(layer.weight.transpose(1,0))
            #transConv.weight = nn.Parameter(layer.weight.transpose(1,0))
            #transConv.bias = layer.bias
            #return transConv(x)
            #x = layer(x)    
            x = torch.nn.functional.conv2d(x, layer.weight.transpose(2,3), layer.bias, padding = 1)
            return x
        
        K = layer.weight.transpose(1,0)
        b = layer.bias

        if direction is not None:
            delW, delb = direction
            K = K + delW.transpose(1,0)
            b = b + delb

        return torch.matmul(x,K) + b
        
