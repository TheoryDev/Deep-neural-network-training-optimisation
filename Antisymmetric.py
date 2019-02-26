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


class AntiSymResNet(res.ResNet):


    def __init__(self, device, N, num_features, num_classes, func_f, func_c, weights = None, bias = None, gamma = 0.01, gpu = False):
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
            gamma - diffusion coefficient
        """
        #use the ResNet superclass initialisation
        super(AntiSymResNet, self).__init__(device, N, num_features, num_classes, func_f, func_c, weights = weights, bias = bias, gpu = gpu)

        self.gamma = gamma


    def forward(self, x, step=0.1, gamma = None, plot=False ):   
        if gamma == None:
            gamma = self.gamma

        i = 0      
        #forward propagation
        self.props = x
        #propagate features through layers
        for layer in self.layers:
            direction = None
            if layer.weight.requires_grad == False and self.directions is not None:
                direction = self.directions[i]
                i += 1

            y = self.antisym(x, layer, gamma, direction)
            x = x + step*self.func_f(y)
            #store each layer output if needed
            if plot == True:
                self.props = torch.cat((self.props, x), 0)
        self.final = x.detach().to(torch.device("cpu")).numpy()
        #classifier
   
        x = self.func_c(self.classifier(x), dim = 1)      

        return x

    def antisym(self, x, layer, gamma = 0.1, direction = None):
        #get imaginary matrix with diffusion
        #note that the torch class does matrices the opposite way round to the Stable Architectures paper so I have done AT -A instead
        mat = layer.weight.transpose(1,0) - layer.weight
        b = layer.bias

        if direction is not None:
            delW, delB = direction
            mat = mat + delW.transpose(1,0) - delW
            b = b + delB
        #create diffusion matrix    
        diffusion = gamma*torch.eye(layer.weight.shape[0])
        if self.gpu == True:
            mat -= diffusion.to(self.device)
        else:
            mat -= diffusion

        result = 0.5*torch.matmul(x, mat) + b
        return result
