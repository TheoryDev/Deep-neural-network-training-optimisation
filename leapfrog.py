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


class Leapfrog(res.ResNet):


    def __init__(self, device, N, num_features, num_classes, func_f, func_c, weights = None, bias = None, gpu = False):
        """
        The leapfrog method uses a new
        """
        #use the ResNet superclass initialisation
        super(Leapfrog, self).__init__(device, N, num_features, num_classes, func_f, func_c, weights = weights, bias = bias, gpu = gpu)
   

    def forward(self, x, step=0.1, plot=False):
        #forward propagation for leapfrog
        self.props = x
        #for first layer (e.h j = 0)
        last_x = x
        step_2 = step*step
        direction = None
        i = 0
        if self.layers[0].weight.requires_grad == False and self.directions is not None:
            direction = self.directions[0]
            i = 1

        x = 2*x + step_2*self.func_f(self.leap_layer(self.layers[0], x))
  
        #propagate through the layers
        for layer in self.layers[1:]: #in range(1,len(self.layers)):
            direction = None
            #skip the first layer as it has a specific propagation
            #update x and last_x at the same time to ensure last_x holds the (j-1)'th example'


            if layer.weight.requires_grad == False and self.directions is not None:
                direction = self.directions[i]
                i += 1

            x , last_x = 2*x - last_x + step_2*self.func_f(self.leap_layer(layer,x,direction)), x
      
            #store each layer output if needed
            if plot == True:
                self.props = torch.cat((self.props, x), 0)
        self.final = x.detach().to(torch.device("cpu")).numpy()
        #classifier
      
        x = self.func_c(self.classifier(x), dim = 1)    
        return x


    def def_neg(self, mat):
        """
        functon to return a negative definite matrix
        """
        return -torch.matmul(mat.transpose(1,0), mat)


    def leap_layer(self, layer, x, direction = None):
        """
        This function is used
        """
        K = self.def_neg(layer.weight)
        b = layer.bias

        if direction is not None:
            K = K + direction[0]
            b = b + direction[1]

        return torch.matmul(x, K) + b
