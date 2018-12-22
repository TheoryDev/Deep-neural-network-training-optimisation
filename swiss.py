from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import itertools
import copy
import time

import ResNet as res
import Antisymmetric as anti
import leapfrog as lp
import Verlet as ver
import ellipse as el

class SwissRoll(el.ellipse):
    """
    class to make swiss rolls- only difference is the data gen function
    """
    def __init__(self, device, N, a, choice = None):
                
       
        torch.manual_seed(11)
        self.choice = choice
        self.device = device

        self.examples, self.valid = self.data_gen(N, 1, a)
                

    def data_gen(self, N, a, b):
        """
        N - number of points
        a - radius of inner
        b - radius of outer
        """
        two_pi = 2*np.pi
        #need to get ellipsoids
        train_data = []
        val_data = []
        train_index = torch.range(0, N-1, 2, dtype = torch.long)
        val_index = torch.range(1, N-1, 2, dtype = torch.long)
        #generate first swiss roll
       
        r = torch.linspace(0, a, N)
        theta = torch.linspace(0, two_pi*2,N) 

        coords = torch.zeros(size = (N,4))
        coords[:,0], coords[:,1] =  r*torch.cos(theta), r*torch.sin(theta)
        coords[:,2], coords[:,3] =  copy.deepcopy(coords[:,0]), copy.deepcopy(coords[:,1])
       
        
        labels = torch.zeros(N, dtype = torch.long)
 
        #add the coords and labels for the current class
        train_data.append([coords[train_index], labels[train_index]])
        val_data.append([coords[val_index], labels[val_index]])

        r = torch.linspace(0, a, N) + b
        coords = torch.zeros(size = (N,4))
        coords[:,0], coords[:,1] =  r*torch.cos(theta), r*torch.sin(theta)
        coords[:,2], coords[:,3] =  copy.deepcopy(coords[:,0]), copy.deepcopy(coords[:,1])
       
        
        labels = torch.zeros(N, dtype = torch.long) + 1
        train_data.append([coords[train_index], labels[train_index]])
        val_data.append([coords[val_index], labels[val_index]])

        return train_data, val_data


def main():
    a = 0.2
    #b = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)


    myS = SwissRoll(device, 500, a)

    num_classes = 2
    num_features = 4

    weights = None
    bias = None

    reg_f = False
    reg_c = False
    gpu = False
    #-----------hyper parameters
    learn_rate = 0.5
    step = 1# 0.00005
    alpha_f = 0.005
    alpha_c = 0.00025

    epochs = 250#0#200#000
    gamma = 0.25
    choice = 'v'

    begin = 0
    end = 10000

    N = 4#50
    batch_size = 64
    func_f = torch.tanh
    func_c = F.softmax
    #error_func=nn.BCELoss()
    error_func = nn.CrossEntropyLoss() 

    myS.set_model(N, num_features, num_classes, func_f, func_c, weights, bias, gpu, choice, gamma)
    
    #myS.load_model("SVN16.pt")
    #myS.model.double_layers()
 
   
    train_time = time.perf_counter()
    myS.train(error_func, learn_rate, epochs, begin, end, step, reg_f, alpha_f, reg_c, alpha_c, batch_size, graph=True)
    train_time = time.perf_counter() - train_time
    #myE.model.double_layers()
    result = myS.test(begin, end, 250, step, graph = False, batch_size = 500)
    print("accuracy_b: ", result)
    print ("train time" , train_time)
   # torch.save(myS.model.state_dict(), "SVN512.pt")


if __name__ == '__main__':
    main()
