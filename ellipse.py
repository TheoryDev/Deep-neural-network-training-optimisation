from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import itertools
import time

import ResNet as res
import Antisymmetric as anti
import leapfrog as lp
import Verlet as ver


class ellipse:
    """
    class to create two concentric ellipse datasets
    """

    def __init__(self, device, N1, N2, a, b, choice = None):
        """
        initilistion for the ellipse
        N - number of points in each class
        a - list of a values
        b - lust of b values
        num_classes - number of classes
        choice - choice of dnn architecture
        """
        torch.manual_seed(11)
        self.choice = choice

        #create training examples
        self.examples = self.data_gen(N1, a, b)
        #create test examples
        self.valid = self.data_gen(N2, a, b)
        self.device = device

    def data_gen(self, N, a, b):
        """
        generates ellipses
        """
        two_pi = 2*np.pi
        #need to get ellipsoids
        data = []
        i = 0
        lims = zip(a,b)
        for x, y in lims:
            uni = torch.distributions.Uniform(x[0],x[1])
            a = uni.sample((1,N))
            uni = torch.distributions.Uniform(y[0],y[1])
            b = uni.sample((1,N))
        
            #generate angles
            theta = torch.rand(N)*two_pi
            #use parametric equations to get ellipse coordinates
            coords = torch.zeros(size = (N,2))
            coords[:,0], coords[:,1] =  a*torch.cos(theta), b*torch.sin(theta)#0.5*i + torch.rand(N), 0.5*i+torch.rand(N)

            labels = torch.zeros(N, dtype = torch.long) + i     
            i += 1

            #add the coords and labels for the current class
            data.append([coords, labels])
            
        return data


    def plot_data(self, data):
        fig, ax = plt.subplots()
        for coord, label in data:
            x, y = coord[:,0], coord[:,1]          
            ax.scatter(x, y)
            ax.axis('equal')    
        plt.axis('equal')
        plt.show()


    def get_examples(self):
        return self.examples

    def set_model(self, N = 3, num_features = 2, num_classes = 2, func_f = torch.tanh, func_c =F.softmax, weights = None, bias = None, gpu=False, choice = None, gamma = 0.01):
        """
        allows the user to set the model choices defaults to resnet
        'a' - antisymmetric resnet
        'v' - verlet integration
        'l' - leapfrog integration
        """
        self.choice = choice
        #choosing model

        if choice == None:
            choice = self.choice

        #choose resnet
        if choice == 'a':
            print("a")
            self.model = anti.AntiSymResNet(self.device,N, num_features, num_classes, func_f, func_c, weights, bias, gamma, gpu)

        elif choice == 'v':
            print("v")
            self.model = ver.Verlet(self.device,N, num_features, num_classes, func_f, func_c, weights, bias,  gpu)

        elif choice == 'l':
            print("l")
            self.model = lp.Leapfrog(self.device,N, num_features, num_classes, func_f, func_c, weights, bias, gpu)

        else:
            print("r")
            self.model = res.ResNet(self.device,N, num_features, num_classes, func_f, func_c, weights, bias, gpu)

        #set parameters to gpu
        if gpu == True:
            self.model.to(self.device)

    def train(self, error_func=nn.CrossEntropyLoss, learn_rate = 0.01, epochs = 10, begin = 0, end = 100,
                            f_step = 0.1, reg_f=True, alpha_f=0.01, reg_c= True, alpha_c = 0.01, batch_size = 1, graph=False):
    
        dataset = self.create_dataset(self.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)

        self.model.train(trainloader, error_func, learn_rate, epochs, begin, end, f_step, reg_f, alpha_f, reg_c, alpha_c, graph)



    def create_dataset(self, dataset):
        datasets = []
        for coords, label in dataset:
            dataset = torch.utils.data.TensorDataset(coords, label)
            datasets.append(dataset)

        return torch.utils.data.ConcatDataset(datasets)


    def test(self, begin, end, N = 100, f_step =0.1, graph = False, batch_size =200):
        
        dataset = self.create_dataset(self.valid)
       # dataset = self.create_dataset(self.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)

        result = self.model.test(trainloader, begin, end, f_step = f_step)

        if graph == True:
            x = self.model.final
                #    print(x.shape)
            plt.plot(x[:N,0], x[:N,1], 'bo')
            plt.plot(x[N:,0], x[N:,1], 'ro')
            plt.axis('equal')
          
            plt.show()

        return result
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    #ellipse dimensions
    a = np.array([[0,0.1],[0.1,0.2]])
#    b = [[0,0.3],[0.6,1.2]]
    a = a*10
    b = a*0.5
    num_classes = 2
    num_features = 2

    weights = None
    bias = None
  
    reg_f = False
    reg_c = False
    gpu = False
    #-----------hyper parameters
    learn_rate = 0.2
    step = .05# 0.00005
    epochs = 10    
   
    
    alpha_f = 0.02001
    alpha_c = 0.00025

    
    gamma = 0.05
    choice = 'r'

    begin = 0
    end = 10000

    N = 512#50
    batch_size = 64
    func_f = torch.tanh
    func_c = F.softmax    
    error_func = nn.CrossEntropyLoss()
  
    myE = ellipse(device, 500, 100, a, b)

    myE.set_model(N, num_features, num_classes, func_f, func_c, weights, bias, gpu, choice, gamma)
  
    #For multilevel training uncmment and then load weights, then the double layers function will prolongate them using averaging:

    #myE.load_model("EVN16.pt")
    #myE.model.double_layers()
    
    start_time = time.time()
    myE.train(error_func, learn_rate, epochs, begin, end, step, reg_f, alpha_f, reg_c, alpha_c, batch_size, graph=True)
    end_time = time.time() - start_time
    
  
    #torch.save(myE.model.state_dict(), "ELN64.pt")
    
   
    result_test = myE.test(begin, end, 100, step, graph = False, batch_size = 200)
    print("test accuracy: ", result_test)
    print("time: ", end_time)
    
   # myE.model.plotter(torch.tensor([[0.1,0.1]]), 0.5)
    

if __name__ == '__main__':
    main()
