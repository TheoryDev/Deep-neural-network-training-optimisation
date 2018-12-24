# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:46:39 2018

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
import time

#dnn codebase
import ResNet as res
import Verlet as ver
import Antisymmetric as anti
import leapfrog as lp
import ellipse as ep
import swiss as sw
#synthetic gradient module
import synthetic as syn

#complex neural network class
class complexNeuralNetwork:
    """
    The complex neural network is built of (M) neural networks
    and (M-1) synthetic gradient modules.     
    """
    
    def __init__(self, device, M=2, gpu=False):
        """
        M - number of neural networks - (M-1) synthetic gradient modules
        """
        self.__M = M
        self.__nnets = []
        self.__sgmodules = []
        self.__optimisers = []
        self.__device = device
        self.__error_func = nn.CrossEntropyLoss
        self.__gpu = gpu       
        print("gpu:", self.__gpu)
        
    def create_net(self, N = 2, num_features = 2, num_classes = 2, func_f = torch.tanh, 
                   func_c =F.softmax, weights = None, bias = None, gpu=False, choice = None, gamma = 0.01, last=False):    
            """
            This function creates a single neural network
                                    
            """
            
            if choice == 'a':
                print("a")
                model = anti.AntiSymResNet(self.__device, N, num_features, num_classes, func_f, func_c, weights, bias, gamma, gpu, last)
    
            elif choice == 'v':
                print("v")
                model = ver.Verlet(self.__device, N, num_features, num_classes, func_f, func_c, weights, bias,  gpu, last)
    
            elif choice == 'l':
                print("l")
                model = lp.Leapfrog(self.__device, N, num_features, num_classes, func_f, func_c, weights, bias, gpu, last)
    
            else:
                print("r")
                model = res.ResNet(self.__device, N, num_features, num_classes, func_f, func_c, weights, bias, gpu, last)
    
            return model
    
    def create_sg(self, function=syn.sgLoss, loss=nn.MSELoss, args=np.array([0.5,0.5,0.5]), gpu=False, num_features=2):
        return syn.synthetic_module(self.__device, function, loss, args, gpu, num_features)
    
    def init_nets(self, N = 2, num_features = 2, num_classes = 2, func_f = torch.tanh, 
                  func_c =F.softmax, weights = None, bias = None, gpu=False, choice = None, gamma = 0.01, mutli = True):
             
              
        #store neural network arguments
        self.__netArgs = (N, num_features, num_classes, func_f, func_c, weights, bias, gpu, choice, gamma)
        #add neural nets to the complex model
        for i in range(self.__M-1):
            #first M-1 nets do not need a classifier
            m = self.create_net(*self.__netArgs) 
            self.__nnets.append(m)
        #last network needs a classifier    
        m = self.create_net(*self.__netArgs, last=True) 
        self.__nnets.append(m)
        
        if self.__gpu == True:
            for net in self.__nnets:
                net.to(self.__device)
            
    def init_sgs(self, function=syn.sgLoss, loss=nn.MSELoss, args=np.array([0.5,0.5,0.5]), 
                 num_features=2, batch_size=2):#, trainloader):  
        #store synthetic gradient module arguments
        self.__sgArgs = (function, loss, args, self.__gpu, num_features)
    
        #add sg modules           
        for i in range(self.__M - 1):
            s = self.create_sg(*self.__sgArgs)
            self.__sgmodules.append(s)       
    
        #for s in self.__sgmodules:
           # s.first_optimise(num_features, batch_size)
    
    def init_optimisers(self, learn_rate = 0.001):        
        """
        It creates a list of optimisers for the sub-neural networks
        The optimisers will only include network parameters that have
        the flag requires_grad set to True    
        """    
        #store optimisers
        self.__optimisers = []
        #iterate through nets
        for net in self.__nnets:
            #get netowrk params - capability to freeze layers
            params = filter(lambda param: param.requires_grad == True, net.parameters())
            #create optimiser for sub-network
            self.__optimisers.append(optim.SGD(params, lr = learn_rate))
        
        for s in self.__sgmodules:
            s.init_optimiser(learn_rate)
    
    def propagate(self, x, step = 0.1, train=True):
        """
        The input is propagated through the series of neural networks
        and sg modules. The output is returned
        """
        #loop through all networks except the last
            #step 1 pass through sub-network 
            #step 2 store output into sg module ahead
        #exit loop
        #propagate through final network
        #calculate loss                  
        for i in range(self.__M-1):
            #propagate through sub-neural network
            x = self.__nnets[i](x, step)
            #print("x", x)
            #save input into synthetic gradient module
            #and get h_after for next round
            #if train == True:
            x = self.__sgmodules[i].propagate(x)
            
                        
        #progate through final network
        x = self.__nnets[-1](x, step)
        
        #print("x", x)
        return x
    
    def backpropgate_SG(self, y, multi=False):
        """
        The error gradient is backpropagated through the complex system of sub 
        neural networks, with the sg modules approximating the gradient for the 
        non-final neural networks
        """
        
        times = []       
        #update with synthetic gradients from last network to the first
        for i in reversed(range(self.__M-1)):
            sub_net_time = time.perf_counter()
            #calculate synthetic gradient and propagate back through sub network 
            #print("i", i)
            self.__sgmodules[i].backward(y, multi)
            #use optimiser to update the weights of each network
            self.__optimisers[i].step()
            sub_net_time = time.perf_counter() - sub_net_time
            times.append(sub_net_time)
            """
            To do, calculate training times
            -add regularisation support at somepoint       
            
            """
        
        return times
        

    def optimise_SG_modules(self, y = None, multi=False):
         """
         This function is used to optimise the SG modules. It works by 
         using the gradient of the instance parameter h_after. 
         
         """
         syn_error = []
         
         for module in reversed(self.__sgmodules):
             #optimises gradient and returns mse(actual grad, pred)
             syn_error.append(module.optimise(y, multi))
         
         return np.array(syn_error)
        
    def train(self, trainloader, error_func, learn_rate, epochs, begin, end
                                ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph, multi=False):
        """
        The function propagates the input through the complex networks and uses
        synthetic gradients to pass the data backwards              
        """
        #time counters
        #total time
        master_t = time.perf_counter()
        #time taken for backprop in series by summing individual times
        master_back_t = 0.0
        #time taken for fastest net to complete backprop each batch of data
        master_para_t = 0.0
        #time time taken for backprop in series by adding total times
        m_loop_back = 0.0     
        #data loading time
        m_load = 0.0
        #create list of optimisers 
        self.init_optimisers(learn_rate)
       
        #loss tracking
        rounds, losses = [], []               
        
        #time for forward pass
        t_forward = .0
        #time for 
        t_out = .0
        t_first = .0
        t_opt = 0.0
    
        t_net = 0.0
    
        #new_t = time.perf_counter()  
        
        t_loss = 0.0
        
        #sub_back_t = 0.0
        
        list_ones = []
        
        #load_t = time.perf_counter()        
        for epoch in range(epochs):              
            
            epoch_loss = 0
            i = 0
            syn_error = np.zeros((self.__M-1,))     
            
            ones = 0.
            
            #iterate over each batch in dataset
            batch_load_time = time.perf_counter()
            for inputs, labels in itertools.islice(trainloader, begin, end):
                #if gpu is being used move tensors to gpu memory
                if self.__gpu == True:
                        inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                      
                inputs = inputs.view(-1, self.__nnets[0].num_features)        
            
                m_load += time.perf_counter() - batch_load_time
                t_in_loop = time.perf_counter()
                #zero_grads
                for opt in self.__optimisers:
                    opt.zero_grad() 
                    
                for s in self.__sgmodules:
                    s.zero_optimiser()
                    
                batch_load_time = time.perf_counter() - batch_load_time
                #print("batch_load_time", batch_load_time)
                #propagate   
                tmp = time.perf_counter()
                out = self.propagate(inputs, f_step)             
                #print("out", out)                       
                #--------------------can add regularisation later----------------
                
                #calculate loss
                tmp_loss_t = time.perf_counter()
                loss = error_func(out, labels)    
                t_loss += time.perf_counter() - tmp_loss_t
                            
                t_forward += time.perf_counter() - tmp
                
                tmp = time.perf_counter()
                #backward on loss and optimiser.step final net
                back_t = time.perf_counter()
                out_net_back_time = time.perf_counter()  
                     
                #update output network weights
                loss.backward()             
                self.__optimisers[-1].step()
                t_out += time.perf_counter() - tmp
                
                out_net_back_time = time.perf_counter() - out_net_back_time                
              
                #for each sub-nn backpropagate sg - makes sense as network 
                #can backprop as soon as it reuturns h^n
                #update sub-neural net weights using sg
                tmp = time.perf_counter()
                sub_back_times = self.backpropgate_SG(labels, multi)    
                t_first += time.perf_counter() - tmp                          
                #the slowest training time of the sub-neural networks is the training time
                sub_back_times.append(out_net_back_time)
                #add to epoch backprop time
               
                m_loop_back += time.perf_counter() - back_t
                #master times
                master_back_t += sum(np.array(sub_back_times))
                master_para_t += np.array(sub_back_times).max()
                ones += np.array(sub_back_times).argmax()            
            
                #optimise sg modules
                tmp = time.perf_counter()
                if multi == False:
                    if i % 4 == 0:     
                        #print(i % 2)
                        syn_error += self.optimise_SG_modules(labels)
                t_opt += time.perf_counter() - tmp        
                epoch_loss += loss.item()
               
                
                
                i += 1
                t_net += time.perf_counter() - t_in_loop 
                batch_load_time = time.perf_counter()
                
                #'print("i", i)
            
            #add the number of ones/ number of batches to list
            list_ones.append(ones/i)            
            
            #track loss
            epoch_freq = i*trainloader.batch_size 
            #print("i")
            if multi == False:
                print("epoch: ", epoch+1, "loss: ", epoch_loss/epoch_freq, "syn_errors", syn_error/epoch_freq)
            #print("epoch_forward_time:" , epoch_forward_time, "epoch_backprop_time:", epoch_backprop_time, "sg_opt_time" , epoch_opt_time)
            if graph == True:
                #make more sophisticated
                losses.append(epoch_loss/epoch_freq)
                rounds.append(epoch)   
          
        if graph == True:
            plt.plot(rounds, losses, '-')
            plt.grid()
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.show()
        
        if multi == True:
            print("mult", multi)
            #store the last batch labels to use for linear fit
            for module in self.__sgmodules:
                module.store_labels(labels)               
  
        #new time measurements
        master_t = time.perf_counter() - master_t
        m_non_back = master_t - master_back_t 
        theoretical_time =  m_non_back+ master_para_t
        adjust_speed_up = (master_t-m_load)/(theoretical_time-m_load)
        
        print("\nblock back_time",  m_loop_back, "\nback_time", master_back_t, "\npara_time", master_para_t)
        print("\nbatch load time", m_load)
        print("\nin_net time", t_net)
        print("\nmean", np.array(list_ones).mean(), "\nmedian" , np.median(list_ones), "\n")
        
        print ("t_forward", t_forward)
        print("t_out" ,t_out)
        print("t_first", t_first)
        print("t_opt", t_opt)
        print("loss_t", t_loss)
    
        
        
        
        return master_t, theoretical_time, master_t/theoretical_time, adjust_speed_up
        #test training accuracy of full network     
       
    def test(self, testloader, begin = 0, end = 100, f_step =0.1):
        #counters for number of positive examples and total
        num_positive = 0.
        num_total = 0.
        #iterate through training set
        for inputs, labels in itertools.islice(testloader, begin, end):

            if self.__gpu == True:
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
            #convert to vector
            inputs = inputs.view(-1, self.__nnets[0].num_features)
            #propagate through network          
            #print("in", inputs)           
            outputs = self.propagate(inputs, step = f_step, train=False)
            #print("outputs", outputs)
            #get predictions- by getting the indices corresponding to max arguments
            pred = torch.max(outputs, 1)[1]
           # print("'",inputs, labels, pred,"'")
            #get number of correct examples in batch
            correct = (pred == labels).sum().item()

            #add to the counters
            num_total += labels.size(0)
            num_positive += correct


        #return total , number positive, classificaiton rate
        return [num_total, num_positive, 100*(num_positive/num_total)]
            
    def get_nets(self):
        return self.__nnets    
    
    def train_multi_level(self,trainloader, error_func, learn_rate, epochs, 
                          begin, end, f_step, reg_f, alpha_f, reg_c, alpha_c, graph=False):
    
        """
        This function trains the coarse model (N/2 layers)
        It then initialises the SG function's linear fit using the final batch
        Then the number of layers in the network is doubled           
        
        """
        #train coarse network
        self.train(trainloader, error_func, learn_rate, epochs, 
                          begin, end, f_step, reg_f, alpha_f, reg_c, alpha_c, graph = graph, multi=True)
        
        #print("sg_mod_b", self.__sgmodules[0].get_coefs())              
        
        #init sg - the last batches' labels were stored in the sg module
        self.optimise_SG_modules(multi=True)
                
    def double_complex_net(self):    
        
        #print("sg_mod_a", self.__sgmodules[0].get_coefs())
        #double the number of layers in each net
        for net in self.__nnets:
            net.double_layers()
            if self.__gpu == True:
                net.to(self.__device)
       
        
    #def multilevel_learn()
        """
        Trains the coarse network.
        ---- sg init to do --- need way to get gradients at the midpoint
        Then network is split in half and the weights are prolongated
        to initialise the first and second sub-neural networks
        
        """
        
        """
        #1.create network with N/2 layers
        #2.train it to a good level of accuracy say 90%
        #3.get 1000 h^n, y and grad values for middle layer of network
        #4.use those values to initialise sg
        #5. initialise networks using weights from coarse network
        # delete network for coarse to save memory
        #6. train and test complex network
      
        NOTE - I can do module every layer by simply adding setting N=1 and choosing M as the number of layers
        """
            
"""
To do - add saving networks      
      - can add regularisation one week off
     
"""

         
            
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    

if __name__ == '__main__':
    main()