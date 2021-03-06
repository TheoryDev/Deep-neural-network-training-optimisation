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
import multiprocessing as mp

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
    
    def __init__(self, device, M=2, gpu=False, conv=False, in_chns=1, n_filters = 6):
        """
        device - gpu
        M - number of neural networks , (M-1) synthetic gradient modules
        gpu - boolean flag for gpu
        conv - if true convolutional nets will be used
        in_chns- number of channels in images
        n_filters - number of filters
        """
        self.__M = M
        self.__nnets = []
        self.__sgmodules = []
        self.__optimisers = []
        self.__device = device
        self.__error_func = nn.CrossEntropyLoss
        self.__gpu = gpu  
        self.__conv = conv
        self.__in_chns = in_chns
        self.__n_filters = n_filters
        
        
    def create_net(self, N = 2, num_features = 2, num_classes = 2, func_f = torch.tanh, 
                   func_c =F.softmax, weights = None, bias = None, choice = None, gamma = 0.01,
                   last=False, first=False):    
            """
            This function creates a single neural network                                    
            """            
            if choice == 'a':
                print("a")
                model = anti.AntiSymResNet(self.__device, N, num_features, num_classes, func_f, func_c, weights, bias, gamma, gpu, last, sg)
    
            elif choice == 'v':
                print("v")
                model = ver.Verlet(self.__device, N, num_features, num_classes, func_f, func_c, weights,
                                   bias, self.__gpu, last, self.__conv, first, self.__in_chns, self.__n_filters)
    
            elif choice == 'l':
                print("l")
                model = lp.Leapfrog(self.__device, N, num_features, num_classes, func_f, func_c, weights, bias, gpu, last, sg)
    
            else:
                print("r")
                model = res.ResNet(self.__device, N, num_features, num_classes, func_f, func_c, weights, bias, 
                     self.__gpu, last, self.__conv, first, self.__in_chns, self.__n_filters)
    
            return model
    
    def create_sg(self, gpu=False, num_features=2):
        #creates sg module
        return syn.synthetic_module(self.__device, gpu, num_features, self.__conv, self.__in_chns, self.__n_filters)
    
    def init_nets(self, N = 2, num_features = 2, num_classes = 2, func_f = torch.tanh, 
                  func_c =F.softmax, weights = None, bias = None, choice = None, gamma = 0.01, mutli = True):
             
              
        #store neural network arguments
        self.__netArgs = (N, num_features, num_classes, func_f, func_c, weights, bias, choice, gamma)
        
        self.__nnets = []
        
        m = self.create_net(*self.__netArgs, first=True)
        self.__nnets.append(m)
        
        #add neural nets to the complex model
        for i in range(self.__M-2):
            #first M-1 nets do not need a classifier
            m = self.create_net(*self.__netArgs) 
            self.__nnets.append(m)
        #last network needs a classifier    
        m = self.create_net(*self.__netArgs, last=True) 
        self.__nnets.append(m)
        
        if self.__gpu == True:
            for net in self.__nnets:
                net.to(self.__device)
            
    def init_sgs(self, num_features=2, batch_size=2): 
        
        #store synthetic gradient module arguments                
        self.__sgArgs = (self.__gpu, num_features)
    
        #add sg modules           
        for i in range(self.__M - 1):
            s = self.create_sg(*self.__sgArgs)
            self.__sgmodules.append(s)    
    
    def init_optimisers(self, learn_rate = 0.001, decay_rate=0):        
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
            self.__optimisers.append(optim.Adam(params, lr = learn_rate, weight_decay=decay_rate))
        
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
            #save input into synthetic gradient module
            #and get h_after for next round            
            x = self.__sgmodules[i].propagate(x)          
                        
        #progate through final network
        x = self.__nnets[-1](x, step)        
        
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
            torch.cuda.synchronize()
            sub_net_time = time.perf_counter()
            #calculate synthetic gradient and propagate back through sub network             
            self.__sgmodules[i].backward(y, multi)
            #use optimiser to update the weights of each network
            self.__optimisers[i].step()
            torch.cuda.synchronize()
            sub_net_time = time.perf_counter() - sub_net_time
            times.append(sub_net_time)
            """
            To do -add regularisation support at somepoint       
            
            """        
        return times        

    def optimise_SG_modules(self, y=None, multi=False):
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
        decay_rate = 0
        if reg_f == True:
            decay_rate = alpha_f  
        
        self.init_optimisers(learn_rate, decay_rate)
       
        #loss tracking
        rounds, losses = [], []               
        
        #time for forward pass
        t_forward = .0
        #time for 
        t_out = .0
        t_first = .0
        t_opt = 0.0    
        t_net = 0.0
        t_loss = 0.0        
        
        list_ones = []
        
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
                
                if self.__conv == False:
                    inputs = inputs.view(-1, self.__nnets[0].num_features*self.__nnets[0].in_chns)                                        
                torch.cuda.synchronize()
                m_load += time.perf_counter() - batch_load_time
                t_in_loop = time.perf_counter()
                #zero_grads
                for opt in self.__optimisers:
                    opt.zero_grad() 
                    
                for s in self.__sgmodules:
                    s.zero_optimiser()
                torch.cuda.synchronize()    
                batch_load_time = time.perf_counter() - batch_load_time
                
                #propagate  
                torch.cuda.synchronize()
                tmp = time.perf_counter()
                out = self.propagate(inputs, f_step)             
                              
                #--------------------may add regularisation later----------------
                
                #calculate loss
                torch.cuda.synchronize()
                tmp_loss_t = time.perf_counter()
               
                loss = error_func(out, labels)  
                torch.cuda.synchronize()
                t_loss += time.perf_counter() - tmp_loss_t
                
                torch.cuda.synchronize()
                t_forward += time.perf_counter() - tmp
                
                torch.cuda.synchronize()
                tmp = time.perf_counter()
                #backward on loss and optimiser.step final net
                back_t = time.perf_counter()
                out_net_back_time = time.perf_counter()            
            
                #update output network weights
                loss.backward()             
                self.__optimisers[-1].step()
                
                torch.cuda.synchronize()
                t_out += time.perf_counter() - tmp                
                out_net_back_time = time.perf_counter() - out_net_back_time                
              
                #for each sub-nn backpropagate sg - makes sense as network 
                #can backprop as soon as it reuturns h^n
                #update sub-neural net weights using sg
                torch.cuda.synchronize()
                tmp = time.perf_counter()
                #if self.__conv == True:
                  #  sub_back_times = self.backpropgate_SG(labels, multi)
                #else:                    
                sub_back_times = self.backpropgate_SG(labels, multi) 
                torch.cuda.synchronize()
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
                torch.cuda.synchronize()
                tmp = time.perf_counter()
                #if multi == False:                    
                syn_error += self.optimise_SG_modules(labels)
                torch.cuda.synchronize()
                t_opt += time.perf_counter() - tmp        
                epoch_loss += loss.item()               
                
                i += 1
                t_net += time.perf_counter() - t_in_loop 
                batch_load_time = time.perf_counter()                
                
            
            #add the number of ones/ number of batches to list
            list_ones.append(ones/i)            
            
            #track loss
            epoch_freq = i*trainloader.batch_size 
           
            print("epoch: ", epoch+1, "loss: ", epoch_loss/epoch_freq, "syn_errors", syn_error/epoch_freq)         
            if graph == True:               
                losses.append(epoch_loss/epoch_freq)
                rounds.append(epoch)   
          
        if graph == True:
            plt.plot(rounds, losses, '-')
            plt.grid()
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.show()
          
        #new time measurements
        master_t = time.perf_counter() - master_t
        m_non_back = master_t - master_back_t 
        theoretical_time =  m_non_back+ master_para_t
        adjust_speed_up = (master_t-m_load)/(theoretical_time-m_load)
        
        """
        FOR TESTING PURPOSES
        print("\nblock back_time",  m_loop_back, "\nback_time", master_back_t, "\npara_time", master_para_t)
        print("\nbatch load time", m_load)
        print("\nin_net time", t_net)
        print("\nmean", np.array(list_ones).mean(), "\nmedian" , np.median(list_ones), "\n")
        
        print ("t_forward", t_forward)
        print("t_out" ,t_out)
        print("t_first", t_first)
        print("t_opt", t_opt)
        print("loss_t", t_loss)       
        """
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
            if self.__conv == False:
                #convert to vector
                inputs = inputs.view(-1, self.__nnets[0].num_features*self.__nnets[0].in_chns)
           
            #propagate through network                                 
            outputs = self.propagate(inputs, step = f_step, train=False)
            
            #get predictions- by getting the indices corresponding to max arguments
            pred = torch.max(outputs, 1)[1]         
            
            #get number of correctly classified examples in the batch
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
        
        #To do double the networks in here and then use the last labels to condition the sg
                
    def double_complex_net(self):                
        #double the number of layers in each net
        for net in self.__nnets:
            net.double_layers()
            if self.__gpu == True:                
                net.to(self.__device)     
                
  
    def getFirstNetParams(self):
        return [x for x in self.__nnets[0].parameters()]
    
    def distTrain(self, trainloader, error_func, learn_rate, epochs, begin, end
                                ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph, multi=False, num_procs=2):

        """
        Distributed training algorithm using multiprocessing
        num_procs - number of processes to distributed training across
        """
         
        #init pipes
        processes = []
        pipes = []
        losses = []
        rounds = []
        #create optimisers
        if self.__gpu == True:
            #send model to cpu to send to child processes
            device = torch.device("cpu")            
            for net in self.__nnets:                
                net.to(device)    
                net.props = None
        
            for sg in self.__sgmodules:
                sg.set_device(device)                          
        
        self.init_optimisers(learn_rate)        
       
        #pipes for synchronisation
        pPipe, cPipe = mp.Pipe()
        print("before start")        
        for i in range(num_procs+1):
            pipes.append(mp.Pipe())
        #first process
        p = mp.Process(target=proc_run, args=(0, pipes[0][1], pipes[1][0], 
                                              self.__nnets[0], self.__optimisers[0], f_step, self.__sgmodules[0], None, cPipe))
        p.start()
        processes.append(p)
        
        #N-2 middle processes
        for i in range(1, num_procs-1):
            p = mp.Process(target=proc_run, args=(i, pipes[i][1], pipes[i+1][0], 
                                                  self.__nnets[i], self.__optimisers[i], f_step, self.__sgmodules[i], None, cPipe))
            p.start()
            processes.append(p)
        #last process
        p = mp.Process(target=proc_run, args=(-1, pipes[-2][1], pipes[-1][0], self.__nnets[-1], 
                                              self.__optimisers[-1], f_step, None, error_func, cPipe))
        p.start()
        processes.append(p)
        print("start")
        torch.cuda.synchronize()
        t = time.perf_counter()
        
        for epoch in range(epochs):        
            
            epoch_loss = 0    
            i = 0
            #iterate over batches and epochs and send to processors
            for inputs, labels in itertools.islice(trainloader, begin, end):
                
                if self.__conv == False:
                    #flatten inputs 
                    inputs = inputs.view(-1, self.__nnets[0].num_features*self.__nnets[0].in_chns)             
                          
                #send batch through pipes
                pipes[0][0].send([inputs, labels])
                
                #receive loss              
                loss = pipes[-1][1].recv()
                
                epoch_loss += loss           
                i += 1               
       
            print("epoch", epoch+1, "loss", epoch_loss/(i*trainloader.batch_size))
            
            if graph == True:
                #make more sophisticated
                losses.append(epoch_loss/i)
                rounds.append(epoch)   
        print("end")  
        torch.cuda.synchronize()
        t = time.perf_counter() - t    
        #send kill signal
        pipes[0][0].send(None)        

        #if using gpu synch parameters with child process
        if self.__gpu == True:
            #iterate through processes
            #collect parameters
            
            params = []
            for i in range(self.__M):
                p = pPipe.recv()
                params.append(p)
                                  
        for p in processes:
            p.join()
                    
        if self.__gpu == True:
            #iterate through processes
            device = torch.device("cuda:0")
            for i in range(self.__M-1):                
                net_params, sg_params = params[i]
                self.__nnets[i].set_net_params(net_params)
                self.__nnets[i].to(device)        
                self.__nnets[i].set_device(device)
                self.__sgmodules[i].set_params(sg_params)
                self.__sgmodules[i].set_device(device)
            #get last net       
            cWeights, cBias = params[-1]
            self.__nnets[-1].set_net_params(cWeights)
            self.__nnets[-1].set_classifier_params(cBias)
            self.__nnets[-1].to(device)
            self.__nnets[-1].set_device(device)
        
        print("ended")       
             
        if graph == True:
            plt.plot(rounds, losses, '-')
            plt.grid()
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.show()        
    
        return t
"""
To do - add saving networks      
      - add regularisation
"""

def proc_run(name, pipeA, pipeB, model, opt, step, sg_module, error_func, parentPipe):
    #receive data     
    data = pipeA.recv()   
        
    if model.gpu == True:
        #send model and sg module to i'th gpu
        if name == -1: 
            #use first gpu for last processor
            device = torch.device("cuda:0")            
        else:            
            device = torch.device("cuda:"+str(name+1))
            sg_module.set_device(device)
        model.to(device)
        model.set_device(device)
   
    scheduler = optim.lr_scheduler.StepLR(opt, 5, 0.9)
    
    while data != None:                   
        
        inputs, labels = data
        
        if model.gpu == True:
            inputs, labels = inputs.to(device), labels.to(device)
        
        if name != 0:
            inputs.requires_grad = True
        
        #zero optimiser
        opt.zero_grad()
        if name != -1:
            sg_module.zero_optimiser()
        
        #propagate through sub neural network
        output = model(inputs, step)
       
        #propagate through sg module and pass output to next net
        if name == -1:
            loss = error_func(output, labels)
            pipeB.send(loss.detach().cpu())
        else:
            output = sg_module.propagate(output, para=True)        
            pipeB.send([output.detach().cpu(), labels.detach().cpu()])            
      
        #use backward pass
        if name == -1:            
            loss.backward()            
            
        #calculate synthetic gradient    
        else:
            sg_module.backward(labels, False)
        #update weights    
        opt.step()                               
               
        #send synthetic gradient back first if not first neural net
        if name != 0:    
            pipeA.send(inputs.grad.detach().cpu())
            
        #if last subnet wait for next batch 
        if name != -1:         
            #get gradient and update sg module
            grad = pipeB.recv()
            #get error
            if model.gpu == True:
                grad = grad.to(device)
            
            sg_error = sg_module.optimise(labels, multi=False, para=True, gradient=grad)            
        
       # scheduler.step()
        #get next batch
        data = pipeA.recv()   
    
    if model.gpu == True:
        #send parameters to the parent process
        device = torch.device("cpu")
        model.to(device)
        if name == -1:            
            parentPipe.send(model.get_net_params())
        else:
            sg_module.set_device(device)
            parentPipe.send([model.get_net_params(), sg_module.get_params()])     
    
    #send terminating none signal to next sub neural network
    pipeB.send(None)  
              
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
       
if __name__ == '__main__':
    main()
