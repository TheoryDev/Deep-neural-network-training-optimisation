from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import itertools
import copy
from torch.autograd import Variable
import time

import ResNet as res
import Antisymmetric as anti
import leapfrog as lp
import Verlet as ver
import ellipse as el
import swiss as sr
import dataloader as dl

class PVD:
    """
    class to perform the PVD algorithm
    """
    def __init__(self, device, num_proc = 2, trainloader = None, testloader = None):
        #TO DO: change loaders
        """ device - cpu or gpu
            num_proc - number of processors
        """
        self.device = device
        self.num_proc = num_proc
       
    def set_models(self, N = 3, num_features = 2, num_classes = 2, func_f = torch.tanh, func_c =F.softmax, weights = None, bias = None, gpu=False, choice = None, gamma = 0.01):
        """
        allows the user to set the model choices defaults to resnet
        'a' - antisymmetric resnet
        'v' - verlet integration
        'l' - leapfrog integration
        """
        models = []

        #make self.num_proc copies of the model
        for i in range(self.num_proc):
            m = self.create_model(N, num_features, num_classes, func_f, func_c, weights, bias, gpu, choice, gamma)
            models.append(m)

        self.models = models

        self.direction_model = self.create_model(N, num_features, num_classes, func_f, func_c, weights, bias, gpu, choice, gamma)

    def create_model(self, N = 3, num_features = 2, num_classes = 2, func_f = torch.tanh, func_c =F.softmax, weights = None, bias = None, gpu=False, choice = None, gamma = 0.01):
        """
        allows the user to set the model choices defaults to resnet
        'a' - antisymmetric resnet
        'v' - verlet integration
        'l' - leapfrog integration
        """
        self.choice = choice
        #choosing model
        
        #gpu is only used for MNIST only want one model on gpu memory at once
        
        if choice == None:
            choice = self.choice

        #choose resnet
        if choice == 'a':
            print("a")
            model = anti.AntiSymResNet(self.device,N, num_features, num_classes, func_f, func_c, weights, bias, gamma, gpu)

        elif choice == 'v':
            print("v")
            model = ver.Verlet(self.device,N, num_features, num_classes, func_f, func_c, weights, bias,  gpu)

        elif choice == 'l':
            print("l")
            model = lp.Leapfrog(self.device,N, num_features, num_classes, func_f, func_c, weights, bias, gpu)

        else:
            print("r")
            model = res.ResNet(self.device,N, num_features, num_classes, func_f, func_c, weights, bias, gpu)

        #set parameters to gpu
        if gpu == True:
            model.to(self.device)

        return model

    def freeze(self):
        for m in self.models:
            m.freeze_all()

    def freeze_blocks(self, steps):
        #first freeze all reset_parameters
        self.freeze()
        i, j = 0, 0
        #freeze step-many layers in each model
        for model in self.models:
            j += steps
            model.unfreeze_range(i, j)
            i = j
            model.unfreeze_classifier()
        #the last model always has the classifier

    def gen_directions(self, model, trainloader, error_func, f_step = 0.1, plot=False):
        #clear gradient buffers
        model.zero_grad()

        #get first batch to calculate directions with- if time try to reset trainloader
        inputs, labels = iter(trainloader).next()
        #get batch of input features and convert from (28*28->784 MNIST)

        if model.gpu == True:
            inputs, labels = inputs.to(model.device), labels.to(model.device)

        inputs = inputs.view(-1, model.num_features)


        outputs = model(inputs, f_step, plot)
        loss = error_func(outputs, labels)

        #print("loss is:" ,loss)

        loss.backward()

        directions = []

        for layer in model.layers:            
            directions.append([layer.weight.grad, layer.bias.grad])

        return directions

    def pvd_algo(self, iterations, trainloader, testloader, error_func=nn.CrossEntropyLoss(), learn_rate = 0.01, epochs = 10, begin = 0, end = 100
                            ,f_step = 0.1, reg_f=False, alpha_f=0.01, reg_c= False, alpha_c = 0.01, graph=False, base_acc = 85):
        """
        The pvd algorithm is implemented here:
        iterations - number of iterations to train for
        trainloader - dataloader object of training examples
        testtloader - dataloader object of testing examples
        base_acc - accuracy level at which to stop the PVD
        """
        timer = 0
        #freeze parameters not in the block
        steps = round(self.models[0].N/self.num_proc)
        self.freeze_blocks(steps)
                
        #run pvd for iterations
        for i in range(iterations):         
            
            #calculate list of directions
            direction_block = self.gen_directions(self.direction_model, trainloader, error_func, f_step, plot = graph)
       
            print("---------------------------iteration---------------------------------", i+1)
           
            #train each model in loop using error as objective function           
            #train each model and calculate errors
            results = []
            
            #parallelisation---------------------------------------------------
            for m in self.models:
                
                r = self.pvd_process(m, direction_block,trainloader, error_func, learn_rate, epochs, begin, end
                                        ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph)
                results.append(r)

            #synchronisation---------------------------------------------------
            torch.cuda.synchronize()
            synch_time = time.perf_counter()

            #get minimum error
            errors = np.array([r[0] for r in results])
            min_index = errors.argmin()

            #now get directions
            direction, mu = results[min_index][1], results[min_index][2]
            
            print("processor (", min_index, ") was the fastest in this epoch")
            #multiply direction blocks by mu
            mu_direction = res.ResNet.mult_mu(direction, mu, steps)

            #update weights
            self.add_direction(self.models[min_index], mu_direction)

            #after weights updated with forget me not terms copy to all others          
            acc = self.models[min_index].test(trainloader, 0, 1000, f_step)

            print("result:", acc)
            
            times = np.array([r[3] for r in results])
            p_time = times.mean()
            
            acc = acc[2]
            if acc >= base_acc:
                torch.cuda.synchronize()
                synch_time = time.perf_counter() - synch_time 
                timer += synch_time + p_time
                print("---------------------PVD finished with accuracy:", acc, "-------------------")
                print("synch time:", synch_time, "time in process:", p_time)               
                return timer
          
            #synchronise the modesl
            self.copy_model(min_index)
            torch.cuda.synchronize()
            synch_time = time.perf_counter() - synch_time
            #print(synch_time, p_time) 
            timer += synch_time + p_time
            
        return timer

    def pvd_process(self, model, direction_block ,trainloader, error_func, learn_rate, epochs, begin, end
                            ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph):
     
        #get time of process
        torch.cuda.synchronize()
        run_time = time.perf_counter()
        #gets block of directions and set as instance of model
   
        directions = self.d_block(direction_block, model)
    
        steps = np.round(model.N/self.num_proc)
        mu = torch.rand((self.num_proc-1), requires_grad = True)

        if model.gpu == True:
            mu = torch.rand((self.num_proc-1), requires_grad = True, device = "cuda")      

        model.directions = copy.deepcopy(directions)

        #train model by optimising weights and also mu
        #t = time.perf_counter()
        error = model.train(trainloader, error_func, learn_rate, epochs, begin, end
                                ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph, mu, steps)
       # print("t: ",time.perf_counter()-t)
        torch.cuda.synchronize()
        run_time = time.perf_counter() - run_time
      
        #print("mu", mu)
       # print("run time of processor is ", run_time)
        return [error, directions, mu, run_time]


    def d_block(self, directions, model):
        #keep track of when loop is in the model's block
        in_block = False
        start = 0
        end = len(model.layers)
        #iterate through layers to find out which ones are in the block
        for i in range(len(model.layers)):
            #get first layer in block
   
            if model.layers[i].weight.requires_grad == True and in_block == False:
                start = i
                in_block = True
                continue
            #get last layer in block
            if model.layers[i].weight.requires_grad == False and in_block == True:
                end = i
                in_block = True
                break
        #useful to know which processor is currently being trained
        #print("start", start, "end", end)
        
        d = directions[:start] + directions[end:]
        return d

    def copy_model(self, index):
        #iterate set all the other models to the original
        best_model = self.models[index]
        range_models = range(len(self.models))
        layer_range = range(len(best_model.layers))

        #iterate through layers
        for i in range_models:
            #skip if best model is current element of self.models
            if i == index:
                continue
            for j in layer_range:
                #copy layers
                self.copy_layer(best_model.layers[j], self.models[i].layers[j])
            #copy classifier
            self.copy_layer(best_model.classifier, self.models[i].classifier)

        #copy into direction_model
        for j in layer_range:
            self.copy_layer(best_model.layers[j], self.direction_model.layers[j])
        self.copy_layer(best_model.classifier, self.direction_model.classifier)


    def copy_layer(self, layer1, layer2):
        layer2.weight.data = layer1.weight.data
        layer2.bias.data = layer2.bias.data



    def add_direction(self, model, mu_directions):
        """
        This multiplies the directions by the corresponding components of mu
        """
        index = 0
        #iterate through the directions
        for layer in model.layers:
            #if layer is frozen use forget me not term
            if layer.weight.requires_grad == False:
                w, b = mu_directions[index]
                layer.weight.data += w
                layer.bias.data += b
                index += 1    

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    #set the flag for the database you want equal to trye
    conv = False #
    dataset_name = "ELLIPSE" #choose from MNIST, CIFAR10, CIFAR100, ELLIPSE, SWISS
    gpu=False

    #architecture choices
    num_proc =  2
    N = 4
    num_features = 2
    num_classes = 2
    batch_size = 8
    
    learn_rate = 0.1
    iterations = 10#00 # max number of iterations of PVD algorithm
    epochs = 100# epochs that each process will train for in each iteration of the PVD
       
    func_f = torch.tanh
    func_c =F.softmax
    weights = None
    bias = None
    error_func=nn.CrossEntropyLoss()
    
    choice = 'v'
    gamma = 0.1   
    
    begin = 0
    end = 100
    f_step =0.5

    reg_f=False
    alpha_f=0.01
    reg_c= False
    alpha_c = 0.01
    graph=False
    acc = 85
      
    dataloader = dl.InMemDataLoader(dataset_name, conv_sg=False) # only uses FCN layers
        
    num_features, num_classes, in_channels = dl.getDims(dataset_name)
    
    #load training dataset                         
    trainloader = dataloader.getDataLoader(batch_size, shuffle = True, num_workers = 0, pin_memory = True, train = True)     
    testloader = dataloader.getDataLoader(batch_size, shuffle = False, num_workers = 0, pin_memory = False, train = False)
    
    pvd = PVD(device, num_proc)
    pvd.set_models(N, num_features, num_classes, func_f, func_c, weights, bias, gpu, choice, gamma)

    torch.cuda.synchronize()
    start_time = time.perf_counter()  
    
    timer =  pvd.pvd_algo(iterations, trainloader, testloader, error_func, learn_rate, epochs, begin, end
                            ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph, acc)

    torch.cuda.synchronize()
    print("total time:", time.perf_counter()-start_time)
    print("theoretical distributed time:", timer)
    torch.cuda.synchronize()
    #print("time_old: ", (time.time()-start_time)/num_proc)
    result = pvd.models[0].test(testloader, begin, end, f_step)
    print("test accuracy result:", result)
    #print("iterations", iterations, "num processors", num_proc, "num layers", N, "epochs", epochs, "acc", acc)
 

if __name__ == '__main__':
    main()
