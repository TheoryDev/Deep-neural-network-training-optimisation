from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import itertools
from torch.autograd import Variable 
import time
import copy

class ResNet(nn.Module):
    #use super to initialise the base class ClassName(object):
        """ The ResNet class inherits from the nn.Module class.
        """
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

            super(ResNet, self).__init__()
            #forward propagation activationn function
            self.num_features = num_features
            self.num_classes = num_classes
            self.func_f = func_f
            self.func_c = func_c
            #stores layers
            self.N = N
            self.device = device
            self.layers = nn.ModuleList([])
            #useful for the PVD
            self.directions = None
            self.mu = None
            self.last = False
            self.n_filters = n_filters
            self.in_chns = in_chns
            self.first=first            
            self.last = last
            self.gpu = gpu   
            self.conv = conv
            #add N layers          
           
            #init layers for CNN
            if conv == True:
                #if first == True:
                    #scales input to correct number of channels
                    #self.firstMask = nn.Conv2d(in_chns,n_filters,3, padding=1) 
                for i in range(0, N):                    
                    self.layers.append(nn.Conv2d(n_filters,n_filters,3, padding=1)) 
                    self.layers.append(nn.BatchNorm2d(n_filters))
                #classifier
                if last == True:
                    self.classifier = nn.Linear(num_features*n_filters,num_classes) 
            #init layers for FCN
            else:
                for i in range(0, N):
                    self.layers.append(nn.Linear(num_features*in_chns, num_features*in_chns))
                #classifier  
                if last == True:                  
                    self.classifier = nn.Linear(num_features*in_chns,num_classes)           
        
            
        def forward(self, x, step=0.1, plot=False):
  
            i = 0    
            #start list of coordinates        
            self.props = x
            
            #if using conv nets need to scale input up to use n_filters
            if self.conv and self.first:
                if self.in_chns == 1:
                    #for MNIST we have 1 channel so copy input channels x6
                    x = torch.cat((x,x,x,x,x,x), dim=1)   
                elif self.in_chns == 3:
                    # for CIFAR we have 3 channels so copy input channels x2
                    x = torch.cat((x,x), dim=1)
            
            #propagate through layers
            for layer in self.layers:
                #if self.directions is not None then PVD is used
                if self.directions == None or layer.weight.requires_grad == True:
                    #print(self.func_f(layer(x)).shape)
                    x = x + step*self.func_f(layer(x))
                else:
                    w , b = self.directions[i]                
                    x = x + step*(self.func_f(layer(x)) + x.matmul(w) + b)
                    #increment 
                    i += 1
                #store each layer output if needed
                if plot == True:
                    self.props = torch.cat((self.props, x), 0)
            #stores final output
            self.final = x.detach().to(torch.device("cpu")).numpy()
            #classifier 
            if self.last == True:
                if self.conv == True:
                      x = x.view(-1, self.num_features*self.n_filters)
                x = self.func_c(self.classifier(x), dim = 1)
           
            return x

            #can assign weights to nn.CrossEntropyLoss in the case of unbalanced training sets
        def train(self, trainloader, error_func=nn.CrossEntropyLoss(), learn_rate = 0.01, epochs = 10, begin = 0, end = 100
                                ,f_step = 0.1, reg_f=False, alpha_f=0.01, reg_c= False, alpha_c = 0.01, graph=False, mu = None, steps = 2):
            """
               trainloader - dataloader object that stores examples
               error_func - error function
               begin - starting batch number (i.e start at n'th batch)
               end - stop iteration after (end - begin) many batches
               reg_f - if true use regularisation on forward propagation
               alpha_f - regularisation parameter for forward prop regularisation
               reg_c - if true use regularisation on classifier
               alpha_c - regularisation parameter for classieir regularisation
               graph - if true shows loss curve
               mu - used for PVD
               steps - used for PVD 
               
            """
            #counters for time
            load_t = 0.0
            forward_t = 0.0
            back_t = 0.0
            view_t = 0.0
            
            #set up optimiser         
            params = filter(lambda param: param.requires_grad == True, self.parameters())     
                         
            #if using PVD a deepcopy of directions are made
            directions = copy.deepcopy(self.directions)          
            
            first_round = False
            
            if mu is not None:
                first_round = True
                params = [p for p in params] 
                params.append(mu)
            
            #create optimiser
            decay_rate = 0
            if reg_f == True:
                decay_rate = alpha_f           
              
            optimiser = optim.Adam(params, lr = learn_rate, weight_decay=decay_rate)           
            scheduler = optim.lr_scheduler.StepLR(optimiser, 5, 0.9)
            
            rounds, losses = [], []
            #train for epochs
            t_net = 0.0
            for epoch in range(epochs):
                #loss per epoch
                epoch_loss = 0
                i = 0              
                batch_t = time.perf_counter() 
                #use islice to sample from [begin] to [end] batches
                
                for inputs, labels in itertools.islice(trainloader, begin, end):
                    #get batch of input features and convert from (28*28->784 MNIST)           
                    #inputs, labels = next(iter_load)
                    
                    if first_round == True:   
                        self.directions = ResNet.mult_mu(copy.deepcopy(directions), mu, steps)
                                    
                    if self.gpu == True:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        torch.cuda.synchronize()
                   
                    if self.conv == False:                   
                        tmp_v = time.perf_counter()
                        #flatten image grid to row vector
                        inputs = inputs.view(-1, self.num_features)
                        view_t += time.perf_counter() - tmp_v
                                      
                    #time taken to load data
                    load_t += time.perf_counter() - batch_t
                  
                    t = time.perf_counter()
                    #clear gradient buffers
                    optimiser.zero_grad()     
                    torch.cuda.synchronize()
                    tmp_forward = time.perf_counter()    
                    outputs = self(inputs, f_step)
                  
                    loss = error_func(outputs, labels)
                    torch.cuda.synchronize()
                    forward_t += time.perf_counter() - tmp_forward
                    #add forward propagation regularisation term
                    #if reg_f == True:                        
                     #   loss += alpha_f*self.layer_reg(f_step)
                    #add classifier regularisation term
                    #if reg_c == True:
                     #   loss +=  alpha_c*self.class_reg()
                  
                    torch.cuda.synchronize()
                    tmp_b = time.perf_counter()
                    #calculate the gradients for the backpropagations
                    loss.backward()
               
                    #update the weights
                    optimiser.step()
               
                    torch.cuda.synchronize()
                    back_t += time.perf_counter() - tmp_b
                    epoch_loss += loss.item()
                    i += 1
                    t_net += time.perf_counter() - t
                    
                    batch_t = time.perf_counter() 
                    
                #multiplies direction by mu   
                if first_round == True:                 
                   self.directions = ResNet.mult_mu(copy.deepcopy(directions), mu, steps)
                   mu.requires_grad = False
                #scheduler.step()
                epoch_freq = i*trainloader.batch_size
                if directions is None:
                    print("epoch: ", epoch+1, "loss: ", epoch_loss/epoch_freq)
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
                print("in_net time", t_net) 
                print("batch_time", load_t)
                print("back_time", back_t)
                print("forward_time", forward_t)
                print("view_time", view_t)
            return epoch_loss/epoch_freq

        def validation(self, trainloader, start_v = 50000, m ="f_step", min=0.1, max=1.0 ,num = 10
                    ,error_func=nn.CrossEntropyLoss(), learn_rate = 0.01, epochs = 10, f_step = 0.1,
                    reg_f=True, alpha_f=0.01, reg_c= True, alpha_c = 0.01):

            #must divide the dataset into the training and validation sets
            num_samples = len(trainloader.dataset)

            #parameters for training
            begin_t = 0
            end_t = round(start_v/trainloader.batch_size)

            #parameters for validation
            begin_v = end_t + 1
            end_v = round(num_samples/trainloader.batch_size)
            # look at grad how to choose variable
            val_variable = []
            val_class = []           

            for i in torch.linspace(min, max, num):
                alpha = i.detach().item()
                print("alpha is: ", alpha)
                self.train(trainloader, error_func, learn_rate, epochs, begin_t, end_t
                                    ,f_step = f_step, reg_f = reg_f, alpha_f = alpha_f, reg_c= reg_c
                                    ,alpha_c = alpha_c)


                val_err = self.test(trainloader, begin_v, end_v)
                val_variable.append(i)
                val_class.append(val_err)
                print("alpha: ", alpha, " a_result: ", val_err)

                #reset parameters
                self.reset_params()
           
            print(val_variable, val_class)
            plt.plot(val_variable, val_class, '-')
            plt.grid()
            plt.xlabel("epochs")
            plt.ylabel("loss")           
            plt.show()
            plt.close()

            return val_variable, val_class

            #allow step to be chosen
        def test(self, testloader, begin = 0, end = 100, f_step =0.1):
            #counters for number of positive examples and total
            num_positive = 0.
            num_total = 0.
            #iterate through training set
            for inputs, labels in itertools.islice(testloader, begin, end):

                if self.gpu == True:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.conv == False:
                    #convert to vector
                    inputs = inputs.view(-1, self.num_features)
                #propagate through network
                outputs = self(inputs, step = f_step)

                #get predictions- by getting the indices corresponding to max arguments
                pred = torch.max(outputs, 1)[1]
                
                #get number of correct examples in batch
                correct = (pred == labels).sum().item()

                #add to the counters
                num_total += labels.size(0)
                num_positive += correct


            #return total , number positive, classificaiton rate
            return [num_total, num_positive, 100*(num_positive/num_total)]


        def layer_reg(self, step):
            #regulariser terms for layers and biases
            reg_w = 0
            reg_b = 0
            for i in range(1, self.N):
                #add terms [current layer- previous layer]
                w_diff = self.layers[i].weight-self.layers[i-1].weight
                b_diff = self.layers[i].bias-self.layers[i-1].bias

                reg_w += torch.sum(torch.pow(w_diff, 2))
                reg_b += torch.sum(torch.pow(b_diff, 2))

            reg = (0.5/step)*(reg_w+reg_b)
            return reg#.item()

        def class_reg(self):
            """regularisation for the classifier
             """
            #calculate the linear operator difference i.e w0-w1,...,wN-w0
            w_diff = self.classifier.weight - self.roll(self.classifier.weight)
            b_diff = self.classifier.bias - self.roll(self.classifier.bias)

            #regularisation term is the sum or the squares
            reg = 0.5*(torch.sum(torch.pow(w_diff, 2)) + torch.sum(torch.pow(b_diff, 2)))
            return reg
        
        def reg_L2(self):
            reg = 0.0
            for p in self.parameters():
                reg += torch.sum(torch.pow(p,2))
            return reg
            

        def reset_params(self):
            for i in range(len(self.layers)):
                self.layers[i].reset_parameters()
            self.classifier.reset_parameters()

        def freeze_layer(self, i):
            self.layers[i].weight.requires_grad = False
            self.layers[i].bias.requires_grad = False

        def unfreeze_layer(self, i):
            self.layers[i].weight.requires_grad = True
            self.layers[i].bias.requires_grad = True

        def freeze_classifier(self):
            self.classifier.weight.requires_grad = False
            self.classifier.bias.requires_grad = False

        def unfreeze_classifier(self):
            self.classifier.weight.requires_grad = True
            self.classifier.bias.requires_grad = True

        def freeze_range(self, start, stop):
            #freeze layers in range
            for i in range(start, stop):
                self.freeze_layer(i)

        def unfreeze_range(self, start, stop):
            #unfreeze layers in range
            for i in range(start, stop):
                self.unfreeze_layer(i)

        def freeze_all(self):
            #freeze all layers
            self.freeze_range(0, self.N)
            #freeze classifier
            self.freeze_classifier()

        def unfreeze_all(self):
            #unfreeze all layers
            self.unfreeze_range(0, self.N)
            #unfreeze classifier
            self.unfreeze_classifier()

        def get_not_frozen(self):
                params = filter(lambda param: param.requires_grad == False, self.parameters())
                return params

        def get_frozen(self):
                params = filter(lambda param: param.requires_grad == True, self.parameters())
                return params

        def plotter(self, x, step=0.1):
            points = []
            for i in x:
                i=i.reshape(1,len(i))
                self.forward(i, step, plot=True)
                points.append(self.props)
                #print(self.props)
            j = 0
            for i in points:
                results = i.detach().numpy()
                print(results)
                plt.plot(results[:,0], results[:,1], '-', label=str(j))
                j += 1
            plt.grid()
            plt.show()

            """TO DO:
            -fix only 9000 validation
            
            """

        def roll(self, mat):
            #get indexs then swap around
            indexes = torch.range(1,mat.shape[0], dtype = torch.long)
            indexes[-1] = 0
            mat = mat[indexes]

            return mat

        def double_layers(self):
            """
            doubles the number of layers
            """
            end = len(self.layers)
            if self.conv == False:
                end = len(self.layers)*2
            #make list of layers because moduleList does not have insert
            layers = [layer for layer in self.layers]

            for i in np.arange(0, end, 2):
                #case for convolutional neural networks
                if self.conv == True:
                    layers.insert(i, nn.Conv2d(self.n_filters,self.n_filters,3, padding=1))
                    layers.insert(i+1, nn.BatchNorm2d(self.n_filters))
                else:            
                    layers.insert(i, nn.Linear(self.num_features, self.num_features))
            
            #re-encapsulate layers within container
            self.layers = nn.ModuleList(layers).to(self.device)
            end -= 1
           
            for i in np.arange(2, end, 2):                
                if self.conv == True:
                    self.layers[i].weight.data = 0.5*(layers[i+2].weight.data+layers[i-2].weight.data)
                    self.layers[i].bias.data = 0.5*(layers[i+2].bias.data+layers[i-2].bias.data)
                else:    
                    self.layers[i].weight.data = 0.5*(layers[i+1].weight.data+layers[i-1].weight.data)
                    self.layers[i].bias.data = 0.5*(layers[i+1].bias.data+layers[i-1].bias.data)
                

        def set_net_params(self, parameters):
            #The parameters returns the weights then parameters for each layer
            for i, p in enumerate(parameters):
                weight, bias = p
                self.layers[i].weight.data = weight.data            
                self.layers[i].bias.data = bias.data                    
       
        def set_classifier_params(self, params):
            weight, bias = params
            self.classifier.weight.data = weight.data
            self.classifier.bias.data = bias.data
        
        def get_net_params(self):
            params = []
            for layer in self.layers:
                params.append([layer.weight.detach().cpu(), layer.bias.detach().cpu()])
            if self.last == True:
                #if last net return layers and classifier
                return params, [self.classifier.weight.detach().cpu(), self.classifier.bias.detach().cpu()]
            #if not last net then return layers
            return params
        
        def set_device(self, device):
            self.device = device
            
        
        def mult_mu(direction, mu, steps):
            index ,counter = 0, 0
            #iterate through the directions    
            for j in range(len(direction)):
                #multiply weight direction by mu
                direction[j][0] *= mu[index]
                #multiply bias direction by mu
                direction[j][1] *= mu[index]
                #increment counter
                counter += 1
                #every (steps) many iterations we move to the next block
                if counter == steps:
                    index += 1
                    counter = 0

            return direction
