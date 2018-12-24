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
        def __init__(self, device, N, num_features, num_classes, func_f, func_c, weights = None, bias = None, gpu=False, last=True):
            """
            It is initialised with
            N - number of layers
            num_features - number of inputs
            num_classes - number of outpurs
            func_f - activation function
            weights - weight matrix if none then a random initlisation will be used
            bias = - bias vector if none then a random initilisation with be used
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
            #add N layers
            for i in range(0, N):
                self.layers.append(nn.Linear(num_features, num_features))#, requires_grad = requires_grad))
                #allows for choosing initial weights manually
                if (weights is not None) and (bias is not None):
                    self.layers[i].weight = torch.nn.Parameter(weights)
                    self.layers[i].bias = torch.nn.Parameter(bias)

            #classifier
            if last == True:
                self.classifier = nn.Linear(num_features,num_classes)
                self.last = last
            self.gpu = gpu


        def forward(self, x, step=0.1, plot=False):
            #forward propagation
            i = 0

            self.props = x
            #self.step = step
            for layer in self.layers:
                if self.directions == None or layer.weight.requires_grad == True:
                    x = x + step*self.func_f(layer(x))
                else:
                    w , b = self.directions[i]                
                    x = x + step*(self.func_f(layer(x)) + x.matmul(w) + b)
                    #increment i
                    i += 1
                #store each layer output if needed
                if plot == True:
                    self.props = torch.cat((self.props, x), 0)
            #stores final output
            self.final = x.detach().to(torch.device("cpu")).numpy()
            #classifier 
            if self.last == True:
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
            load_t = 0.0
            forward_t = 0.0
            back_t = 0.0
            
            
            #set up optimiser         
            params = filter(lambda param: param.requires_grad == True, self.parameters())                
                    
            directions = copy.deepcopy(self.directions)
            #self.directions = None
            
            first_round = False
            
            if mu is not None:
                first_round = True
                params = [p for p in params] 
                params.append(mu)
            
            #create optimiser
            optimiser = optim.SGD(params, lr = learn_rate)
            
            rounds, losses = [], []
            #train for epochs
            t_net = 0.0
            for epoch in range(epochs):
                #get loss per epoch
                epoch_loss = 0
                i = 0              
                batch_t = time.perf_counter() 
                #use islice to sample from [begin] to [end] batches
                for inputs, labels in itertools.islice(trainloader, begin, end):
                    #get batch of input features and convert from (28*28->784 MNIST)                
                    
                    
                    
                    if first_round == True:   
                        self.directions = ResNet.mult_mu(copy.deepcopy(directions), mu, steps)
                                    
                    if self.gpu == True:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                    inputs = inputs.view(-1, self.num_features)
                    
                    #print("inputs", inputs.shape)
                    #print("labels", labels.shape)
                    
                    
                    load_t += time.perf_counter() - batch_t

                    t = time.perf_counter()
                    #clear gradient buffers
                    optimiser.zero_grad()                 
                        
                    outputs = self(inputs, f_step)
                   # print("o:", outputs.shape)
                    loss = error_func(outputs, labels)

                    #add forward propagation regularisation term
                    if reg_f == True:
                        #reg_loss =  alpha*self.layer_reg(f_step) + alph
                        loss += alpha_f*self.layer_reg(f_step)
                    #add classifier regularisation term
                    if reg_c == True:
                        loss +=  alpha_c*self.class_reg()
                  #  print("before backward")
                  #  for p in params:
                  #      print(p)
                    tmp_b = time.perf_counter()
                    #calculate the gradients for the backpropagations
                    loss.backward()
                 #   print(loss.grad)
                 #   print("after backward")
                #    for p in params:
                #        print(p)
                    #update the weights
                    optimiser.step()
                #    print("after optimiser step")             
                 #   for p in params:
                 #       print(p)
                    back_t += time.perf_counter() - tmp_b
                    epoch_loss += loss.item()
                    i += 1
                    t_net += time.perf_counter() - t
                    
                    batch_t = time.perf_counter() 
                    #print("i", i)
                    
                if first_round == True:                 
                   self.directions = ResNet.mult_mu(copy.deepcopy(directions), mu, steps)
                   mu.requires_grad = False
                
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
                #convert to vector
                inputs = inputs.view(-1, self.num_features)
                #propagate through network
                outputs = self(inputs, step = f_step)

                #get predictions- by getting the indices corresponding to max arguments
                pred = torch.max(outputs, 1)[1]
                #print(inputs, labels, pred)
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
            -set flags to inputs to gpu if not cpu
            -set device as an instance object then use self.to(device)
            -then no need to pass parameters just use self.device
            -lower number of reset_parameters
            -validation make selection-possibly function
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
            end = len(self.layers)*2
            #make list of layers because moduleList does not have insert
            layers = [layer for layer in self.layers]

            for i in np.arange(0, end, 2):
            #    print("adding")
                layers.insert(i, nn.Linear(self.num_features, self.num_features))
            #re-encapsulate layers
            self.layers = nn.ModuleList(layers)
            end -= 1

            for i in np.arange(2, end, 2):
                layers[i].weight.data = 0.5*(layers[i+1].weight.data+layers[i-1].weight.data)
                layers[i].bias.data = 0.5*(layers[i+1].bias.data+layers[i-1].bias.data)


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
