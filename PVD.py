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
        #self.models = self.set_models(N, num_features, num_classes, func_f, func_c, weights, bias, gamma, gpu)
        #self.models = []

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
        #self.models[-1].unfreeze_classifier()
        #made classifier get backproped in all cases only first- messed up the mu

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

        print("loss is:" ,loss)

        loss.backward()

        directions = []

        for layer in model.layers:
            #if layers.requires_grad == True
    #        print("gradw", layer.weight.grad, "gradb", layer.bias.grad)
            directions.append([layer.weight.grad, layer.bias.grad])

        #directions.append([model.classifier.weight.grad, model.classifier.bias.grad])

    #    print("direct", directions)

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
        
      #  for model in self.models:
      ##      for layer in model.layers:
        #        print (layer.weight.requires_grad)
       #     print (model.classifier.weight.grad)
        
        #run pvd for iterations
        for i in range(iterations):         
            
            #calculate list of directions
            direction_block = self.gen_directions(self.direction_model, trainloader, error_func, f_step, plot = graph)
       
            print("i------------------------------------", i+1)
           
            #train each model in loop using error as objective function           
            #train each model and calculate errors
            results = []
            
            #parallelisation---------------------------------------------------
            for m in self.models:
                
                r = self.pvd_process(m, direction_block,trainloader, error_func, learn_rate, epochs, begin, end
                                        ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph)
                results.append(r)

            #synchronisation---------------------------------------------------

            synch_time = time.time()

            #get minimum error
            errors = np.array([r[0] for r in results])
            min_index = errors.argmin()


            #now get directions
            direction, mu = results[min_index][1], results[min_index][2]
            
            print("min", min_index)
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
                synch_time = time.time() - synch_time 
                timer += synch_time + p_time
                print(synch_time, p_time)
                print ("PVD finished", acc)
                return timer
          
            #synchronise the modesl
            self.copy_model(min_index)
            synch_time = time.time() - synch_time
            print(synch_time, p_time) 
            timer += synch_time + p_time
            
        return timer

    def pvd_process(self, model, direction_block ,trainloader, error_func, learn_rate, epochs, begin, end
                            ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph):
     
        #get time of process
        run_time = time.time()
        #gets block of directions and set as instance of model
   
        directions = self.d_block(direction_block, model)
    
        steps = np.round(model.N/self.num_proc)
        mu = torch.rand((self.num_proc-1), requires_grad = True)

        if model.gpu == True:
            mu = torch.rand((self.num_proc-1), requires_grad = True, device = "cuda")
      

        model.directions = copy.deepcopy(directions)

        #train model by optimising weights and also mu
        t = time.time()
        error = model.train(trainloader, error_func, learn_rate, epochs, begin, end
                                ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph, mu, steps)
        print("t: ",time.time()-t)
        run_time = time.time() - run_time
      
        print("mu", mu)
        print("run time of processor is ", run_time)
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
        print("start", start, "end", end)
        
        d = directions[:start] + directions[end:]
#        print("d", d)
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

    num_proc =  2
    N = 4
    num_features = 2
    num_classes = 2

    func_f = torch.tanh
    func_c =F.softmax
    weights = None
    bias = None
    error_func=nn.CrossEntropyLoss()


    gpu=False
    choice = 'r'
    gamma = 0.1

    learn_rate = 0.05
    epochs = 100
    begin = 0
    end = 100
    f_step =0.1


    reg_f=False
    alpha_f=0.01
    reg_c= False
    alpha_c = 0.01
    graph=False
    acc = 85

    iterations = 1000

    #set up dataset-ellipse problem
    a = np.array([[0,0.1],[0.1,0.2]])
    a = a*10
    b = a*0.5

    
    
    E = False
    S = False
    MNIST = True

    if E:
        myE = el.ellipse(device, 500, 100, a, b)
        dataset = myE.create_dataset(myE.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)
        datasetT = myE.create_dataset(myE.valid)
        testloader = torch.utils.data.DataLoader(datasetT, batch_size = 10)

    if S:
        num_features = 4
        myS = sr.SwissRoll(device, 500, 0.2)
        dataset = myS.create_dataset(myS.examples)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)
        datasetT = myS.create_dataset(myS.valid)
        testloader = torch.utils.data.DataLoader(datasetT, batch_size = 10)

    


    
    if MNIST:
        num_features = 784
        num_classes = 10
        batch_size = 250
        begin = 0
        end = 100
        gpu = True
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                  shuffle=False, num_workers=8, pin_memory = True)
    
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                                 shuffle=False, num_workers=8, pin_memory = True)

    pvd = PVD(device, num_proc)
    pvd.set_models(N, num_features, num_classes, func_f, func_c, weights, bias, gpu, choice, gamma)

    start_time = time.time()  

    timer =  pvd.pvd_algo(iterations, trainloader, testloader, error_func, learn_rate, epochs, begin, end
                            ,f_step, reg_f, alpha_f, reg_c, alpha_c, graph, acc)

    print("new_time", timer)
    print("time_old: ", (time.time()-start_time)/num_proc)
    result = pvd.models[0].test(testloader, begin, end, f_step)
    print("result :", result)
    print("iterations", iterations, "num processors", num_proc, "num layers", N, "epochs", epochs, "acc", acc)
 

if __name__ == '__main__':
    main()
