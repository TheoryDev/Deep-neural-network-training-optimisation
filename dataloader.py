import numpy as np
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

import ellipse as el
import swiss as sw


class InMemDataLoader:   
    """
    This class is used to download the datasets and converts them into the HDF5 format. 
    The new files will then be used to hold the dataset in memory and reduce the taken to load data 
    during training
    """

    def __init__(self, dataset = 'MNIST', driver = None, root = './data/', conv_sg=False):
        self.driver = driver
        self.dataset = dataset
        self.root = root
        self.conv_sg = conv_sg
        if conv_sg == True:
            self.root = self.root + "conv"   
            #self.dataString = root + "sg" + dataset + ".h5"
        #else:
        self.dataString = root  + dataset + ".h5"     
        
    def loadData(self):
        """
        This loads the dataset and creates the train loader and test loaders
        """
        batch_size = 256
        
        #if self.conv_sg == True:
        #    batch_size = 1        
        
        download = True
        root = self.root + self.dataset
        if self.dataset == "MNIST": 
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            trainset = torchvision.datasets.MNIST(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.MNIST(root, train=False, download=download, transform=transform)
        
        if self.dataset == "CIFAR10":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010,))])
            trainset = torchvision.datasets.CIFAR10(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.CIFAR10(root, train=False, download=download, transform=transform)
        
        if self.dataset == "CIFAR100":
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.CIFAR100(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.CIFAR100(root, train=False, download=download, transform=transform)
            
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                      shuffle=False, num_workers=0, pin_memory = False)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size,
                                             shuffle=False, num_workers=2, pin_memory = False)
        
        return trainloader, testloader
    
    def storeAsHDF5(self):
        
        myFile = h5py.File(self.dataString, 'w', driver= self.driver)
        
        num_classes = 10
        
        if self.dataset == "CIFAR100":
            num_classes = 100           
        
        trainloader, testloader = self.loadData()       
        print("downloading done")
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            #if self.conv_sg == True:
                #tmp = torch.zeros(inputs.shape).float()
                #labels = tmp + labels[0].float()/num_classes
            
            #print(i)
            if i == 0:               
                store_inputs = inputs
                store_labels = labels
                continue
            
            store_inputs = torch.cat((store_inputs, inputs))
            store_labels = torch.cat((store_labels, labels))           
        
        myFile.create_dataset("train_labels", data = store_labels.numpy())    
        myFile.create_dataset("train_inputs", data = store_inputs.numpy(), dtype=np.float)     
        
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            
           # if self.conv_sg == True:
             #   tmp = torch.zeros(inputs.shape).float()
            #    labels = tmp + labels[0].float()/num_classes
            
           # print(i)
            if i == 0:                
                store_inputs = inputs
                store_labels = labels
                continue
            
            store_inputs = torch.cat((store_inputs, inputs))
            store_labels = torch.cat((store_labels, labels))    
            
            
        myFile.create_dataset("test_labels", data = store_labels.numpy())    
        myFile.create_dataset("test_inputs", data = store_inputs.numpy(), dtype=np.float)        
        
        myFile.close()
        
    def getDataset(self, train=True):
        """
        returns the dataset for the given database, for test dataset set train = False
        """
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        
        if self.dataset == "ELLIPSE":
            a = np.array([[0,1.0],[1.0,2.0]])            
            b = a*0.5                            
            myE = el.ellipse(device, 500, 100, a, b)            
            if train == True:
                return myE.create_dataset(myE.examples)
            return myE.create_dataset(myE.valid)                     
    
        if self.dataset == "SWISS":            
            myS = sw.SwissRoll(device, 500, 0.2)            
            if train == True:
                return myS.create_dataset(myS.examples)
            return myS.create_dataset(myS.valid)
                       
               
        #open file
        myFile = h5py.File(self.dataString, 'r', self.driver)
        
        if train == True:         
            inputString = "train_inputs"
            labelsString = "train_labels"
        
        else:
            inputString = "test_inputs"
            labelsString = "test_labels"
        
        #get hdf5 datsets
        features = myFile.get(inputString)
        labels = myFile.get(labelsString)
       
        #convert to tensors
        features = torch.from_numpy(np.array(features))
        labels = torch.from_numpy(np.array(labels))
        
        #close file to ensure dataset is in memory
        myFile.close()
        
        #conver to correct datatypes
        features = features.float()
        
        if self.conv_sg == False:
            labels = labels.long()       
                     
        dataset = torch.utils.data.TensorDataset(features, labels)
        
        return dataset
        
    
    def getDataLoader(self, batch_size = 64, shuffle=True, num_workers=0, pin_memory=True, train=True):
        dataset = self.getDataset(train)
       
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle= shuffle
                                              , num_workers=num_workers, pin_memory = pin_memory)
            

def getDims(dataset):
    
    in_channels = 1
    
    if dataset == "ELLIPSE":
        num_classes = 2
        num_features = 2        
   
    if dataset == "SWISS":
        num_classes = 2
        num_features = 4
        
    if dataset == "MNIST":
        num_classes = 10
        num_features = 784
        
    if dataset == "CIFAR10":
        num_classes = 10
        num_features = 1024 
        in_channels = 3
        
    if dataset == "CIFAR100":
        num_classes = 100
        num_features = 1024  
        in_channels = 3
   
    return num_features, num_classes, in_channels



"""
To do 
    add support for labels for convnets
    add support for transforms

"""