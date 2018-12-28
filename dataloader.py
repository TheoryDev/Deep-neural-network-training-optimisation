# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 22:36:31 2018

@author: Corey

"""

import numpy as np
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

import ellipse as el
import swiss as sw


class InMemDataLoader:    

    def __init__(self, dataset = 'MNIST', driver = None, root = './data/'):
        self.driver = driver
        self.dataset = dataset
        self.root = root
        self.dataString = root  + dataset + ".h5"     
        
    def loadData(self):
        """
        This loads the dataset and creates the train loader and test loaders
        """
        download = True
        root = self.root + self.dataset
        if self.dataset == "MNIST": 
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.MNIST(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.MNIST(root, train=False, download=download, transform=transform)
        
        if self.dataset == "CIFAR10":
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.CIFAR10(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.CIFAR10(root, train=False, download=download, transform=transform)
        
        if self.dataset == "CIFAR100":
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.CIFAR100(root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.CIFAR100(root, train=False, download=download, transform=transform)
            
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100,
                                                      shuffle=False, num_workers=0, pin_memory = False)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size= 100,
                                             shuffle=False, num_workers=2, pin_memory = False)
        
        return trainloader, testloader
    
    def storeAsHDF5(self):
        
        myFile = h5py.File(self.dataString, 'w', driver= self.driver)
        
        trainloader, testloader = self.loadData()       
        print("downloading done")
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            #print(i)
            if i == 0:               
                store_inputs = inputs
                store_labels = labels
                continue
            
            store_inputs = torch.cat((store_inputs, inputs))
            store_labels = torch.cat((store_labels, labels))           
        #print("loop done")
        myFile.create_dataset("train_labels", data = store_labels.numpy())    
        myFile.create_dataset("train_inputs", data = store_inputs.numpy(), dtype=np.float)     
        
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            #print(i)
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
        
        
        #put code for Ellipse/Swiss roll
        if self.dataset == "EllIPSE":
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
        labels = labels.long()
        
        dataset = torch.utils.data.TensorDataset(features, labels)
        
        return dataset
        
    
    def getDataLoader(self, batch_size = 64, shuffle=True, num_workers=0, pin_memory=True, train=True):
        dataset = self.getDataset(train)
       
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle= shuffle
                                              , num_workers=num_workers, pin_memory = pin_memory)
            


"""
To do 
    - 1 . load dataset from torch vision
    - 2 . save in hdf5 format
    - 3 . re-open in memory
    - 4 . return dataloader to use in standard PyTorch code
    
    - flags if exists

"""