# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:07:02 2018

@author: Corey

"""

def main():

    import dataloader as dl
    
    #store MNIST in HDF5
    myMemLoader = dl.InMemDataLoader(dataset= "MNIST")
    myMemLoader.storeAsHDF5()
    print("MNIST stored in HDF5")
    
    #store CIFAR 10
    myMemLoader = dl.InMemDataLoader(dataset= "CIFAR10")
    myMemLoader.storeAsHDF5()
    print("CIFAR10 stored in HDF5")
    
    #STORE CIFAR 100
    myMemLoader = dl.InMemDataLoader(dataset= "CIFAR100")
    myMemLoader.storeAsHDF5()
    print("CIFAR100 stored in HDF5")
    
    
if __name__ == '__main__':
    main()