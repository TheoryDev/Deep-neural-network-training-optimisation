"""
This script downloads the MNIST, CIFAR10 and CIFAR100 datasets and converts them 
to the HDF5 format and saves the result.
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
    
    
    #THE CODE BELOW WILL BE USED LATER FOR CNNs
    """
    #store MNIST in HDF5
    myMemLoader = dl.InMemDataLoader(dataset= "MNIST", conv_sg=True)
    myMemLoader.storeAsHDF5()
    print("MNIST stored in HDF5")   
    
    
    #store CIFAR 10
    myMemLoader = dl.InMemDataLoader(dataset= "CIFAR10", conv_sg=True)
    myMemLoader.storeAsHDF5()
    print("CIFAR10 stored in HDF5")
    
    #STORE CIFAR 100
    myMemLoader = dl.InMemDataLoader(dataset= "CIFAR100", conv_sg=True)
    myMemLoader.storeAsHDF5()
    print("CIFAR100 stored in HDF5")
    """
    
if __name__ == '__main__':
    main()