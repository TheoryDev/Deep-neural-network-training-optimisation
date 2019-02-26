import numpy as np
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

download = True


myFile = h5py.File('./data/MNIST.h5', 'w', driver='core')

#'myFile:write('./data/write.h5', torch.rand(5, 5))
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=download, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64,
                                              shuffle=False, num_workers=0, pin_memory = True)

first = False

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    
    if first == False:
        first = True
        store_inputs = inputs
        store_labels = labels
        continue
    
    store_inputs = torch.cat((store_inputs, inputs))
    store_labels = torch.cat((store_labels, labels))    


myFile.create_dataset('MNIST_train_labels', data = store_labels.numpy())    
myFile.create_dataset('MNIST_train_inputs', data = store_inputs.numpy(), dtype=np.float)       

myFile.close()    
    