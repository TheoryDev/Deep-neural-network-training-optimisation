#Optimising the Training of Stable Deep Neural Network Architectures using Synthetic Gradients in PyTorch

This work is an extension of the Thesis component of MSc Computing Science which I studied at Imperial College London. 
The original project focused on implementing stable deep neural network architectures and optimising 
the training process's computational efficiency using the parallel variable distribution (PVD) algorithm.

The focus has shifted to achieving parallelisation by utilising synthetic gradients that work as seen in the paper ***[Decoupled Neural Interfaces using Synthetic Gradients](http://arxiv.org/abs/1608.05343)***.
In the training process for neural networks, the input data is first propgated in the forward direction, then the error is calculated and finally the error gradients are propagated backwards
through the network and the weights are updated. However, the process is locked as layers (which can be grouped in "modules") must wait for both input features 
to flow through earlier sections of the network and for the error gradient to propagate backwards through the layers ahead. This results in forward and backward locking.

The synthetic gradients speed-up backpropagation by approximating the error between modules of layers and this unlocks the backward connection
and allows for the distribution of training across multiple processors.

For this project you will need Python 3 and the following libraries:
Pytorch 0.4.1
torch
torchvision
autograd
multiprocessing
h5py
sklearn
numpy
scipy
os

Modules:
ResNet.py
Antisymmetric.py
Leapfrog.py
ellipse.py

PVD.py - also has script - main() inside - run pvd on ellipse, swiss roll, or MNIST
ellipse.py - also has script - main() inside - run ellipse example
swiss.py- also has script - main() inside - run swiss roll example


##Scripts for synthetic gradients:

The scripts below build, train and test a DNN. The user is free to specify the 
DNN architecture, hyperparameters and dataset within the script.

### Runs standard DNN with no synthetic gradients
```
fullmodel.py  
```
### Distributes training over multiple processes
```
distSg.py 
```
### Distributes training over multiple processes and also uses a multilevel learning scheme

```
distMult.py  
```

##The PVD code was not distributed across multiple processors but theoretical training times were calculated.

```
PVD.py trains the DNN using the PVD algorithm. 
```

The function train() is used to train deep neural networks for a chosen 
number of epochs of training. It applies the backpropagation algorithm
parameters:   trainloader - dataloader object that stores examples
           error_func - error function
           begin - starting batch number (i.e start at n'th batch)
           end - stop iteration after (end - begin) many batches
           reg_f - if true use regularisation on forward propagation
           alpha_f - regularisation parameter for forward prop regularisation
           reg_c - if true use regularisation on classifier
           alpha_c - regularisation parameter for classifier regularisation
           graph - if true shows loss curve
           mu - used for PVD
           steps - used for PVD 


ParallelNetworks.py
-----------------------------------------------------------------------
This module contains the complexNeuralNetwork class and it is used to model
a complex neural network of sub-neural networks and synthetic gradient modules.

The complexNeuralNetwork class is initialised with:
	device - i.e cpu or gpu
	M - the number of neural networks, there will be (M-1) synthetic gradient modules
	
	The class has methods to train the models using synthetic gradients. 
	

----------------------------------
The PVD code is not used for synthetic gradients
-----------------------------------------------------------------------


