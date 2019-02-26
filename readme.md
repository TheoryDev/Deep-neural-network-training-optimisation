# Dependencies

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

# Summary
This project initially started as my MSc project at Imperial College London with Dr Panos Parpas as my supervisor and it has since
been extended with further research. In the core component of the project, I implemented the Stable Deep Neural Network (DNN) Architectrues
that are described in (Haber, Ruthotto et at., 2017) which can be found at [https://arxiv.org/abs/1705.03341]. The MSc project culminated with the
an implementation of the Parallel Variable Distribution (PVD) (Solodov, 1997) located at [http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.158.2869&rank=4] 
to parallelise the training of the stable DNNs across multiple processors and this resulted in a significant theoretical speed-up. 
The extension is ongoing work and features the use of synthetic gradients (Jaderberg et al., 2016) which can be found at [https://arxiv.org/abs/1608.05343] 
to parallelise the DNN training algorithm and achieve significant speed-ups while still achieving acceptable levels of performance. 
The synthetic gradients have been incorporated in a distributed implementation using Python's multiprocessing library.   

# Code Base

##Core Modules:
ResNet.py
Antisymmetric.py
Leapfrog.py
ellipse.py
ellipse.py - also has script - main() inside - run ellipse example
swiss.py- also has script - main() inside - run swiss roll example

## Core Scripts:

The core scripts 

verletTest.py - MNIST for verlet
resTest.py - MNIST for resNet
leapTest.py - MNIST for Leapfrog
antiTest.py - MNIST for antisymmetric

##PVD Scripts
PVD.py - also has script - main() inside - run pvd on ellipse, swiss roll, or MNIST

The files are generally quite well commented and easy to understand:

ResNet.py
---------------------------------------------------------------------
The core module is ResNet and it contains the ResNet class:
This is the base neural network that the other three architectures 
inherit from. The core functions are the forward propagation 
(which also includes the classifier step) and its training algorithm.
The only changes that need to be made to build a new deep neural network
architecture dervived from it is with the forward propagation when inheriting
from the class. 

The ResNet class is initialised with:
            N - number of layers
            num_features - number of inputs
            num_classes - number of outpurs
            func_f - activation function
            weights - weight matrix if none then a random initlisation will be used
            bias = - bias vector if none then a random initilisation with be used


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


