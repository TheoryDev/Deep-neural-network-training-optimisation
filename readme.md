# Optimising the Training of Stable Deep Neural Network Architectures using Synthetic Gradients in PyTorch

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
distMulti.py  
```

##The PVD code was not distributed across multiple processors but theoretical training times were calculated.

```
PVD.py trains the DNN using the PVD algorithm. 
```

