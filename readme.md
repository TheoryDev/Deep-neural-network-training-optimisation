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
been extended with further research. In the core component of the project, I implemented the Stable Deep Neural Network (DNN) Architectures
that are described in (Haber, Ruthotto et at., 2017) which can be found at [https://arxiv.org/abs/1705.03341]. The MSc project culminated with an implementation of the Parallel Variable Distribution (PVD) (Solodov, 1997) located at [http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.158.2869&rank=4] to parallelise the training of the stable DNNs across multiple processors and this resulted in a significant theoretical speed-up. Note, that a multilevel training scheme was used to initialise the network weights.

The extension is ongoing work and features the use of synthetic gradients (Jaderberg et al., 2016) which can be found at [https://arxiv.org/abs/1608.05343] to parallelise the DNN training algorithm and achieve significant speed-ups while still achieving acceptable levels of performance. The speed-ups were further increased by implementing a multilevel training scheme to initialise the fine model using a coarse model. The synthetic gradients have been incorporated in a distributed implementation using Python's multiprocessing library.   


Core Scripts:

To train and test the models run the scripts, you can change the variables in the scripts (see comments)

verletTest.py - MNIST for verlet

resTest.py - MNIST for resNet

leapTest.py - MNIST for Leapfrog

antiTest.py - MNIST for antisymmetric

ellipse.py - also can run- main() inside - run ellipse example with different DNN architectures

swiss.py- also can run - main() inside - run swiss roll example with different DNN architectures

PVD:
PVD.py - contains the PVD implementation and can run- main() function to run pvd on ellipse, swiss roll, or MNIST datasets

Training with synthetic gradients:

run_standard_sg.py - runs training algorithm

testParallelNet.py - runs training algorithm in series with multilevel training scheme

distSg.py - runs distributed training algorithm  

distMulti.py - runs training algorithm across with multilevel training scheme
