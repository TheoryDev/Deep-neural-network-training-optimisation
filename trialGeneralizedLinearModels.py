# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:26:55 2018

@author: koryz
"""

#generalised linear models test


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
import itertools


from sklearn import linear_model

reg = linear_model.LinearRegression(normalize=True)

reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
                                    
h = np.random.uniform(size=(10,2))
y = np.random.uniform(size=(10,1))

grad = np.random.uniform(size=(10,2))

reg.fit(np.concatenate((h,y), axis=1), grad)



reg.coef_

a=reg.predict(np.concatenate((h,y), axis=1))

print(mse(a, grad))
