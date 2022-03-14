#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:56:44 2020

@author: QI
"""
# mnist data:
input_layer_size  = 400;                                                       # 20x20 Input Images of Digits
num_labels = 10;                                                               # 10 labels, from 1 to 10

import os
from scipy.io import loadmat
import numpy as np

#%% import data
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
#print(filename) #['ex3data1.mat', 'ex2data2.txt', 'ex2data1.txt', 'ex3weights.mat', 'ex1data1.txt', 'ex1data2.txt']
rawdata = loadmat(codepath+'/RawData/ex3data1.mat')
X = rawdata['X']
y = rawdata['y']

#%% plot bunch of datas （mnist)
from random import sample
from matplotlib import pyplot as plt
selectednumber = 25
seletedfigures = sample(list(range(X.shape[0])), selectednumber)
seletedfigures[0] = 0
fig = plt.figure()
fig.subplots_adjust(hspace=0, wspace=-0.7)
for i in range(selectednumber):
    ax = fig.add_subplot(selectednumber**0.5, selectednumber**0.5, i+1)
    ax.imshow(np.reshape(X[seletedfigures[i],:], (20,20), order = 'F'), interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
    
#%% multiclass classifier
import pandas as pd
dummy_y = pd.get_dummies(y.flatten())

def Gradient(theta, X, y): ## the only change from linear reg
    n = len(X)
    designmatrix = np.append(np.ones([n,1]), X, axis=1)
    yfit = (1+np.exp(-designmatrix.dot(theta)))**(-1)
    ydiff = yfit - y
    gradient = (designmatrix.T).dot(ydiff)/n
    return gradient

def GradientDescentMultiClassifier(X, y, alpha, iterations):
    """ Gradient descent
    Args:
        X, y: matrix
        alpha: learning rate
        iterations: max number of iterations
    Returns:
        theta: minimized point
    Required func:
        FeatureNorm - NormReverse
        Gradient: gradient calculation
        Cost
    """
    n,m = X.shape
    _,c = y.shape
    theta = np.zeros((m+1,c))
    for i in range(iterations):
        gradient = Gradient(theta, X, y)
        theta = theta - alpha*gradient
    return theta

theta= GradientDescentMultiClassifier(X, dummy_y, alpha=0.3, iterations=10000)
print(theta)

# prediction
x_new = np.reshape(np.append(1,X[0,:]), (1,401))
y_pred_prob = (1+np.exp(-x_new.dot(theta)))**(-1)
y_pred = np.argmax(y_pred_prob)
print(y_pred)


#%% Feedforward Propagation
import pandas as pd
dummy_y = pd.get_dummies(y.flatten())

def ForwardPropagation(X, theta, g, same_activation = True):
    """
    Parameters
    ----------
    X : numpy matrix
        data input.
    theta : a list of numpy matrix
        parameters of each layer.
    g : a list of functions
        activation functions.

    Returns
    -------
    output : list [a0,a1,a2,...]
    """
    layer_num = len(theta)
    output = []
    tempX = X
    for i in range(layer_num):
        tempX = np.append(np.ones([tempX.shape[0],1]), tempX, axis=1)
        output.append(tempX)
        tempX = tempX.dot(theta[i].T)
        if same_activation:
            tempX = g(tempX)
        else:
            tempX = g[i](tempX)
    output.append(tempX)
    return output

def CostFun(X, theta, g, y, regularization = False, lamb = None):
    pred = ForwardPropagation(X, theta, g)[-1]
    reg = 0
    if regularization:
        reg = lamb/(2*len(y)) * np.sum([np.sum(tt[:,1:]**2) for tt in theta])
    return -np.sum(np.sum(y*np.log(pred)+(1-y)*np.log(1-pred))/len(y))+reg

def SigmoidFunc(ma):
    return (1+np.exp(-ma))**-1

# test cost func using given thetas
rawdata = loadmat(codepath+'/RawData/ex3weights.mat')#25*401,10*26
theta_test = [rawdata['Theta1'], rawdata['Theta2']]
dummy_y = pd.get_dummies(y.flatten())
print(CostFun(X, theta_test, SigmoidFunc, dummy_y)) #should be 0.28762916516131876
print(CostFun(X, theta_test, SigmoidFunc, dummy_y, regularization = True, lamb = 1)) #should be 0.3837698590909235


#%% Backpropagation

def RandomWeightInit(nnodein, nnodeout, epsilon = None):
    """

    Parameters
    ----------
    nnodein : number of node of this(input) level
    nnodeout : number of node of next level
    epsilon : range of generation

    Returns
    -------
    weights initailization : list (flattened)

    """
    if not epsilon:
        epsilon = (6/(nnodein+nnodeout))**0.5 # a good choice of the range
    return np.random.rand((1+nnodein)*nnodeout)*2*epsilon-epsilon
# test
# print(RandomWeightInit(2, 3))

def BackPropagation(X, y, theta, activation, regularization = False, lamb = 1):
    '''
    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    y : same as output (dummy_y if exists)
    theta : TYPE
        DESCRIPTION.
    activation_gradident : function
        derivative of activation, eg.SigmoidGradient.
    cost_gradient : function
        derivative of cost function, eg. yhat-y for OLS and logistic
    
    Returns
    -------
    theta : TYPE
        DESCRIPTION.
    '''
    n = len(X)
    num_level = len(theta)
    deltas = [0]*(num_level+1)
    gradient = [0]*(num_level)
    deltas[num_level] = np.array(a[-1] - y)
    for l in range(num_level-1, 0, -1):
        gprime = activation[l]*(1-activation[l]) 
        tt = np.array(deltas[l+1].dot(theta[l]))
        deltas[l] = (tt * gprime)[:,1:]                                        # remove gradient for bias term (first row)
        gradient[l] = deltas[l+1].T.dot(activation[l])/n
        if regularization:
            gradient[l] = gradient[l] + lamb/n * theta[l]
    gradient[0] = deltas[1].T.dot(a[0])/n
    if regularization:
        gradient[0][:,1:] = gradient[0][:,1:] + lamb/n * theta[0][:,1:]
    return gradient

a = ForwardPropagation(X, theta_test, SigmoidFunc) 
gradient = BackPropagation(X, dummy_y, theta_test, activation = a)


#%% gradient checking
def GradientChecking(X, theta, SigmoidFunc, dummy_y, CostFun, gradient):
    epsilon = 10**(-4)
    error = 10**(-9)
    gg = [0]*len(theta)
    for l in range(len(theta)):
        theta_n1, theta_n2 = theta[l].shape
        gg[l] = np.ones((theta_n1, theta_n2))
        for i in range(theta_n1):
            for j in range(theta_n2):
                theta[l][i,j] = theta[l][i,j] + epsilon
                costplus = CostFun(X, theta, SigmoidFunc, dummy_y)
                theta[l][i,j] = theta[l][i,j] - 2*epsilon
                costminus = CostFun(X, theta, SigmoidFunc, dummy_y)
                theta[l][i,j] = theta[l][i,j] + epsilon
                gg[l][i,j] = (costplus-costminus)/(2*epsilon)
                if abs(gg[l][i,j] - gradient[l][i,j])>error:
                    return False        
    return True

print(GradientChecking(X, theta_test, SigmoidFunc, dummy_y, CostFun, gradient)) 

#%% 
from scipy.optimize import minimize
layer_size = [400, 25, 10]
epsilon = 0.1

def Roll(theta, layer_size):
    layer_num = len(layer_size)
    formal_theta = [0]*(layer_num-1)
    loc_start = 0
    for l in range(layer_num-1):
        loc_stop = loc_start + layer_size[l+1]*(layer_size[l]+1)
        formal_theta[l] = np.array(theta[loc_start:loc_stop]).reshape(layer_size[l+1],layer_size[l]+1)
        loc_start = loc_stop
    return formal_theta

def Unroll(theta):
    out = []
    for xx in theta:
        out += list(xx.flatten())
    return np.array(out)

def FlattenGradient(theta):
    tt = Roll(theta, layer_size)
    a = ForwardPropagation(X, tt, SigmoidFunc) 
    gradient = BackPropagation(X, dummy_y, tt, activation = a, regularization = True, lamb = 1)
    return Unroll(gradient)

total_para_number = 0
for l in range(len(layer_size)-1):
    total_para_number = total_para_number + layer_size[l+1]*(layer_size[l]+1)
x0 = np.random.rand(total_para_number)*2*epsilon-epsilon

def Cost(theta):
    return CostFun(X, Roll(theta, layer_size), SigmoidFunc, dummy_y, regularization = True, lamb = 1)

# too slow
print(Cost(x0))
result = minimize(Cost, x0, method='BFGS', jac=FlattenGradient, options = {'maxiter': 50})
print(Cost(result['x']))

#%%
predict = ForwardPropagation(X, Roll(result['x'], layer_size), SigmoidFunc)
# predict = ForwardPropagation(X, Roll(x0, layer_size), SigmoidFunc)
predictnumber = [row.argmax()+1 for row in predict[-1]]
print(sum([predictnumber[i] == y[i] for i in range(len(y))]))
