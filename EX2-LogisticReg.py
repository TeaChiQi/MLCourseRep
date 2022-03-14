#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 2020
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%% Data Preparation
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
#print(filename) #['ex2data2.txt', 'ex2data1.txt', 'ex1data1.txt', 'ex1data2.txt']

file = open(codepath+'/RawData/ex2data1.txt')
rawdata = file.read()
file.close()
X, y = [], []
for x in rawdata.split('\n')[:-1]:
    obser = np.array(x.split(','), dtype = float)
    X.append(obser[:-1])
    y.append(obser[-1])
X, y = np.array(X), np.array(y)

#%% I. logistic regression
#%% 1. Visualizing
fig, ax = plt.subplots()
ax.scatter(X[:,0][y==1], X[:,1][y==1], c = 'tab:blue', marker = 'o', label = 'Admitted')
ax.scatter(X[:,0][y==0], X[:,1][y==0], c = 'tab:orange', marker = 'x', label = 'Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
ax.legend(loc='upper right')
plt.show()

#%% 2. Implementation
def FeatureNorm(X):
    """
    Args:
        X: feature numpy array(no intercept)
    Returns:
        meanarray, stdarray, X_normed
    """
    n,m = X.shape
    meanarray = np.mean(X, axis = 0)
    stdarray = np.std(X, axis = 0)
    X_normed = (X-meanarray)/stdarray
    return [meanarray, stdarray, X_normed]

def NormReverse(theta, meanlist, stdlist):
    """Calculate the theta for the original data (not normed ones)
    Args:
        theta: numpy array, will be changed
    """
    theta[1:] = theta[1:]/np.array(stdlist)
    theta[0] = theta[0] - np.sum(theta[1:]*np.array(meanlist))

def Gradient(theta, X, y): ## the only change from linear reg
    n = len(X)
    designmatrix = np.append(np.ones([n,1]), X, axis=1)
    yfit = (1+np.exp(-np.sum(designmatrix*theta, axis = 1)))**(-1)
    ydiff = yfit - y
    gradient = np.sum(ydiff.reshape(n,1) * designmatrix, axis=0)/n
    return gradient

def Cost(theta, X, y):
    """ Compute current cost
    Returns:
        cost of current theta
    """
    n = len(X)
    designmatrix = np.append(np.ones([n,1]), X, axis=1)
    yfit = (1+np.exp(-np.sum(designmatrix*theta, axis = 1)))**(-1)
    ydiff = yfit - y
    return np.sum(ydiff**2)/(2*n)

def GradientDescent(X, y, alpha, iterations, epsilon = 0, costtracking = False):
    """ Gradient descent
    Args:
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
    costlist = []
    converge = False
    if m>1:
        meanarray, stdarray, X_normed = FeatureNorm(X)
    else:
        X_normed = X
    theta = np.zeros(m+1)
    for i in range(iterations):
        gradient = Gradient(theta, X_normed, y)
        if np.sum(np.abs(gradient)) <= epsilon:
            print('Gradient descent converges')
            converge = True
            break
        theta = theta - alpha*gradient
        if costtracking:
            costlist.append(Cost(theta, X_normed, y))
    if not converge:
        print('Gradient descent reaches the maximum iterations')
    NormReverse(theta, meanarray, stdarray)
    if costtracking:
        return [theta, costlist]
    return theta

# Fit logistic regression using gradient descent
theta, costlist = GradientDescent(X, y, alpha=0.2, iterations=300, costtracking = True)
plt.plot(list(range(len(costlist))),costlist)

#%% 3. plot fit
fig, ax = plt.subplots()
ax.scatter(X[:,0][y==1], X[:,1][y==1], c = 'tab:blue', marker = 'o', label = 'Admitted')
ax.scatter(X[:,0][y==0], X[:,1][y==0], c = 'tab:orange', marker = 'x', label = 'Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
ax.legend(loc='upper right')
plt.plot(X[:,0], -X[:,0]*theta[1]/theta[2]-theta[0]/theta[2], c="r")
plt.show()

#%% II. Regularized logistic regression
#%% 1. visualizing
file = open(codepath+'/RawData/ex2data2.txt')
rawdata = file.read()
file.close()
X, y = [], []
for x in rawdata.split('\n')[:-1]:
    obser = np.array(x.split(','), dtype = float)
    X.append(obser[:-1])
    y.append(obser[-1])
X, y = np.array(X), np.array(y)
fig, ax = plt.subplots()
ax.scatter(X[:,0][y==1], X[:,1][y==1], c = 'tab:blue', marker = 'o', label = 'y=1')
ax.scatter(X[:,0][y==0], X[:,1][y==0], c = 'tab:orange', marker = 'x', label = 'y=0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
ax.legend(loc='upper right')
plt.show()

#%% 2. feature mapping
def AllFeature(X):
    x = []
    for i in range(7):
        for j in range(i+1):
            x.append(X[:,0]**j * X[:,1]**(i-j))
    return np.array(x).T

allX = AllFeature(X)
allX.shape

#%%
def Gradient(theta, X, y): ## the only change from linear reg
    lamda = 4
    n = len(X)
    designmatrix = np.append(np.ones([n,1]), X, axis=1)
    yfit = (1+np.exp(-np.sum(designmatrix*theta, axis = 1)))**(-1)
    ydiff = yfit - y
    gradient = np.sum(ydiff.reshape(n,1) * designmatrix, axis=0)/n + theta*lamda/n
    return gradient

theta= GradientDescent(allX[:,1:], y, alpha=0.3, iterations=1000)
theta

#%% plot thetas
delta = 0.025
xrange = np.arange(-1., 1.5, delta)
yrange = np.arange(-0.8, 1.2, delta)
Xmesh, Ymesh = np.meshgrid(xrange,yrange)
nx, ny = len(xrange), len(yrange)
Zmesh = AllFeature(np.append(Xmesh.reshape(nx*ny,1),Ymesh.reshape(nx*ny,1),axis=1))
Z = np.sum(Zmesh * theta, axis = 1)

fig, ax = plt.subplots()
ax.scatter(X[:,0][y==1], X[:,1][y==1], c = 'tab:blue', marker = 'o', label = 'y=1')
ax.scatter(X[:,0][y==0], X[:,1][y==0], c = 'tab:orange', marker = 'x', label = 'y=0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
ax.legend(loc='upper right')
plt.contour(Xmesh, Ymesh, Z.reshape(ny, nx), [0])
fig.show()






