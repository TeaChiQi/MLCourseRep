#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:42:42 2020

@author: QI
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def LoadDataSet(filename, delim = ','):
    """ Import Data
    Args:
        filename: name of file
        delim: punctuation used to seperate datas
    Returns:
        X,y: training data
    """
    file = open(filename)
    rawdata = file.read()
    file.close()
    X,y = [],[]
    for x in rawdata.split('\n'):
        if not x:
            return ([np.array(X), np.array(y)])
        observation = np.array(x.split(delim), dtype = float)
        X.append(observation[:-1])
        y.append(observation[-1])
        
def PlotDataUniVar(X,y,theta = None):
    """ Plot Data for univariate model with regression line
    """
    plt.scatter(X,y,
                c="g",                                                         #color
                alpha=1,                                                       #size of point
                marker=r'$\clubsuit$',                                         #Style of point
                label='Luck')
    plt.xlabel("Population")
    plt.ylabel("Profit")
    plt.legend(loc='upper left')
    if theta is not None:
        plt.plot(X, X*theta[1]+theta[0], c="r")
    plt.show()
    

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

def UpdateForLinear(theta, X, y):
    n = len(X)
    designmatrix = np.append(np.ones([n,1]), X, axis=1)
    ydiff = np.sum(designmatrix*theta, axis = 1) - y
    gradient = np.sum(ydiff.reshape(n,1) * designmatrix, axis=0)/n
    return gradient

def GradientDescent(X, y, alpha, iterations, epsilon = 1, \
                    stochastic = False, costtracking = False):
    """ Gradient descent
    Args:
        alpha: learning rate
        iterations: max number of iterations
    Returns:
        theta: minimized point
    Required func:
        FeatureNorm - NormReverse
        UpdateForLinear: gradient calculation
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
        if stochastic:
            j = i%n
            xdata = np.append(1,X_normed[j,:])
            gradient = (np.sum(xdata*theta)-y[j])*xdata
        else:
            gradient = UpdateForLinear(theta, X_normed, y)
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

def GDforLWR(X, y, alpha, iterations, centralpoint, tau = 1, epsilon = 0):
    n,m = X.shape
    converge = False
    if m>1:
        meanarray, stdarray, X_normed = FeatureNorm(X)
        centralpoint = (centralpoint-meanarray)/stdarray
    else:
        X_normed = X
    theta = np.zeros(m+1)
    for i in range(iterations):
        weight = np.exp(-np.sum((X_normed-centralpoint)**2, axis=0)/(2*tau**2))
        gradient = UpdateForLinear(theta, X_normed, y)*weight
        if np.sum(np.abs(gradient)) <= epsilon:
            print('Gradient descent converges')
            converge = True
            break
        theta = theta - alpha*gradient
    if not converge:
        print('Gradient descent reaches the maximum iterations')
    NormReverse(theta, meanarray, stdarray)
    return theta

def Cost(theta, X, y):
    """ Compute current cost
    Returns:
        cost of current theta
    """
    n = len(X)
    designmatrix = np.append(np.ones([n,1]), X, axis=1)
    ydiff = np.sum(designmatrix*theta, axis = 1) - y
    return np.sum(ydiff**2)/(2*n)

def ComputeCostGrid(xstart, xend, ystart, yend, X, y):
    """ 
    Returns:
        list of theta0, theta1, costs
    Required func:
        Cost: compute the z value
    """
    xgrid = np.linspace(xstart,xend,100)
    ygrid = np.linspace(ystart,yend,100)
    grid_x, grid_y = np.meshgrid(xgrid, ygrid)
    x_flatten, y_flatten = grid_x.flatten(), grid_y.flatten()
    itercost = lambda theta0,theta1: Cost(np.array([theta0,theta1]), X, y)
    costsmap = map(itercost, x_flatten, y_flatten)
    return [list(x_flatten), list(y_flatten), list(costsmap)]

def Plots3D(x,y,z, scatter = False, contour = False, surface = False,
            flatcontour = False):
    """
    need "from mpl_toolkits import mplot3d"
    """
    n = int(len(x)**0.5)
    fig = plt.figure()
    if flatcontour:
        plt.contour(np.array(x).reshape(n,n),np.array(y).reshape(n,n),\
                        np.array(z).reshape(n,n), 100, cmap='RdGy');
        plt.colorbar()
        fig.show()
        return;
    ax = plt.axes(projection='3d')
    if contour:
        ax.contour3D(np.array(x).reshape(n,n),np.array(y).reshape(n,n),\
                     np.array(z).reshape(n,n), 50, cmap='binary')
    elif surface:
        ax.plot_surface(np.array(x).reshape(n,n),np.array(y).reshape(n,n),\
                        np.array(z).reshape(n,n), rstride=1, cstride=1,\
                        cmap='viridis', edgecolor='none')
    else:
        ax.plot3D(x,y,z, 'gray')
        if scatter:
            ax.scatter3D(x,y,z, c=z, cmap='Greens')
    #ax.view_init(60, 35) #can change angle of view
    fig.show()

def NormalEquations(X, y):
    designmatrix = np.append(np.ones([len(y),1]), X, axis=1)
    desginmatrixtranspose = designmatrix.T
    temp1 = np.linalg.inv(desginmatrixtranspose.dot(designmatrix))
    temp2 = desginmatrixtranspose.dot(y)
    return temp1.dot(temp2)

#### import data ####
#import os
#codepath = os.path.abspath(os.getcwd())                                        #get path[or: os.path.dirname(os.path.abspath(__file__)) ]
#DataFileName = os.listdir(codepath+'/RawData')  
#X,y = LoadDataSet(codepath+'/RawData/'+DataFileName[1])

#### scatterplot ####                           
# PlotDataUniVar(X,y)

#### Gradient descent ####
#interations = 500
#alpha = 0.03
#theta = GradientDescent(X, y, alpha, interations)
#print(theta)
# PlotDataUniVar(X,y,theta)

#### Compare with explicit solution ####
#print("True value is", NormalEquations(X, y))

#### Visualizing cost ####
#print(Cost(theta, X, y))
#gridx, gridy, gridz = ComputeCostGrid(-10, 10, -1, 4, X, y)
#Plots3D(gridx, gridy, gridz)

#### learning rate debug ####
#iterations = 1500
#alpha = 0.03
#theta, costlist = GradientDescent(X, y, alpha, iterations, costtracking = True)
#plt.scatter(np.arange(len(costlist)),costlist, s=1)

#### Stochastic Gradient descent ####
#interations = 50000
#alpha = 0.3
#theta = GradientDescent(X, y, alpha, interations, stochastic = True)
#print(theta)

