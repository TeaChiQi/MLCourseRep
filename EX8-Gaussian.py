#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 08:56:20 2020

@author: QI

Exercise 8-1 | Anomaly Detection and Collaborative Filtering
"""

import os 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

#%% ================== Part 1: Load Example Dataset  ===================
'''
%  Our example case consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  This exercise will help us find possibly faulty (or very fast) machines.
'''

print('Visualizing example dataset for outlier detection.\n\n');

rawpath = os.path.abspath(os.getcwd())
rawdata = loadmat(rawpath+'/RawData/ex8data1.mat')
X = rawdata['X']
Xval = rawdata['Xval']
yval = rawdata['yval']

plt.plot(X[:, 0], X[:, 1], 'bx');
plt.xlabel('Latency (ms)');
plt.ylabel('Throughput (mb/s)');


#%% ================== Part 2: Estimate the dataset statistics ===================
print('Visualizing Gaussian fit.\n\n');

def estimateGaussian(X):
    return np.mean(X, axis = 0), np.cov(X.T)

mu, sigma2 = estimateGaussian(X)

def multivariateGaussian(X, mu, sigma2):
    return st.multivariate_normal.pdf(X, mu, sigma2)
p = multivariateGaussian(X, mu, sigma2);

def visualizeFit(X, mu, sigma2):
    n = 10
    xstart, xend = mu[0] - sigma2[0,0]**0.5 * 3, mu[0] + sigma2[0,0]**0.5 * 3
    ystart, yend = mu[1] - sigma2[1,1]**0.5 * 3, mu[1] + sigma2[1,1]**0.5 * 3
    delta = (yend - ystart) / n
    xrange = np.arange(xstart, xend, delta)
    yrange = np.arange(ystart, yend, delta)
    Xmesh, Ymesh = np.meshgrid(xrange,yrange)
    nx, ny = len(xrange), len(yrange)
    Zmesh = multivariateGaussian(np.append(Xmesh.reshape(nx*ny,1),\
                                           Ymesh.reshape(nx*ny,1),axis=1), mu, sigma2)
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c = 'tab:blue', marker = 'o')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.contour(Xmesh, Ymesh, Zmesh.reshape(ny, nx))
    plt.show()
 
visualizeFit(X, mu, sigma2)

#%% ================== Part 3: Find Outliers ===================

pval = multivariateGaussian(Xval, mu, sigma2);

def selectThreshold(yval, pval):
    yval = yval.T
    bestEpsilon, bestF1 = 0, 0
    stepsize = (max(pval) - min(pval)) / 1000
    for ep in np.arange(min(pval)+stepsize, max(pval), stepsize):
        positive = (pval<ep)
        cases = (positive + yval)[0]
        tp = sum(cases == 2)
        p = sum(positive)
        fn = sum(cases == 1) - p + tp
        prec = tp/p
        rec = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)
        if F1>bestF1:
            bestEpsilon, bestF1 = ep, F1
    return bestEpsilon, bestF1
        

epsilon, F1 = selectThreshold(yval, pval);
print('Best epsilon found using cross-validation:', epsilon);
print('Best F1 on Cross Validation Set:', F1, '\n');
print('   (you should see a value epsilon of about 8.99e-05)');
print('   (you should see a Best F1 value of  0.875000)\n\n');

outliers = (p < epsilon)

plt.plot(X[outliers, 0], X[outliers, 1], 'ro')
plt.scatter(Xval[:, 0], Xval[:, 1], c = yval[:,0], cmap = plt.cm.Paired, marker = 'x')


#%% ================== Part 4: Multidimensional Outliers ===================
rawdata = loadmat(rawpath+'/RawData/ex8data2.mat')
X = rawdata['X']
Xval = rawdata['Xval']
yval = rawdata['yval']


[mu, sigma2] = estimateGaussian(X);

p = multivariateGaussian(X, mu, sigma2);

pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon, F1] = selectThreshold(yval, pval);

print('Best epsilon found using cross-validation: ', epsilon);
print('Best F1 on Cross Validation Set:  ', F1,'\n');
print('   (you should see a value epsilon of about 1.38e-18)\n');
print('   (you should see a Best F1 value of 0.615385)\n');
print('# Outliers found: ', sum(p < epsilon));