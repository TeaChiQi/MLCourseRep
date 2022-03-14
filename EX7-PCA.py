#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:41:21 2020

@author: QI

Exercise 7-2 | Principle Component Analysis and K-Means Clustering
"""

import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

#%% ================== Part 1: Load Example Dataset  ===================

rawpath = os.path.abspath(os.getcwd())
rawdata = loadmat(rawpath+'/RawData/ex7data1.mat')
X = rawdata['X']

print('Visualizing example dataset for PCA...\n');

plt.plot(X[:,0], X[:,1], 'bo')

#%% =============== Part 2: Principal Component Analysis ===============
print('Running PCA on example dataset...\n');

def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu)/sigma
    return [X_norm, mu, sigma]
    
#  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);

#  Run PCA
[U, S, V] = svd(np.cov(X_norm.T))

fig = plt.figure(figsize=(8,8))
plt.xlim([1,7])
plt.ylim([1,7])
plt.plot(X[:,0], X[:,1], 'bo')
vec1 = np.array([mu, mu + 1.5 * S[0] * U[:,0]])
vec2 = np.array([mu, mu + 1.5 * S[1] * U[:,1]])
plt.plot(vec1[:,0], vec1[:,1], '-k')
plt.plot(vec2[:,0], vec2[:,1], '-k')

print('Top eigenvector: \n');
print(' U(:,1) = ', U[0,0], U[1,0]);
print('\n(you should expect to see -0.707107 -0.707107)\n');

#%% =================== Part 3: Dimension Reduction ===================
K = 1
projectData = X_norm.dot(U[:,:K])

fig = plt.figure(figsize=(8,8))
plt.xlim([-2.5,2.5])
plt.ylim([-2.5,2.5])
for i in range(len(X)):
    plt.plot([X_norm[i,0],projectData[i,0]*U[0,0]], [X_norm[i,1],projectData[i,0]*U[0,1]], '--ko')

#%% =============== Part 4: Face Data =============
rawdata = loadmat(rawpath+'/RawData/ex7faces.mat')
X = rawdata['X']

def PCA(X, K = 1):
    [X_norm, mu, sigma] = featureNormalize(X)
    [U, S, _] = svd(np.cov(X_norm.T))
    featurevector = X_norm.dot(U[:,:K])
    return featurevector.dot(U[:,:K].T)*sigma+mu

Xrec = PCA(X, K = 100)
plt.imshow(X[1,:].reshape((32,32)))
plt.imshow(Xrec[1,:].reshape((32,32)))

#%% =============== Part 5: with tools ==============
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)
principalComponents = pca.fit_transform(X)










