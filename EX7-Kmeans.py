#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:46:54 2020

@author: QI

Machine Learning Online Class
%  Exercise 7-1 | K-Means Clustering
"""

import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

input('pause')
#%% ============= Part 1: Find Closest Centroids & Compute Means ==============

print('Finding closest centroids...')

codepath = os.path.abspath(os.getcwd())
rawdata = loadmat(codepath + '/RawData/ex7data2.mat')
X = rawdata['X']

intialcentroids = np.array([[3,3],[6,2],[8,5]])

distances = np.array(list(map(lambda y: np.sum((y-X)**2, axis = 1), intialcentroids)))
idx = [np.argmin(x)+1 for x in distances.T]

print('Closest centroids for the first 3 examples: {}'.format(idx[:3]))
print('(the closest centroids should be 1, 3, 2 respectively)')

print('Computing centroids means...')

df = pd.DataFrame(X)
df['idx'] = idx
centroids = np.array(df.groupby('idx').mean())

print('Centroids computed after initial finding of closest centroids: {}'.format(centroids))
print('(the centroids should be\n \
      [[ 2.428301 3.157924 ],[ 5.813503 2.633656 ],[ 7.119387 3.616684 ]]')
      
#%% =================== Part 3: K-Means Clustering ======================

def RandomInit(X,K):
    '''
    Parameters
    ----------
    X : data matrix.
    K : int, number of groups.

    Returns
    -------
    centroids : numpy array.
    '''
    n = len(X)
    choose = np.random.choice(n, K)
    return X[choose, :]

def Kmeans(X, K, max_iters = 10, plotprogress = False):
    if plotprogress:
        fig = plt.figure()
        ims = []
    print('Running K-Means clustering on example dataset...');
    df = pd.DataFrame(X)
    centroids = RandomInit(X, K)
    centroid_history = list(centroids)
    # initial index
    distances = np.array(list(map(lambda y: np.sum((y-X)**2, axis = 1), centroids)))
    idx = [np.argmin(x)+1 for x in distances.T]
    df['idx'] = idx
    for _ in range(max_iters):
        #find centers
        centroids = np.array(df.groupby('idx').mean())
        #update index
        distances = np.array(list(map(lambda y: np.sum((y-X)**2, axis = 1), centroids)))
        idx = [np.argmin(x)+1 for x in distances.T]
        df['idx'] = idx
        if plotprogress:
            centroid_history = [np.vstack((centroid_history[i],c)) for i,c in enumerate(centroids)]
            im = plt.scatter(X[:,0],X[:,1], c = idx, cmap = plt.cm.Paired, animated=True)
            for i in range(len(centroid_history)):
                plt.plot(centroid_history[i][:,0],centroid_history[i][:,1], '-kx', animated=True)
            ims.append([im])
    print('\nK-Means Done.');
    if plotprogress:
        ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                repeat_delay=1000)
        ani.save('kmeanprogress.gif')
    return [idx, centroids]
    
# Kmeans(X, 3, plotprogress = True)

#%% ============= Part 4: K-Means Clustering on Pixels ===============
'''
%  In this exercise, you will use K-Means to compress an image. To do this,
%  you will first run K-Means on the colors of the pixels in the image and
%  then you will map each pixel onto its closest centroid.
'''

rawdata = plt.imread(codepath + '/RawData/monet.png')
# N * M * 3, '3' containing the Red, Green and Blue pixel values

X = rawdata.reshape((rawdata.shape[0]*rawdata.shape[1], 3))

K = 16

[idx, centroids] = Kmeans(X, K, max_iters = 10, plotprogress = False)
X_recovered = np.zeros(X.shape)
for i in range(len(X)):
    X_recovered[i, :] = centroids[idx[i]-1]

compressedimage = X_recovered.reshape(rawdata.shape)

plt.figure()
ax = plt.subplot(211)
ax.imshow(rawdata)
ax = plt.subplot(212)
ax.imshow(compressedimage)
plt.show()

#%% =============== Part 5: with tools ==============

from sklearn.cluster import KMeans
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)









