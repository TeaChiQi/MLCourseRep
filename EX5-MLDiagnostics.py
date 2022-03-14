import os
from scipy.io import loadmat
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


#%% import data
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
#print(filename) #['ex3data1.mat', 'ex2data2.txt', 'ex2data1.txt', 'ex3weights.mat', 'ex1data1.txt', 'ex1data2.txt']
rawdata = loadmat(codepath+'/RawData/ex5data1.mat')
X, Xtest, Xval = rawdata['X'], rawdata['Xtest'], rawdata['Xval']
y, ytest, yval = rawdata['y'], rawdata['ytest'], rawdata['yval']

#%% Learning curve

def Error(ypred, y):
    return np.sum((ypred-y)**2)/len(y)/2

def LearningCurve(X, y, Xval, yval, Costfun):
    regressor = LinearRegression()
    e_cv, e_train = [], []
    for samplesize in range(2,len(y)):
        regressor.fit(X[:samplesize,:], y[:samplesize])
        e_cv.append(Error(regressor.predict(Xval), yval))
        e_train.append(Error(regressor.predict(X[:samplesize,:]), y[:samplesize]))
        
    plt.plot(list(range(2,len(y))), e_train)
    plt.plot(list(range(2,len(y))), e_cv)
    
LearningCurve(X, y, Xval, yval, Costfun = Error)