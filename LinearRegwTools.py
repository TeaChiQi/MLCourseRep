#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Preparation: Data import
import numpy as np

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

import os
codepath = os.path.abspath(os.getcwd()) #os.path.dirname(os.path.abspath(__file__))
DataFileName = os.listdir(codepath+'/RawData')  
X,y = LoadDataSet(codepath+'/RawData/'+'ex1data1.txt')#DataFileName[2])

#%% split data to the training set and test set 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.2, random_state=0)
#%% linear regression 
regressor = LinearRegression() # fit
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test) # predict

#%% visualizing
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#%% Gradient descent

# optimatimization scipy 
#from scipy.optimize import minimize

#minimize(cost, theta_0, method = 'Newton-CG', jac = gradient)








