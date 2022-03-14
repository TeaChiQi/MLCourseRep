#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 2020
"""
import os
import numpy as np

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
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


