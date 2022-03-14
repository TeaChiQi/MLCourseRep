#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:37:47 2020
%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs
@author: QI
"""

import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import re
from nltk.stem import PorterStemmer

#%% ============ Part 1&2: Email Preprocessing & Feature Extraction=============
'''
%  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
%  to convert each email into a vector of features. In this part, you will
%  implement the preprocessing steps for each email. You should
%  complete the code in processEmail.m to produce a word indices vector
%  for a given email. you will convert each email into a vector of features in R^n. 
%  You should complete the code in emailFeatures.m to produce a feature
%  vector for a given email.
'''

def GetVocab():
    '''
    Returns
    -------
    A list of highly occured spam words.
    '''
    codepath = os.path.abspath(os.getcwd())
    # get vocablist
    with open(codepath+'/RawData/vocab.txt') as f:
        fid = f.read()
        
    spamvoaclist = re.sub('\d', ' ', fid)
    return re.split('[ |\n|\t]+', spamvoaclist)[1:-1]
    
def processEmail(file_contents):
    '''
    Parameters
    ----------
    file_contents : str, input email.

    Returns
    -------
    A sorted list of words in the email.
    '''
    file_regu = file_contents.lower()                                              #Lower case
    file_regu = re.sub('[0-9]+', 'number', file_regu)                              #Handle Numbers
    file_regu = re.sub('<[^<>]+>', ' ', file_regu)                                 #Strip all HTML
    #Looks for any expression that starts with < and ends with > and does not have any < or > in the tag and replace it with a space
    file_regu = re.sub('(http|https)://[^\s]*', 'httpaddr', file_regu)             #Handle URLS
    file_regu = re.sub('[^\s]+@[^\s]+', 'emailaddr', file_regu)                    #Handle email, @ in the middle
    file_regu = re.sub('[$]+', 'dollar', file_regu)                                #Handle $ sign
    #Tokenize and also get rid of any punctuation, Remove any non alphanumeric characters
    file_regu = re.sub('[^\w]', ' ', file_regu)
    words = re.split('[ ]+', file_regu)
    ps = PorterStemmer()
    return sorted([ps.stem(i) for i in words if len(i)>0])

def emailFeatures(words, spamvoaclist):
    '''
    Parameters
    ----------
    words : output of processEmail.
    spamvoaclist : output of GetVocab.

    Returns
    -------
    spamdummy : list, dummy feature vector of size n*1.

    '''
    spamdummy = [0]*len(spamvoaclist)
    spampointer, wordpointer = 0, 0
    while spampointer<len(spamvoaclist) and wordpointer<len(words):
        if spamvoaclist[spampointer] > words[wordpointer]:
            wordpointer += 1
        else:
            if spamvoaclist[spampointer] == words[wordpointer]:
                spamdummy[spampointer] = 1
            spampointer += 1
    return spamdummy

#%%
print('Loading Data ...\n')
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
with open(codepath+'/RawData/emailSample1.txt') as f:
    file_contents = f.read()

print('Extracting features from sample email (emailSample1.txt)...\n');
spamvoaclist = GetVocab();
words = processEmail(file_contents);
features = emailFeatures(words, spamvoaclist);

print('Length of feature vector: {}'.format(len(features)))
print('Number of non-zero entries: {}'.format(sum(features)) )
#%% ======== Part 3&4: Train Linear SVM And Test for Spam Classification ========
'''
%  In this section, you will train a linear classifier to determine if an
%  email is Spam or Not-Spam.
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat
'''

rawdata = loadmat(codepath+'/RawData/spamTrain.mat')
X = np.array(rawdata['X'])
y = np.array(rawdata['y'])
rawdata = loadmat(codepath+'/RawData/spamTest.mat')
Xtest = np.array(rawdata['Xtest'])
ytest = np.array(rawdata['ytest'])

print('Training Linear SVM (Spam Classification)\n')

C = 1;
svc = svm.LinearSVC(C = C)
svc.fit(X, y.ravel())
# ypred = svc.predict(Xtest)
print('Training Accuracy: {}%\n'.format(svc.score(Xtest, ytest) * 100))

# ?? linearsvc has a different result with svc(kernel = 'linear') ??

#%% ================= Part 5: Top Predictors of Spam ====================
'''
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
'''

weight = pd.DataFrame(spamvoaclist, columns=['word'])
weight['weight'] = svc.coef_[0]
sortedwords = weight.sort_values(['weight'], ascending=False)
print(sortedwords['word'][:10])

#%% =================== Part 6: Try Your Own Emails =====================

print( svc.predict(np.array(features).reshape(1,1899)) )




