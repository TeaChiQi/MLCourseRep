import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#%% =============== Part 1: Loading and Visualizing Data ================
'''
  We start the exercise by first loading and visualizing the dataset. 
  The following code will load the dataset into your environment and plot
  the data.
'''

print('Loading and Visualizing Data ...\n')
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
rawdata = loadmat(codepath+'/RawData/ex6data1.mat')
X = np.array(rawdata['X'])
y = np.array(rawdata['y'])
label = (y==1).T[0]
# method 1
plt.scatter(X[:,0][label], X[:,1][label], c = 'tab:blue', marker = 'o', label = '1')
plt.scatter(X[:,0][~label], X[:,1][~label], c = 'tab:orange', marker = 'x', label = '0')
plt.legend()
plt.show()
# method 2
plt.scatter(X[:, 0], X[:, 1], c=label, s=30, cmap=plt.cm.Paired)
plt.show()

input('Program paused. Press enter to continue.\n')


#%% ==================== Part 2: Training Linear SVM ====================
'''
  The following code will train a linear SVM on the dataset and plot the
  decision boundary learned.
'''

print('\nTraining Linear SVM ...\n')

def LinearSVMBoundary(X, y, C):
    fig = plt.figure()
    N = len(C)
    for i in range(N):
        ax = plt.subplot(N,1,i+1)
        ax.scatter(X[:, 0], X[:, 1], c=label, s=30, cmap=plt.cm.Paired)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        linear_svc = svm.SVC(kernel='linear', C = C[i])
        linear_svc.fit(X, y.ravel())
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = linear_svc.decision_function(xy).reshape(XX.shape)
        
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
    plt.show()
    
LinearSVMBoundary(X, y, C = [1,100])

input('Program paused. Press enter to continue.\n')

#%% =============== Part 3: Implementing Gaussian Kernel ===============
'''
%  You will now implement the Gaussian kernel to use
%  with the SVM. You should complete the code in gaussianKernel.m
'''

def gaussianKernel(x1, x2, sigma):
    x1, x2 = np.array(x1), np.array(x2)
    return np.exp(-np.sum((x1-x2)**2)/sigma**2/2)

print('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1, 2, 1]; x2 = [0, 4, -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {}:\n\t{}.\
      \n(for sigma = 2, this value should be about 0.324652)\n'.format( sigma, sim))

input('Program paused. Press enter to continue.\n')


#%% =============== Part 4: Visualizing Dataset 2 ================
'''
%  The following code will load the next dataset into your environment and 
%  plot the data. 
'''

print('Loading and Visualizing Data ...\n')

# % Load from ex6data2: 
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
rawdata = loadmat(codepath+'/RawData/ex6data2.mat')
X = np.array(rawdata['X'])
y = np.array(rawdata['y'])

# plt.scatterplot(X, y);
label = (y==1).T[0]
plt.scatter(X[:, 0], X[:, 1], c=label, s=30, cmap=plt.cm.Paired)

input('Program paused. Press enter to continue.\n')

#%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
'''
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
'''

print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

def visualizeBoundary(X, y, decisionfunc):
    ax = plt.subplot(1,1,1)
    ax.scatter(X[:, 0], X[:, 1], c=label, s=30, cmap=plt.cm.Paired)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
        
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = decisionfunc(xy).reshape(XX.shape)
        
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
    plt.show()
    
# % Use data from ex6data2
C = 1; sigma = 0.1;
def gaussianKernelwithsigma(X,Y):
    kernel = []
    for i in range(X.shape[0]):
        kernel.append( list(np.exp(-np.sum((X[i,:] - Y)**2, axis = 1)/sigma**2/2)))
    return np.array(kernel)

svc = svm.SVC(kernel=gaussianKernelwithsigma, tol = 1, C = C)
svc.fit(X, y.ravel())
visualizeBoundary(X, y, decisionfunc = svc.decision_function)

input('Program paused. Press enter to continue.\n')

#%% =============== Part 6: Visualizing Dataset 3 ================
print('Loading and Visualizing Data ...\n')

# % Load from ex6data2: 
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
rawdata = loadmat(codepath+'/RawData/ex6data3.mat')
X = np.array(rawdata['X'])
y = np.array(rawdata['y'])
Xval = np.array(rawdata['Xval'])
yval = np.array(rawdata['yval'])

# plt.scatterplot(X, y);
label = (y==1).T[0]
plt.scatter(X[:, 0], X[:, 1], c=label, s=30, cmap=plt.cm.Paired)

input('Program paused. Press enter to continue.\n')

#%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

def dataset3Params(X, y, Xval, yval):
    candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    C, sigma, optimal = 0.01, 0.01, 0
    for c in candidates:
        for sig in candidates:
            svc = svm.SVC(kernel='rbf', gamma = (2*sig**2)**(-1), C = c)
            svc.fit(X, y.ravel())
            correct = sum((svc.predict(Xval)==yval.T[0]))
            if correct>optimal:
                C, sigma, optimal = c, sig, correct
    return C, sigma
# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)
print('optimal value is C = {}, sigma = {}'.format(C,sigma))

svc = svm.SVC(kernel='rbf', gamma = (2*sigma**2)**(-1), tol = 1, C = C)
svc.fit(X, y.ravel())
visualizeBoundary(X, y, decisionfunc = svc.decision_function)







