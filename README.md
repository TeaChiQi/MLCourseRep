# MLCourseRep
Python Replication For Matlab Based ML Course

**Summary:**
	There are two groups of code. The ones starting with ‘EX’ are the exercises of the Machine Learning course on Coursera. The ones ending with ‘wTools’ are replications of the main algorithms using libraries.

## **1.‘EX1-GDForLinearReg’ : Gradient Descent with Linear Regression**
*  Data importation
	* LoadDataSet
*  Plot scatterplot and regression line for univariate model
	* PlotDataUniVar
*  Gradient Descent with Normalization and Learning rate debugging
	* GradientDescent (can be batch or stochastic)
*  3Dplot for cost function
	* Cost - ComputeCostGrid - Plots3D
*  Locally weighted linear regression
	* GDforLWR

## **‘EX2-LogisticReg’: Gradient Descent with Logistic Regression**
1. Logistic regression and gradient descent
	GradientDescent
2. Scatterplot of multi-catergories

## **‘EX3-NeuralNet’**
1. Data importation of matlab format
2. Pixel plot of images
3. Multiclass classifier
	GradientDescentMultiClassifier
4. Forward Propagation and Back Propagation
	ForwardPropagation, BackPropagation
5. Using limit to check if back propagation process computes the correct gradient
	GradientChecking

## **‘EX5-MLDiagnostics’**
1. plot learning curve

## **‘EX6-SVM’ & ‘EX6-SVMspam’**
SVM: sklearn -> svm

## **‘EX7-Kmeans’**
Kmeans with random initialization: sklearn.cluster -> KMeans

## **‘EX7-PCA’**
PCA: sklearn.decomposition -> PCA

## **‘LinearRegwTools’**
Linear regression: sklearn.linear_model -> LinearRegression
Gradient descent: scipy.optimize -> minimize 

## **‘LogisticRegwTools’**
Logistic regression: sklearn.linear_model -> LogisticRegression

## **‘NeuralNetwTools’**
Example code from book ‘Tensorflow 2.0’ & test using Mnist data
Neural Network: tensorflow -> keras


# A Verticle Summary 
1. Data importation
	* EX1-GDForLinearReg.py: LoadDataSet
	* EX3-NeuralNet.py: for mat format
2. Plot data
	* 3D plot (EX1-GDForLinearReg.py)
	* Scatterplot of multi-categories (EX6-SVM.py)
	* Contour plot (EX2-LogisticReg.py)
	* Pixel plot (EX3-NeuralNet.py)
	* Multiple figures (EX3-NeuralNet.py)
	* Animation (EX7-Kmeans.py)
3. Regularize Expression
	* (EX6-SVMspam.py)
