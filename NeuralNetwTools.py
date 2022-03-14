'''
Code on 'Learn TensorFlow 2.0'

Neural Networks (Chapter 3)
'''

#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

(training_images, training_labels), (test_images, test_labels) = ks.datasets.fashion_mnist.load_data()

print('Training Images Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Images Dataset Labels: {}'.format(len(training_labels)))
print('Test Images Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Images Dataset Labels: {}'.format(len(test_labels)))

# pixel values in [0,255], we have to rescale these values to [0,1] 
training_images = training_images / 255.0
test_images = test_images / 255.0

input_data_shape = training_images.shape[1:] #(28, 28)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'

#%% build nn
nn_model = ks.models.Sequential()  # ?? difference with ks.Sequential()?
nn_model.add(ks.layers.Flatten(input_shape=input_data_shape, name='Input_layer'))
nn_model.add(ks.layers.Dense(32, activation=hidden_activation_function, name='Hidden_layer'))
nn_model.add(ks.layers.Dense(10, activation=output_activation_function, name='Output_layer'))
nn_model.summary()

#%% fit
optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy']
nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
nn_model.fit(training_images, training_labels, epochs=10)

training_loss, training_accuracy = nn_model.evaluate(training_images, training_labels)
print('Training Data Accuracy {}'.format(round(float(training_accuracy),2)))

#%% test
test_loss, test_accuracy = nn_model.evaluate(test_images, test_labels)
print('Test Data Accuracy {}'.format(round(float(test_accuracy),2)))


#%% deep nn
input_data_shape = training_images.shape[1:] #(28, 28)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'
dnn_model = ks.Sequential()
dnn_model.add(ks.layers.Flatten(input_shape=input_data_shape, name='Input_layer'))
dnn_model.add(ks.layers.Dense(256, activation=hidden_activation_function, name='Hidden_layer_1'))
dnn_model.add(ks.layers.Dense(192, activation=hidden_activation_function, name='Hidden_layer_2'))
dnn_model.add(ks.layers.Dense(128, activation=hidden_activation_function, name='Hidden_layer_3'))
dnn_model.add(ks.layers.Dense(10, activation=output_activation_function, name='Output_layer'))
dnn_model.summary()
optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy']
dnn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
dnn_model.fit(training_images, training_labels, epochs=20)
training_loss, training_accuracy = dnn_model.evaluate(training_images, training_labels)
print('Training Data Accuracy {}'.format(round(float(training_accuracy),2)))
test_loss, test_accuracy = dnn_model.evaluate(test_images, test_labels)
print('Test Data Accuracy {}'.format(round(float(test_accuracy),2)))

#%% Custom estimators Using the Keras Model ??
import tensorflow_datasets as tf_ds

def data_input():
  train_test_split = tf_ds.Split.TRAIN
  iris_dataset = tf_ds.load('iris', split=train_test_split, as_supervised=True)
  iris_dataset = iris_dataset.map(lambda features, labels: ({'dense_input':features}, labels))
  iris_dataset = iris_dataset.batch(32).repeat()
  return iris_dataset

activation_function = 'relu'
input_shape = (4,)
dropout = 0.2
output_activation_function = 'sigmoid'
keras_model = ks.models.Sequential([ks.layers.Dense(16, activation=activation_function, 
                                                    input_shape=input_shape), 
                                    ks.layers.Dropout(dropout), 
                                    ks.layers.Dense(1, activation=output_activation_function)])

loss_function = 'categorical_crossentropy'
optimizer = 'adam'
keras_model.compile(loss=loss_function, optimizer=optimizer)
keras_model.summary()

model_path = "/keras_estimator/"
estimator_keras_model = ks.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_path)

estimator_keras_model.train(input_fn=data_input, steps=25)
evaluation_result = estimator_keras_model.evaluate(input_fn=data_input, steps=10)
print('Fianl evaluation result: {}'.format(evaluation_result))


#%%
'''
Try code on Mnist
'''
import os
from scipy.io import loadmat
import numpy as np
codepath = os.path.abspath(os.getcwd())
filename = os.listdir(codepath+'/RawData')
rawdata = loadmat(codepath+'/RawData/ex3data1.mat')
X = rawdata['X']
y = rawdata['y']
rawdata = loadmat(codepath+'/RawData/ex3weights.mat')
theta = [rawdata['Theta1'], rawdata['Theta2']]

input_data_shape = X.shape[1]
hidden_activation_function = 'sigmoid'
output_activation_function = 'softmax'
nn_model = ks.models.Sequential()
nn_model.add(ks.Input(shape = input_data_shape))
nn_model.add(ks.layers.Dense(255, activation=hidden_activation_function, name='Hidden_layer'))
nn_model.add(ks.layers.Dense(10, activation=output_activation_function, name='Output_layer'))
nn_model.summary()

optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy']
nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
nn_model.fit(X, y-1, epochs=10)

training_loss, training_accuracy = nn_model.evaluate(X, y-1)
print('Training Data Accuracy {}'.format(round(float(training_accuracy),2)))

