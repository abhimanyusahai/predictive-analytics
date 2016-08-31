# This code predicts handwritten digits using the MNIST image dataset, found at
# http://yann.lecun.com/exdb/mnist/ , using a neural network with one hidden
# layer. The neural network has been coded practically from scratch, with the code being
# vectorized, as well as parametrized to be able to re-use the code for other
# applications. The current prediction accuracy is ~93.7%, which can possibly
# be improved by tweaking (increasing) the regularization parameter as well as
# the number of hidden nodes. The code will likely take a long time to run
# (~5-10 mins on my Mac Book Air with 1.4 GHz Intel Core i5 processor).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.io as sio
from scipy.optimize import minimize
import struct

# --- Helper functions

# Helper function to load MNIST data
def load_mnist():
    # Load training images
    train_data_file = open('data/train-images-idx3-ubyte', 'r')
    magic_images_train = np.fromfile(train_data_file, '>i4', 1, '')
    num_images_train = np.fromfile(train_data_file, '>i4', 1, '')
    num_rows_train = np.fromfile(train_data_file, '>i4', 1, '')
    num_cols_train = np.fromfile(train_data_file, '>i4', 1, '')
    train_images = np.fromfile(train_data_file, 'uint8')
    train_images = train_images.reshape(num_images_train, num_rows_train * num_cols_train)
    train_data_file.close()
    # Load training labels
    train_label_file = open('data/train-labels-idx1-ubyte', 'r')
    magic_labels_train = np.fromfile(train_label_file, '>i4', 1, '')
    num_labels_train = np.fromfile(train_label_file, '>i4', 1, '')
    train_labels = np.fromfile(train_label_file, 'uint8').reshape(num_labels_train, 1)
    train_label_file.close()
    # Load test images
    test_data_file = open('data/t10k-images-idx3-ubyte', 'r')
    magic_images_test = np.fromfile(test_data_file, '>i4', 1, '')
    num_images_test = np.fromfile(test_data_file, '>i4', 1, '')
    num_rows_test = np.fromfile(test_data_file, '>i4', 1, '')
    num_cols_test = np.fromfile(test_data_file, '>i4', 1, '')
    test_images = np.fromfile(test_data_file, 'uint8')
    test_images = test_images.reshape(num_images_test, num_rows_train * num_cols_test)
    test_data_file.close()
    # Load test labels
    test_label_file = open('data/t10k-labels-idx1-ubyte', 'r')
    magic_labels_test = np.fromfile(test_label_file, '>i4', 1, '')
    num_labels_test = np.fromfile(test_label_file, '>i4', 1, '')
    test_labels = np.fromfile(test_label_file, 'uint8').reshape(num_labels_test, 1)
    test_label_file.close()
    return train_images, train_labels, test_images, test_labels

# Helper function to train the neural network
def train_nn():
    # Initialize theta
    num_thetas = ((n + 1) * num_hidden_nodes) + ((num_hidden_nodes + 1) * num_output_nodes)
    initial_theta = np.random.random(num_thetas).reshape(num_thetas,) * 2 * 0.12 - 0.12 # epsilon taken to be 0.12
    return minimize(cost, initial_theta, method='CG', jac=cost_grad, options={'maxiter': num_iters}).x

# Helper function to compute sigmoid
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

# Helper function to compute network cost as a function of network parameters
def cost(theta):
    theta_1 = theta[0:((n + 1) * num_hidden_nodes)].reshape(num_hidden_nodes, (n + 1))
    theta_2 = theta[((n + 1) * num_hidden_nodes):].reshape(num_output_nodes, (num_hidden_nodes + 1))

    # Feedforward and compute network cost
    X_ = np.c_[np.ones((m,1)), X]
    z_2 = np.dot(X_, np.transpose(theta_1))
    a_2 = sigmoid(z_2)
    a_2 = np.c_[np.ones((m,1)), a_2]
    z_3 = np.dot(a_2, np.transpose(theta_2))
    a_3 = sigmoid(z_3)
    cost = ((-1.0/m) * (np.sum((y * np.log(a_3))) + np.sum((1 - y) * np.log(1 - a_3)))) + ((reg_lambda / (2 * m)) * (np.sum(theta_1[:,1:]**2) + np.sum(theta_2[:,1:]**2)))

    return cost

# Helper function to compute cost gradient as a function of network parameters
def cost_grad(theta):
    theta_1 = theta[0:((n + 1) * num_hidden_nodes)].reshape(num_hidden_nodes, (n + 1))
    theta_2 = theta[((n + 1) * num_hidden_nodes):].reshape(num_output_nodes, (num_hidden_nodes + 1))

    # Feedforward and compute network cost
    X_ = np.c_[np.ones((m,1)), X]
    z_2 = np.dot(X_, np.transpose(theta_1))
    a_2 = sigmoid(z_2)
    a_2 = np.c_[np.ones((m,1)), a_2]
    z_3 = np.dot(a_2, np.transpose(theta_2))
    a_3 = sigmoid(z_3)
    cost = ((-1.0/m) * (np.sum((y * np.log(a_3))) + np.sum((1 - y) * np.log(1 - a_3)))) + ((reg_lambda / (2 * m)) * (np.sum(theta_1[:,1:]**2) + np.sum(theta_2[:,1:]**2)))

    # Compute gradient of cost using backpropagation
    D_1, D_2 = np.zeros(theta_1.shape), np.zeros(theta_2.shape) # Gradients
    delta_3 = a_3 - y
    delta_2 = np.dot(delta_3, theta_2)[:,1:] * a_2[:, 1:] * (1 - a_2[:, 1:])
    D_1 = np.dot(np.transpose(delta_2), X_)
    D_2 = np.dot(np.transpose(delta_3), a_2)
    theta1_grad = ((1.0/m) * D_1) + ((reg_lambda/m) * theta_1)
    theta2_grad = ((1.0/m) * D_2) + ((reg_lambda/m) * theta_2)
    theta1_grad[:, 0] = (1.0/m) * D_1[:, 0]
    theta2_grad[:, 0] = (1.0/m) * D_2[:, 0]

    # Unroll the gradients into a vector
    theta_grad = np.r_[theta1_grad.reshape(((n + 1) * num_hidden_nodes),), theta2_grad.reshape(((num_hidden_nodes + 1) * num_output_nodes),)]

    return theta_grad

# Helper function to predict with a given set of network parameters
def nn_predict(X):
    m = X.shape[0]
    # Roll the parameter vector
    theta_1 = theta_optimal[0:((n + 1) * num_hidden_nodes)].reshape(num_hidden_nodes, (n + 1))
    theta_2 = theta_optimal[((n + 1) * num_hidden_nodes):].reshape(num_output_nodes, (num_hidden_nodes + 1))
    # Perform predictions
    X = np.c_[np.ones((m,1)), X]
    z_2 = np.dot(X, np.transpose(theta_1))
    a_2 = sigmoid(z_2)
    a_2 = np.c_[np.ones((m,1)), a_2]
    z_3 = np.dot(a_2, np.transpose(theta_2))
    a_3 = sigmoid(z_3)
    predictions = np.zeros((m, 1))
    for i in xrange(0,m):
        predictions[i] = np.argmax(a_3[i,:])
    return predictions


# --- Load data, train model, and predict on training set

# Load the input data
X_train, y_train, X_test, y_test = load_mnist()
X = X_train
y_labels = y_train

# Set network parameters
num_hidden_nodes = 200
num_output_nodes = 10
reg_lambda = 1 # Regularization factor
num_iters = 150 # number of iterations of the cost minimization algorithm

# Train the network with the above set of parameters
m = X.shape[0]
n = X.shape[1]
# Recast y as a binary matrix with 10 classes, each class representing one digit
y = np.zeros((m, num_output_nodes))
for i in xrange(0, num_output_nodes):
    y[:, i] = (y_labels == i).astype(int).ravel()
theta_optimal = train_nn()

# Predict on the training dataset using the optimal theta detected above
nn_predictions = nn_predict(X_test)
nn_prediction_accuracy = float(np.sum((y_test == nn_predictions).astype(int)))/y_test.shape[0]

# Print prediction accuracy
print "The prediction accuracy on the test dataset is %r%%" % (nn_prediction_accuracy * 100)
