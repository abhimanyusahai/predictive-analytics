# This Python 2.7 code implements part of the programming assignment from Week 4 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build multi-class logistic regression classifiers in Python using the One-vs-Rest method.
# It uses the numpy, pandas, scipy and an off-the-shelf module for logistic regression in scikit-learn. For
# the details of the programming assignment sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/
# and access Week 4.

import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy.io as sio

# Load the input data as numpy arrays
In = sio.loadmat('data/week-4/ex3data1.mat')
X = In['X']
y = In['y']
y = y.reshape(y.shape[0], ) # Recast as 1-D array to prevent warning from being raised in regression model

# Train a multiclass logistic regression classifier on the input data
multilogreg = linear_model.LogisticRegression(C=10, multi_class='ovr')
multilogreg.fit(X, y)

# Predict labels over the training set and print the accuracy
predictions = multilogreg.predict(X)
prediction_accuracy = float(np.sum((predictions == y).astype(float)))/y.shape[0]
print "The prediction accuracy over the training set is %.2f%%" % (prediction_accuracy*100)
