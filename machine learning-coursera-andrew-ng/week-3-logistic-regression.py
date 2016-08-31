# This Python 2.7 code implements the programming assignment from Week 3 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build logistic regression classifiers in Python as well as how to use regularization
# on these models. It also demonstrates how to build higher polynomial features of input factors for improving the fit.
# It uses the numpy, pandas and off-the-shelf modules for logistic regression and polynomial features in scikit-learn. For
# the details of the programming assignment sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/
# and access Week 3.

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

# --- Exercise 1: Logistic regression

# Load the input data
X = pd.read_table('data/week-3/ex2data1.txt', usecols=[0, 1], sep=',', names=['Test 1 Score', 'Test 2 Score'])
y = pd.read_table('data/week-3/ex2data1.txt', usecols=[2], sep=',', names=['Admission Result'], squeeze=True)

# Train a logistic regression model using the above data
logreg = linear_model.LogisticRegression(C=1e4) # No regularization in this part
logreg.fit(X,y)

# Predict and print the probability of admission for a given test score set
X_test = np.array([[45, 85]])
y_test = logreg.predict_proba(X_test)
print "Results of Unregularized Logistic Regression Exercise"
print "For a student with a score of %i in Test 1 and %i in Test 2, the probability of admission is %.2f%%" % (X_test[0,0], X_test[0,1], y_test[0,1]*100)
print type(y_test[0,1])

# Predict and print accuracy of prediction over training set
y_test = logreg.predict(X)
prediction_accuracy = float(np.sum((y_test == y).astype(int)))/y.shape[0]
print "The prediction accuracy over the training set is %.2f%%" % (prediction_accuracy*100)


# --- Exercise 2: Regularized Logistic Regression

# Load the input data
X = pd.read_table('data/week-3/ex2data2.txt', usecols=[0, 1], sep=',', names=['Test 1', 'Test 2'])
y = pd.read_table('data/week-3/ex2data2.txt', usecols=[2], sep=',', names=['Chip Quality Result'], squeeze=True)

# Create polynomial features of input data
poly_6 = preprocessing.PolynomialFeatures(6, include_bias=False)
X_poly_6 = poly_6.fit_transform(X)

# Train a logistic regression classifier on the above data
logreg = linear_model.LogisticRegression() # Regularization by default is 1
logreg.fit(X_poly_6, y)

# Predict and print accuracy of prediction over training set
y_test = logreg.predict(X_poly_6)
prediction_accuracy = float(np.sum((y_test == y).astype(int)))/y.shape[0]
print "Results of regularized Logistic Regression exercise"
print "The prediction accuracy over the training set is %.2f%%" % (prediction_accuracy*100)
