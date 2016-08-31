# This Python 2.7 code implements the programming assignment from Week 2 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build both univariate and multivariate linear regression regression systems
# in Python. It uses the numpy, pandas and the off-the-shelf module for linear regression in scikit-learn. For
# the details of the programming assignment sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/
# and access Week 2.

import numpy as np
import pandas as pd
from sklearn import linear_model

# --- Exercise 1: Univariate linear regression

# Load input data
X = pd.read_table('data/week-2/ex1data1.txt', usecols=[0], sep=',', names=['Population (10,000s)'])
y = pd.read_table('data/week-2/ex1data1.txt', usecols=[1], sep=',', names=['Profits ($10,000)'], squeeze=False)

# Train a linear regression model on the above data
linreg = linear_model.LinearRegression()
linreg.fit(X, y)

# Print out the coefficients found from the linear regression model
intercept = linreg.intercept_[0]
theta = linreg.coef_[0][0]
print "The intercept value is %r" % intercept
print "The theta value is %r" % theta

# Predict values at certain datapoints and print out the predicted values
X_test = np.array([[3.5], [7]])
predictions = linreg.predict(X_test)
print "Results of Univariate Linear Regression exercise"
print "For a population of %r we predict a profit of %r$" % (X_test[0,0]*10000, predictions[0,0]*10000)
print "For a population of %r we predict a profit of %r$\n" % (X_test[1,0]*10000, predictions[1,0]*10000)


# --- Exercise 2: Multivariate Linear Regression

# Load input data
X = pd.read_table('data/week-2/ex1data2.txt', usecols=[0,1], sep=',', names=['Area', '# of Bedrooms'])
y = pd.read_table('data/week-2/ex1data2.txt', usecols=[2], sep=',', names=['Price in $'])

# Train a linear regression model on the above data
linreg = linear_model.LinearRegression(normalize=True)
linreg.fit(X, y)

# Print out the coefficients found from the linear regression model
intercept = linreg.intercept_[0]
theta = linreg.coef_[0]
print "Results of Multivariate Linear Regression exercise"
print "The intercept value is %r" % intercept
print "The theta values are theta1: %r and theta2: %r" % (theta[0], theta[1])

# Predict values at certain datapoints and print out the predicted values
X_test = np.array([[1650, 3]])
predictions = linreg.predict(X_test)
print "For a house of size %r sqft. and with %r bedrooms, we predict a price of %r$" % (X_test[0,0], X_test[0,1], predictions[0,0])
