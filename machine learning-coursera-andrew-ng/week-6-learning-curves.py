# This Python 2.7 code implements the programming assignment from Week 6 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build learning curves to make decisions on model improvement.
# It uses the numpy, pandas, matplotlib, scipy and off-the-shelf modules for linear regression and polynomial features
# in scikit-learn. For the details of the programming assignment sign up for this course on Coursera
# at https://www.coursera.org/learn/machine-learning/ and access Week 6.

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
import scipy.io as sio
import matplotlib.pyplot as plt

# Helper function to compute cost over the predictions for a given dataset
def cost(predictions, y):
    return (0.5/y.shape[0]) * np.sum((predictions - y) ** 2)

# Load input data
In = sio.loadmat('data/week-6/ex5data1.mat')
# Training set
X = In['X']
y = In['y']
# Cross validation set
Xval = In['Xval']
yval = In['yval']
# Test set
Xtest = In['Xtest']
ytest = In['ytest']

# Visualize training data
plt.scatter(X, y, marker='x', s=40, color='red')
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the dam')

# Fit a linear regression model on the training data
linreg = linear_model.LinearRegression(normalize=True)
linreg.fit(X, y)
# plot the best fit line
theta0 = linreg.intercept_[0]
theta1 = linreg.coef_[0][0]
xx = np.linspace(1.1*X.min(), 1.1*X.max())
yy = theta0 + theta1*xx
plt.plot(xx, yy)
plt.title('Linear regression with single power of X (straight line fit)')
plt.show()

# Calculate and plot the learning curves for the above model
training_error = []
cv_error = []
for i in xrange(1, (X.shape[0] + 1)):
    linreg.fit(X[:i, :], y[:i, :])
    training_predictions = linreg.predict(X[:i, :])
    cv_predictions = linreg.predict(Xval)
    training_error.append(cost(training_predictions, y[:i, :]))
    cv_error.append(cost(cv_predictions, yval))
plt.plot(training_error, color='blue')
plt.plot(cv_error, color='green')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(('Training error', 'Cross validation error'))
plt.title('Learning curves on linear regression with single power of X (straight line fit)')
plt.show()

# Since the above learning curves diagnosed the model to be a high-bias one,
# we will try to fit higher powers to increase the model variance

# Create feature vector by creating higher powers of X - till the power of 8
poly8 = preprocessing.PolynomialFeatures(degree=8)
X_8 = poly8.fit_transform(X)[:, 1:]
# Fit the linear regression model
linreg.fit(X_8, y)
xx_8 = poly8.fit_transform(xx.reshape(xx.shape[0],1))[:, 1:]
yy_8 = linreg.predict(xx_8)
plt.scatter(X, y, marker='x', s=40, color='red')
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the dam')
plt.plot(xx, yy_8, linestyle='-')
plt.title('Linear regression with higher powers of X - uptill power 8\nWithout regularization')
plt.show()

# Now plot learning curves for the above fit
Xval_8 = poly8.fit_transform(Xval)[:, 1:]
training_error = []
cv_error = []
for i in xrange(1, (X.shape[0] + 1)):
    linreg.fit(X_8[:i, :], y[:i, :])
    training_predictions = linreg.predict(X_8[:i, :])
    cv_predictions = linreg.predict(Xval_8)
    training_error.append(cost(training_predictions, y[:i, :]))
    cv_error.append(cost(cv_predictions, yval))
plt.plot(training_error, color='blue')
plt.plot(cv_error, color='green')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(('Training error', 'Cross validation error'))
plt.title('Learning curves on linear regression with higher powers of X - uptill power 8\nWithout regularization')
plt.ylim((0, 100))
plt.show()

# Now add a regularization parameter of 0.1 to the above model to reduce cross-validation error
linreg = linear_model.Ridge(alpha=0.1, normalize=True)
linreg.fit(X_8, y)
xx_8 = poly8.fit_transform(xx.reshape(xx.shape[0],1))[:, 1:]
yy_8 = linreg.predict(xx_8)
plt.scatter(X, y, marker='x', s=40, color='red')
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the dam')
plt.plot(xx, yy_8, linestyle='-')
plt.title('Linear regression with higher powers of X - uptill power 8\nWith regularization')
plt.show()

# Plot the learning curve of the above model
training_error = []
cv_error = []
for i in xrange(1, (X.shape[0] + 1)):
    linreg.fit(X_8[:i, :], y[:i, :])
    training_predictions = linreg.predict(X_8[:i, :])
    cv_predictions = linreg.predict(Xval_8)
    training_error.append(cost(training_predictions, y[:i, :]))
    cv_error.append(cost(cv_predictions, yval))
plt.plot(training_error, color='blue')
plt.plot(cv_error, color='green')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(('Training error', 'Cross validation error'))
plt.title('Learning curves on linear regression with higher powers of X - uptill power 8\nWith regularization')
plt.ylim((0, 100))
plt.show()

# Now experiment with different values of regularization parameter and plot the validation curves
alpha_array = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
training_error = []
cv_error = []
for alpha in alpha_array:
    linreg = linear_model.Ridge(alpha=alpha, normalize=True)
    linreg.fit(X_8, y)
    training_predictions = linreg.predict(X_8)
    cv_predictions = linreg.predict(Xval_8)
    training_error.append(cost(training_predictions, y))
    cv_error.append(cost(cv_predictions, yval))
plt.plot(alpha_array, training_error, color='blue')
plt.plot(alpha_array, cv_error, color='green')
plt.xlabel('Regularization parameter')
plt.ylabel('Error')
plt.legend(('Training error', 'Cross validation error'))
plt.title('Validation curves - value of errors with different values of regularization parameter')
plt.ylim((0, 100))
plt.show()
