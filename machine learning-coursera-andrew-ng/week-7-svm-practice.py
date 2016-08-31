# This Python 2.7 code implements the first part of the programming assignment from Week 7 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build Support Vector Machines in Python using both Linear and Gaussian kernels.
# It uses the numpy, pandas, scipy, matplotlib and an off-the-shelf module for SVMs in scikit-learn. For
# the details of the programming assignment sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/
# and access Week 7.

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

# --- Exercise 1: Implement simpler version of SVM (without using kernels) on a 2D dataset
# with 2 classes that are linearly separable

# Load and visualize input data
X = sio.loadmat('data/week-7/ex6data1.mat')['X']
y = sio.loadmat('data/week-7/ex6data1.mat')['y'].reshape(X.shape[0],)
plt.scatter(X[np.where(y == 0)][:, 0], X[np.where(y == 0)][:, 1], marker='o', color='yellow', edgecolor='black', s=40)
plt.scatter(X[np.where(y == 1)][:, 0], X[np.where(y == 1)][:, 1], marker='+', color='black', s=40)
plt.title('Dataset 1')
plt.show()

# Train an SVM model with C = 1
svm_model = svm.SVC(C=1, kernel='linear')
svm_model.fit(X, y)
theta0 = svm_model.intercept_[0]
theta1 = svm_model.coef_[0][0]
theta2 = svm_model.coef_[0][1]
xx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
yy = (-theta1/theta2)*xx + (-theta0/theta2)
plt.subplot(2,1,1)
plt.scatter(X[np.where(y == 0)][:, 0], X[np.where(y == 0)][:, 1], marker='o', color='yellow', edgecolor='black', s=40)
plt.scatter(X[np.where(y == 1)][:, 0], X[np.where(y == 1)][:, 1], marker='+', color='black', s=40)
plt.plot(xx, yy)
plt.title('Dataset 1 with SVM classification boundary (C = 1)')

# Now train an SVM model with C = 100 and visualize the decision boundaries
svm_model = svm.SVC(C=100, kernel='linear')
svm_model.fit(X, y)
theta0 = svm_model.intercept_[0]
theta1 = svm_model.coef_[0][0]
theta2 = svm_model.coef_[0][1]
xx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
yy = (-theta1/theta2)*xx + (-theta0/theta2)
plt.subplot(2,1,2)
plt.scatter(X[np.where(y == 0)][:, 0], X[np.where(y == 0)][:, 1], marker='o', color='yellow', edgecolor='black', s=40)
plt.scatter(X[np.where(y == 1)][:, 0], X[np.where(y == 1)][:, 1], marker='+', color='black', s=40)
plt.plot(xx, yy)
plt.title('Dataset 1 with SVM classification boundary (C = 100)')
plt.show()


# --- Exercise 2: Implement SVM with Gaussian kernel on a 2D dataset with 2 classes
# that are not linearly separable

# Load the input dataset
# Load and visualize input data
X = sio.loadmat('data/week-7/ex6data2.mat')['X']
y = sio.loadmat('data/week-7/ex6data2.mat')['y'].reshape(X.shape[0],)
plt.scatter(X[np.where(y == 0)][:, 0], X[np.where(y == 0)][:, 1], marker='o', color='yellow', edgecolor='black', s=40)
plt.scatter(X[np.where(y == 1)][:, 0], X[np.where(y == 1)][:, 1], marker='+', color='black', s=40)
plt.title('Dataset 2')
plt.show()

# Train an SVM with Gaussian kernel on the above dataset
svm_model = svm.SVC(C=10, gamma=1) # Default kernel is Gaussian
svm_model.fit(X, y)

# Plot the decision boundary of this Gaussian kernel
plt.scatter(X[np.where(y == 0)][:, 0], X[np.where(y == 0)][:, 1], marker='o', color='yellow', edgecolor='black', s=40)
plt.scatter(X[np.where(y == 1)][:, 0], X[np.where(y == 1)][:, 1], marker='+', color='black', s=40)
xx = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
yy = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
xc, yc = np.meshgrid(xx, yy)
X_contour = np.c_[xc.ravel(), yc.ravel()]
Z = svm_model.predict(X_contour)
Z = Z.reshape(yc.shape[0], xc.shape[1])
plt.contour(xc, yc, Z, linestyles='solid', cmap='Blues')
plt.title('Dataset 2 with SVM classification boundary (Gaussian kernel)')
plt.show()


# --- Exercise 3: Using cross-validation along with a Gaussian kernel

# Load and visualize the input dataset
X = sio.loadmat('data/week-7/ex6data3.mat')['X']
y = sio.loadmat('data/week-7/ex6data3.mat')['y'].reshape(X.shape[0],)
Xval = sio.loadmat('data/week-7/ex6data3.mat')['Xval']
yval = sio.loadmat('data/week-7/ex6data3.mat')['yval'].reshape(Xval.shape[0],)
plt.scatter(X[np.where(y == 0)][:, 0], X[np.where(y == 0)][:, 1], marker='o', color='yellow', edgecolor='black', s=40)
plt.scatter(X[np.where(y == 1)][:, 0], X[np.where(y == 1)][:, 1], marker='+', color='black', s=40)
plt.title('Dataset 3')
plt.show()

# Train an SVM with Gaussian kernel and figure out the best C and gamma to use,
# using the cross-validation set
gamma_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
best_gamma = best_C = best_accuracy = 0
for C in C_array:
    for gamma in gamma_array:
        svm_model = svm.SVC(C=C, gamma=gamma)
        svm_model.fit(X, y)
        predictions = svm_model.predict(Xval)
        classification_accuracy = np.sum((predictions == yval).astype(int))
        if classification_accuracy > best_accuracy:
            best_accuracy = classification_accuracy
            best_gamma = gamma
            best_C = C
svm_model = svm.SVC(C=best_C, gamma=best_gamma)
svm_model.fit(X, y)

# Plot the decision boundary of the above model with the best C. gamma
plt.scatter(X[np.where(y == 0)][:, 0], X[np.where(y == 0)][:, 1], marker='o', color='yellow', edgecolor='black', s=40)
plt.scatter(X[np.where(y == 1)][:, 0], X[np.where(y == 1)][:, 1], marker='+', color='black', s=40)
xx = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2)
yy = np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2)
xc, yc = np.meshgrid(xx, yy)
X_contour = np.c_[xc.ravel(), yc.ravel()]
Z = svm_model.predict(X_contour)
Z = Z.reshape(yc.shape[0], xc.shape[1])
plt.contour(xc, yc, Z, linestyles='solid', cmap='Blues')
plt.title('Dataset 3 with SVM classification boundary (Gaussian kernel) \nand C, gamma optimized using cross-validation')
plt.show()
