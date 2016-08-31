# This Python 2.7 code implements the first part of the programming assignment from Week 9 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build anomaly detection systems in Python using a Gaussian probability distribution.
# It uses numpy, pandas, scipy, matplotlib and scipy.stats for the actual Gaussian model. For the details of the programming assignment,
# sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/ and access Week 9.

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt

# Helper function to compute best epsilon
def best_epsilon(predicted_p, y_cv):
    epsilon = 0
    best_F1 = 0
    predictions =  np.zeros(predicted_p.shape)

    for i in np.linspace(predicted_p.min() + (predicted_p.max() - predicted_p.min())/1000, predicted_p.max(), 1000):
        predictions = (predicted_p < i).astype(int)
        tp = np.sum((predictions == 1) & (y_cv == 1))
        fp = np.sum((predictions == 1) & (y_cv == 0))
        fn = np.sum((predictions == 0) & (y_cv == 1))
        precision = float(tp)/(tp + fp)
        recall = float(tp)/(tp + fn)
        F1 = (2 * precision * recall)/(precision + recall)
        if F1 > best_F1:
            best_F1 = F1
            epsilon = i

    return epsilon

# Load the input data
In = sio.loadmat('data/week-9/ex8data1.mat')
X = In['X']

# Visualize the input data
plt.scatter(X[:,0], X[:,1], marker='x', color='blue')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (Mb/s)')

# Fit the input data into a Gaussian distribution and draw a contour plot of the probabilities of X
p1 = stats.norm(X[:,0].mean(), X[:,0].std())
p2 = stats.norm(X[:,1].mean(), X[:,1].std())
xx = np.linspace(1.1*X[:,0].min(), 1.1*X[:,0].max())
yy = np.linspace(1.1*X[:,1].min(), 1.1*X[:,1].max())
xc, yc = np.meshgrid(xx, yy)

Z = np.zeros((yy.shape[0], xx.shape[0]))
for i in xrange(0, xx.shape[0]):
    for j in xrange(0, yy.shape[0]):
        Z[j, i] = p1.pdf(xx[i]) * p2.pdf(yy[j])
plt.contour(xc, yc, Z)

# Now use the cross-validation set to estimate the best value of epsilon
X_cv = In['Xval']
y_cv = In['yval'].reshape(X_cv.shape[0],)
predicted_p = p1.pdf(X_cv[:,0]) * p2.pdf(X_cv[:,1])
epsilon = best_epsilon(predicted_p, y_cv)

# Circle the predicted anomalies in the original dataset with a red circle
X_anomalies = X[np.where((p1.pdf(X[:, 0]) * p2.pdf(X[:, 1])) < epsilon)]
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], s=80, marker='o', facecolors='none', edgecolors='red')
plt.show()

# Now run the algorithm on the much larger dataset

# Load the data
In = sio.loadmat('data/week-9/ex8data2.mat')
X = In['X']
X_cv = In['Xval']
y_cv = In['yval'].reshape(X_cv.shape[0],)

# Create probability distribution model
p = [] # List to hold the probability density objects
for i in xrange(0, X.shape[1]):
    p.append(stats.norm(X[:,i].mean(), X[:,i].std()))

# Use cross validation set to estimate best value of epsilon
predicted_p = np.ones(y_cv.shape[0],)
for i in xrange(0, X_cv.shape[1]):
    predicted_p *= p[i].pdf(X_cv[:,i])
epsilon = best_epsilon(predicted_p, y_cv)

# Predict and print out the number of anomalies
predicted_p = np.ones(X.shape[0],)
for i in xrange(0, X.shape[1]):
    predicted_p *= p[i].pdf(X[:,i])
X_anomalies = X[np.where(predicted_p < epsilon)]
print "The number of anomalies in the larger dataset of size %i are %i" % (X.shape[0], X_anomalies.shape[0])
