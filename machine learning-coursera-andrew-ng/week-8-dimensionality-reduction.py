# This Python 2.7 code implements the second part of the programming assignment from Week 8 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build dimensionality reduction systems in Python using the Principal Components Analysis (PCA)
# algorithm. The first part of the code demonstrates how to use numpy, pandas, scipy, matplotlib and an off-the-shelf library for PCA
# on scikit learn to transform a simple 2D dataset to 1D. The second part then uses PCA to compress a dataset of face images by using PCA
# for reducing data from 1024 dimensions (32x32 images) to 100 dimensions. For the details of the programming assignment, sign up for this
# course on Coursera at https://www.coursera.org/learn/machine-learning/ and access Week 8.

import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Exercise 1: Simple conversion of 2D data to 1D

# Load input data
X = sio.loadmat('data/week-8/ex7data1.mat')['X']

# Perform PCA on the 2D dataset, reducing it to 1D
pca = PCA(1)
pca.fit(X)
X_1D = pca.transform(X)
X_recovered = pca.inverse_transform(X_1D)

# Print the first element of the transformed X
print X_1D[0]

# --- Exercise 2: Dimensionality reduction of faces dataset

# Load input data
X = sio.loadmat('data/week-8/ex7faces.mat')['X']

# Visualize the first 100 faces
for i in xrange(0, 100):
    plt.subplot(20,10,i+1)
    x = X[i,:].reshape(32, 32)
    plt.imshow(np.rot90(x, 3), cmap='gray')
    plt.xticks(()), plt.yticks(())

# Perform PCA on the dataset retaining only 100 dimensions, the recover the data back
pca = PCA(100)
pca.fit(X)
X_100D = pca.transform(X)
X_recovered = pca.inverse_transform(X_100D)

# Visualize the recovered faces
for i in xrange(0, 100):
    plt.subplot(20,10,i+101)
    x = X_recovered[i,:].reshape(32, 32)
    plt.imshow(np.rot90(x, 3), cmap='gray')
    plt.xticks(()), plt.yticks(())
plt.show()
