# This Python 2.7 code implements the first part of the programming assignment from Week 8 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build clustering systems in Python using the kmeans algorithm. The first part of the
# code demonstrates how to use numpy, pandas, scipy, matplotlib and an off-the-shelf library for kmeans on scikit learn. The
# second part then uses kmeans to compress the image of a bird from 256 x 256 x 256 colors to 16 colors. For the details of the
# programming assignment, sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/ and access Week 8.

import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Exercise 1: k-means clustering on a 2D dataset

# Load and visualize the input data
X = sio.loadmat('data/week-8/ex7data2.mat')['X']
plt.scatter(X[:,0], X[:,1], s=20)
plt.title("Initial (Unclustered) data")
plt.show()

# Perform k-means clustering on the input data with 3 clusters
kmeans = KMeans(3)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Replot the data indicating the discovered clusters
plt.scatter(X[clusters == 0][:,0], X[clusters == 0][:, 1], s=20, color='red')
plt.scatter(X[clusters == 1][:,0], X[clusters == 1][:, 1], s=20, color='blue')
plt.scatter(X[clusters == 2][:,0], X[clusters == 2][:, 1], s=20, color='green')
plt.show()


# --- Exercise 2: Compressing an image using k-means clustering

# Load the input image
X = sio.loadmat('data/week-8/bird_small.mat')['A']
X = X.astype(float)/255

# Create a (n_pixels * 3) dimension matrix to perform k-means clustering
X_temp = np.zeros((X.shape[0]*X.shape[1], X.shape[2]))
for i in xrange(0, X.shape[2]):
    X_temp[:,i] = X[:,:,i].reshape(X.shape[0]*X.shape[1],) # Create 1 column vector for each color

# Perform k-means clustering to determine the 16 R,G,B coordinates that best represent the picture
kmeans = KMeans(16)
kmeans.fit(X_temp)
color_clusters = kmeans.predict(X_temp)

# Create compressed output image matrix
y_temp = np.zeros((X_temp.shape[0], X_temp.shape[1]))
for i in xrange(0, y_temp.shape[1]):
    for j in xrange(0, y_temp.shape[0]):
        y_temp[j, i] = kmeans.cluster_centers_[color_clusters[j], i]
y = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
for i in xrange (0, y.shape[2]):
    y[:, :, i] = y_temp[:, i].reshape(X.shape[0], X.shape[1])

# Visualize the input and the compressed image
plt.subplot(1,2,1)
plt.title("Input Image")
plt.imshow(X)
plt.xticks(())
plt.yticks(())
plt.subplot(1,2,2)
plt.title("Compressed Image")
plt.imshow(y)
plt.xticks(())
plt.yticks(())
plt.show()
