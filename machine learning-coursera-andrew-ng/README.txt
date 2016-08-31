Date: 31 Aug, 2016

This directory contains Python 2.7 code for solving the programming assignments in Andrew Ng's course on Machine Learning on Coursera.

It demonstrates the use of staple Python data science libraries (numpy, pandas, matplotlib and scikit-learn) to implement some of the most common machine learning techniques:
1. Linear regression (Univariate as well as Multivariate)
2. Regularization of regression
3. Logistic regression (2-class as well as Multiclass)
4. Support Vector Machines (linear as well as Gaussian kernels) [Also includes code for a spam email classifier built using SVM]
5. Clustering using kmeans algorithm [Also includes code for an image compressing application built using kmeans]
6. Dimensionality Reduction using Principal Components Analysis
7. Anomaly Detection using Gaussian probability modeling
8. Collaborative Filtering [Also includes code for a movie recommender application built using Collaborative Filtering]

Each file contains a further description of the code within it. All these files have been tested successfully and have results that match closely with those that you would get if you use the MATLAB/Octave implementations of the assignments provided on the course website. Please note that the assignments corresponding to Neural Networks are not included in this directory as I could not yet implement an off-the-shelf neural network module. As an FYI, sklearn v0.17 (the current stable version) does not have any off-the-shelf module for neural networks. Such modules are present in sklearn v0.18 (dev), PyBrain and PyLearn. However, do check out the directory "neural-networks-from-scratch" where I have written Python 2.7 code that implements a neural network from scratch and uses it to predict the MNIST handwritten digits.
