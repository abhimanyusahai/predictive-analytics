# This Python 2.7 code implements the second part of the programming assignment from Week 9 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build a movie recommender system in Python using the collaborative filtering algorithm.
# It uses numpy, pandas and scipy, and the actual collaborative filter has been coded from scratch. It does, however use
# an off-the-shelf module from scipy.optimize to implement the CG solver for minimizing the cost function. For the details of the
# programming assignment, sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/ and access Week 9.

import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.optimize import minimize

# Helper function to compute cost as a function of Xtheta
def compute_cost(Xtheta):
    X = Xtheta[:(num_movies * n)].reshape(num_movies, n) # Roll out X
    Theta = Xtheta[(num_movies * n):].reshape(num_users, n) # Roll out Theta
    cost = 0.5 * np.sum(R * ((np.dot(X, np.transpose(Theta)) - Y) ** 2)) + (float(reg_lambda)/2) * (np.sum(Theta ** 2) + np.sum(X ** 2))
    return cost

# Helper function to compute gradient of cost as a function of Xtheta
def compute_gradient(Xtheta):
    X = Xtheta[:(num_movies * n)].reshape(num_movies, n) # Roll out X
    Theta = Xtheta[(num_movies * n):].reshape(num_users, n) # Roll out Theta
    X_grad = np.dot((R * (np.dot(X, np.transpose(Theta)) - Y)), Theta) + float(reg_lambda) * X
    Theta_grad = np.dot(np.transpose(R * (np.dot(X, np.transpose(Theta)) - Y)), X) + float(reg_lambda) * Theta
    Xtheta_grad = np.concatenate((X_grad.reshape((num_movies * n),), Theta_grad.reshape((num_users * n),)))
    return Xtheta_grad

# Load the input data
In = sio.loadmat('data/week-9/ex8_movies.mat')
Y = In['Y']
R = In['R']
movies = pd.read_table('data/week-9/movie_ids.txt', names=['movie_name'])
# Remove movie index numbers from the names
for i in xrange(0, movies.shape[0]):
    movies['movie_name'][i] = movies['movie_name'][i].split(' ', 1)[1]

# Create our own user column with our preferences and concatenate them to Y and R
my_ratings = np.zeros((Y.shape[0],1))
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5
r = (my_ratings != 0).astype(int)
Y = np.c_[my_ratings, Y]
R = np.c_[r, R]

# Normalize Y
Ymean_movies = np.zeros((Y.shape[0], 1))
for i in xrange(0, Y.shape[0]):
    Ymean_movies[i, 0] = np.mean(Y[i, :][R[i, :] == 1])
    Y[i, :][R[i, :] == 1] = Y[i, :][R[i, :] == 1] - Ymean_movies[i, 0]

# Define some useful parameters and matrices
num_movies = Y.shape[0]
num_users = Y.shape[1]
n = 10 # Number of parameters defining each user and each movie
reg_lambda = 10 # Regularization parameter
num_Xthetas = (num_movies * n) + (num_users * n) # Total number of model parameters
num_iters = 100 # Number of iterations of the minimization algorithm

# Perform collaborative filtering and find optimal parameters
# Initialize Xtheta
initial_Xtheta = np.random.random(num_Xthetas).reshape(num_Xthetas,) * 2 * 0.12 - 0.12 # epsilon taken to be 0.12
# Minimize cost to find optimal Xtheta
optimal_Xtheta = minimize(compute_cost, initial_Xtheta, method='CG', jac=compute_gradient, options={'maxiter': num_iters}).x
optimal_X = optimal_Xtheta[:(num_movies * n)].reshape(num_movies, n)
optimal_Theta = optimal_Xtheta[:(num_users * n)].reshape(num_users, n)

# Compute predictions matrix and print out top 10 movie recommendations for us
predictions = np.dot(optimal_X, np.transpose(optimal_Theta)) + Ymean_movies
my_predictions = predictions[:,0]
movies['Predicted rating'] = my_predictions
movies.sort(columns=['Predicted rating'], ascending=False, inplace=True)
print "The top 10 movie recommendations for you are: \n", movies.head(10)['movie_name'].to_string(index=False)
