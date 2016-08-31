# This Python 2.7 code implements the second part of the programming assignment from Week 7 of Andrew Ng's course on Machine Learning
# on Coursera. It demonstrates how to build an email spam classifier using Support Vector Machines in Python. The first part of the
# code demonstrates how to use Python regex and an off-the-shelf word-stemming module to process email text and convert it into a feature
# vector that can be fed into the model. The latter part of the code then uses numpy, pandas, scipy, matplotlib and an off-the-shelf
# module for SVMs in scikit-learn to do spam classification for a dataset of 6000 emails. For the details of the programming assignment
# sign up for this course on Coursera at https://www.coursera.org/learn/machine-learning/ and access Week 7.

import numpy as np
import pandas as pd
import scipy.io as sio
import re
from stemming.porter2 import stem
from sklearn import svm

# Helper function to pre-process input email. The output is a feature vector
# of the same size as the reference spam vocabulary list, with 1s where the
# spam word (at the corresponding index) is present and 0s otherwise
def email_feature_vector(email, vocab_list):
    # Preprocess the email
    # Convert contents to lower case
    hold = email.lower()
    # Replace HTML tags with space
    hold = re.sub(r'<[^<>]+>', ' ', hold)
    # Normalize URLs
    hold = re.sub(r'(http|https)://[\S]+', 'httpaddr', hold)
    # Normalize email addresses
    hold = re.sub(r'[\S]+@[\S]+', 'emailaddr', hold)
    # Normalize numbers
    hold = re.sub(r'[0-9]+', 'number', hold)
    # Normalize dollar sign(s)
    hold = re.sub(r'[$]+', 'dollar', hold)
    # Now iterate through the contents to further process each word and compare
    # with the reference vocabulary list to see if there is a match
    hold_array = re.split(r'@|/|#|\.|-|:|&|\*|\+|=|\[|]|\?|!|\(|\)|{|}|,|\'|"|>|_|<|;|%|\s', hold) # Split on all these characters
    word_indices = np.array([]) # Will be used to capture indices of words in the email that also appear in vocab list
    for word in hold_array:
        # Remove all non-alphanumeric characters
        word = re.sub(r'[^a-z0-9]', '', word)
        # Stem the word using Porter stemmer
        try:
            word = stem(word)
        except:
            continue
        # Skip the word if it's too short
        if len(word) < 1:
            continue
        index = np.where(vocab_list == word)[0]
        if index.size != 0:
            word_indices = np.append(word_indices, index).astype(int)
    word_indices = word_indices.flatten()
    # Create the feature vector for the email
    feature_vector = np.zeros((vocab_list.shape[0],))
    for i in word_indices:
        feature_vector[i] = 1
    return feature_vector.astype(int)

# Load the input data
vocab_list = pd.read_table('data/week-7/vocab.txt', usecols=[1], names=['Word'])
vocab_list = np.array(vocab_list).reshape(vocab_list.shape[0],) # Convert to 1D numpy array
# Create feature vector for the sample email file
f = open('data/week-7/emailSample1.txt')
email = f.read()
f.close()
v1 = email_feature_vector(email, vocab_list)

# Above code was only to demonstrate how emails are coded as input for spam filters.
# Now load processed email train and test data for SVM implementation
X = sio.loadmat('data/week-7/spamTrain.mat')['X']
y = sio.loadmat('data/week-7/spamTrain.mat')['y'].reshape(X.shape[0],)
Xtest = sio.loadmat('data/week-7/spamTest.mat')['Xtest']
ytest = sio.loadmat('data/week-7/spamTest.mat')['ytest'].reshape(Xtest.shape[0],)

# Train an SVM classifer on the training data
svm_model = svm.SVC(C=0.1, kernel='linear')
svm_model.fit(X, y)

# Predict using the above model on both training and test sets and report accuracy
training_predictions = svm_model.predict(X)
test_predictions = svm_model.predict(Xtest)
training_accuracy = float(np.sum(training_predictions == y))/y.shape[0]
test_accuracy = float(np.sum(test_predictions == ytest))/ytest.shape[0]
print "Accuracy on training set is %0.1f%% and accuracy on test set is %0.1f%%" % ((training_accuracy*100), (test_accuracy*100))
