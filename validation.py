# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:58:16 2023

@author: Selina Blijleven
"""

#%% Imports

# Dataset from Scikit Learn
from sklearn.datasets import load_iris

# Clustering algorithm
from sklearn.neighbors import KNeighborsClassifier

# Cross-validation helper methods
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

#%% Load dataset
iris = load_iris()

#%% Prep-processing

# Use this if you used the load_iris function
X, y = iris.data, iris.target

#%% Clustering

model = KNeighborsClassifier(n_neighbors=1)

#%% N-fold cross-validation

# Number of times to split the data and get an accuracy for a specific 
# split.
n = 5

# Scikit Learn will make different splits of the data for us and return a 
# list of n accuracy scores.
nscores = cross_val_score(model, X, y, cv=n)
print(nscores)
print(nscores.mean())

#%% Leave One Out cross-validation
looscores = cross_val_score(model, X, y, cv=LeaveOneOut())
print(looscores)
print(looscores.mean())
