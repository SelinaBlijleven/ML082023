# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:34:46 2023

@author: selin
"""

#%% Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

#%% Load data

# Draw some data on drawdata.xyz and import with Pandas
# 1. What happens if you draw a (semi-straight) line and fit a linear regression?
# 2. What happens if you draw a polynomial function and fit a linear regression?
clip_data = pd.read_clipboard(sep=',')

#%% Prepare data

# We need to reshape because we only have one feature. If we do not reshape 
# it looks like we are providing a list of features instead.
X = clip_data['x'].to_numpy().reshape(-1, 1)

#%% Fitting the model

# Create the model. We also fit an intercept, since we might have a 
# minimum/maximum value for the data.
model = LinearRegression(fit_intercept=True)

# Fit the model to data
model.fit(X, clip_data['y'])

# The model results in weights: we can use these to interpret the function, 
# but plotting is easier.
#print(model.coef_)
#print(model.intercept_)

yfit = model.predict(X)

#%% Plot model + original data points

# Create a scatterplot for the original data
# Note: we only need the reshaped data for the model
plt.scatter(clip_data['x'], clip_data['y']);

# Plot the model that we fit in the same plot, illustrating the 
# correctness (or lack thereof) of our model
plt.plot(clip_data['x'], yfit, color='red');

