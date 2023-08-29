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
clip_data = pd.read_clipboard(sep=',')

#%% Apply some linear regression

# Create the model
model = LinearRegression(fit_intercept=True)

# We need to reshape because we only have one feature. If we do not reshape 
# it looks like we are providing a list of features instead.
X = clip_data['x'].to_numpy().reshape(-1, 1)

# Fit the model
model.fit(X, clip_data['y'])

print(model.coef_)
print(model.intercept_)

yfit = model.predict(X)

#%% Plot the data we drew

plt.scatter(clip_data['x'], clip_data['y']);
plt.plot(clip_data['x'], yfit);

