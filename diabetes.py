# -*- coding: utf-8 -*-
"""
An exploration of a diabetes dataset.

Created on Mon Aug 28 09:08:54 2023

@author: (Se)lina Blijleven
"""
#%% Imports
# This cell can hold all of our imports, so we only have to run them once.

# NUMerical PYthon for statistics and mathematical operations
import numpy as np

# Pandas for DataFrames and Series (importing as pd is convention)
import pandas as pd

# Seaborn for visualization
import seaborn as sns
sns.set_theme()
sns.set_palette("viridis")

# Scikit Learn: Learning library
from sklearn.preprocessing import StandardScaler

#%% Load data

# Load the CSV in the same folder containing diabetes data into a Pandas Dataframe.
diabetes = pd.read_csv('./Data/diabetes.csv')

# Print the names of the columns we have in our DataFrame.
print(diabetes.columns)

#%% Pre-processing

''' 
Removing zero/NaN (because this data is unusable and skews our results)

- Some patients have no glucose measurements
- Some have no blood pressure measurements 
    (often no skin thickness or insulin measurements either)
'''
db_ppd = diabetes[(diabetes.Glucose != 0) & (diabetes.BloodPressure != 0)]

'''
Some of our values are distributed inequally, with exceptionally large or small 
values occurring. We use the sklearn StandardScaler, to reduce the impact of any 
feature that uses big numbers.
'''
db_scaled = pd.DataFrame(
    StandardScaler().fit_transform(db_ppd), 
    columns = db_ppd.columns
)

#%% Exploration: Uni-variate
'''
Look at every column available in the DataFrame so we can 
plot a distribution plot.
'''
for column in diabetes.columns:
    
    # Plot a distribution of the 
    sns.displot(db_scaled, x=column, binwidth=1)    # Bin-width helps us with continuous variables

#%% Exploration 2: Multi-variate

# Determine the correlation matrix between variables
corr_mat = db_scaled.corr()

sns.heatmap(corr_mat, cmap='coolwarm', annot=True)

#%% Fitting

#%% Evaluation

#%% Fine-tuning

#%% Prediction