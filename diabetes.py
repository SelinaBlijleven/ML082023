# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:08:54 2023

@author: (Se)lina Blijleven
"""
#%% Imports

# This cell can hold all of our imports, so we only have to run them once.

# Pandas for DataFrames and Series (importing as pd is convention)
import pandas as pd

#%% Load data

# Load the CSV in the same folder containing diabetes data into a Pandas Dataframe.
diabetes = pd.read_csv('diabetes.csv')

# Print the names of the columns we have in our DataFrame.
print(diabetes.columns)

