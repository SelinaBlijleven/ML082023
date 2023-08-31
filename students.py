# -*- coding: utf-8 -*-
"""
students.py

Created on Thu Aug 31 10:31:48 2023

@author: 
"""

#%% Imports

# Helper methods (for alphabet)
import string

# NumPy for linear spaces
import numpy as np

# Pandas for data structures
import pandas as pd

# Pre-processors
from sklearn.preprocessing import StandardScaler

# Pipeline
from sklearn.pipeline import Pipeline

# GridSearch for parameters incl. cross-validation
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.svm import SVC

import seaborn as sns
sns.set()

#%% Load data

unmapped_students = pd.read_csv('Data/student_data.csv')

#%% Feature config

# Re-run this cell to empty the features
X = pd.DataFrame()

# The columns that contain numerical values without meaningful relationships
mappable_columns = [
    "Marital status",
    "Nacionality",
    "Application mode",
    "Course",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Gender",
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "International"
]

# Columns we can (and should) scale
scaling_columns = [
    "Application order",
    "Age at enrollment",
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)', 
    'Unemployment rate',
    'Inflation rate', 
    'GDP'
]

#%% Data exploration

sns.pairplot(unmapped_students[scaling_columns[:4] + ['Target']], hue='Target');

#%% Pre-processing: categorical variables

# Copy the unmapped version of the data
students = unmapped_students.copy()

# A dictionary of the alphabet, so we can apply OneHotEncoding more easily.
alphabet = dict(zip(range(0, 52), string.ascii_lowercase + string.ascii_uppercase))

# Helper function to map a number to a corresponding letter in the alphabet
def map_to_alphabet(x): 
    return alphabet[x]

# Loop over the columns that contain the numerical data
for column in mappable_columns:
    
    # Map the numerical value to a letter, since the numbers in these columns 
    # are categorical.
    students[column] = students[column].apply(map_to_alphabet)
    
    # One-hot encoding for this feature
    enc = pd.get_dummies(students[column], prefix=column)
    
    X = pd.concat([X, enc], axis=1)
    
#%% Pre-processing: normalization

X[scaling_columns] = StandardScaler().fit_transform(students[scaling_columns])
    
#%% Pipeline steps
# Dimensionality reduction?
# Evaluation?

estimator = SVC(kernel="linear")

pipe = Pipeline(steps=[
    ("estimator", estimator)
])

param_grid= {
     'estimator__C': np.linspace(1, 1.2, 3),         # Regularization parameter
     'estimator__degree': range(1, 5),
     'estimator__gamma': np.linspace(0.1, 0.5, 5)
}

model = GridSearchCV (
    pipe,
    param_grid=param_grid,
    verbose=10
)

# Try to fit our features to the target column, which indicates whether a student is still 
# enrolled, dropped out or graduated.
model.fit(X, students['Target'])

# Find best model parameters
print(model.best_params_)

# # find best model score
print(model.score(X, students['Target']))
