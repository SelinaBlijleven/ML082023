# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:23:07 2023

@author: (Se)lina Blijleven
"""

#%% Imports
import pandas as pd

import seaborn as sns
sns.set()

# Pipeline utility
from sklearn.pipeline import Pipeline

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Cross-validator (also does fitting, prediction and scoring)
from sklearn.model_selection import GridSearchCV

# Dimensionality reduction
from sklearn.decomposition import TruncatedSVD

# Machine learning models
from sklearn.neighbors import KNeighborsClassifier

#%% Loading data

# Load the CSV in the same folder containing diabetes data into a Pandas Dataframe.
tweets = pd.read_csv('./Data/political_social_media.csv', encoding="latin-1")

#%% Putting it all together

svd = TruncatedSVD()
print(svd.get_params())

knn = KNeighborsClassifier()
tfidf = TfidfVectorizer()

#%% 
pipe = Pipeline(steps=[("tfidf", tfidf),
                       ("svd", svd), 
                       ("knn", knn)])

param_grid= {
    'svd__n_components': range(42, 103, 5),
    'knn__n_neighbors': range(50, 101, 5)
}

model = GridSearchCV (
    pipe,
    param_grid=param_grid
)

model.fit(tweets['text'], tweets['message'])

# Find best model parameters
print(model.best_params_)

# find best model score
print(model.score(tweets['text'], tweets['message']))
