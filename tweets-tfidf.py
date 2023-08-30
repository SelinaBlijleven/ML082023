# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:23:07 2023

@author: (Se)lina Blijleven
"""

#%% Imports

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Cross-validator (also does fitting, prediction and scoring)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Dimensionality reduction
from sklearn.decomposition import PCA

# Machine learning models
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score

#%% Loading data

# Load the CSV in the same folder containing diabetes data into a Pandas Dataframe.
tweets = pd.read_csv('./Data/political_social_media.csv', encoding="latin-1")

#%% Data pre-processing

# We use TF-IDF for relative word frequency, since some texts might be longer 
# than others.
vec = TfidfVectorizer()

# Transform the text into word vectors we can use as features.
X = vec.fit_transform(tweets['text'])

# Put them into a labeled DataFrame for convenient access
text_vecs = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

# Our tweets contain 18220 unique tokens. We can still use the vectors to run simple 
# algorithms, but ideally would like to reduce and determine more useful features.
pca = PCA(n_components=20)
reduced_vecs = pca.fit_transform(text_vecs)

#%% Exploration: message distribution

# If we want to classify the type of message we are receiving, we should 
# look into the distribution to see if there are any oddities. If one or more 
# categories are underrepresented or overrepresented they will hurt performance.
sns.histplot(tweets, x='message')

# Rotate the labels for readability
plt.xticks(rotation=45)

#%% Prediction with GridSearch

estimator = KNeighborsClassifier()
params = estimator.get_params()

param_grid= {
    'n_neighbors': range(50, 101, 5)
}

model = GridSearchCV (
    estimator=estimator,
    param_grid=param_grid
)

model.fit(reduced_vecs, tweets['message'])

# Find best model parameters
print(model.best_params_)

# find best model score
print(model.score(reduced_vecs, tweets['message']))

#%% Prediction without GridSearch

# Which column would we like to predict?
# In this case source/message (type) are options, but feel free to experiment.
pred_column = 'message'

# n for n-fold cross validation
n = 10

# Type of ML model to use, just pick one available from sklearn.


# Execute fitting, prediction and scoring.
# We could make our lives even easier by using a scikit-learn pipeline
scores = cross_val_score(
    model,
    reduced_vecs,       # https://github.com/scikit-learn/scikit-learn/issues/27180 
    tweets[pred_column],
    cv=n
)

# Average the accuracy scores for each model
print(f"Average score after cross-validation: {scores.mean()}")