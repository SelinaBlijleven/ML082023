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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#%% Loading data

# Load the CSV in the same folder containing diabetes data into a Pandas Dataframe.
tweets = pd.read_csv('./Data/political_social_media.csv', encoding="latin-1")

#%% TF-IDF on the message

vec = TfidfVectorizer()
X = vec.fit_transform(tweets['text'])
message_vecs = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

#%% Correlation matrix

corr_mat = tweets.corr()

sns.heatmap(corr_mat)

#%% Exploration: message distribution

sns.histplot(tweets, x='message')
plt.xticks(rotation=45)

#%%


#%% 

# Which column would we like to predict?
pred_column = 'source'

# n for n-fold cross validation
n = 10
model = GaussianNB()

scores = cross_val_score(model, message_vecs, tweets[pred_column], cv=n)

#%%
scores.mean()