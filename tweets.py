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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#%% Loading data

# Load the CSV in the same folder containing diabetes data into a Pandas Dataframe.
tweets = pd.read_csv('./Data/political_social_media.csv', encoding="latin-1")

#%% Sentiment analysis on text

# Create the analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    '''
    A function to return the sentiment scores for a piece of text, using VaderSentiment
    
    @param  str   The text to analyze
    @return dict  Negative (neg), neutral (neu), positive (pos) and compound (compound) scores
    '''
    sentiment_scores = analyzer.polarity_scores(text)
    return pd.Series(sentiment_scores)

# This is another example function that we could use with apply to flag certain words.
def is_controversial(text):
    return "potus" in text.lower() or "obama" in text.lower()

# For every message in the text column we apply the function to get the sentiment scores, 
# which are then appended to the original dataframe, giving us more features.
tweets[['neg', 'neu', 'pos', 'compound']] = tweets['text'].apply(get_sentiment_scores)

#%% Exploration: univariate distributions for sentiment analysis scores
corr_mat = tweets.corr()

sns.heatmap(corr_mat)

for score in ['neg', 'neu', 'pos', 'compound']:
    
    # Plot a distribution of the different scores
    sns.displot(tweets, x=score)
    plt.yscale('log')

#%% Exploration: message distribution

sns.histplot(tweets, x='message')
plt.xticks(rotation=45)

#%% Pre-process data

# We split into a training and test set, as well as their corresponding labels.
X_train, X_test, y_train, y_test = train_test_split(
    tweets[['neg', 'neu', 'pos', 'compound']], 
    tweets['message']
)

#%% Classification

# Construct the Naive Bayes Classifier
gnb = GaussianNB()

# Fit the model to the training data and labels
gnb.fit(X_train, y_train)

# Predict the results
y_pred = gnb.predict(X_test)

print(accuracy_score(y_test, y_pred))
