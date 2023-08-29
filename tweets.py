# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:23:07 2023

@author: selin
"""

#%% Imports

import pandas as pd

import seaborn as sns
sns.set()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#%% Loading data

# Load the CSV in the same folder containing diabetes data into a Pandas Dataframe.
tweets = pd.read_csv('./Data/political_social_media.csv', encoding="latin-1")

#%% Exploration
corr_mat = tweets.corr()

sns.heatmap(corr_mat)

#%% Sentiment analysis on text

analyzer = SentimentIntensityAnalyzer()
analyzer.polarity_scores(["i don't like anything ever :("])

#%% Pre-process data

