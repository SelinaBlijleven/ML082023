# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:30:13 2023

@author: selin
"""

#%% Load data 2
# Seaborn for visualization and datasets
import seaborn as sns

# Load the car crash data from seaborn (will use the internet to download the set as well)
car_crashes = sns.load_dataset('car_crashes')