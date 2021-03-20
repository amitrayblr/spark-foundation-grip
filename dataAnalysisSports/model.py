# Importing the required libraries
import pandas as pd
import numpy as np

# Reading data 
matchData = pd.read_csv('data/matches.csv')
deliveryData = pd.read_csv('data/deliveries.csv')
print(matchData.head(10))
print(deliveryData.head(10))