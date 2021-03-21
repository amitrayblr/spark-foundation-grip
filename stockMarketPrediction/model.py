# Importing the required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Reading data 
headlineData = pd.read_csv('data/indianewsheadlines.csv')
# print(headlineData.head(10))

# Cleaning data
headlineData['publish_date'] = pd.to_datetime(headlineData['publish_date'], format='%Y%m%d')
headlineData.drop('headline_category', axis = 1, inplace = True)
headlineData.replace("[^a-zA-Z']", ' ', regex=True, inplace=True)
print(headlineData.head(10))

# Grouping data with respect to date
headlineData['headline_text'] = headlineData.groupby(['publish_date']).transform(lambda x : ' '.join(x)) 
headlineData = headlineData.drop_duplicates() 
headlineData.reset_index(inplace = True, drop = True)
# print(headlineData.head(10))

# # Calculating polarity and subjectivity
headlineData['polarity'] = headlineData.apply(lambda row: TextBlob(row['headline_text']).sentiment.polarity, axis=1)
headlineData['subjectivity'] = headlineData.apply(lambda row: TextBlob(row['headline_text']).sentiment.subjectivity, axis=1)

headlineData['compound'] = headlineData.apply(lambda row: SentimentIntensityAnalyzer().polarity_scores(row['headline_text'])['compound'], axis=1)
headlineData['negativity'] = headlineData.apply(lambda row: SentimentIntensityAnalyzer().polarity_scores(row['headline_text'])['neg'], axis=1)
headlineData['neutrality'] = headlineData.apply(lambda row: SentimentIntensityAnalyzer().polarity_scores(row['headline_text'])['neu'], axis=1)
headlineData['positivity'] = headlineData.apply(lambda row: SentimentIntensityAnalyzer().polarity_scores(row['headline_text'])['pos'], axis=1)

print(headlineData.head())
print(headlineData.tail())