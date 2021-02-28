#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:59:53 2021

@author: elisabethjones
"""


###########################################################
# This script was created to clean and make a word cloud of tweets collected by the  
#   HDMA Center on the COVID-19 Vaccine
#Author: Elisabeth Jones
#Date Created: 1-7-21

#possible thigns to add:
    #merge all dfs to simplify
    #automate wordcloud process with loop
###########################################################

###########################################################
#Import data
###########################################################

#import libraries
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string
import re

#import data frames
data = pd.read_csv(r"/Users/elisabethjones/GoogleDrive/Thesis/DataFiles/LA_local.csv")

###########################################################
#data cleaning
###########################################################

#create function to clean tweets
def processTweet(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase
    tweet = tweet.lower()
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ') 
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet

#run function
data['cleaned_text'] = data['full_text'].apply(processTweet)

# remove special characters, numbers, punctuations
data['tidy_tweet'] = data['cleaned_text'].str.replace("[^a-zA-Z#]", " ")
data['tidy_tweet'] = data['cleaned_text'].str.replace("\\", " ");
data['bow'] = data['tidy_tweet'].str.split()
data_df = data[data['bow'].isnull()==False] #creating bag of words, needed for PMI
text = data_df['tidy_tweet'].tolist() #tidy tweets for word clouds
text = " ".join(text)

###########################################################
#Word Cloud
###########################################################
#stopwords for LA - change for each city
more_stopwords = ['http','https','via','amp','_', 'coronavirus', 'Wuhan',
'quarantine', 'nCoV', 'pandemic', 'COVID-19', 'covid', 'covid-19', 'garcetti',
 'mayor', 'eric', 'angeles', "lacity", "mayorofla", "countyofla", "lapublichealth"]
stopwords = list(STOPWORDS) + more_stopwords 

#generating word cloud
wordcloud = WordCloud(width = 800, height = 800, 
                prefer_horizontal=1,
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(text) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

###########################################################
#Save word cloud
###########################################################
wordcloud.to_file("/Users/elisabethjones/GoogleDrive/Thesis/DataFiles/Visuals/LA_wordcloud.png")

