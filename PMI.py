#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################################
###########################################################
# This script was created to cconduct a pointwise mutual information (PMI) analysis on local government dat related to COVID-19
#Author: Elisabeth Jones
#Date Created: 2-18-21
###########################################################
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
import numpy as np
from nltk.corpus import stopwords

#import data frames
DATA = pd.read_csv(r"C:\\Users\\ejones3387\\Desktop\\Thesis\\DataFiles\\ALLCITY_local.csv")

#taking a sample to test:
#DATA = DATA.sample(1000) #you might want to change this var name to lowercase, I regret the all caps


###########################################################
#Data cleaning
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
DATA['cleaned_text'] = DATA['full_text'].apply(processTweet)

# remove special characters, numbers, punctuations
DATA['tidy_tweet'] = DATA['cleaned_text'].str.replace("[^a-zA-Z#]", " ")
DATA['tidy_tweet'] = DATA['cleaned_text'].str.replace("\\", " ");
DATA['bow'] = DATA['tidy_tweet'].str.split()
DATA = DATA[DATA['bow'].isnull()==False] #creating bag of words, needed for PMI

#stopwords - update for your data
more_stopwords = ['http','https','via','amp','_', 'coronavirus', 'Wuhan',                
'quarantine', 'nCoV', 'pandemic', 'COVID-19', 'covid', 'covid-19', 'covid19', 
'covid19', 'los', 'garcetti','mayor', 'eric', 'angeles', "lacity", "@lacity",
"mayorofla", "@mayorofla", "countyofla", "@countyofla", "@lapublichealth", '@ericgarcetti'
"lapublichealth", 'angelenos', 'nycgov', '@nycgov', 'Blasio', 'blasio', 'NYCMayor', '@nycmayor', 'nycmayor',
'BilldeBlasio', 'billdeblasio', '@billdeblasio', '@nychealthy''nycHealthy', 
'nychealthy','nycmayorsoffice', '@nycmayorsoffcie','deblasio', 'nyc', 'nycmayor',
'york', 'bill', 'mayor', 'new', 'city', 'yorkers', '@kevinfaulconer', 'are…',
'CityofSanDiego', '@cityofsandiego', 'Faulconer', 'SDMayorsOffice', 
'@sdmayorsoffice', 'SanDiegoCounty', '@sandiegocounty','SDCountyHHSA', 
'@sdcountyhhsa', 'san', 'diego', 'sandiegocounty', 'faulconer', 'kevin', 'mayor',
'county', 'sdcountyhhsa', 'cityofsandiego','sandiego', 'diegans', '300', "didn’",
'CityofDetroit','cityofdetroit', 'duggan', 'Duggan', 'MayorMikeDuggan', 'mayormikeduggan',
'waynecountymi', '@waynecountymi', 'DetHealth', '@dethealth',
'detroit', 'Detroit', 'mike', '@cityofdetroit', '@mayormikeduggan']
stoplist = list(stopwords.words("english")) + more_stopwords 


###########################################################
#PMI calculation
###########################################################

#define subset based on 'CITY' - in this case Detoroit 
CITY = DATA[DATA['CITY']=='DT'] 

# This part is optional depending on the data   
DATA['CITY'].fillna("unknown", inplace= True)
Unknown = DATA[DATA['CITY'] == 'unknown']
#view
CITY.head(2)
CITY['bow'].head(3)

#get the concatenated strings of word in each subset (city)
list_of_tweets_all = DATA['bow'].to_list()
new_list_all = []

for a in list_of_tweets_all:
    for b in a:
        new_list_all.append(b)

list_of_tweets_CITY = CITY["bow"].to_list()
new_list_CITY = []

for a in list_of_tweets_CITY:
    for b in a:
        new_list_CITY.append(b)

#remove stop words
def remove_stopwords(texts):
    return [word for word in texts if word not in stoplist] 

new_list_all = remove_stopwords(new_list_all)
new_list_CITY = remove_stopwords(new_list_CITY)


#PMI columns
all = pd.value_counts(new_list_all)
freq = pd.DataFrame({'All': all})

A = pd.value_counts(new_list_CITY)
freq['CITY'] = A

#view
freq.head()


#PMI calculation column
a = freq['CITY'] * np.sum(freq['All'])
b = freq['All'] * np.sum(freq['CITY'])
c = a / b
freq['PMI'] = abs(np.log2(c))

#view
freq.head()

#The limit below may need to be changed to a smaller number for small datasets or those with tweets with few words
freq2 = freq[freq['CITY'] > 3] # limit = 3. Try different numbers.
freq2.sort_values(by = 'PMI', ascending = False)
words = [w for w in freq2.index.values]

freq3 = freq2.loc[words] # Update freq2 to freq3 by adding 'words'
freq3.sort_values(by = 'PMI', ascending = False)
new = freq3.sort_values(by = 'PMI', ascending = False)
new[10:20]

freq3['words'] = freq3.index
freq3.reset_index()
freq_hi = freq3[freq3['PMI'] > 1]


###########################################################
#PMI wordcloud, PMI > 1
###########################################################

#all 
freq_hi.sort_values(by = 'PMI', ascending= False)[:15]

w = dict(freq_hi['PMI'])
j = WordCloud(width=1000,height=1000, 
              background_color='white',
              prefer_horizontal=1).generate_from_frequencies(w)

plt.imshow(j, interpolation = 'bilinear')
plt.axis('off')

j.to_file(r"C:\Users\ejones3387\Desktop\Thesis\DataFiles\PMI_wclds\DT\ALLpmiover1.png")

#city 
freq_hi.sort_values(by = 'CITY', ascending= False)[:15]
freq_hi['CITY'] = freq_hi['CITY'].astype(float)
v = dict(freq_hi['CITY'])
i = WordCloud(width=1000,height=1000, 
              background_color='white', 
              prefer_horizontal=1).generate_from_frequencies(v)

plt.imshow(i, interpolation = 'bilinear')
plt.axis('off')

i.to_file(r"C:\Users\ejones3387\Desktop\Thesis\DataFiles\PMI_wclds\DT\DTpmiover1.png")

###########################################################
#PMI wordcloud, PMI > 1 and All > city
###########################################################

freq_high = freq_hi[freq_hi['All'] > freq_hi['CITY']] 

#all
freq_high.sort_values(by = 'PMI', ascending= False)[:15]

w = dict(freq_high['PMI'])
j = WordCloud(width=1000,height=1000, 
              background_color='white',
              prefer_horizontal=1).generate_from_frequencies(w)

plt.imshow(j, interpolation = 'bilinear')
plt.axis('off')

j.to_file(r"C:\Users\ejones3387\Desktop\Thesis\DataFiles\PMI_wclds\DT\ALL_pmiover1alloverciy.png")

#city
freq_high.sort_values(by = 'CITY' , ascending= False)[:15] 

freq_high['CITY'] = freq_high['CITY'].astype(float) 
v = dict(freq_high['CITY'])
i = WordCloud(width=1000,height=1000, 
              background_color='white',
              prefer_horizontal=1).generate_from_frequencies(v)

plt.imshow(i, interpolation = 'bilinear')
plt.axis('off')

i.to_file(r"C:\Users\ejones3387\Desktop\Thesis\DataFiles\PMI_wclds\DT\DT_pmiover1alloverciy.png")


