#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################################
###########################################################
# This script was created to cconduct a sentiment analysis using TextBlob on local government tweets
#Author: Elisabeth Jones
#Date Created: 2-19-21
###########################################################
###########################################################

#############################################################
#subjecctivity: (0-1) where 0 indicats the text is object and 1 indactes subjectivity
#polarity: (-1-1) where negative score is a negative comment and positive score a positive comment
#############################################################

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


#import data frame and create df excluding retweets (dfrt) - need to update for each city 
df = pd.read_csv(r"C:\\Users\\ejones3387\\Desktop\\Thesis\\DataFiles\\DT_local.csv")
df.columns

df['duplicates'] = df.duplicated('full_text')
df = df.loc[df['duplicates'] == False] #521 excluding retweets


#df['rtflag'] = df['retweeted_id'].notnull()
#df['rtflag'].head()

#dfrt = df.loc[df['rtflag']== False]

####################### Sentiment Analysis #########################

# The x in the lambda function is a row (because I set axis=1)
# Apply iterates the function to [‘text’] field  across the dataframe's rows
df['polarity'] = df.apply(lambda x: TextBlob(x['full_text']).sentiment.polarity, axis=1)
df['subjectivity'] = df.apply(lambda x: TextBlob(x['full_text']).sentiment.subjectivity, axis=1)

#creating boolean positive and subjective columns
df['positive'] = df['polarity'] > 0
df['subjective'] = df['subjectivity'] > .5

#set sns color palatte
colors = ["#2F5F8A", "#2F5F8A"]
sns.set_palette(sns.color_palette(colors))
sns.set_style('white')

##bar graph of overall pos vs neg
g = sns.factorplot(x='positive', 
                   data=df, 
                   aspect=1.5, 
                   kind='count')
g.set(xticklabels = ['negative', 'positive'])
g.set_xlabels(" ")
#plt.title("Local Government Positivity")

g.savefig(r'C:\Users\ejones3387\Desktop\Thesis\DataFiles\sentiment\DT\nort_pos.png')

###bar graph of overall subjectivity
sns.set_style('white')
f = sns.factorplot(x='subjective', 
                   data=df, 
                   aspect=1.5, 
                   kind='count')
f.set(xticklabels = ['objective', 'subjective'])
f.set_xlabels(" ")

f.savefig(r'C:\Users\ejones3387\Desktop\Thesis\DataFiles\sentiment\DT\nort_subj.png')


####################### Mayoral slice - Sentiment Analysis #########################


#mayroal slice - update for each city 
df['duggan'] = df["full_text"].str.contains("Duggan|MayorMikeDuggan")
duggan = df.loc[df["duggan"] == True]


#set sns color palatte turq =#00868B purp=#270e59 grn=#83d070 ylw#f8cf1d
colors = ["#2F5F8A", "#2F5F8A"]
sns.set_palette(sns.color_palette(colors))
sns.set_style('white')

##bar graph of overall pos vs neg
g = sns.factorplot(x='positive', 
                   data=duggan, 
                   aspect=1.5, 
                   kind='count')
g.set(xticklabels = ['negative', 'positive'])
g.set_xlabels(" ")
#plt.title("Local Government Positivity")

g.savefig(r'C:\Users\ejones3387\Desktop\Thesis\DataFiles\sentiment\DT\Duggan_nort_pos.png')

##bar graph of overall subjectivity
sns.set_style('white')
f = sns.factorplot(x='subjective', 
                   data=duggan, 
                   aspect=1.5, 
                   kind='count') 
f.set(xticklabels = ['objective', 'subjective'])
f.set_xlabels(" ")

f.savefig(r'C:\Users\ejones3387\Desktop\Thesis\DataFiles\sentiment\DT\duggan_nort_subj.png')

















