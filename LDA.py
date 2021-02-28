#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:13:28 2021

@author: elisabethjones
"""

###########################################################
###########################################################
# This script was created to cconduct a LDA topic model on local government dat related to COVID-19
#Author: Elisabeth Jones
#Date Created: 1-23-21
###########################################################
###########################################################


###########################################################
#pandas
import pandas as pd
#import data frames - select for each city
ALLCITY_local = pd.read_csv(r"C:\Users\ejones3387\Desktop\Thesis\DataFiles\ALLCITY_local.csv")
SD = ALLCITY_local[ALLCITY_local["CITY"] == "SD"]

tweets = SD['full_text'] 
tweets.describe()

################### Basic Cleaning ##########################

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string
import re

#cleaning function
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
cleaned_tweet = tweets.apply(processTweet)

# remove special characters, numbers, punctuations
tidy_tweet = cleaned_tweet.str.replace("[^a-zA-Z#]", " ")
tidy_tweet = tidy_tweet.str.replace("\\", " ");

#check tidy tweet 
tidy_tweet.head()

############### Stop words and creating word list #####################
 
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#update stopwords basesd on city 
stop_words = stopwords.words('english')
stop_words.extend(['http','https','via','amp','_', 'coronavirus', 'Wuhan',
'quarantine', 'nCoV', 'pandemic', 'COVID-19', 'covid', 'covid-19', 'CityofSanDiego',
'Faulconer', 'SDMayorsOffice', 'SanDiegoCounty', 'SDCountyHHSA','san', 'diego', 'covid19',
'sandiegocounty', 'faulconer', 'kevin', 'mayor', 'county', 'sdcountyhhsa',
'public', 'health', 'cityofsandiego','sandiego', 'diegans'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]


data = tidy_tweet.tolist()
data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)

####################### Create Dictionary ##########################
import gensim.corpora as corpora

id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1][0][:30])

####################### LDA Model Training #########################
from pprint import pprint

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics= 5,
                                       chunksize=100,
                                       passes = 15,
                                       random_state=10) #tune parameters as needed

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())


doc_lda = lda_model[corpus]

################### perplexity and coherence metrics #####################

#perplexity, the lower the better
print('\nPerplexity: ', lda_model.log_perplexity(corpus,total_docs=10000))  

# Compute Coherence Score, the higher the better
from gensim.models.coherencemodel import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda) #higher the better

################### OPTIONAL: Tuning with c_v #####################
##ran on vpn eng computer took 7 hrs to get to 41% compelte. 

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=k, 
                                           random_state=10,
                                           chunksize=10,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=texts, 
                                         dictionary=id2word, 
                                         coherence='c_v')
    
    return coherence_model_lda.get_coherence()


import tqdm
import numpy as np

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
               gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
               corpus]
corpus_title = ['75% Corpus', '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run - need to run on a higher functioning computer
if 1 == 1:
    pbar = tqdm.tqdm(total=540)
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv(r'C:/Users/ejones3387/Desktop/Thesis/DataFiles/LA_lda_tuning_results.csv', index=False)
    pbar.close()

################### visualizing with pyLDAvis #####################

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
pyLDAvis.show(vis)



