#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:23:37 2019

@author: krazy
"""

'''
Importing required libraries
'''
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Global Parameters
stop_words = set(stopwords.words('english'))


'''
Loading dataset for the given location and columns

Target -> (0 = negative, 2 = neutral, 4 = positive)
'''
def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

'''
Removing unwanted cols from the dataset
'''
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

'''
Processing tweet text data.
Converting to Lower case, removing urls, removing @users and #, removing puntuations, removeing stopwords from nltk library

Important:
    I have implemented the Porter stemming. The operation is itself computational intensive and also model's accuracy is not improved.
    Not applying Leamatization
'''
# Preprocessing tweets
def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    '''
    Removing it from the process. With Porter I am getting 2-3% less accuracy + It is taking 3 more 
    minutes on my system to execute this step
    '''
    #ps = PorterStemmer()
    #stemmed_words = [ps.stem(w) for w in filtered_words]
    
    return " ".join(filtered_words)

'''
Create feature vector of text.
I am using m=the testing data as the corpus to make tf-idf vector. Same vector structure is used for training and testing purpose.
Important:
    Use "vector" for transforming your text if you want to use the trained model
'''
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

'''
Converting sentiment integer to string.
Target -> (0 = negative, 2 = neutral, 4 = positive)
'''
def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"
    

'''
*******************Building Model***************************
'''
# Load dataset
dataset = load_dataset("data/training.csv", ['target', 't_id', 'created_at', 'query', 'user', 'text'])
# Remove unwanted columns from dataset
n_dataset = remove_unwanted_cols(dataset, ['t_id', 'created_at', 'query', 'user'])
#Preprocess data
dataset.text = dataset['text'].apply(preprocess_tweet_text)
# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print(accuracy_score(y_test, y_predict_nb))

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)
print(accuracy_score(y_test, y_predict_lr))



'''
*******************Integrating the output received from TWITTER BOT to predicting the sentiments for the trending Hashtags***************************
'''
# Loading dataset
test_ds = load_dataset("trending_tweets/30-11-2019-1575134256-tweets.csv", ["t_id", "hashtag", "created_at", "user", "text"])
test_ds = remove_unwanted_cols(test_ds, ["t_id", "created_at", "user"])

# Creating text feature
test_ds.text = test_ds["text"].apply(preprocess_tweet_text)
test_feature = tf_vector.transform(np.array(test_ds.iloc[:, 1]).ravel())

# Using Logistic Regression model for prediction
test_prediction_lr = LR_model.predict(test_feature)

# Averaging out the hashtags result
test_result_ds = pd.DataFrame({'hashtag': test_ds.hashtag, 'prediction':test_prediction_lr})
test_result = test_result_ds.groupby(['hashtag']).max().reset_index()
test_result.columns = ['heashtag', 'predictions']
test_result.predictions = test_result['predictions'].apply(int_to_string)

print(test_result)