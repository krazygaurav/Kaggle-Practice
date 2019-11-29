#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:57:31 2019

@author: krazy
"""
import tweepy
import json
import schedule
import time
import datetime
import os
import csv

# Loading the API properties
def initiate_api():
    try: 
        with open('config.json', 'r') as f:
            config = json.load(f)        
        auth = tweepy.OAuthHandler(config["CONSUMER_KEY"], config["CONSUMER_SECRET"])
        auth.set_access_token(config["ACCESS_KEY"], config["ACCESS_SECRET"])
        api = tweepy.API(auth)
        return api
    except:
        print("Problems with config.json")
        return None

def get_woeid(api, locations):
    twitter_world = api.trends_available()
    #print(twitter_world)
    places = {loc['name'].lower() : loc['woeid'] for loc in twitter_world};
    woeids = []
    for location in locations:
        if location in places:
            woeids.append(places[location])
        else:
            print("err: ",location," woeid does not exist in trending topics")
    return woeids
    
def get_tweets(api, query):
    tweets = []
    #file = open("tweets.csv", "w+")
    #csv_writer = csv.writer(file)
    for status in tweepy.Cursor(api.search,
                       q=query,
                       count=1000,
                       result_type='popular',
                       include_entities=True,
                       monitor_rate_limit=True, 
                       wait_on_rate_limit=True,
                       lang="en").items():
     
        tweets.append([status.id_str, query, status.created_at.strftime('%d-%m-%Y %H:%M'), status.user.screen_name.encode('utf8'), status.text.encode('utf-8')])
        #csv_writer.writerow()
    #file.close()
    return tweets
    
def get_trending_hashtags(api, location):
    woeids = get_woeid(api, location)
    trending = set()
    for woeid in woeids:
        try:
            raise Exception("here")
            trends = api.trends_place(woeid)
        except:
            print("API limit exceeded. Waiting for next hour")
            time.sleep(3605) # change to 5 for testing
            trends = api.trends_place(woeid)
        topics = [trend['name'] for trend in trends[0]['trends'] if trend['name'].find('#') == 0]
        trending.update(topics)
    
    return trending
    
def twitter_bot(api, locations):
    today = datetime.datetime.today().strftime("%d-%m-%Y-%s")
    if not os.path.exists("trending_tweets"):
        os.makedirs("trending_tweets")
    file_tweets = open("trending_tweets/"+today+"-tweets.csv", "a+")
    file_hashtags = open("trending_tweets/"+today+"-hashtags.csv", "w+")
    writer = csv.writer(file_tweets)
    
    hashtags = get_trending_hashtags(api, locations)
    file_hashtags.write("\n".join(hashtags))
    file_hashtags.close()
    
    for hashtag in hashtags:
        try:
            raise Exception()
            tweets = get_tweets(api, hashtag)
        except:
            print("API limit exceeded. Waiting for next hour")
            time.sleep(3605) # change to 0.2 sec for testing
            tweets = get_tweets(api, hashtag)
        for tweet in tweets:
            writer.writerow(tweet)
    
    file_tweets.close()
def main():
    #locations = ['new york', 'los angeles', 'philadelphia', 'barcelona', 'canada', 'united kingdom', 'india']        
    locations = ['india']
    api = initiate_api()
    
    #schedule.every().day.at("00:00").do(job,'It is 01:00')
    schedule.every(10).seconds.do(twitter_bot, api, locations)
    while True:
        schedule.run_pending()
        time.sleep(1)
        
if __name__ == "__main__":
    main()