# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:26:03 2021

@author: Kumanan
"""

!pip install tweepy

!pip install textblob

#python -m textblob.download_corpora

import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = 'xD4TJGJI8Dr2iOa6B0fExR640'
        consumer_secret = 'pcEvHl5sxEzjvdY4eym5YVrKNqO9DzpF44ORBDDy14Kx9dd8cS'
        access_token = '1430132319025733633-Vts1qcGlgL0ybFjs45n76zomhWsqUR'
        access_token_secret = 'hSLzwNrVXPmxnAeOXoBsykawBMrSO4CalmDuxoTgfMC4t'
        
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
    
    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        #create Textblob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        #set sentiment 
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def get_tweets(self, query, count=10):
        '''
        Main function to fetch and parse them
        '''
        # empty list to store parsed tweets
        tweets = []
        
        try:
            fetched_tweets = self.api.search(q=query, count=count)
            
            # parsing tweets one by one
            for tweet in fetched_tweets:
                #empty dictionary to store required params of a tweet
                parsed_tweet = {}
                
                #saving text of the tweet
                parsed_tweet['text'] = tweet.text
                #savig sentiment of the tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                
                #appending the parsed tweet to tweet list
                if tweet.retweet_count > 0:
                    # if tweet has retweet, ensure it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
        
            #return parsed tweets
            return tweets
        
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
    
def main():
    #creating object of twitter client class
    api = TwitterClient()
    
    #calling fucntion to get tweets
    tweets = api.get_tweets(query = 'Modi', count = 200)
    
    #picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    #percentage of positive tweets
    print("Positive tweet percentage : {} %".format(100 * len(ptweets)/len(tweets)))
    
    #picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    #percentage of positive tweets
    print("Negative tweet percentage : {} %".format(100 * len(ntweets)/len(tweets)))
    
    #percentage of neutral tweets
    print("Neutral tweets percentage: {} %".format(100 * (len(tweets) - (len(ptweets)+len(ntweets)))/len(tweets)))
    
    #printing first 5 positive tweets
    print("\n\nPositive tweets:")
    for tweet in ptweets[:5]:
        print(tweet['text'])
    
    #printing first 5 negative tweets
    print("\n\nNegative tweets:")
    for tweet in ntweets[:5]:
        print(tweet['text'])
    

if __name__ == "__main__":
    #calling main fucntion
    main()
            
