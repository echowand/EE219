import json
import numpy as np
import collections
import logging as logger
import statsmodels.api as sm
from datetime import datetime

logger.basicConfig(level=logger.INFO, format='%(asctime)-15s - %(message)s')

# hash tags
hash_tags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']

for hash_tag in hash_tags:
    tweets = open('./tweet_data/tweets_#{!s}.txt'.format(hash_tag), 'rb')
    first_tweet = json.loads(tweets.readline())
    start_time = first_tweet.get('firstpost_date')  # get the first tweet post time for window creation

    # features for model construction
    number_of_tweets_hour = 0
    number_of_retweets_hour = 0
    number_of_followers_hour = 0
    max_number_of_followers = 0

    tweets.seek(0, 0)  # set start point to 0 again

    current_window = 1
    end_time_of_window = start_time + current_window * 3600

    tweet_features = []
    tweet_class = []
    logger.info("Extracting features from tweets")

    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data.get('firstpost_date')

        if end_time < end_time_of_window:
            number_of_tweets_hour += 1
            number_of_retweets_hour += tweet_data.get('metrics').get('citations').get('total')
            follower_count = tweet_data.get('author').get('followers')
            number_of_followers_hour += follower_count
            if follower_count > max_number_of_followers:
                max_number_of_followers = follower_count

        else:
            features = [number_of_retweets_hour, number_of_followers_hour, max_number_of_followers,
                        int(datetime.fromtimestamp(tweet_data.get('firstpost_date')).strftime("%H"))]
            tweet_class.append(number_of_tweets_hour)

            number_of_tweets_hour = 1
            number_of_retweets_hour = tweet_data.get('metrics').get('citations').get('total')
            number_of_followers_hour = tweet_data.get('author').get('followers')
            max_number_of_followers = tweet_data.get('author').get('followers')
            current_window += 1
            end_time_of_window = start_time + current_window * 3600

            tweet_features.append(features)

    # roll the class variable to get tweet count of the next hour
    tweet_class = np.roll(np.array(tweet_class), -1)
    tweet_class=np.delete(tweet_class,-1)
    del(tweet_features[-1])
    # get transpose of the class variable
    tweet_class = collections.deque(tweet_class)
    tweet_class.rotate(-1)
    tweet_class = list(tweet_class)

    # train regression model OLS
    result = sm.OLS(tweet_class, tweet_features).fit()

    logger.info('Parameters: {}'.format(result.params))
    logger.info('Standard errors: {}'.format(result.bse))
    logger.info('p values: {}'.format(result.pvalues))
    logger.info('t values: {}'.format(result.tvalues))
    logger.info('Accuracy: {}'.format(result.rsquared * 100))
    logger.info("------------------------------")
