import json
import logging as logger
from collections import defaultdict
from matplotlib import pyplot as plt

logger.basicConfig(level=logger.INFO, format='%(asctime)-15s - %(message)s')

# hash tags
hash_tags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']

for hash_tag in hash_tags:

    tweets = open('./tweet_data/tweets_#{!s}.txt'.format(hash_tag), 'rb')
    first_tweet = json.loads(tweets.readline())
    userid_followers_dict = defaultdict()
    tweets_count = len(tweets.readlines())
    retweets_count = 0
    tweets.seek(0, 0)

    # tweets are sorted with respect to their posting time, get fisrtpost_date of the first tweet
    current_window = 1
    start_time = first_tweet.get('firstpost_date')
    userid_followers_dict[first_tweet.get('tweet').get('user').get('id')] = first_tweet.get('author').get('followers')
    end_time_of_window = start_time + current_window * 3600

    tweets_count_hour = []
    current_window_count = 0

    for tweet in tweets:
        tweet_json = json.loads(tweet)
        end_time = tweet_json.get('firstpost_date')

        retweets_count += tweet_json.get('metrics').get('citations').get('total')
        if end_time < end_time_of_window:
            current_window_count += 1
        else:
            tweets_count_hour.append(current_window_count)
            current_window += 1
            current_window_count = 0
            end_time_of_window = start_time + current_window * 3600  # non-overlapping window

        user_id = tweet_json.get('tweet').get('user').get('id')
        userid_followers_dict[user_id] = tweet_json.get('author').get('followers')

    logger.info("Statistics for #{}".format(hash_tag))
    logger.info("Total number of tweets: {}".format(tweets_count))
    logger.info("Average number of tweets per hour: {}".format(tweets_count / ((end_time - start_time) / 3600.0)))
    logger.info("Average number of followers: {}".format(
        sum(userid_followers_dict.values()) / float(len(userid_followers_dict.keys()))))
    logger.info("Average number of retweets: {}".format(retweets_count / float(tweets_count)))

    # Plot "number of tweets in hour" over time for #SuperBowl and #NFL (a histogram with 1-hour bins).
    if hash_tag == 'superbowl' or hash_tag == "nfl":
        plt.figure(1)
        plt.ylabel('Number of Tweets')
        plt.xlabel('Hour')
        plt.title('Number of Tweets in Hour for #{!s}'.format(hash_tag))
        plt.bar(range(len(tweets_count_hour)), tweets_count_hour)
        plt.savefig('pics/number_of_tweets_{!s}.png'.format(hash_tag), format='png')
        plt.clf()

    logger.info("------------------------------")
