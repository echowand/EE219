import json
from utility import *
import logging as logger

logger.basicConfig(level=logger.INFO, format='%(asctime)-15s - %(message)s')

hash_tags = ['gopatriots', 'gohawks', 'nfl', 'patriots', 'sb49', 'superbowl']

def cross_validation():
    for hash_tag in hash_tags:
        tweets = open('./tweet_data/tweets_#{!s}.txt'.format(hash_tag), 'rb')
        first_tweet = json.loads(tweets.readline())
        start_time = first_tweet.get('firstpost_date')

        # features for model construction
        features = extra_feature_dict()
        current_window = 1
        end_time_of_window = start_time + current_window * 3600
        tweets.seek(0, 0)

        # store data
        tweet_features = []
        tweet_class = []
        logger.info("Extracting features from tweets")

        for tweet in tweets:
            tweet_data = json.loads(tweet)
            end_time = tweet_data.get('firstpost_date')

            if end_time < end_time_of_window:
                features = calculate_statistic(features, tweet_data)
            else:
                '''
                features : retweets, followers, max_followers, impressions, favorite_count,
                           ranking_score, hour_of_day, number_of_users, number_of_long tweets
                '''
                # append features to data variables
                extracted_features = get_features(features)
                tweet_class.append(extracted_features[0])
                tweet_features.append(extracted_features[1:])

                features = reset_features_dict() # reset features for new window calculation
                features = calculate_statistic(features, tweet_data)  # update stats of tweet

                current_window += 1
                end_time_of_window = start_time + current_window * 3600  # update window

        logger.info("Performing 10 Folds Cross Validation")

        tweet_class = np.roll(np.array(tweet_class), -1)
        tweet_class = collections.deque(tweet_class)
        tweet_class = np.delete(tweet_class, -1)
        del (tweet_features[-1])

        perform_classification(np.array(tweet_features), np.array(tweet_class))  # 10 fold cross validation
        logger.info("------------------------------")

def superbowl():
    for hash_tag in hash_tags:
        tweets = open('./tweet_data/tweets_#{!s}.txt'.format(hash_tag), 'rb')
        first_tweet = json.loads(tweets.readline())
        start_time = first_tweet.get('firstpost_date')

        # features for model construction
        features = extra_feature_dict()
        flag_between = True  # flag to set new window start point
        flag_after = True  # flag to set new window start point
        current_window = 1
        end_time_of_window = start_time + current_window * 3600
        tweets.seek(0, 0)

        # store data
        tweet_features_before = []
        tweet_features_between = []
        tweet_features_after = []
        tweet_class_before = []
        tweet_class_between = []
        tweet_class_after = []

        logger.info("Extracting features from tweets")

        for tweet in tweets:
            tweet_data = json.loads(tweet)
            end_time = tweet_data.get('firstpost_date')

            '''
            features : retweets, followers, max_followers, impressions, favorite_count,
                       ranking_score, hour_of_day, number_of_users, number_of_long tweets
            '''

            if end_time < 1422777600:  # tweeted Before Feb. 1, 8:00 a.m.
                if end_time < end_time_of_window:
                    features = calculate_statistic(features, tweet_data)
                else:
                    extracted_features = get_features(features)
                    tweet_class_before.append(extracted_features[0])
                    tweet_features_before.append(extracted_features[1:])
                    features = reset_features_dict()
                    features = calculate_statistic(features, tweet_data)
                    current_window += 1
                    end_time_of_window = start_time + current_window * 3600

            elif (end_time > 1422777600) and (end_time < 1422820800):  # tweeted Between Feb. 1, 8:00 a.m to 8 p.m
                if flag_between:  # reset start_time
                    flag_between = False
                    current_window = 1
                    start_time = end_time
                    end_time_of_window = start_time + (current_window * 3600)

                if end_time < end_time_of_window:
                    features = calculate_statistic(features, tweet_data)
                else:
                    extracted_features = get_features(features)
                    tweet_class_between.append(extracted_features[0])
                    tweet_features_between.append(extracted_features[1:])
                    features = reset_features_dict()
                    features = calculate_statistic(features, tweet_data)
                    current_window += 1
                    end_time_of_window = start_time + current_window * 3600

            else:  # tweeted After Feb. 1, 8 p.m
                if flag_after:  # reset start_time
                    flag_after = False
                    current_window = 1
                    start_time = end_time
                    end_time_of_window = start_time + current_window * 3600

                if end_time < end_time_of_window:
                    features = calculate_statistic(features, tweet_data)
                else:
                    extracted_features = get_features(features)
                    tweet_class_after.append(extracted_features[0])
                    tweet_features_after.append(extracted_features[1:])
                    features = reset_features_dict()
                    features = calculate_statistic(features, tweet_data)
                    current_window += 1
                    end_time_of_window = start_time + current_window * 3600

        tweet_class_before = np.roll(np.array(tweet_class_before), -1)
        tweet_class_between = np.roll(np.array(tweet_class_between), -1)
        tweet_class_after = np.roll(np.array(tweet_class_after), -1)

        # perform 10 fold cross validation for each time-period
        logger.info("Before")
        perform_classification(np.array(tweet_features_before), np.array(tweet_class_before))
        logger.info("Between")
        perform_classification(np.array(tweet_features_between), np.array(tweet_class_between))
        logger.info("After")
        perform_classification(np.array(tweet_features_after), np.array(tweet_class_after))

        logger.info("------------------------------")

#cross_validation()

superbowl()