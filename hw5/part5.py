import json
from utility import *


hash_tags = ['gopatriots', 'gohawks', 'nfl', 'patriots', 'sb49', 'superbowl']
test_samples = ['sample1_period1', 'sample2_period2', 'sample3_period3', 'sample4_period1', 'sample5_period1', 'sample6_period2', 'sample7_period3', 'sample8_period1', 'sample9_period2', 'sample10_period3']

error = []
for hash_tag in hash_tags:
    tweets = open('./tweet_data/tweets_#{!s}.txt'.format(hash_tag), 'rb')
    first_tweet = json.loads(tweets.readline())
    start_time = first_tweet.get('firstpost_date')

    # features for model construction
    features = extra_feature_dict()
    flag_between = True
    flag_after = True
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

    logger.info("Extracting training features from tweets")

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

    # create linear regression model of all time-periods
    model_before = sm.OLS(list(tweet_class_before), tweet_features_before).fit()
    model_between = sm.OLS(list(tweet_class_between), tweet_features_between).fit()
    model_after = sm.OLS(list(tweet_class_after), tweet_features_after).fit()

    test_before = []
    test_between = []
    test_after = []

    # for each file in the test_data folder
    for t_file in test_samples:
        tweets = open('./test_data/{0}.txt'.format(t_file), 'rb')
        first_tweet = json.loads(tweets.readline())
        start_time = first_tweet.get('firstpost_date')
        logger.info("Testing File {}".format(t_file))
        # features for model construction
        features = extra_feature_dict()
        current_window = 1
        end_time_of_window = start_time + current_window * 3600
        tweets.seek(0, 0)

        # store data
        test_tweet_features = []
        test_tweet_class = []
        # calculate features using one-hour window
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
                extracted_features = get_features(features)
                test_tweet_class.append(extracted_features[0])
                test_tweet_features.append(extracted_features[1:])

                features = reset_features_dict()
                features = calculate_statistic(features, tweet_data)

                current_window += 1
                end_time_of_window = start_time + current_window * 3600

        predicted_value_before = model_before.predict(test_tweet_features)
        predicted_value_between = model_between.predict(test_tweet_features)
        predicted_value_after = model_after.predict(test_tweet_features)

        test_tweet_class = np.roll(np.array(test_tweet_class), -1)

        # check for which period is the file for and predicted using present hash-tag model
        if(t_file =='sample1_period1' or t_file=='sample4_period1' or t_file=='sample8_period1' or t_file=='sample5_period1') :  # test data is for set 1 (before Feb 1. 8:am)
            logger.info("Predicted # of Tweets for 7th Hour: {}".format(predicted_value_before[-1]))
            logger.info("Error Value for Model: {}".format(np.mean(predicted_value_before[0:len(predicted_value_before) - 1] - test_tweet_class[0:len(test_tweet_class) - 1])))
        elif(t_file =='sample2_period2' or t_file=='sample6_period2' or t_file=='sample9_period2') :  # test data is for set 2 (from Feb 1. 8:am to 8:00 pm)
            logger.info("Predicted # of Tweets for 7th Hour: {}".format(predicted_value_between[-1]))
            logger.info("Error Value for Model: {}".format(np.mean(predicted_value_before[0:len(predicted_value_before) - 1] - test_tweet_class[0:len(test_tweet_class) - 1])))
        else:  # test data is for set 3 (from Feb 1. after 8:00 pm)
            logger.info("Predicted # of Tweets for 7th Hour: {}".format(predicted_value_after[-1]))
            logger.info("Error Value for Model: {}".format(np.mean(predicted_value_before[0:len(predicted_value_before) - 1] - test_tweet_class[0:len(test_tweet_class) - 1])))