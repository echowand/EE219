import re
import string
import collections
import numpy as np
from nltk import bigrams
import logging as logger
import statsmodels.api as sm
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.cross_validation import KFold

logger.basicConfig(level=logger.INFO, format='%(asctime)-15s - %(message)s')

extra_features = ['Number of Tweets', 'Number of Retweets', 'Number of Followers', 'Max Number of Followers',
                  'Impression Count', 'Favourite Count', 'Ranking Score', 'Hour of Day', 'Number of Users tweeting',
                  'Number of Long Tweets']


def extra_feature_dict():
    """
    :return: create feature dictionary for features calculation
    """
    features = dict.fromkeys(extra_features, 0)
    features['Number of Users tweeting'] = []

    return features


def calculate_statistic(features, tweet_data):
    """
    :return: return calculated features of the tweet data
    """
    features['Number of Tweets'] += 1
    features['Number of Retweets'] += tweet_data.get('metrics').get('citations').get('total')
    follower_count = tweet_data.get('author').get('followers')
    features['Number of Followers'] += follower_count
    features['Impression Count'] += tweet_data.get('metrics').get('impressions')
    if follower_count > features['Max Number of Followers']:
        features['Max Number of Followers'] = follower_count
    features['Ranking Score'] += tweet_data.get('metrics').get('ranking_score')
    features['Hour of Day'] = int(datetime.fromtimestamp(tweet_data.get('firstpost_date')).strftime("%H"))
    features['Favourite Count'] += tweet_data.get('tweet').get('favorite_count')
    features['Number of Users tweeting'].append(tweet_data.get('tweet').get('user').get('id'))
    features['Number of Long Tweets'] += 1 if len(tweet_data.get('title')) > 100 else 0

    return features


def get_features(features):
    """
    :return: get feature values in specified order of extra_features
    """
    extract = []
    for i in extra_features:
        extract.append(features[i])

    extract[8] = len(set(extract[8]))

    return extract


def reset_features_dict():
    """
    :return: reset the features dict for new window calculation
    """
    features = dict.fromkeys(extra_features, 0)
    features['Number of Users tweeting'] = []

    return features


def perform_classification(X, Y):
    """
    :return: 10 fold cross validation of X and Y
    """

    average_error = []
    kf = KFold(len(X), n_folds=10, shuffle=False, random_state=None)

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        y_train = collections.deque(y_train)
        y_train.rotate(-1)
        y_train = list(y_train)
        X_train = list(X_train)

        result = sm.OLS(y_train, X_train).fit()
        test_prediction = result.predict(X_test)
        average_error.append(np.mean(abs(test_prediction - y_test)))

    logger.info("10 Fold Error : {}".format(average_error))
    logger.info("Average error : {}".format(np.mean(average_error)))


stopWords = stopwords.words("english")
punctuation = list(string.punctuation)
remove_words = stopWords + punctuation

top_word_count = Counter()
top_bigrams_counter = Counter()
top_hash_tags = Counter()
all_tweets = []


def preprocess_data(hourly_tweet_texts):
    """
    :return: preprocessed data of tweets from one hour window
    """
    # count top words
    word_count = Counter()
    other_hash_tags = Counter()
    bigrams_counter = Counter()
    word_list = []
    terms_only = Counter()

    for tweet_text in hourly_tweet_texts:
        temp = tweet_text
        temp = re.sub("[,.-:/()?{}*$&]", " ", temp)  # remove all symbols
        temp = "".join([ch for ch in temp if ch not in string.punctuation])  # remove all punctuation
        temp = "".join(ch for ch in temp if ord(ch) < 128)  # remove all non-ascii characters
        temp = temp.lower()  # convert to lowercase
        text = [word for word in temp.split() if word not in remove_words]
        word_count.update(text)
        all_tweets.append(text)

        # count hash-tags only
        terms_hash = [term.lower() for term in tweet_text.split()
                      if term.startswith('#')]
        other_hash_tags.update(terms_hash)

        # count terms only (no hashtags, no mentions)
        temp = [term for term in text if not term.startswith(('#', '@'))]
        terms_only.update(temp)
        top_word_count.update(temp)

        bigrams_counter.update(bigrams(temp))
        top_bigrams_counter.update(bigrams(temp))
        top_hash_tags.update([term.lower() for term in tweet_text.split()
                              if term.startswith('#')])
        word_list.append(terms_only)

    return word_list, other_hash_tags, terms_only, bigrams_counter

