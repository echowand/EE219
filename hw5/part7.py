import datetime
import json
import math
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import nltk
import logging as logger
import re
import string
from nltk.corpus import stopwords
from collections import Counter

logger.basicConfig(level=logger.INFO, format='%(asctime)-15s - %(message)s')

hash_tags = ['gohawks', 'patriots']
start_date = datetime.datetime(2015, 02, 01, 03, 30, 0)
end_date = datetime.datetime(2015, 02, 02, 03, 30, 0)

tweets_hawks = [[] for i in range(24)]
tweets_patriots = [[] for i in range(24)]

for hash_tag in hash_tags:
    filename = './tweet_data/tweets_#{!s}.txt'.format(hash_tag)
    f = open(filename, 'rb')
    line = f.readline()
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple()))
    while len(line) != 0:
        tweet = json.loads(line)
        if tweet['firstpost_date'] < mintime:
            line = f.readline()
            continue
        if tweet['firstpost_date'] > maxtime:
            break
        index = (tweet['firstpost_date'] - mintime) / 3600
        if hash_tag == 'gohawks':
            tweets_hawks[index].append(tweet)
        else:
            tweets_patriots[index].append(tweet)
        line = f.readline()

logger.info("Finished loading Hawks and Patriots data")

emoji_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

special_str = [
    emoji_str,
    r'<[^>]+>',  # HTML
    r'(?:@[\w_]+)',  # @
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_regex = re.compile(r'(' + '|'.join(special_str) + ')', re.VERBOSE | re.IGNORECASE)
emoji_regex = re.compile(r'^' + emoji_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_regex.findall(s)


def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoji_regex.search(token) else token.lower() for token in tokens]
    return tokens


punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

# term list and term co-occurrence list for hawks
count_list_hawks = []
cooccur_list_hawks = []
for tweets in tweets_hawks:
    count_single_stop = Counter()
    co_occur = defaultdict(lambda: defaultdict(int))
    for tweet in tweets:
        terms_stop = [term for term in preprocess(tweet['tweet']['text']) if term not in stop]
        count_single_stop.update(terms_stop)

        for i in range(len(terms_stop) - 1):
            for j in range(i + 1, len(terms_stop)):
                w1, w2 = sorted([terms_stop[i], terms_stop[j]])
                if (w1 != w2):
                    co_occur[w1][w2] += 1

    count_list_hawks.append(count_single_stop)
    cooccur_list_hawks.append(co_occur)

logger.info("Hawks count: " + str(len(count_list_hawks)))
logger.info("Hawks cooccurrence count: " + str(len(cooccur_list_hawks)))

# term list and term co-occurrence list for patriots
count_list_patriots = []
cooccur_list_patriots = []
for tweets in tweets_patriots:
    count_single_stop = Counter()
    co_occur = defaultdict(lambda: defaultdict(int))
    for tweet in tweets:
        terms_stop = [term for term in preprocess(tweet['tweet']['text']) if term not in stop]
        count_single_stop.update(terms_stop)

        for i in range(len(terms_stop) - 1):
            for j in range(i + 1, len(terms_stop)):
                w1, w2 = sorted([terms_stop[i], terms_stop[j]])
                if (w1 != w2):
                    co_occur[w1][w2] += 1

    count_list_patriots.append(count_single_stop)
    cooccur_list_patriots.append(co_occur)

logger.info("Patriots count: " + str(len(count_list_patriots)))
logger.info("Patriots cooccurrence count: " + str(len(cooccur_list_patriots)))

pmi_hawks_list = []
pmi_patriots_list = []

positive_vocab = []
negative_vocab = []
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
with open('./positive.txt', 'rb') as vocabulary:
    for line in vocabulary:
        for word in line.split():
            if word in english_vocab:
                positive_vocab.append(word)

logger.info("Positive vocabulary length: " + str(len(positive_vocab)))

with open('./negative.txt', 'rb') as vocabulary:
    for line in vocabulary:
        for word in line.split():
            if word in english_vocab:
                negative_vocab.append(word)

logger.info("Negative vocabulary length: " + str(len(negative_vocab)))

# calculate the PMI for terms
for i in range(len(count_list_hawks)):
    p_t = {}
    p_t_com = defaultdict(lambda: defaultdict(int))
    length = float(len(tweets_hawks[i]))
    for term, n in count_list_hawks[i].items():
        p_t[term] = n / length
        for t2 in cooccur_list_hawks[i][term]:
            p_t_com[term][t2] = cooccur_list_hawks[i][term][t2] / length

    pmi = defaultdict(lambda: defaultdict(int))
    for t1 in p_t:
        for t2 in cooccur_list_hawks[i][t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = math.log(p_t_com[t1][t2] / denom, 2)
    semantic_orientation = {}
    for term, n in p_t.items():
        pos_asso = sum(pmi[term][tx] for tx in positive_vocab)
        neg_asso = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = pos_asso - neg_asso
    pmi_hawks_list.append(
        sum(semantic_orientation[term] for term in semantic_orientation) / float(len(semantic_orientation)))

logger.info("Hawks PMI count: " + str(len(pmi_hawks_list)))

for i in range(len(count_list_patriots)):
    p_t = {}
    p_t_com = defaultdict(lambda: defaultdict(int))
    length = float(len(tweets_patriots[i]))
    for term, n in count_list_patriots[i].items():
        p_t[term] = n / length
        for t2 in cooccur_list_patriots[i][term]:
            p_t_com[term][t2] = cooccur_list_patriots[i][term][t2] / length

    pmi = defaultdict(lambda: defaultdict(int))
    for t1 in p_t:
        for t2 in cooccur_list_patriots[i][t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = math.log(p_t_com[t1][t2] / denom, 2)
    semantic_orientation = {}
    for term, n in p_t.items():
        pos_asso = sum(pmi[term][tx] for tx in positive_vocab)
        neg_asso = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = pos_asso - neg_asso
    pmi_patriots_list.append(
        sum(semantic_orientation[term] for term in semantic_orientation) / float(len(semantic_orientation)))

logger.info("Hawks PMI count: " + str(len(pmi_patriots_list)))

# plot
plt.figure(1)
x = range(24)
plt.plot(x, pmi_hawks_list, lw=3, label="Hawks")
plt.plot(x, pmi_patriots_list, lw=3, label="Patriots")
plt.xlabel("Time(hour)")
plt.ylabel("Emotion")
plt.legend()
plt.title("Emotion change on Super Bowl day")
plt.savefig('pics/part7/sb_day.png', format='png')
plt.clf()

plt.figure(1)
x = range(9)
plt.plot(x, pmi_hawks_list[12:21], lw=3, label="Hawks")
plt.plot(x, pmi_patriots_list[12:21], lw=3, label="Patriots")
plt.xlabel("Time(hour)")
plt.ylabel("Emotion")
plt.legend()
plt.title("Emotion change in Super Bowl")
plt.savefig('pics/part7/game.png', format='png')
plt.clf()
