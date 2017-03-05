import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.ticker import FormatStrFormatter
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import SnowballStemmer


def preprocess(sentence, stop_words):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer("english")
    tokens = tokenizer.tokenize(sentence)
    filter1 = [w for w in tokens if w not in stop_words and len(w) is not 1]
    filter2 = [stemmer.stem(plural) for plural in filter1]
    filter3 = [i for i in filter2 if not i.isdigit()]
    return " ".join(filter3)


def load():
    stop_words = text.ENGLISH_STOP_WORDS
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    all_data_set = f20(subset='all', categories=categories, shuffle=True, random_state=42,
                       remove=('headers', 'footers', 'quotes'))

    size, = all_data_set.filenames.shape
    for item in range(0, size):
        sentence = all_data_set.data[item]
        all_data_set.data[item] = preprocess(sentence, stop_words)

    vectorizer = CountVectorizer(min_df=1)
    vector = vectorizer.fit_transform(all_data_set.data)
    transformer = TfidfTransformer()
    tfidf_fit = transformer.fit_transform(vector.toarray())
    docs, terms = tfidf_fit.shape

    print "TFIDF matrix:"
    print "The number of terms: " + str(terms)
    print "The number of docs: " + str(docs)

    return all_data_set, tfidf_fit, vector, vectorizer


load()
