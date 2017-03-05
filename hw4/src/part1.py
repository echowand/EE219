import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.ticker import FormatStrFormatter
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(r'\w+')
        self.stemmer = LancasterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(token) for token in self.tok.tokenize(doc)]


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

all_data_set = f20(subset='all', categories=categories, shuffle=True, random_state=42,
                   remove=('headers', 'footers', 'quotes'))

stop_words = text.ENGLISH_STOP_WORDS
vectorizer = CountVectorizer(tokenizer=Tokenizer(),
                             stop_words=stop_words,
                             min_df=1)
vector = vectorizer.fit_transform(all_data_set.data)
transformer = TfidfTransformer()
tfidf_fit = transformer.fit_transform(vector.toarray())
docs, terms = tfidf_fit.shape

print "TFIDF matrix:"
print "The number of terms: " + str(terms)
print "The number of docs: " + str(docs)
