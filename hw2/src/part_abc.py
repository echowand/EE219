import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.ticker import FormatStrFormatter
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def plot_by_categories(cat_count, train):
    x = np.arange(0, cat_count + 1, 1)
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(train.target, bins=x, edgecolor='gray')
    ax.set_xticks(bins)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Categories', x=1)
    ax.set_ylabel('Count')
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # counts
        ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # percentages
        percent = '%0.0001f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points', va='top', ha='center')
    plt.title('Distribution of Documents')
    plt.subplots_adjust(bottom=0.15)
    plt.show()


class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(r'\b([a-zA-Z]+)\b')
        self.stemmer = LancasterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(token) for token in self.tok.tokenize(doc)]

#Task A
def A():
    # plot a histogram of the number of documents per topic
    train = f20(subset='train', shuffle=True, random_state=42)
    plot_by_categories(20, train)
    # plot the number of documents in Computer Technology and Recreational Activity
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    train = f20(subset='train', categories=categories, shuffle=True, random_state=42)
    plot_by_categories(8, train)

#Task B
def B():
    train = f20(subset='train', shuffle=True, random_state=42)
    stop_words = text.ENGLISH_STOP_WORDS
    categories = train.target_names
    vectorizer = CountVectorizer(tokenizer=Tokenizer(),
                                 stop_words=stop_words,
                                 min_df=1)
    vector = vectorizer.fit_transform(train.data)
    transformer = TfidfTransformer()
    tfidf_fit = transformer.fit_transform(vector.toarray())
    tfidf = tfidf_fit.toarray()
    print 'number of terms: ' + str(len(tfidf[0]))

#Task C
def C():
    categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
    train = f20(subset='train', shuffle=True, random_state=42)
    datalist = []
    for i in range(0, 20):
        datalist.append('')
    for i in range(0, len(train.data)):
        datalist[train.target[i]] += (' ' + train.data[i])

    # get the count vector
    stopwords = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(tokenizer=Tokenizer(),
                                 stop_words=stopwords,
                                 min_df=1)
    vector = vectorizer.fit_transform(datalist)
    count = vector.toarray()
    # get the if and icf
    index = {0: 3, 1: 4, 2: 6, 3: 15}
    tf = np.ndarray([4, len(count[0])], dtype=float)
    icf = np.ndarray([4, len(count[0])], dtype=float)
    # calculate if
    for i in index:
        s = float(np.sum(count[index[i]]))
        for j in range(0, len(count[0])):
            tf[i][j] = 0.5 + 0.5 * count[index[i]][j] / s
    # calculate icf
    for i in index:
        for j in range(0, len(count[0])):
            c = 0
            for x in range(0, 20):
                if (count[x][j] != 0):
                    c = 1 + c
            icf[i][j] = scipy.log(float(20) / c)
    # if * icf => if.icf
    # get the most 10 significent terms in the 4 classes
    tficf = tf * icf
    top_n = 10
    features = vectorizer.get_feature_names()
    for i in range(0, 4):
        print categories[i]
        indices = np.argsort(tficf[i])[::-1]
        top_features = [features[i] for i in indices[:top_n]]
        print top_features


C()


