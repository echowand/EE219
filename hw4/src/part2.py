from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn import metrics
from nltk import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans

# Get stop words
def get_stop_words():
    stop_words = [text.ENGLISH_STOP_WORDS]
    with open('NLTK_StopWords.txt', 'r') as file:
        lists = file.readlines()

    for i in range(len(lists)):
        lists[i] = lists[i].rstrip()  # remove trailing spaces

    NLTK_stop_words = lists

    return stop_words, NLTK_stop_words

# Proprocess data
def preprocess(sentence, stop_words, NLTK_stop_words):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer("english")

    tokens = tokenizer.tokenize(sentence)
    filter1 = [w for w in tokens if not w in NLTK_stop_words and w not in stop_words and len(w) is not 1]
    filter2 = [stemmer.stem(plural) for plural in filter1]
    filter3 = [i for i in filter2 if not i.isdigit()]
    return " ".join(filter3)

# Calculate TFIDF
def TFIDF(categories, train_test, stop_words, NLTK_stop_words):
    # Get the data
    twenty_data = fetch_20newsgroups(subset=train_test,categories=categories, remove=('headers','footers','quotes'))
    # Save size of the data
    size, = twenty_data.filenames.shape
    # Preprocess all documents
    for item in range(0, size):
        sentence = twenty_data.data[item]
        twenty_data.data[item] = preprocess(sentence, stop_words, NLTK_stop_words)

    # Transfer processed data to a TDM
    count_vectorizer = CountVectorizer()
    X_train_number = count_vectorizer.fit_transform(twenty_data.data)

    # Calculate TFIDF for every term in the document
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_number)
    X_train_tfidf = tf_transformer.transform(X_train_number)
    docs, terms = X_train_tfidf.shape
    print(100 * '_')
    print("TFIDF Matrix Created")
    print("Final number of terms: ", terms)
    print("Final number of documents: ", docs)
    print(100 * '_')
    return twenty_data, X_train_tfidf, X_train_number, count_vectorizer

# Load data
def load_data():
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    stop_words, NLTK_stop_words = get_stop_words()
    data, X_tfidf, X_counts, count_vectorizer = TFIDF(categories, 'all', stop_words, NLTK_stop_words)
    return data, X_tfidf

# Label data
def label_data(Y_data):
    labels = []
    for y in Y_data:
        labels.append(0 if (y <= 3) else 1)
    return labels

# Benchmark K means
def k_means(estimator, name, data, labels):
    estimator.fit(data)
    print('% 9s   %.3f   %.3f   %.3f   %.3f'
          % (name,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             ))

def p2():
    data, X_tfidf = load_data()
    number_of_samples, number_of_features = X_tfidf.shape
    number_of_digits = 2
    labels = label_data(data.target)
    print("no.digits: %d, \t no.samples %d, \t n_features %d"
	      % (number_of_digits, number_of_samples, number_of_features))
    print('% 9s' % 'init'
                   '        homogeneity    completeness     ARS    AMI')
    k_means(KMeans(init='k-means++', n_clusters=number_of_digits,
                         n_init=10, max_iter=200, random_state=42, tol=1e-5),
                  name="k-means++", data=X_tfidf, labels=labels)

    k_means(KMeans(init='random', n_clusters=number_of_digits,
                         n_init=10, max_iter=200, random_state=42, tol=1e-5),
                  name="random", data=X_tfidf, labels=labels)

    print(100 * '_')

p2()

# ____________________________________________________________________________________________________
# TFIDF Matrix Created
# Final number of terms:  67764
# Final number of documents:  7882
# ____________________________________________________________________________________________________
# no.digits: 2, 	 no.samples 7882, 	 n_features 67764
# init        homogeneity    completeness     ARS    AMI
# k-means++   0.421   0.455   0.440   0.421
#    random   0.438   0.469   0.462   0.438
# ____________________________________________________________________________________________________
