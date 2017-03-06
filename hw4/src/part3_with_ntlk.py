from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn import metrics
from nltk import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import FunctionTransformer

k_value = 5
n_iter = 5
m_iter = 200
random_state = 42
np.random.seed(304145309)

# Get stop words
def get_stop_words():
    stop_words = [text.ENGLISH_STOP_WORDS]
    with open('NLTK_StopWords.txt', 'r') as file:
        lists = file.readlines()

    for i in range(len(lists)):
        lists[i] = lists[i].rstrip()

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
    all_data = f20(subset=train_test,categories=categories, remove=('headers','footers','quotes'))
    # Save size of the data
    size, = all_data.filenames.shape
    # Preprocess all documents
    for item in range(0, size):
        sentence = all_data.data[item]
        all_data.data[item] = preprocess(sentence, stop_words, NLTK_stop_words)

    # Transfer processed data to a TDM
    count_vectorizer = CountVectorizer()
    X_train_number = count_vectorizer.fit_transform(all_data.data)

    # Calculate TFIDF for every term in the document
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_number)
    X_train_tfidf = tf_transformer.transform(X_train_number)
    docs, terms = X_train_tfidf.shape
    print("TFIDF")
    print("Number of terms: ", terms)
    print("Number of documents: ", docs)
    return all_data, X_train_tfidf, X_train_number, count_vectorizer

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
def print_k_means(estimator, name, data, labels):
    estimator.fit(data)
    print('% 9s   %.4f   %.4f   %.4f   %.4f'
          % (name,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             ))

# Calculate sigma
def get_sigma(X_tfidf, k_values):
    print(100 * '_')
    print("Sigma Matrix")
    U, sigma, VT = randomized_svd(X_tfidf, n_components=k_values,
                                  n_iter=200,
                                  random_state=42)
    print(sigma)
    return sigma

def get_sigma(X_tfidf):
    sigma = get_sigma(X_tfidf, 200)
    plt.figure(1)
    plt.title('Sigma Matrix Values')
    plt.scatter(range(len(sigma)), sigma, marker='o')
    plt.plot(range(len(sigma)), sigma)
    plt.show()

def p3():
    data, X_tfidf = load_data()
    n_samples, n_features = X_tfidf.shape
    n_clusters = 2
    labels = label_data(data.target)

    # Check sigma values
    # sigma = help_functions.calculate_sigma(X_tfidf,200)
    # plt.figure(1)
    # plt.title('Sigma Matrix Values')
    # plt.scatter(range(len(sigma)), sigma, marker='o')
    # plt.plot(range(len(sigma)), sigma)
    # plt.show()

    # PCA
    print("PCA")
    svd = TruncatedSVD(n_components=k_value, n_iter=n_iter)
    normalizer_nmf = Normalizer(copy=True)
    normalizer_pca = Normalizer(copy=True)
    lsa_pca = make_pipeline(svd, normalizer_pca)
    X_tfidf_pca = svd.fit_transform(X_tfidf)

    # NMF
    print("NMF")
    nmf = NMF(n_components=k_value, random_state=random_state,
              alpha=.01, l1_ratio=0, max_iter=m_iter).fit(X_tfidf)
    lsa_nmf = make_pipeline(nmf, normalizer_nmf)
    X_tfidf_nmf = nmf.fit_transform(X_tfidf)

    #print out unnormalized scores for PCA,NMF
    print("n_clusters: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))

    print(79 * '_')
    print('% 9s' % 'init'
                   '        homo    compl     ARI    AMI')

    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="Original", data=X_tfidf, labels=labels)
    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA", data=X_tfidf_pca, labels=labels)
    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF", data=X_tfidf_nmf, labels=labels)

    X_tfidf_pca = log(X_tfidf_pca)
    X_tfidf_nmf = log(X_tfidf_nmf)

    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA_log", data=X_tfidf_pca, labels=labels)
    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF_log", data=X_tfidf_nmf, labels=labels)

    # print out normalized score for PCA,NMF
    X_tfidf_pca_norm = lsa_pca.fit_transform(X_tfidf)
    X_tfidf_nmf_norm = lsa_nmf.fit_transform(X_tfidf)

    print("n_clusters: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))

    print(79 * '_')
    print('% 9s' % 'init'
                   '        homo    compl     ARI    AMI')

    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA_norm", data=X_tfidf_pca_norm, labels=labels)
    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF_norm", data=X_tfidf_nmf_norm, labels=labels)

    X_tfidf_pca_norm = log(X_tfidf_pca_norm)
    X_tfidf_nmf_norm = log(X_tfidf_nmf_norm)
    #
    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA_norm_log", data=X_tfidf_pca_norm, labels=labels)
    print_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF_norm_log", data=X_tfidf_nmf_norm, labels=labels)

    # nmf_2 = NMF(n_components=2, random_state=random_state,
    #             alpha=.01, l1_ratio=0, max_iter=m_iter).fit(X_tfidf)
    # X_tfidf_nmf_2 = nmf_2.fit_transform(X_tfidf)
    # log_base = np.max(X_tfidf_nmf_2) - np.min(X_tfidf_nmf_2)
    # X_tfidf_nmf_log = log2(X_tfidf_nmf_2, log_base)
    # plt.figure(2)
    # plt.title('NMF Embedding with Log Transformation')
    # plt.scatter(X_tfidf_nmf_log[:, 0], X_tfidf_nmf_log[:, 1], marker='o')
    # plt.show()

def log(tfidf):
    transformer = FunctionTransformer(np.log1p)
    for (x,y), value in np.ndenumerate(tfidf):
        transformer.transform(value)
        tfidf[x, y] = value
    return tfidf

def log2(tfidf, log_base):
    print (log_base)
    for (x, y), value in np.ndenumerate(tfidf):
        if value > 0:
            tfidf[x,y] = np.log(value)/np.log(log_base)
        elif value < 0:
            tfidf[x,y] = np.log(abs(value))/np.log(log_base)
        else:
            tfidf[x,y] = value
    return tfidf

p3()