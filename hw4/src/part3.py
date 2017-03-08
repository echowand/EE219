from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


##### reduce feature dimension: LSI/LSA
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
def LSA(X):
    # n_components = [2, 5, 10, 50, 100, 200, 500, 1000]
    n_components = [100, 300, 500]
    homogeneity = []
    completeness = []
    vmeasure = []
    ari = []
    for dimensionality in np.array(n_components):
        print("Desired dimensionality: %d" % dimensionality)
        svd = TruncatedSVD(dimensionality)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        Y = lsa.fit_transform(X)

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

        # print("Clustering sparse data with %s" % km)
        km.fit(Y)

        homogeneity.append(metrics.homogeneity_score(labels, km.labels_))
        completeness.append(metrics.completeness_score(labels, km.labels_))
        vmeasure.append(metrics.v_measure_score(labels, km.labels_))
        ari.append(metrics.adjusted_rand_score(labels, km.labels_))

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
        # print("Adjusted Mutual Info: %.3f" % metrics.adjusted_mutual_info_score(labels, km.labels_)) === homogeneity
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, km.labels_, sample_size=1000))
        print()

    l1, = plt.plot(n_components, homogeneity, label='homogeneity')
    l2, = plt.plot(n_components, completeness, label='completeness')
    l3, = plt.plot(n_components, vmeasure, label='vmeasure')
    l4, = plt.plot(n_components, ari, label='ari')
    plt.legend(handles=[l1, l2, l3, l4])
    plt.show()
    print("------------------------------------------------------------------")


##### reduce feature dimension: NMF
def myNMF(X):
    # n_components = range(2, 21, 1)
    n_components = [600, 300]
    homogeneity = []
    completeness = []
    vmeasure = []
    ari = []
    for dimensionality in np.array(n_components):
        print("Desired dimensionality: %d" % dimensionality)
        nmf = NMF(n_components=dimensionality, init='nndsvd', random_state=0, alpha=.1, l1_ratio=0)
        Y = nmf.fit_transform(X)
        # km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        km = KMeans(n_clusters=2, max_iter=1000, random_state=42).fit(X)

        km.fit(Y)

        homogeneity.append(metrics.homogeneity_score(labels, km.labels_))
        completeness.append(metrics.completeness_score(labels, km.labels_))
        vmeasure.append(metrics.v_measure_score(labels, km.labels_))
        ari.append(metrics.adjusted_rand_score(labels, km.labels_))

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
        # print("Adjusted Mutual Info: %.3f" % metrics.adjusted_mutual_info_score(labels, km.labels_))
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, km.labels_, sample_size=1000))
        print()

    l1, = plt.plot(n_components, homogeneity, label='homogeneity')
    l2, = plt.plot(n_components, completeness, label='completeness')
    l3, = plt.plot(n_components, vmeasure, label='vmeasure')
    l4, = plt.plot(n_components, ari, label='ari')
    plt.legend(handles=[l1, l2, l3, l4])
    plt.show()
    print("------------------------------------------------------------------")


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
dataset = f20(subset='all', categories=categories, shuffle=True, random_state=42,
              remove=('headers', 'footers', 'quotes'))

'''
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
'''

labels = dataset.target
# true_k = np.unique(labels).shape[0]
true_k = 2

##### reduce feature dimension: min_df=2
print("Reduce feature dimension by setting min_df = 2")
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
#vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', max_features=10000)

X = vectorizer.fit_transform(dataset.data)
samples, features = X.shape
print("Number of samples: %d, number of features: %d" % (samples, features))
print()
# LSA(X)
myNMF(X)

'''
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
'''
