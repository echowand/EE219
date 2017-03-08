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

def mySVD(X):
    n_components = [200, 300, 500, 1000]
    homogeneity = []
    completeness = []
    vmeasure = []
    ari = []
    for dimensionality in np.array(n_components):
        print("Desired dimensionality: %d" % dimensionality)

        svd = TruncatedSVD(n_components=dimensionality, algorithm='arpack')
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        Y = lsa.fit_transform(X)

        kmLSI = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

        kmLSI.fit(Y)

        homogeneity.append(metrics.homogeneity_score(labels, kmLSI.labels_))
        completeness.append(metrics.completeness_score(labels, kmLSI.labels_))
        vmeasure.append(metrics.v_measure_score(labels, kmLSI.labels_))
        ari.append(metrics.adjusted_rand_score(labels, kmLSI.labels_))

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmLSI.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, kmLSI.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(labels, kmLSI.labels_))
        print("Adjusted_Mutual_Info_Score: %.3f"
              % metrics.adjusted_mutual_info_score(labels, kmLSI.labels_))
        print()

    l1, = plt.plot(n_components, homogeneity, label='homogeneity')
    l2, = plt.plot(n_components, completeness, label='completeness')
    l3, = plt.plot(n_components, vmeasure, label='vmeasure')
    l4, = plt.plot(n_components, ari, label='ari')
    plt.legend(handles=[l1, l2, l3, l4])
    plt.show()

    print("------------------------------------------------------------------")


def myNMF(X):
    n_components = [2, 5, 10, 15, 20]
    homogeneity = []
    completeness = []
    vmeasure = []
    ari = []

    for dimensionality in np.array(n_components):
        print("Desired dimensionality: %d" % dimensionality)
        nmf = NMF(n_components=dimensionality, init='nndsvd', random_state=0, alpha=.1, l1_ratio=0)
        Y = nmf.fit_transform(X)
        kmNMF = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

        kmNMF.fit(Y)

        homogeneity.append(metrics.homogeneity_score(labels, kmNMF.labels_))
        completeness.append(metrics.completeness_score(labels, kmNMF.labels_))
        vmeasure.append(metrics.v_measure_score(labels, kmNMF.labels_))
        ari.append(metrics.adjusted_rand_score(labels, kmNMF.labels_))

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmNMF.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, kmNMF.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(labels, kmNMF.labels_))
        print("Adjusted_Mutual_Info_Score: %.3f"
              % metrics.adjusted_mutual_info_score(labels, kmNMF.labels_))
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

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

# all 20 subclasses data
dataset = f20(subset='all', categories=categories, shuffle=True, random_state=42,
              remove=('headers', 'footers', 'quotes'))
labels = dataset.target
true_k = 20
X = vectorizer.fit_transform(dataset.data)
samples, features = X.shape

print("Number of samples: %d, number of features: %d" % (samples, features))
print ()

mySVD(X)
myNMF(X)