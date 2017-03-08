from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

# all 20 subclasses data
dataset = f20(subset='all', categories=categories, shuffle=True, random_state=42,
              remove=('headers', 'footers', 'quotes'))
labels = dataset.target

# map labels
dict = {1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0,
        7 : 1, 8 : 1, 9 : 1, 10 : 1,
        11 : 2, 12 : 2, 13 : 2, 14 : 2,
        6 : 3,
        16 : 4, 17 : 4, 18 : 4,
        0 : 5, 15 : 5, 19 : 5}

i = 0
for label in labels:
    labels[i] = dict[labels[i]]
    i = i+1

X = vectorizer.fit_transform(dataset.data)

km = KMeans(n_clusters=6, max_iter = 1000, random_state=42).fit(X)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted_Mutual_Info_Score: %.3f"
      % metrics.adjusted_mutual_info_score(labels, km.labels_))
print()

# # SVD
# n_components = [2, 5, 10, 50, 100]
#
# for dimensionality in np.array(n_components):
#     print("Desired dimensionality: %d" % dimensionality)
#
#     svd = TruncatedSVD(n_components=dimensionality, algorithm='arpack')
#     dataset_LSI = svd.fit_transform(X)
#     kmLSI = KMeans(n_clusters=20, max_iter=1000, random_state=42).fit(dataset_LSI)
#
#     print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmLSI.labels_))
#     print("Completeness: %0.3f" % metrics.completeness_score(labels, kmLSI.labels_))
#     print("Adjusted Rand-Index: %.3f"
#           % metrics.adjusted_rand_score(labels, kmLSI.labels_))
#     print("Adjusted_Mutual_Info_Score: %.3f"
#           % metrics.adjusted_mutual_info_score(labels, kmLSI.labels_))
#     print()
#
# print("------------------------------------------------------------------")

# NMF

n_components = [2, 5, 10, 15, 20]

for dimensionality in np.array(n_components):
    print("Desired dimensionality: %d" % dimensionality)

    nmf = NMF(n_components=dimensionality, init='nndsvd', random_state=0, alpha=.1, l1_ratio=0)
    dataset_NMF = nmf.fit_transform(X)
    kmNMF = KMeans(n_clusters=6, max_iter=1000, random_state=42).fit(dataset_NMF)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmNMF.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, kmNMF.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, kmNMF.labels_))
    print("Adjusted_Mutual_Info_Score: %.3f"
          % metrics.adjusted_mutual_info_score(labels, kmNMF.labels_))
    print()