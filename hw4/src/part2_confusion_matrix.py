import numpy as np
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from nltk.stem.snowball import SnowballStemmer

class1 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'];
class2 = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'];
class1_train = fetch_20newsgroups(subset = 'train', categories = class1, shuffle=True, random_state=42, remove=('headers','footers','quotes'));
class2_train = fetch_20newsgroups(subset = 'train', categories = class2, shuffle=True, random_state=42, remove=('headers','footers','quotes'));

stop_words = text.ENGLISH_STOP_WORDS

stemmer = SnowballStemmer("english")

class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(r'\w+')
        self.stemmer = LancasterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(token) for token in self.tok.tokenize(doc)]

total_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories = class1+class2, remove=('headers','footers','quotes'))
total_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, categories = class1+class2, remove=('headers','footers','quotes'))

# normalize features
n_features = 10000
TFxIDF = TfidfVectorizer(analyzer='word',tokenizer=Tokenizer(), stop_words=stop_words,token_pattern='[a-zA-Z]{2,}', norm = 'l2', max_df=0.95, min_df=2, max_features=n_features)
TFxIDF_train = TFxIDF.fit_transform(total_train.data)
TFxIDF_test  = TFxIDF.transform(total_test.data)

tfidf_feature_names = TFxIDF.get_feature_names()

kmeans = KMeans(n_clusters=2, max_iter = 1000, random_state=42).fit(TFxIDF_train)

# Group the subclasses into 2 superclasses
test_target_group = [ int(x / 4) for x in total_test.target]

kmeans_predict = kmeans.predict(TFxIDF_test)

if sum(abs(kmeans_predict - test_target_group)) > len(kmeans_predict) / 2:
    i = 0
    for candidate in kmeans_predict:

        if candidate == 1:
            kmeans_predict[i] = 0
        elif candidate == 0:
            kmeans_predict[i] = 1
        i = i + 1

class_names = ['Computer Tech', 'Recreation']

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_target_group, kmeans_predict)
np.set_printoptions(precision=2)

print cnf_matrix
print homogeneity_score(test_target_group, kmeans_predict)
print completeness_score(test_target_group, kmeans_predict)
print adjusted_rand_score(test_target_group, kmeans_predict)
print normalized_mutual_info_score(test_target_group, kmeans_predict)