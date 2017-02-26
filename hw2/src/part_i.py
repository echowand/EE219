from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as TSVD
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
import nltk.stem

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.multiclass import OneVsOneClassifier as OVOC
from sklearn.multiclass import OneVsRestClassifier as OVRC

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def I():
    cat = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
           'misc.forsale', 'soc.religion.christian']

    train = f20(subset='train', categories=cat, shuffle=True, random_state=42)

    stopwords = text.ENGLISH_STOP_WORDS
    vectorizer = StemmedTfidfVectorizer(
        min_df=1, stop_words=stopwords, decode_error='ignore')
    vector_train = vectorizer.fit_transform(train.data)
    tfidf_train = vector_train.toarray()
    svd = TSVD(n_components=50, n_iter=10, random_state=42)
    tfidf_train_reduced = svd.fit_transform(tfidf_train)

    svm_train_data = tfidf_train_reduced
    svm_train_tag = train.target

    test = f20(subset='test', categories=cat, shuffle=True, random_state=42)
    vector_test = vectorizer.transform(test.data)
    tfidf_test = vector_test.toarray()
    tfidf_test_reduced = svd.transform(tfidf_test)
    svm_test_data = tfidf_test_reduced
    svm_test_tag = test.target

    OneVSOne(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
    OneVSRest(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
    Gaussian(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
    Bernoulli(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)


def OneVSOne(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    svc = SVC(kernel='linear', C=100)
    svc_ovoc = OVOC(svc)
    svc_ovoc.fit(svm_train_data, svm_train_tag)
    svc_ovoc_predict = svc_ovoc.predict(svm_test_data)
    score = svc_ovoc.score(svm_test_data, svm_test_tag)
    precision = precision_score(svm_test_tag, svc_ovoc_predict, average='weighted')
    recall = recall_score(svm_test_tag, svc_ovoc_predict, average='weighted')
    print "1 VS 1 SVC"
    print "confusion matrix:", "\n", confusion_matrix(svm_test_tag, svc_ovoc_predict)
    print "score=", score
    print "precision=", precision
    print "recall=", recall
    print '\n'

def OneVSRest(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    svc = SVC(kernel='rbf', C=100)
    svc_ovrc = OVRC(svc)
    svc_ovrc.fit(svm_train_data, svm_train_tag)
    svc_ovrc_predict = svc_ovrc.predict(svm_test_data)
    score = svc_ovrc.score(svm_test_data, svm_test_tag)
    precision = precision_score(svm_test_tag, svc_ovrc_predict, average='weighted')
    recall = recall_score(svm_test_tag, svc_ovrc_predict, average='weighted')
    print "1 VS Rest SVC"
    print "confusion matrix:", "\n", confusion_matrix(svm_test_tag, svc_ovrc_predict)
    print "score=", score
    print "precision=", precision
    print "recall=", recall
    print '\n'

def Gaussian(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    gnb = GaussianNB()
    gnb.fit(svm_train_data, svm_train_tag)
    gnb_predict = gnb.predict(svm_test_data)
    score = gnb.score(svm_test_data, svm_test_tag)
    precision = precision_score(svm_test_tag, gnb_predict, average='weighted')
    recall = recall_score(svm_test_tag, gnb_predict, average='weighted')
    print "GaussianNB"
    print "confusion matrix:", "\n", confusion_matrix(svm_test_tag, gnb_predict)
    print "score=", score
    print "precision=", precision
    print "recall=", recall
    print '\n'

def Bernoulli(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    bnb = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    bnb.fit(svm_train_data, svm_train_tag)
    bnb_predict = bnb.predict(svm_test_data)
    # precision, recall, thresholds = precision_recall_curve(svm_test_tag, ovoc_predict)
    # BernoulliNB(alpha=1.0, binarize=0.5, class_prior=None, fit_prior=True)
    score = bnb.score(svm_test_data, svm_test_tag)
    precision = precision_score(svm_test_tag, bnb_predict, average='weighted')
    recall = recall_score(svm_test_tag, bnb_predict, average='weighted')
    print "BernoulliNB"
    print "confusion matrix:", "\n", confusion_matrix(svm_test_tag, bnb_predict)
    print "score=", score
    print "precision=", precision
    print "recall=", recall
    print '\n'


I()