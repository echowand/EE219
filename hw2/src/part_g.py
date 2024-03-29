import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as TSVD
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import nltk.stem

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def G():
    cat = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    train = f20(subset='train', categories=cat,shuffle =False, random_state = 42)

    stopwords = text.ENGLISH_STOP_WORDS
    vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words=stopwords, decode_error='ignore')
    vector_train = vectorizer.fit_transform(train.data)
    tfidf_train=vector_train.toarray()
    svd = TSVD(n_components = 50, n_iter = 10, random_state = 42)
    tfidf_train_reduced = svd.fit_transform(tfidf_train)
    svm_train_data = tfidf_train_reduced

    svm_train_tag = []
    for i in train.target:
        if(i < 4):
            svm_train_tag.append(0)
        else:
            svm_train_tag.append(1)
    svm_train_tag=np.array(svm_train_tag)

    test = f20(subset='test',categories=cat, shuffle = True, random_state = 42)
    vector_test = vectorizer.transform(test.data)
    tfidf_test=vector_test.toarray()
    tfidf_test_reduced = svd.transform(tfidf_test)
    svm_test_data = tfidf_test_reduced
    svm_test_tag = []
    for i in test.target:
        if(i < 4):
            svm_test_tag.append(0)
        else:
            svm_test_tag.append(1)
    svm_test_tag=np.array(svm_test_tag)
    # Gaussian Bayes
    Gaussian(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
    # Bernoulli Bayes
    Bernoulli(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)

def Gaussian(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    gnb = GaussianNB()
    gnb.fit(svm_train_data, svm_train_tag)
    gnb_predict=gnb.predict(svm_test_data)
    test_score = gnb.predict_proba(svm_test_data)
    precision, recall, thresholds = precision_recall_curve(svm_test_tag,gnb_predict)
    score = gnb.score(svm_test_data, svm_test_tag)
    gnb_fpr,gnb_tpr,gnb_thr=roc_curve(svm_test_tag,test_score[:,1])
    gnb_auc = auc(gnb_fpr, gnb_tpr)
    plt.figure()
    plt.plot(gnb_fpr, gnb_tpr, lw = 1)
    plt.title("ROC of Gaussian")
    plt.show()
    print "GaussianNB"
    print "confusion matrix:","\n",confusion_matrix(svm_test_tag, gnb_predict)
    print "score=",score
    print "precision=",precision[1]
    print "recall=",recall[1]
    print "auc=",gnb_auc
    print "\n"

def Bernoulli(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    binarizes=np.linspace(-0.007,-0.008,3)#best -0.0075
    for b in binarizes:
        bnb = BernoulliNB(alpha=1.0, binarize=b, class_prior=[0.5047619048,0.4952380952], fit_prior=True)
        bnb.fit(svm_train_data, svm_train_tag)
        bnb_predict=bnb.predict(svm_test_data)
        test_score=bnb.predict_proba(svm_test_data)
        precision, recall, thresholds = precision_recall_curve(svm_test_tag, bnb_predict)
        bnb_fpr,bnb_tpr,bnb_thr=roc_curve(svm_test_tag,test_score[:,1])
        bnb_auc = auc(bnb_fpr, bnb_tpr)
        plt.figure()
        plt.plot(bnb_fpr, bnb_tpr, lw = 1)
        plt.title("ROC of Bernoulli")
        plt.show()

        score=bnb.score(svm_test_data,svm_test_tag)
        print "BernoulliNB,",b
        print "confusion matrix:","\n",confusion_matrix(svm_test_tag, bnb_predict)
        print "score=",score
        print "precision=",precision[1]
        print "recall=",recall[1]
        print "auc=",bnb_auc
        print "\n"


G()