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
from sklearn.linear_model import LogisticRegression

english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def H1():
    cat = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
           'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    train = f20(subset='train', categories=cat, shuffle=False, random_state=0)

    stopwords = text.ENGLISH_STOP_WORDS
    vectorizer = StemmedTfidfVectorizer(
        min_df=1, stop_words=stopwords, decode_error='ignore')
    vector_train = vectorizer.fit_transform(train.data)
    tfidf_train = vector_train.toarray()
    svd = TSVD(n_components=50, n_iter=10, random_state=42)
    tfidf_train_reduced = svd.fit_transform(tfidf_train)
    svm_train_data = tfidf_train_reduced
    svm_train_tag = []
    for i in train.target:
        if (i < 4):
            svm_train_tag.append(0)
        else:
            svm_train_tag.append(1)
    svm_train_tag = np.array(svm_train_tag)

    test = f20(subset='test', categories=cat, shuffle=True, random_state=42)
    vector_test = vectorizer.transform(test.data)
    tfidf_test = vector_test.toarray()
    tfidf_test_reduced = svd.transform(tfidf_test)
    svm_test_data = tfidf_test_reduced
    svm_test_tag = []
    for i in test.target:
        if (i < 4):
            svm_test_tag.append(0)
        else:
            svm_test_tag.append(1)
    svm_test_tag = np.array(svm_test_tag)

    # L1 Regularization
    # L1(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
    # L2 Regularization
    # L2(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
    # L3(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
    coef = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    for c in coef:
        L1(c, svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)
        L2(c, svm_train_data, svm_train_tag, svm_test_data, svm_test_tag)


def L1(c, svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    c1 = c
    lr1 = LogisticRegression('l1', False, C=c1, warm_start=True, solver='liblinear')
    lr1.fit(svm_train_data, svm_train_tag)
    lr1_predict = lr1.predict(svm_test_data)
    precision, recall, thresholds = precision_recall_curve(svm_test_tag, lr1_predict)
    score = lr1.score(svm_test_data, svm_test_tag)
    test_score = lr1.predict_proba(svm_test_data)
    lr1_fpr, lr1_tpr, lr1_thr = roc_curve(svm_test_tag, test_score[:, 1])
    lr1_auc = auc(lr1_fpr, lr1_tpr)
    plt.figure(1)
    plt.title("ROC of Naive Logistic Regression with L1 Regularization")
    plt.plot(lr1_fpr, lr1_tpr, lw=1)
    plt.savefig("../new_pic/l1_c=" + str(c) + ".png", format='png')
    plt.clf()
    print "LR1: " + str(c)
    print "confusion matrix:", "\n", confusion_matrix(svm_test_tag, lr1_predict)
    print "score=", score
    print "precision=", precision[1]
    print "recall=", recall[1]
    print "auc=", lr1_auc
    print "\n"


def L2(c, svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    c2 = c
    lr2 = LogisticRegression('l2', False, C=c2, warm_start=True, solver='liblinear')
    lr2.fit(svm_train_data, svm_train_tag)
    lr2_predict = lr2.predict(svm_test_data)
    precision, recall, thresholds = precision_recall_curve(svm_test_tag, lr2_predict)
    score = lr2.score(svm_test_data, svm_test_tag)
    test_score = lr2.predict_proba(svm_test_data)
    lr2_fpr, lr2_tpr, lr2_thr = roc_curve(svm_test_tag, test_score[:, 1])
    lr2_auc = auc(lr2_fpr, lr2_tpr)
    plt.figure(1)
    plt.plot(lr2_fpr, lr2_tpr, lw=1)
    plt.title("ROC of Naive Logistic Regression with L2 Regularization")
    plt.savefig("../new_pic/l2_c=" + str(c) + ".png", format='png')
    plt.clf()
    print "LR2: " + str(c)
    print "confusion matrix:", "\n", confusion_matrix(svm_test_tag, lr2_predict)
    print "score=", score
    print "precision=", precision[1]
    print "recall=", recall[1]
    print "auc=", lr2_auc
    print "\n"


def L3(svm_train_data, svm_train_tag, svm_test_data, svm_test_tag):
    lr2 = LogisticRegression(C=10)
    lr2.fit(svm_train_data, svm_train_tag)
    lr2_predict = lr2.predict(svm_test_data)
    precision, recall, thresholds = precision_recall_curve(svm_test_tag, lr2_predict)
    score = lr2.score(svm_test_data, svm_test_tag)
    test_score = lr2.predict_proba(svm_test_data)
    lr2_fpr, lr2_tpr, lr2_thr = roc_curve(svm_test_tag, test_score[:, 1])
    lr2_auc = auc(lr2_fpr, lr2_tpr)
    plt.figure()
    plt.plot(lr2_fpr, lr2_tpr, lw=1)
    plt.title("ROC of Naive Logistic Regression with L2 Regularization")
    plt.show()
    print "LR2"
    print "confusion matrix:", "\n", confusion_matrix(svm_test_tag, lr2_predict)
    print "score=", score
    print "precision=", precision[1]
    print "recall=", recall[1]
    print "auc=", lr2_auc
    print "\n"


H1()
