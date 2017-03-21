import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sc

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


file_dir ="tweet_data/tweets_#superbowl.txt"

data = []
label = []
filedname =[u'author', u'tweet', u'original_author', u'metrics']
for line in open(file_dir, 'r'):
    tweet = json.loads(line)
    loc = tweet["tweet"]["user"]["location"]
    if 'WA' in loc or 'Washington' in loc:
        data.append(tweet["highlight"])
        label.append(0)
        
    elif 'MA' in loc or 'Massachusetts' in loc:
        data.append(tweet["highlight"])
        label.append(1)
print len(data)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
print X_train[2]

vectorizer = TfidfVectorizer(min_df=1, stop_words='english', strip_accents='ascii', decode_error='ignore')
train = vectorizer.fit_transform(X_train)
test = vectorizer.transform(X_test)
print train.shape
print test.shape

def roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, None)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def stats(expected, predicted, learning_algo):
    print('Classification Report of %s\n'%learning_algo)
    print(metrics.classification_report(expected, predicted))
    print('Confusion Matrix of %s\n'%learning_algo)
    print(metrics.confusion_matrix(expected, predicted))
    print('Accuracy of %s\n'%learning_algo)
    print(metrics.accuracy_score(expected, predicted))
    print '\n'

k = 50
U_km,S_km,V_km = sc.sparse.linalg.svds(train.T,k = k)
train_k = U_km.T * train.T
test_k = U_km.T * test.T

# SVC
svm = SVC(kernel='linear', probability=True, random_state=40)
svm_fit = svm.fit(train_k.T, y_train)
svm_predict = svm_fit.predict(test_k.T)
stats(y_test, svm_predict, 'SVM')
roc(y_test, svm_fit.predict_proba(test_k.T)[:,1])

# Naive Bayes
model_nb = GaussianNB()
model_fitted_nb = model_nb.fit(train_k.T, y_train)
nb_predicted = model_fitted_nb.predict(test_k.transpose())
stats(y_test, nb_predicted, 'Naive Bayes')
roc(y_test, model_fitted_nb.predict_proba(test_k.transpose())[:, 1])

# Logistic
model = LogisticRegression(fit_intercept = False)
model_fitted_logistic = model.fit(train_k.transpose(), y_train)
logistic_predicted = model_fitted_logistic.predict(test_k.transpose())
stats(y_test, logistic_predicted, 'Logistic Regression')
roc(y_test, model_fitted_logistic.predict_proba(test_k.transpose())[:, 1])