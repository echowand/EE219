import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
twenty_all = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
print len(twenty_all.data)
print twenty_all.target[0]

#count term

twenty_all_X = twenty_all.data
count_vect = CountVectorizer(stop_words='english')
X_counts = count_vect.fit_transform(twenty_all_X)
X = X_counts.toarray()

vocab_new={}
vocab = count_vect.vocabulary_
stemmer = PorterStemmer()
for i in vocab:
    word = stemmer.stem(i)
    if word in vocab_new:
        vocab_new[word].append(vocab[i])
    else:
        vocab_new[word]=[vocab[i]]
term_num = len(vocab_new)
print term_num