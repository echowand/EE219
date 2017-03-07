from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

dataset = f20(subset='all', categories=categories, shuffle=True, random_state=42,
              remove=('headers', 'footers', 'quotes'))

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))

##### reduce feature dimension
# vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
X = vectorizer.fit_transform(dataset.data)
print("number of samples: %d, number of features: %d" % X.shape)
