import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# TODO: Using Bag of words

data = pd.read_csv('data/Restaurant_Reviews.tsv', sep='\t', quoting=3)  # quoting is to ignore double quotes
# print(data.head())

# clean the data
corpus = []
ps = PorterStemmer()
nltk.download('stopwords')
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])  # take alphabetical characters only
    review = review.lower()
    review = review.split()
    review = [word for word in review if word not in stopwords.words('english')]  # remove stopwords
    review = " ".join(review)
    corpus.append(review)

# bag of word model creation
bow = CountVectorizer(max_features=1500)
X = bow.fit_transform(corpus).toarray()  # text review
y = data.iloc[:, 1].values  # take only first column (1/0 label)

# apply Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
cfier = GaussianNB()
cfier.fit(X_train, y_train)

# predict
y_pred = cfier.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy score:", score)