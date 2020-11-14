import pandas as pd
import spacy
from spacy.symbols import punct
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from spacy.lang.en.stop_words import STOP_WORDS
import string


def locatePATH(PATH):
    global lines
    lines = open(PATH).read().split('\n')


def reviewClassification():
    # custom function to remove punctuation and stopwords.  Also make everything lc and produce lemma
    def text_cleaning(sentence):
        doc = nlp(sentence)
        stopwords = list(STOP_WORDS)
        tokens = []
        cleaned_tokens = []
        punct = string.punctuation

        for token in doc:
            if token.lemma_ != "-PRON-":
                temp = token.lemma_.lower().strip()
            else:
                temp = token.lower_
            tokens.append(temp)

        for token in tokens:
            if token not in stopwords and token not in punct:
                cleaned_tokens.append(token)

        return cleaned_tokens

    data = pd.read_csv('data/yelp_labelled.txt', sep='\t', header=None)  # quoting is to ignore double quotes
    nlp = spacy.load("en_core_web_sm")  # model (pretrained)

    # assign column names
    column_name = ['Review', 'Sentiment']
    data.columns = column_name
    X = data['Review']
    y = data['Sentiment']

    # vectorization feature engineering (TF-IDF)
    tfidf = TfidfVectorizer(
        tokenizer=text_cleaning)  # convert all the original records into numbers. tf-idf is a measure..
    cfier = LinearSVC()

    # train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # fit
    clf = Pipeline([('tfidf', tfidf), ('clf', cfier)])  # first do vectorization and then classification
    clf.fit(X_train, y_train)

    results = []
    predictions = []
    # predict
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Accuracy score:", score)

    for i in range(0, len(lines)):
        results.append(lines[i])
        pred = clf.predict([lines[i]])
        if 0 in pred:
            predictions.append("bad")
        else:
            predictions.append("good")

    return score, results, predictions