import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def locatePATH(PATH):
    try:
        global lines
        lines = open(PATH).read().split('\n')
    except ValueError:
        return None
    except FileNotFoundError:
        return None


def SimpleSpamDetection():
    df = pd.read_csv('data/spamTRAIN.tsv', sep='\t')
    print(df.head())
    results = []
    predictions = []

    # Check dataset
    # print(df.isna().sum())  # theres no null values
    print(df['label'].value_counts() / len(df) * 100)  # it's not balanced

    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']
    ham = ham.sample(spam.shape[0])  # Balance it
    data = ham.append(spam, ignore_index=True)  # combine the data

    plt.hist(data[data['label'] == 'ham']['length'], bins=100, alpha=0.8)
    plt.hist(data[data['label'] == 'spam']['length'], bins=100, alpha=0.8)
    # plt.show()

    # separate data. We don't need a big test_size because we mostly test on our custom data.
    X_train, X_test, y_train, y_test = \
        train_test_split(data['message'], data['label'], test_size=0.1, random_state=0)

    # build the model
    cfier = Pipeline([(("tfidf"), TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=100))])
    cfier.fit(X_train, y_train)

    y_pred = cfier.predict(X_test)  # predict results
    accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred))
    print("\n\nModel Accuracy:", accuracy)
    # uncomment below to test on messages (test split) from the spamTRAIN.tsv...
    # for i in range(0, len(y_test)):
    #     results.append(X_test[i:i + 1])
    #     predictions.append(y_pred[i])

    pd.options.display.max_colwidth = 80  # show a part of the message

    # ...else, make prediction on your custom txt file with text
    for i in range(0, len(lines)):
        results.append(lines[i])
        predictions.append(cfier.predict([lines[i]]))

    return accuracy, results, predictions
