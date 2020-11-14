import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import GlobalMaxPooling1D


def locatePATH(PATH):
    try:
        global lines
        lines = open(PATH).read().split('\n')
    except ValueError:
        return None
    except FileNotFoundError:
        return None


def SpamDectectionCNN():

    def prepData(text, size):
        # Convert to array
        textDataArray = [text]

        # Convert into list with word ids
        Features = tokenizer.texts_to_sequences(textDataArray)
        Features = pad_sequences(Features, size, padding='post')

        return Features

    results = []
    predictions = []
    # read ds
    df = pd.read_csv('data/spam.csv', encoding='ISO-8859-1')
    # print(df.head()) # theres some missing data here

    # drop garbage columns
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

    # rename columns
    df.columns = ['labels', 'data']

    # create binary labels (ham/spam)
    df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
    y = df['b_labels'].values

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['data'], y, test_size=0.1)

    # convert sentences to sequences (embedding technique: bag of words)
    # : every unique character will be given one index position
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(X_train)

    seq_train = tokenizer.texts_to_sequences(X_train)  # each number is unique and corresponds to a unique word
    seq_test = tokenizer.texts_to_sequences(X_test)

    # create one big matrix to pass to the CNN
    #  --> we have to pad it because the CNN accepts a FIXED length but the above process produces a dynamic length due to
    # each sentence having a different length.
    data_train = pad_sequences(seq_train)
    T = data_train.shape[1]
    word2idx = tokenizer.word_index
    V = len(word2idx)  # total number of unique words

    # pad the test set
    data_test = pad_sequences(seq_test, maxlen=T)  # we dont know the maxlen

    # build the model
    D = 20  # this is a hyper parameter -> word vector size
    input_layer = Input(shape=T)
    x = Embedding(V + 1, D)(input_layer)  # N * T * D array. Returns sequence of word vectors
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPool1D(3)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPool1D(3)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input_layer, x)

    # compile model & train
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # ...binary problem
    hist = model.fit(x=data_train, y=y_train, epochs=5)
    # model.summary()

    accuracy = hist.history['accuracy'][-1]
    # test model
    pd.options.display.max_colwidth = 80  # show a part of the message

    # make prediction on your custom txt file with text
    for i in range(0, len(lines)):
        results.append(lines[i])
        textTokenizedTest = prepData(results[i], T)
        predictions.append(model.predict(textTokenizedTest).item())

    return accuracy, results, predictions


