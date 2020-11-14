import requests
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical


def locatePATH(PATH):
    try:
        global lines
        lines = open(PATH).read().split('\n')
    except ValueError:
        return None
    except FileNotFoundError:
        return None


def clean_text(doc):
    tokens = doc.split()
    tokens = [word for word in tokens if word.isalpha()]  # leave only alpha characters
    tokens = [word.lower() for word in tokens]  # make it lowercase

    return tokens


def generate_text(model, tokenizer, text_seq_length, seed_text, n_words):
    text = []

    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=text_seq_length, truncating='pre')
        y_predict = model.predict_classes(encoded)


        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_predict:
                predicted_word = word
                break
        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)
    return ' '.join(text)


# get a text to learn from
text = requests.get('http://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
data = text.text.split('\n')  # get info line by line
data = data[253:]  # start from index 253
data = " ".join(data)  # join them so that it converts to text

ftokens = clean_text(data)

# create data sequence
# use 100 sets of words to predict the next word
length = 50 + 1  # +1 for output
lines = []

for i in range(length, len(ftokens)):
    seq = ftokens[i-length:i]  # start from 51
    line = ' '.join(seq)
    lines.append(line)
    if i > 300000:   # cap it so it executes faster
        break

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)  # fit lines on tokenization -> unique words will be embedded as an integer
sequences = tokenizer.texts_to_sequences(lines)

# convert to numpy array
sequences = np.array(sequences)

# prepare for training
x = sequences[:, :-1]  # [all except the last]
y = sequences[:, -1]   # [only last]

vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)

# Build LSTM model
model = Sequential()
seq_len = x.shape[1]
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_len))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=vocab_size, activation='softmax'))   # 73

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # multiclass
model.fit(x, y, batch_size=256, epochs=500)
model.save('LSTM_TextGeneration.h5')

# predict
testLine = "Sometimes I like to go to the"
result = generate_text(model, tokenizer, seq_len, testLine, 10)  # 10 indicates how many words we want to predict
print(result)