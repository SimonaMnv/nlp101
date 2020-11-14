import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


def locatePATH(PATH):
    global lines
    lines = open(PATH).read().split('\n')


# custom text load from .txt
text = []
lines = open('data/mobydick.txt').read().split('\n')
text = ''.join(lines)
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

tokens = [token.text for token in doc]
punctuation = punctuation + '\n'

# clean text
frequency = {}
stop_words = list(STOP_WORDS)

for word in doc:
    if word.text.lower() not in stop_words:
        if word.text.lower() not in punctuation:
            if word.text not in frequency.keys():
                frequency[word.text] = 1
            else:
                frequency[word.text] += 1

max_freq = max(frequency.values())
for word in frequency.keys():
    frequency[word] = frequency[word] / max_freq
# print(frequency)

# sentence tokenization -- get a score for every sentence
sent_tokens = [sent for sent in doc.sents]
sent_score = {}
for sent in sent_tokens:
    for word in sent:
        if word.text.lower() in frequency.keys():
            if sent not in sent_score.keys():
                sent_score[sent] = frequency[word.text.lower()]
            else:
                sent_score[sent] += frequency[word.text.lower()]

# get the summary - select 5% of sentences with max score
largest = round(len(sent_score) * 0.05)
summary = nlargest(n=largest, iterable=sent_score, key=sent_score.get)
final = [word.text for word in summary]
summary = " ".join(final)
print(summary)
