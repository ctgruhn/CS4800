import nltk
from nltk.corpus import stopwords, reuters
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def tokenize():
    words = []
    for word in reuters.words():
        w = word.lower()
        word_tokenize(w)
        if w not in stop_words:
            words.append(w)
    return words

print(tokenize()) #For testing purposes