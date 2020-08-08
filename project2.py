import nltk
from nltk.corpus import stopwords, reuters
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def tokenize():
    words = []
    for word in reuters.words():
        word_tokenize(word.lower())
        if word not in stop_words:
            words.append(word)
    return words

print(tokenize()) #For testing purposes