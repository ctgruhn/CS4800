import nltk
from nltk.corpus import stopwords, reuters
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def tokenize():
    # words = []
    # for word in reuters.words():
    #     word_tokenize(word.lower())
    #     if word not in stop_words:
    #         words.append(word)
    # return words

    """
    The above code did nothing, reuters.words() is already tokenized.
    Replace w/ text from reuters docs.
    """
    pass


# TODO: Separate Documents
print(reuters.fileids())
#Test
documents = {}
for docIDs in reuters.fileids():
    documents[docIDs] = reuters.raw(docIDs)

print(documents)
# TODO: Tokenize Words
# TODO: Term freq w/in each doc
# TODO: Doc freq for each term
# TODO: TF-IDF Calc
# TODO: One more weight system

# TODO: Analyze Queries
# TODO: Calculate Precision 
# TODO:           & Recall
# TODO: Graph Precision & Recall

