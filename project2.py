import nltk
from nltk.corpus import stopwords, reuters
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words("english"))

# TODO: Tokenize Words
def tokenize(document):
    words = []
    tokens = word_tokenize(document)
    lemmatizer = WordNetLemmatizer()
    for word in tokens:
        if word not in stop_words and word.isalpha():
            words.append(lemmatizer.lemmatize(word.lower()))
    return words

# TODO: Separate Documents
# print(reuters.fileids())
#Test
def get_docs():
    documents = {}
    for docID in reuters.fileids():
        words = reuters.raw(docID)
        documents[docID] = tokenize(words)
    return documents

# TODO: Term freq w/in each doc
def term_frequency(text):
    freq_distribution = FreqDist(text)
    return freq_distribution
# TODO: Doc freq for each term
# TODO: TF-IDF Calc
def tf_idf(doc):
    tfidf = TfidfVectorizer()
    text_tf = tfidf.fit_transform(doc)
    return text_tf
    
# TODO: One more weight system
# TODO: Analyze Queries
# TODO: Calculate Precision 
# TODO:           & Recall
# TODO: Graph Precision & Recall


""" Test samples """
# sample = "This is a sentence of a bunch of words and more words and a butt plus second butt of a kind of thing and stuff love sentence."
# words = reuters.raw(file_ids[0])

documents = get_docs()
# term_per_doc = {}
# for doc_id in documents.keys():
#     term_per_doc[doc_id] = term_frequency(documents[doc_id])

# print(term_per_doc.items()) # Test