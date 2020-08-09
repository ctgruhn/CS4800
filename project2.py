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
def get_docs():
    documents = {}
    for docID in reuters.fileids():
        words = reuters.raw(docID)
        documents[docID] = tokenize(words)
    return documents

# TODO: Term freq w/in each doc
def term_frequency(corpus):
    freq_dict = {}
    for doc_id in corpus:
        term_per_doc = []
        for word in corpus[doc_id]:
            count = corpus[doc_id].count(word)
            if (word, count) not in term_per_doc:
                term_per_doc.append((word, count))
        freq_dict[doc_id] = term_per_doc
    return freq_dict
def freq_dist(text):
    freq_distribution = FreqDist(text)
    return freq_distribution
# TODO: Doc freq for each term
def df(term_frequency):
    doc_freq ={}
    for doc_id in term_frequency:
        print(doc_id)
        for (word, count)in term_frequency[doc_id]:
            print(word)
            if word in doc_freq:
                doc_freq[word] += count
            else:
                doc_freq[word] = count
    return doc_freq

def idf(corpus):
    pass


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
sample = {"doc1": "This is a sentence of a bunch of words and more words and a butt plus second butt of a kind of thing and stuff love sentence.",
        "doc2": "A second sentence for testing purposes. This sentence has words and is about a shark for some reason.",
        "doc3":"I have a blue water bottle and two pencils. One is broken though. The other is not blue, unfortunately."
        }
# words = reuters.raw(file_ids[0])


""" Term Freq Test"""
"""
token_sample = {}
for doc_id in sample:
    token_sample[doc_id] = tokenize(sample[doc_id])
term_per_doc = term_frequency(token_sample)
print(term_per_doc)
doc_freq = df(term_per_doc)
print(doc_freq)
    # term_per_doc[doc_id] = term_frequency(documents[doc_id])
# print(term_per_doc)
"""

"""
tfidf = TfidfVectorizer()

fitted_vector = tfidf.fit(documents)
tfidf_vect = fitted_vector.transform(documents)

print(tfidf_vect)
# text_tf = tfidf.fit_transform(doc)



# print(term_per_doc.items()) # Test
"""