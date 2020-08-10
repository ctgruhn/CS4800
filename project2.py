import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import re
import math

stop_words = set(stopwords.words("english"))

# TODO: Tokenize Words
def tokenize(document):
    words = []
    tokens = word_tokenize(document)
    lemmatizer = WordNetLemmatizer()
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            words.append(lemmatizer.lemmatize(word.lower()))
    return words

# TODO: Separate Documents
def get_docs():
    corpus_dict = {}
    raw_corpus = []
    for files in glob.glob("Project2/lisa/LISA[0-9].*"):   
        text =open(files,'r')
        split_text = re.split('[*]+\n', text.read())
        split_text = filter(None, split_text)
        raw_corpus.extend(split_text) 
    for docs in raw_corpus:
        temp_arr = []
        contents = re.split('(Document\s+[0-9]+)', docs)
        doc_name = contents[1]
        text = contents[2]
        corpus_dict[doc_name] = tokenize(text)
    return corpus_dict

def get_queries():
    query_dict = {}
    query_doc = open("Project2/lisa/LISA.QUE", "r")
    doc = query_doc.read()
    split_query = re.split('\s+#\n', doc)
    split_query = list(filter(None, split_query))
    for item in split_query:
        temp_list = re.split('^\n?([0-9]+)\n', item)
        temp_list = list(filter(None, temp_list))
        query_id = temp_list[0]
        text = temp_list[1]
        query_dict[query_id] = text
    return query_dict

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

def weight_tf(term_frequency):
    weighted_terms = {}
    for doc_id in term_frequency:
        weight_list = []
        _,count = zip(*term_frequency[doc_id])
        N = sum(count)
        for term, count in term_frequency[doc_id]:
            weight_list.append((term, count / N))
        weighted_terms[doc_id] = weight_list
    print (weighted_terms)

def freq_dist(text):
    freq_distribution = FreqDist(text)
    return freq_distribution

# TODO: Doc freq for each term
def doc_freq_per_term(term_frequency):
    doc_freq ={}
    for doc_id in term_frequency:
        for (word, count)in term_frequency[doc_id]:
            if word in doc_freq:
                doc_freq[word] += 1
            else:
                doc_freq[word] = 1
    return doc_freq

def corpus_term_freq(term_frequency):
    doc_freq ={}
    for doc_id in term_frequency:
        for (word, count)in term_frequency[doc_id]:
            if word in doc_freq:
                doc_freq[word] += count
            else:
                doc_freq[word] = count
    return doc_freq

def idf(corpus):
    inv_doc_freq = {}
    N = len(corpus)
    tf = term_frequency(corpus)
    doc_freq = doc_freq_per_term(tf)
    for word in doc_freq:
        inv_doc_freq[word] = math.log10(N/(doc_freq[word]))
    return inv_doc_freq

# TODO: TF-IDF Calc
def tf_idf(weighted_tf_dict, idf_dict):
    tf_idf_dict = {}
    for doc_id in weighted_tf_dict:
        for term, weight in weighted_tf_dict[doc_id]:
            tf_idf_dict[term] = weight * idf_dict[term]
    return tf_idf_dict

def auto_tf_idf(doc):
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


""" Term Freq Test"""
"""
token_sample = {}
for doc_id in sample:
    token_sample[doc_id] = tokenize(sample[doc_id])
term_per_doc = term_frequency(token_sample)
# print(term_per_doc)
doc_freq = weight_tf(term_per_doc)
inv_doc_freq = idf(token_sample)
# inv_doc_freq = idf(token_sample)
print(inv_doc_freq)
# print(doc_freq)

# """

"""
tfidf = TfidfVectorizer()

fitted_vector = tfidf.fit(documents)
tfidf_vect = fitted_vector.transform(documents)

print(tfidf_vect)
# text_tf = tfidf.fit_transform(doc)



# print(term_per_doc.items()) # Test
"""

"""
documents = get_docs()
print(documents)
# query_dict = get_queries()
# print(query_dict)
"""