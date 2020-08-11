import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer, PorterStemmer
import glob
import re
import math
import heapq

stop_words = set(stopwords.words("english"))

# TODO: Tokenize Words
def tokenize(document):
    words = []
    tokens = word_tokenize(document)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            # lemmed = lemmatizer.lemmatize(word.lower())
            # stemmed = stemmer.stem(lemmed)
            # words.append(stemmed)
            words.append(word)
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
        query_dict[query_id] = tokenize(text)
    return query_dict

# TODO: Term freq w/in each doc
def term_frequency(corpus):
    freq_dict = {}
    for doc_id in corpus:
        term_per_doc = []
        for word in corpus[doc_id]:
            count = corpus[doc_id].count(word)
            if (count, word) not in term_per_doc:
                term_per_doc.append((count, word))
        freq_dict[doc_id] = term_per_doc
    return freq_dict

def weight_tf(term_frequency):
    weighted_terms = {}
    for doc_id in term_frequency:
        weight_list = []
        count, _ = zip(*term_frequency[doc_id])
        N = sum(count)
        for count, term in term_frequency[doc_id]:
            weight_list.append((count / N, term))
        weighted_terms[doc_id] = weight_list
    return weighted_terms

def freq_dist(text):
    freq_distribution = FreqDist(text)
    return freq_distribution

# TODO: Doc freq for each term
def doc_freq_per_term(term_frequency):
    doc_freq ={}
    for doc_id in term_frequency:
        for (count, word)in term_frequency[doc_id]:
            if word in doc_freq:
                doc_freq[word] += 1
            else:
                doc_freq[word] = 1
    return doc_freq

def corpus_term_freq(term_frequency):
    doc_freq ={}
    for doc_id in term_frequency:
        for (count, word)in term_frequency[doc_id]:
            if word in doc_freq:
                doc_freq[word] += count
            else:
                doc_freq[word] = count
    return doc_freq

def idf(doc_freq, N):
    inv_doc_freq = {}
    for word in doc_freq:
        inv_doc_freq[word] = math.log10(N/(doc_freq[word]))
    return inv_doc_freq

# TODO: TF-IDF Calc
def tf_idf(weighted_tf_dict, idf_dict):
    tf_idf_dict = {}
    for doc_id in weighted_tf_dict:
        doc_tf_idf = {}
        for weight, term in weighted_tf_dict[doc_id]:
            tf_idf_value = weight * idf_dict[term]
            doc_tf_idf[term] = weight * idf_dict[term]
        tf_idf_dict[doc_id] = doc_tf_idf
    return tf_idf_dict

def auto_tf_idf(doc):
    tfidf = TfidfVectorizer()
    text_tf = tfidf.fit_transform(doc)
    return text_tf
    
# TODO: One more weight system
def get_cosine(query, query_tf_idf, doc_tf_idf):
    cos_heap = []
    heapq.heapify(cos_heap)
    for corpus_doc in doc_tf_idf:
        cos_val = 0.0
        for terms in query:
            if terms in doc_tf_idf[corpus_doc]:
                cos_val += (doc_tf_idf[corpus_doc][terms] * query_tf_idf[terms])
        heapq.heappush(cos_heap, (cos_val, corpus_doc))
    return cos_heap

# TODO: Analyze Queries
def tf_idf_retrieval(query, tf_idf):
    rel_doc_heap = []
    heapq.heapify(rel_doc_heap)
    for corpus_doc in tf_idf:
        tf_idf_val = 0.0
        for terms in query:
            if terms in tf_idf[corpus_doc]:
                tf_idf_val += tf_idf[corpus_doc][terms]
        heapq.heappush(rel_doc_heap, (tf_idf_val, corpus_doc))
    return rel_doc_heap

# TODO: Calculate Precision 
# TODO:           & Recall
def get_precision(tf_idf_dict, rel_dict):
    prec_dict = {}
    for query in tf_idf_dict:
        TP= 0
        for doc in rel_dict[query]:
            print(doc)
            if doc in tf_idf_dict[query]:
                TP += 1
            prec_dict[query] = TP / (10)

    return prec_dict

def get_recall(tf_idf_dict, rel_dict):
    recall_dict = {}
    for query in tf_idf_dict:
        TP= 0
        total_rel = len(rel_dict[query])
        for (_, doc)in tf_idf_dict[query]:
            if "Document {}".format(doc)  in rel_dict[query]:
                TP += 1
            recall_dict[query] = TP /total_rel
    return recall_dict
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

"""

# """
# Retrieve Documents and Queries
documents = get_docs()
query_dict = get_queries()


# """
N = len(documents)

# Documents TF and Weighted TF
doc_tf = term_frequency(documents)
query_tf = term_frequency(query_dict)
# print("TERM FREQUENCY")
# print(doc_tf)
doc_weighted_tf = weight_tf(doc_tf)
query_weighted_tf = weight_tf(query_tf)
# print("DOCUMENT WEIGHTED TERM FREQUENCY:")
# print(weighted_doc_tf)

# Corpus DF and IDF
doc_df = doc_freq_per_term(doc_tf)
query_df = doc_freq_per_term(query_tf)
# print("DOCUMENT FREQUENCY")
# print(doc_df)
doc_idf = idf(doc_df, N)
query_idf = idf(query_df, N)
# print("DOCUMENT IDF")
# print(doc_idf)

# Calculate Corpus Weights (TF-IDF)
corpus_tf_idf = tf_idf(doc_weighted_tf, doc_idf)
query_tf_idf = tf_idf(query_weighted_tf, query_idf)
# print("CORPUS TF-IDF")
# print(corpus_tf_idf)

# # token_query = tokenize("I AM INTERESTED IN ALMOST ANYTHING TO DO WITH OCCUPATIONAL HEALTH INFORMATION SERVICES AVAILABLE, SUCH AS LIBRARIES. I AM INTERESTED IN BOTH OCCUPATIONAL NURSES AND OCCUPATIONAL DOCTORS. HEALTH, OCCUPATIONAL HEALTH, NURSES, DOCTORS, MEDICINE.")
# # # # print(token_query)
# # document_ret = tf_idf_retrieval(token_query, corpus_tf_idf)
relevant_query_docs = {}
# print("TOKENIZE ONLY")
MAX_RELEVANT_DOCS = 10
for query_id in query_dict:
    # print("Document ID %s: %s" %(query_id, query_dict[query_id]))
    # query_results = tf_idf_retrieval(query_dict[query_id], corpus_tf_idf)
    query_results = get_cosine(query_dict[query_id], query_tf_idf[query_id], corpus_tf_idf)
    # print(query_tf_idf[query_id])
    relevant_query_docs[query_id] = heapq.nlargest(MAX_RELEVANT_DOCS, query_results)
    print("Query %s:" %(query_id))
    print(relevant_query_docs[query_id])
    print()

# print("TF-IDF RETRIEVAL")
# print(relevant_query_docs)

# # # print(doc_df)
# # # print(len(doc_df))
# # # print(len(documents))
# """

