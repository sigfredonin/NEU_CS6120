"""
p4_utils.py
Utilities to support the solution of Assigment 2 Problem 4, Summary Evaluation

In this question, you will design classifiers to evaluate summary quality based
on its non-redundancy and fluency. You will be given a training data and a test
data, both in the following format:
1. Column 1: summary for a news article
2. Column 2: non-redundancy score
  [a score that indicates the conciseness of the summary]
  Non-redundancy scores range from -1 [highly redundant] to 1 [no redundancy].
3. Column 3: fluency
  [a score that indicates whether a sentence is grammatically correct in the summary]
  Fluency can range from -1 [grammatically poor] to 1 [fluent and grammatically correct].

The file is in .csv spreadsheet format - the summary is a quoted string, and it
is followed by the scores, with commas between, as in ...

"The 41-year-old was the first female official on the ▃ level and the first to
 retain a bowl game. She was the first female official on the ▃ level and the
 first to retain a bowl game, the 2009 little caesars pizza bowl in Detroit.",-1,0

Sig Nin
October 26, 2018
"""

import os
import re
import numpy as np
import nltk

from datetime import datetime
from nltk import FreqDist
from gensim.models import KeyedVectors
from scipy import spatial

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# Input summary data ---
# ------------------------------------------------------------------------

PATH_TRAIN = "data/a2_p4_train_set.csv"
PATH_TEST = "data/a2_p4_test_set.csv"

def get_text_from_file(filePath):
    """
    Read all of the text from a file.
    This is expected to be a file containing the summaries,
    one quoted sentence per line, with non-redundancy and fluency
    scores at the end, separated by commas from the sentence and
    each other.
    Example:
      "The movie really stunk ! The movie proved bad bad .",-1,0
    """
    with open(filePath) as f:
        text = f.read()
    return text

re_summary = re.compile(r'^\"(.+)\",(.+),(.+)$', re.MULTILINE)

def get_summary_training_data(text):
    records = re_summary.findall(text)
    summaries, non_redundancies, fluencies = zip(*records)
    np_non_redundancies_float = np.array(non_redundancies).astype(np.float)
    np_fluencies_float = np.array(fluencies).astype(np.float)
    return summaries, np_non_redundancies_float, np_fluencies_float

# ------------------------------------------------------------------------
# Data preprocessing ---
# ------------------------------------------------------------------------

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = { ',', '.', '?', '!', ';', ':' }

def get_ngrams_words(words, N):
    grams = [tuple(words[i:i+N]) for i in range(len(words)-N+1)]
    return grams

def get_words_in_sents_summary(summary, NO_STOPS=False, NO_PUNCT=False):
    sents = nltk.sent_tokenize(summary)
    words_in_sents = [ [ w for w in nltk.word_tokenize(s) \
                            if not (NO_STOPS and w in STOPWORDS) and \
                               not (NO_PUNCT and w in PUNCTUATION) \
                       ] for s in sents ]
    return words_in_sents

def get_words_summary(summary, NO_STOPS=True, NO_PUNCT=True):
    """
    Get the words in a summary, with stop words and punctuation filtered out.
    """
    words_in_sents = \
        get_words_in_sents_summary(summary, NO_STOPS=NO_STOPS, NO_PUNCT=NO_PUNCT)
    words = [ w for s in words_in_sents for w in s ]
    return words

def get_bigrams_summary(summary, NO_STOPS=False, NO_PUNCT=False):
    """
    Get the bigrams in a summary, with stop words filtered out
    """
    words = get_words_summary(summary, NO_STOPS=NO_STOPS, NO_PUNCT=NO_PUNCT)
    bigrams = get_ngrams_words(words, 2)
    return bigrams

def get_most_frequent(items):
    """
    Get the most frequent item and its count from the items.
    Each item must be hashable, such as a word, string, or tuple.
    A list is not hashable, so items cannot be a list of lists.
    """
    fd = FreqDist(items)
    most_freq = fd.most_common(1)
    item, count = most_freq[0]
    return item, count

# ------------------------------------------------------------------------
# Word vector embeddings ---
# ------------------------------------------------------------------------

WORD_VECTORS_FILE = "../p3/data/GoogleNews-vectors-negative300.bin"

def load_embeddings_gensim():
    """
    Load pre-trained word embeddings from the Google News dataset.
    Returns -
        vectors - the KeyedVectors object loaded from the Google News file.
    """
    vectors = KeyedVectors.load_word2vec_format(WORD_VECTORS_FILE, binary=True)
    return vectors

def get_vocabulary(vectors, fd_words):
    """
    Compile a dictionary of vector index -> word, for the words given.
    Compile a reverse dictionary: vector index -> word
    Filter out words not in the vectors.
    Input:
        vectors - a gensim KeyedVectors object containing the word embeddings
        fd_words - vocabulary words in an NLTK frequency distribution
    Returns -
        wv_vocabulary - [ word, ... ] for review words included in vectors
        wv_dictionary - { word : word vector index, ... }
        wv_reverse_dictionary - { word vector index : word , ... }
    """
    v = vectors
    wv_vocabulary = ['</s'] + [ w for w in fd_words if w in v.vocab ]
    wv_dictionary = { w : v.vocab[w].index for w in fd_words \
        if w in v.vocab }
    wv_dictionary['</s>'] = v.vocab['</s>'].index
    wv_reverse_dictionary = { v.vocab[w].index : w for w in fd_words \
        if w in v.vocab }
    wv_reverse_dictionary[v.vocab['</s>'].index] = '</s>'
    return wv_vocabulary, wv_dictionary, wv_reverse_dictionary

def get_embeddings(vectors, words_in_sents):
    """
    Convert words to pre-trained embedding vectors from the Google News dataset.
    Filter out words not in the vectors.
    Inputs -
        vectors - a gensim KeyedVectors object containing the word embeddings
        words_in_sents - list of words, grouped by sentence
            [
                [ word, word, ...]    # sentence 1
                [ word, word, ...]    # sentence 2
                ...
                [ word, word, ...]    # sentence N
            ]
    Returns -
        vw_data - [
                [ word vector index, word vector index, ...]    # sentence 1
                [ word vector index, word vector index, ...]    # sentence 2
                ...
                [ word vector index, word vector index, ...]    # sentence N
            ]
        vw_vectors - [
                [ word vector, word vector, ...]                # sentence 1
                [ word vector, word vector, ...]                # sentence 2
                ...
                [ word vector, word vector, ...]                # sentence N
            ]
        vw_sentence_average_vectors = [
                average word vector,                            # sentence 1
                average word vector,                            # sentence 2
                ...
                average word vector,                            # sentence n
            ]
    NOTE: append '</s>' to each review to ensure all have at least one word
          in the vocabulary of the word embeddings.
    NOTE: the word embeddings include the NLTK English stop words, except:
            'a', 'and', 'mightn', "mightn't", 'mustn', "mustn't",
            "needn't", 'of', "shan't", 'to'
          The word embeddings do not include punctuation marks.
    """
    v = vectors
    wv_data = [ [ v.vocab[w].index for w in s+['</s>'] if w in v.vocab ] \
        for s in words_in_sents ]
    wv_vectors = [ np.array([ v[w] for w in s+['</s>'] if w in v.vocab ]) \
        for s in words_in_sents ]
    wv_sentence_average_vectors = [ np.mean(s, axis=0) for s in wv_vectors ]
    return wv_data, wv_vectors, wv_sentence_average_vectors

def max_cosine_similarity(wva_sents, DEBUG=False):
    """
    Find the maximum pairwise cosine similarity between the sentences
    listed, which are represented as the average of embedding vectors
    of the words in them.
    Note:
    If there is only one sentence, returns max_similarity = 0,
    which is probably the most favorable value for a summary.
    """
    max_similarity = 0.0
    max_similarity_indices = (0, 0)
    for i, s1 in enumerate(wva_sents):
        for j in range(i+1, len(wva_sents)):
            if DEBUG:
                print("Trying %d:%d" % (i, j))
            s2 = wva_sents[j]
            similarity = 1 - spatial.distance.cosine(s1, s2)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_indices = (i, j)
    return max_similarity, max_similarity_indices

# ------------------------------------------------------------------------
# Non-redundancy features (for problem 4.1) ---
# ------------------------------------------------------------------------

def get_non_redundancy_features(vectors, summary, DEBUG=False):
    """
    Get the non-redundancy features for a summary.
    Returns -
        - max unigram frequency (int)
            with stop words and punctuation removed
        - max bigram frequency (int)
            with stop words and punctuation still present
        - max sentence similarity (float in 0.0 ... 0.1)
            between average word embeddings
    """
    words = get_words_summary(summary)
    mf_unigram, mf_unigram_count = get_most_frequent(words)
    if DEBUG:
        print("Most frequent unigram: %s : %d" % (mf_unigram, mf_unigram_count))
    bigrams = get_bigrams_summary(summary)
    mf_bigram, mf_bigram_count = get_most_frequent(bigrams)
    if DEBUG:
        print("Most frequent bigram: %s : %d" % (mf_bigram, mf_bigram_count))
    words_in_sents = get_words_in_sents_summary(summary)
    _data, _vectors, wva_sents = get_embeddings(vectors, words_in_sents)
    max_similarity, (s1, s2) = max_cosine_similarity(wva_sents)
    if DEBUG:
        print("Max cosine similarity: %7.4f : (%d,%d)" % (max_similarity, s1, s2))
    return mf_unigram_count, mf_bigram_count, max_similarity

def plot_similarity_hist(similarities):
    plt.figure()
    plt.hist(similarities, bins=51)
    plt.title("Max Sentence Cosine Similarity")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plotName = "tests/p4_utils_max_sent_cos_sim_"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(plotName + timestamp + '.png')
