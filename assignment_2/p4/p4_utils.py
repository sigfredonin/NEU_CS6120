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
November 9, 2018
"""

import os
import re
import numpy as np
import random
import nltk
import csv

from datetime import datetime
from nltk import FreqDist
from gensim.models import KeyedVectors
from scipy import spatial
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# Input summary data ---
# ------------------------------------------------------------------------

PATH_TRAIN = "data/a2_p4_train_set.csv"
PATH_TEST = "data/a2_p4_test_set.csv"

def remove_extra_whitespace(token_str):
   no_new_line = re.sub(r'\n', " ", token_str)
   no_dup_spaces = re.sub(r'  +', " ", no_new_line)
   return no_dup_spaces

def get_summary_data(filePath):
    """
    Read all of the records from the training file.
    This is a csv file containing the summaries,
    one sentence per line, with non-redundancy and fluency
    scores at the end, separated by commas from the sentence and
    each other.
    The sentence is quoted if it contains quotation marks.
    Examples:
        The movie really stunk ! The movie proved bad bad .,-1,0
        "Mary's new movie is a real showcase of her talents . Hurrah !",-1,0.5
    Note:
        Drop the first and second records from the training data.
        The first record is a heading ...
            Summary,Non-Redundancy,Fluency
        The second record has no ratings at the end, so is useless for training ...
            "Nepalese, ▃, sustained severe injuries ... facing in the. .",,null
    """
    with open(filePath) as f:
        records = list(csv.reader(f))[2:]
    summaries, non_redundancies, fluencies = zip(*records)
    summaries = [ remove_extra_whitespace(s) for s in summaries ]
    np_non_redundancies_float = np.array(non_redundancies).astype(np.float)
    np_fluencies_float = np.array(fluencies).astype(np.float)
    return summaries, np_non_redundancies_float, np_fluencies_float

def load_summary_training_data():
    return get_summary_data(PATH_TRAIN)

def load_summary_test_data():
    return get_summary_data(PATH_TEST)

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

def get_bigrams_summary_no_stops(summary, NO_STOPS=True, NO_PUNCT=True):
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

def get_embeddings_vocabulary(vectors, fd_words):
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
    NOTE: append '</s>' to each review to ensure all have at least one word
          in the vocabulary of the word embeddings.
    """
    v = vectors
    wv_vocabulary = ['</s>'] + [ w for w in fd_words if w in v.vocab ]
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
        wv_data - [
                [ word vector index, word vector index, ...]    # sentence 1
                [ word vector index, word vector index, ...]    # sentence 2
                ...
                [ word vector index, word vector index, ...]    # sentence N
            ]
        wv_vectors - [
                [ word vector, word vector, ...]                # sentence 1
                [ word vector, word vector, ...]                # sentence 2
                ...
                [ word vector, word vector, ...]                # sentence N
            ]
        wv_sentence_average_vectors = [
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
          Of these, the words 'a', 'and', 'of', and 'to' are used in the training data.
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

# ------------------------------------------------------------------------
# Prepare training and evaluation data ---
# ------------------------------------------------------------------------

def shuffle_data(review_data, review_labels):
    """
    Shuffle the reviews and labels.
    Return:
        shuffled_indices -     list of original indices, e.g.,
                            review_data[shuffled_indices[0]] == shuffled_data[0]
        shuffled_data -     list of shuffled sentences
        shuffled_labels -   list of shuffle ratings
    """
    # Use the same seed for all the shuffles,
    # so the data, labels, and indices are all shuffled the same way
    seed = 123

    # Shuffle the data
    random.seed(seed)
    shuffled_data = review_data.copy()
    random.shuffle(shuffled_data)

    # Shuffle the labels
    random.seed(seed)
    shuffled_labels = review_labels.copy()
    random.shuffle(shuffled_labels)

    # Construct an index to the shuffled data:
    # where did the original reviews end up?
    shuffled_indices = list(range(len(review_labels)))
    random.seed(seed)
    random.shuffle(shuffled_indices)

    return shuffled_indices, shuffled_data, shuffled_labels

def split_training_data_for_cross_validation(review_data, review_labels, number_of_trials):
    """
    Spit the data and labels into N equal size sets, for N trials, after shuffling.
    Return:
        shuffled_indices - list of original indices, e.g.,
                        review_data[shuffled_indices[0]] == data[0][0][0]
        data = [
                    ( training_data, validation_data )      # set 1
                    ( training_data, validation_data )      # set 2
                    ...
                    ( training_data, validation_data )      # set N
                ]
        where each set has a different (N-1)/N training and 1/N validation data.
    """
    # Shuffle the data and labels
    shuffled_indices, shuffled_data, shuffled_labels = \
        shuffle_data(review_data, review_labels)

    # Subdivide the data and labels into # NOTE:  ~ equal size sets each
    N = number_of_trials
    set_count = len(shuffled_data) // N
    xval_sets = []
    for iSet in range(N-1):
        xD = iSet * set_count
        xval_sets.append(( \
            shuffled_data[xD : xD + set_count], \
            shuffled_labels[xD : xD + set_count] ))
    xval_sets.append(( \
        shuffled_data[(N-1) * set_count : len(shuffled_data)], \
        shuffled_labels[(N-1) * set_count : len(shuffled_data)] ))

    return shuffled_indices, xval_sets

def assemble_cross_validation_data(xval_sets, index_val):
    """
    Assemble training data and labels, and validation data and labels,
    for cross-validation training.
    Inputs:
        xval_sets-   sets of data and labels, one to be used for validation.
        index-  whidh set to use for validation.
    Returns:
        cross_validation data and labels -
            training_set -  ( training_data, training_labels)
            val_set -       ( val_data, val_labels )
    """
    val_set = xval_sets[index_val]
    training_data = []
    training_labels = []
    for iSet, data_labels in enumerate(xval_sets):
        if iSet != index_val:
            _data_, _labels_ = data_labels
            training_data += _data_
            training_labels += _labels_
    training_set = ( training_data, training_labels)
    return training_set, val_set

def assemble_full_training_data(xval_sets):
    """
    Assemble training data and labels for training with the full training set.
    Inputs:
        xval_sets-   sets of data and labels.
    Returns:
        full training data and labels -
            training_set -  ( training_data, training_labels)
    """
    training_data = []
    training_labels = []
    for iSet, data_labels in enumerate(xval_sets):
        _data_, _labels_ = data_labels
        training_data += _data_
        training_labels += _labels_
    training_set = ( training_data, training_labels)
    return training_set

# ------------------------------------------------------------------------
# Visualize results ---
# ------------------------------------------------------------------------

def plot_results(np_train_loss, np_train_mse, np_val_loss, np_val_mse, \
        heading, subheading, plotName='tests/p4_tf_MLP_test_plot_'):

    figure, axis_1 = plt.subplots()

    plt.suptitle(heading, size=12)
    plt.title(subheading, size=10)

    # Plot loss for both training and validation
    axis_1.plot(np_train_loss, 'r--')
    axis_1.plot(np_val_loss, 'b--')
    axis_1.set_xlabel('Epoch')
    axis_1.set_ylabel('Avg Loss')
    axis_1.legend(['Training Loss', 'Validation Loss'], loc='upper left')

    # Plot MSE for both training and validation
    axis_2 = axis_1.twinx()
    axis_2.plot(np_train_mse, 'r')
    axis_2.plot(np_val_mse, 'b')
    axis_2.set_ylabel('Avg MSE')
    axis_2.legend(['Training MSE', 'Validation MSE'], loc='upper right')

    figure.subplots_adjust(top=0.9)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(plotName + timestamp + '.png')
