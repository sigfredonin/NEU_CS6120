"""
p3_utils.py
Utilities to support the solution of Assigment 2 Problem 3, Sentiment Analysis

Sig Nin
October 26, 2018
"""

import os
import re
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from struct import unpack
from nltk import FreqDist

# ------------------------------------------------------------------------
# Exceptions ---
# ------------------------------------------------------------------------

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InvalidFileException(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

# ------------------------------------------------------------------------
# Input review data ---
# ------------------------------------------------------------------------

VOCABULARY_SIZE = 10000

def get_text_from_file(filePath):
    """
    Read all of the text from a file.
    This is expected to be a file containing the reviews,
    one tokenized sentence per line, with a rating at the end,
    following "|".
    Example:
      "This movie really stunk !|1"
    """
    with open(filePath) as f:
        text = f.read()
    return text

def get_words_and_ratings(text):
    """
    Get the words and ratings in the reviews.
    Also get the vocabulary, sorted with most frequent words first.
    Input: raw text
    Output:
        words: all of the words in all of the reviews, in order of occurrence.
        review_words: words in each review
            [ [word_1, word_2, ...], ... ]
        review_data: indices in dictionary of words in each review
            [ [word_index_1, word_index_2, ..., word_index_n, 0, ... ], ... ]
            each review is zero-padded to a vector length that is
            a multiple of 10 >= length of longest review
        review_labels: ratings from each review
            [ r_1, r_2, ... ], where r_i is a decimal number in { 0 .. 4 }
        vocabulary: the VOCABULARY_SIZE most frequent wordS,
            in order of decreasing frequency
        dictionary: a mapping word -> index in the vocabulary
        reverse_dictionary: a mapping index -> word
        review_data: review words encoded as indices into the vocabulary
    """
    # collect words and ratings from the reviews
    re_reviews = re.compile(r'^(.+)\|(.+)$', re.MULTILINE)
    reviews = re_reviews.findall(text.lower())
    review_words = [ s.split() for s, r in reviews ]
    review_labels = [int(r) for s, r in reviews ]
    words = [ w for ws in review_words for w in ws ]
    # compile vocabulary
    fd_words = FreqDist(words)
    frequents = [ ('UNK', 0) ] + fd_words.most_common(VOCABULARY_SIZE)
    # set the count of the unknown words (not in top VOCABULARY_SIZE)
    count_UNK = 0
    for word in fd_words:
        if word not in frequents:
            count_UNK += 1
    frequents[0] = ( 'UNK', count_UNK )
    # compile the dictionary and reverse dictionary
    vocabulary, word_counts = zip(*frequents)
    dictionary = {}
    reverse_dictionary = {}
    for iWord, word in enumerate(vocabulary):
        dictionary[word] = iWord
        reverse_dictionary[iWord] = word
    # compile review data as the indices of the words into the vocabulary
    review_data = []
    max_len = max(len(review) for review in review_words)
    len_review_vector = ((max_len + 9) // 10) * 10  # multiple of 10
    for review in review_words:
        words_in_review = [0] * len_review_vector
        for i, word in enumerate(review):
            words_in_review[i] = dictionary.get(word, 0)
        review_data.append(words_in_review)
    # return the results
    return words, review_words, review_data, review_labels, \
        fd_words, vocabulary, dictionary, reverse_dictionary

# ------------------------------------------------------------------------
# Transform review representation ---
# ------------------------------------------------------------------------

def one_hot_vectors(review_data, vocabulary_size):
    """
    Returns a vector for each review that has
    a one for each word that occurs in the review.
    Repeated words still get just a one.
    Returns:
        review_data_one_hot:
            one list per review,
            review list has zero or one for each word in the vocabulary,
            one if it occurs in the review, zero if it does not.
    """
    review_data_one_hot = []
    for review in review_data:
        review_one_hot = [0] * vocabulary_size  # zero per word in vocabulary
        for word_index in review:
            review_one_hot[word_index] = 1
        review_data_one_hot.append(review_one_hot)
    return review_data_one_hot

def one_hot_label_vectors(review_labels, number_of_classes):
    """
    Returns a vector for each label that has
    a one for the rating (class label) that occurs in the label.
    Returns:
        review_labels_one_hots:
            one list per review,
            review list has zero or one for each possible rating (class label),
            one if it is the rating in the label, zero if it is not.
    """
    review_labels_one_hots = []
    for label in review_labels:
        label_one_hot = [0] * number_of_classes  # zero per possible label
        label_one_hot[label] = 1
        review_labels_one_hots.append(label_one_hot)
    return review_labels_one_hots

def count_vectors(review_data, vocabulary_size):
    """
    Returns a vector for each review that has
    a count for each word that occurs in the review.
    Repeated words get a count greater than one.
    Returns:
        review_data_counts:
            one list per review,
            review list has zero or a count for each word in the vocabulary,
            count if it occurs in the review, zero if it does not.
    """
    review_data_counts = []
    for review in review_data:
        review_counts = [0] * vocabulary_size  # zero per word in vocabulary
        for word_index in review:
            review_counts[word_index] += 1
        review_data_counts.append(review_counts)
    return review_data_counts

def word_idfs(review_data, vocabulary_size):
    """
    Compute inverse document frequency for the words in a document collection.
        idf = log(N / n_i)
            N = number of documents (reviews)
            n_i = number of reviews in which word i occurs
    Returns:
        word_idfs - idf for each word in the document, by word index
    """
    word_doc_counts = [0] * vocabulary_size # vocabulary size
    for review in review_data:
        for word_index in set(review):
            word_doc_counts[word_index] += 1
    N = float(len(review_data))    # numnber of documents, e.g., reviews
    word_idfs = [ math.log(N / c) if c > 0 else 0 for c in word_doc_counts ]
    return word_idfs

def tdIdf_vectors(review_data, vocabulary_size):
    """
    Returns a vector for each review that has
    a td-idf weight for each word that occurs in the review.
        idf:    inverse document frequency = log(N / n_i)
                    N = number of reviews
                    n_i = number of reviews in which word i occurs
        tf:     term frequency = count word i in a review
    Repeated words get a multiple of the idf for the word.
    Returns:
        review_data_tdIdfs:
            one list per review,
            review list has zero or a td-idf for each word in the vocabulary,
            td-idf if it occurs in the review, zero if it does not.
    """
    idfs = word_idfs(review_data, vocabulary_size)
    review_data_tdIdfs = []
    for review in review_data:
        review_tdIdfs = [0] * vocabulary_size  # zero per word in vocabulary
        for word_index in review:
            review_tdIdfs[word_index] += idfs[word_index]
        review_data_tdIdfs.append(review_tdIdfs)
    return review_data_tdIdfs

def load_embeddings(filePath, vocabulary, DEBUG=False):
    """
    Load embeddings for words in the vocabulary.
    Source: https://code.google.com/archive/p/word2vec/
    Format:
        <count> <dims>\n    - number of words, space, dimensions, newline
        <word> [<dims>f]    - word, space, dimensions x 4-byte floats
        ...
        <word> [<dims>f]    - word, space, dimensions x 4-byte floats
        NOTES: <count>, <dims>, and <word> are ascii strings
    Returns:
        embeddings - { word : tuple of floats, ... }
    """
    EXPECTED_HEADER = b'3000000 300\n'
    COUNT_WORDS = 3000000
    VECTOR_DIMENSIONS = 300
    F = "%df" % VECTOR_DIMENSIONS
    SEEK_CURR = 1
    embeddings = {}
    with open(filePath, 'rb') as f:
        header = f.read(len(EXPECTED_HEADER))   # skip over header
        if header != EXPECTED_HEADER:
            raise InvalidFileException("Invalid header: ", header)
        offset = len(header)
        for i in range(COUNT_WORDS):
            wordBytes = b''
            while(True):
                aByte = f.read(1)
                offset += 1
                if aByte != b' ':
                    wordBytes += aByte
                else:
                    try:
                        word = wordBytes.decode()
                        if word in vocabulary:
                            if DEBUG: print("%d : %s" % (offset, word))
                            vectors = unpack(F, f.read(4 * VECTOR_DIMENSIONS))
                            embeddings[word] = vectors    # save only vocabulary words
                        else:
                            f.seek(4 * VECTOR_DIMENSIONS, SEEK_CURR)
                        offset += 4 * VECTOR_DIMENSIONS
                    except UnicodeDecodeError as err:
                        print("ERROR decoding", wordBytes, \
                            "at offset ", offset)
                        print(err)
                    break
        if DEBUG: print("Ending offset: %d: " % offset)
    return embeddings

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

def split_data(review_data, review_labels):
    """
    Spit the data and labels into a training set, and a validation set.
    Shuffle the reviews, then select the first 9/10 for training and
    the remaining reviews for validation.
    Returns:
        shuffled_indices - list of original indices, e.g.,
                            review_data[shuffled_indices[0]] == train_data[0]
        train_data -    list of review sentences for training
        train_labels -  list of review ratings for training
        val_data -      list of review sentences for validation
        val_labels -    list of review ratings for validations
    """
    # Shuffle the data and labels
    shuffled_indices, shuffled_data, shuffled_labels = \
        shuffle_data(review_data, review_labels)

    # Select 9/10 of the data/labels for training, 1/10 for validation
    train_count = 9 * (len(review_data) // 10)
    train_data = shuffled_data[:train_count]
    train_labels = shuffled_labels[:train_count]
    val_data = shuffled_data[train_count:]
    val_labels = shuffled_labels[train_count:]

    return shuffled_indices, train_data, train_labels, val_data, val_labels

def split_training_data_for_cross_validation(review_data, review_labels):
    """
    Spit the data and labels into 10 equal size sets, after shuffling.
    Return:
        shuffled_indices - list of original indices, e.g.,
                        review_data[shuffled_indices[0]] == data[0][0][0]
        data = [
                    ( training_data, validation_data )      # set 1
                    ( training_data, validation_data )      # set 2
                    ...
                    ( training_data, validation_data )      # set 10
                ]
        where each set has a different 9/10 training and 1/10 validation data.
    """
    # Shuffle the data and labels
    shuffled_indices, shuffled_data, shuffled_labels = \
        shuffle_data(review_data, review_labels)

    # Subdivide the data and labels into 10 ~ equal size sets each
    set_count = len(shuffled_data) // 10
    xval_sets = []
    for iSet in range(9):
        xD = iSet * set_count
        xval_sets.append(( \
            shuffled_data[xD : xD + set_count], \
            shuffled_labels[xD : xD + set_count] ))
    xval_sets.append(( \
        shuffled_data[9 * set_count : len(shuffled_data)], \
        shuffled_labels[9 * set_count : len(shuffled_data)] ))

    return shuffled_indices, xval_sets

def assemble_cross_validation_data(xval_sets, index_val):
    """
    Assemble training data and labels, and validation data and labels,
    for cross-validation trainingself.
    Inputs:
        data-   sets of data and labels, one to be used for validation.
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

# ------------------------------------------------------------------------
# Visualize results ---
# ------------------------------------------------------------------------

def plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
        val_acc_min, val_acc_mean, val_acc_max, \
        input_type, h1_units, h1_f, h2_f, epochs, \
        plotName='tests/p3_tf_MLP_test'):
    plt.figure(1)
    plt.suptitle("Keras MLP: %s:Lin, %d:%s, 10:%s, 5:Softmax; epochs=%d" % \
        (input_type, h1_units, h1_f, h2_f, epochs))
    plt.title("validation accuracy: %7.4f %7.4f %7.4f" % \
        (val_acc_min, val_acc_mean, val_acc_max))
    plt.plot(np_train_loss, 'r--')
    plt.plot(np_train_acc, 'r')
    plt.plot(np_val_loss, 'b--')
    plt.plot(np_val_acc, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss / Avg Acc')
    plt.legend(['Training Loss', 'Training Accuracy', \
        'Validation Loss', 'Validation Accuracy'], loc='upper left')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(plotName + timestamp + '.png')

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    filePath = "data/a2_p3_train_data.txt"
    text = get_text_from_file(filePath)
    print("Read %d bytes from '%s' as text" % (len(text), filePath))
    print("Text begins : '%s'" % text[:30])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    words, review_words, review_data, review_labels, \
        fd_words, vocabulary, dictionary, reverse_dictionary = \
        get_words_and_ratings(text)
    vocabulary_size = len(vocabulary)
    print("Vocabulary size: %d" % vocabulary_size)
    print("Number of reviews: %d" % len(review_words))
    print("Number of words in text: %d" % len(words))
    print("Number of unique words: %d" % len(set(words)))
    print("Check len(review_words) == len(review_data) == len(review_labels) : %s" \
        % ((len(review_words) == len(review_data)) and \
           (len(review_words) == len(review_labels))))
    print("Check len(vocabulary) == len(dictionary) == len(reverse_dictionary) : %s" \
        % ((len(vocabulary) == len(dictionary)) and \
           (len(vocabulary) == len(reverse_dictionary))))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    one_hots = one_hot_vectors(review_data, vocabulary_size)
    print("Count one hot vectors: %d" % len(one_hots))
    print(" one hot indices 1..10 ".center(50, '-'))
    for i in range(10):
        indices = [ iX for iX, c in enumerate(one_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" one hot indices 101..110 ".center(50, '-'))
    for i in range(100, 110):
        indices = [ iX for iX, c in enumerate(one_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" one hot indices 1001..1010 ".center(50, '-'))
    for i in range(1000, 1010):
        indices = [ iX for iX, c in enumerate(one_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" one hot indices 9475..9483 ".center(50, '-'))
    for i in range(9474, 9484):
        indices = [ iX for iX, c in enumerate(one_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    number_of_classes = 5
    one_hot_labels = one_hot_label_vectors(review_labels, number_of_classes)
    print("Count one hot label vectors: %d" % len(one_hot_labels))
    print(" one_hot_labels 1..10 ".center(50, '-'))
    for i in range(10):
        indices = [ iX for iX, c in enumerate(one_hot_labels[i]) if c > 0 ]
        print("%4d: %s %s" % (i+1, indices, one_hot_labels[i]) )
    print(" one_hot_labels 101..110 ".center(50, '-'))
    for i in range(100, 110):
        indices = [ iX for iX, c in enumerate(one_hot_labels[i]) if c > 0 ]
        print("%4d: %s %s" % (i+1, indices, one_hot_labels[i]) )
    print(" one_hot_labels 1001..1010 ".center(50, '-'))
    for i in range(1000, 1010):
        indices = [ iX for iX, c in enumerate(one_hot_labels[i]) if c > 0 ]
        print("%4d: %s %s" % (i+1, indices, one_hot_labels[i]) )
    print(" one_hot_labels 9475..9483 ".center(50, '-'))
    for i in range(9474, 9484):
        indices = [ iX for iX, c in enumerate(one_hot_labels[i]) if c > 0 ]
        print("%4d: %s %s" % (i+1, indices, one_hot_labels[i]) )

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    count_hots = count_vectors(review_data, vocabulary_size)
    print("Count count vectors: %d" % len(count_hots))
    print(" hot index : count 1..10 ".center(50, '-'))
    for i in range(10):
        indices = [ (iX, c) for iX, c in enumerate(count_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" hot index : count 101..110 ".center(50, '-'))
    for i in range(100, 110):
        indices = [ (iX, c) for iX, c in enumerate(count_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" hot index : count 1001..1010 ".center(50, '-'))
    for i in range(1000, 1010):
        indices = [ (iX, c) for iX, c in enumerate(count_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" hot index : count 9475..9483 ".center(50, '-'))
    for i in range(9474, 9484):
        indices = [ (iX, c) for iX, c in enumerate(count_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    idfs = word_idfs(review_data, vocabulary_size)
    print("Count idfs: %d" % len(idfs))
    print(" idfs 1..10 ".center(50, '-'))
    for i in range(10):
        print("%-14s %f" % (list(dictionary.keys())[i], idfs[i]))
    print(" idfs 101..110 ".center(50, '-'))
    for i in range(100, 110):
        print("%-14s %f" % (list(dictionary.keys())[i], idfs[i]))
    print(" idfs 1001..1010 ".center(50, '-'))
    for i in range(1000, 1010):
        print("%-14s %f" % (list(dictionary.keys())[i], idfs[i]))
    print(" idfs 9991..10001 ".center(50, '-'))
    for i in range(9992, 10001):
        print("%-14s %f" % (list(dictionary.keys())[i], idfs[i]))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    tdIdf_hots = tdIdf_vectors(review_data, vocabulary_size)
    print("Count tdIdf vectors: %d" % len(tdIdf_hots))
    print(" hot index : tdIdf 1..10 ".center(50, '-'))
    for i in range(10):
        indices = [ (iX, c) for iX, c in enumerate(tdIdf_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" hot index : tdIdf 101..110 ".center(50, '-'))
    for i in range(100, 110):
        indices = [ (iX, c) for iX, c in enumerate(tdIdf_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" hot index : tdIdf 1001..1010 ".center(50, '-'))
    for i in range(1000, 1010):
        indices = [ (iX, c) for iX, c in enumerate(tdIdf_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" hot index : tdIdf 9475..9483 ".center(50, '-'))
    for i in range(9474, 9484):
        indices = [ (iX, c) for iX, c in enumerate(tdIdf_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    shuffled_indices, shuffled_data, shuffled_labels = \
        shuffle_data(count_hots, review_labels)
    print("Shuffle indices: %s ..., length=%d" % (shuffled_indices[:10], len(shuffled_indices)))
    print((" Shuffled data : %d " % len(shuffled_data)).center(50, '-'))
    for i in range(0, 5):
        print(shuffled_data[i][:10], '...' if len(shuffled_data[i]) > 10 else '')
    print((" Shuffled labels : %d " % len(shuffled_labels)).center(50, '-'))
    for i in range(0, 5):
        print(shuffled_labels[i])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    shuffled_indices, train_data, train_labels, val_data, val_labels = \
        split_data(count_hots, review_labels)
    print("Shuffle indices: %s ..., length=%d" % (shuffled_indices[:10], len(shuffled_indices)))
    print((" Training data : %d " % len(train_data)).center(50, '-'))
    for i in range(0, 5):
        print(train_data[i][:10], '...' if len(train_data[i]) > 10 else '')
    print((" Training labels : %d " % len(train_labels)).center(50, '-'))
    for i in range(0, 5):
        print(train_labels[i])
    print((" Validation data : %d " % len(val_data)).center(50, '-'))
    for i in range(0, 5):
        print(val_data[i][:10], '...' if len(val_data[i]) > 10 else '')
    print((" Validation labels : %d " % len(val_labels)).center(50, '-'))
    for i in range(0, 5):
        print(val_labels[i])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    shuffled_indices, data = \
        split_training_data_for_cross_validation(review_data, review_labels)
    print("Shuffle indices: %s ..., length=%d" % (shuffled_indices[:10], len(shuffled_indices)))
    print((" Set data and labels : %s " % str([len(s[0]) for s in data])).center(80, '-'))
    for i in range(len(data)):
        set_data, set_labels = data[i]
        print(set_data[i][:10], '...' if len(set_data[i]) > 10 else '')
        print(set_labels[i])
        print(20*'-')

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    a = [ \
      ([1, 2, 3],    ['a', 'b', 'c']), \
      ([4, 5, 6],    ['d', 'e', 'f']), \
      ([7, 8, 9],    ['g', 'h', 'i']), \
      ([10, 11, 12], ['j', 'k', 'l'])  \
    ]
    print("a: %s" % str(a))
    for i in range(len((a))):
        t, v = assemble_cross_validation_data(a, i)
        print("%d: %s %s" % (i, str(t), str(v)))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    train_loss = [[ math.exp(-(l/20.0))+(0.002*(random.random()-0.5)) \
        for l in range(10) ] for i in range(20) ]
    train_accuracy = [[ 0.8 * (1 - math.exp(-l/20.0)+(0.002*(random.random()-0.5))) \
        for l in range(10) ] for i in range(20) ]
    val_loss = [[ math.exp(-l/20.0)+(0.002*(random.random()-0.5)) \
                + math.exp(0.5 * (l/20.0))+(0.002*(random.random()-0.5)) \
        for l in range(10) ] for i in range(20) ]
    val_accuracy = [[ 0.31 * (1 - math.exp(-l/20.0)+(0.001*(random.random()-0.5))) \
        for l in range(10) ] for i in range(20) ]

    np_train_loss = np.array(train_loss).mean(axis=0)
    np_train_acc = np.array(train_accuracy).mean(axis=0)
    np_val_loss = np.array(val_loss).mean(axis=0)
    np_val_acc = np.array(val_accuracy).mean(axis=0)

    np_val_acc_finals = np.array(val_accuracy)[:,-1] # last value from each trial
    val_acc_min  = np_val_acc_finals.min()
    val_acc_mean = np_val_acc_finals.mean()
    val_acc_max  = np_val_acc_finals.max()

    plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
        val_acc_min, val_acc_mean, val_acc_max, \
        input_type='td-idf-hot', h1_units=60, h1_f='relu', h2_f='relu', epochs=20, \
        plotName='tests/p3_utils_test_')

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Reading embeddings ...")
    FILE = "GoogleNews-vectors-negative300.bin"
    filePath = os.path.join("data", FILE)
    embeddings = load_embeddings(filePath, fd_words)
    print("Length embeddings: %d" % len(embeddings))
    for word, vector in list(embeddings.items())[:10]:
        print("%10s : %s" % (word, vector[:4]))
    for word, vector in list(embeddings.items())[-10:]:
        print("%10s : %s" % (word, vector[:4]))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
