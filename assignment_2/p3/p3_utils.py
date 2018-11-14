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
import nltk

from datetime import datetime
from struct import unpack
from gensim.models import KeyedVectors

PATH_TRAIN = "data/a2_p3_train_data.txt"
PATH_TEST = "data/a2_p3_test_data.txt"
WORD_VECTORS_FILE = "data/GoogleNews-vectors-negative300.bin"

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

class InvalidArgumentException(ValueError):
    """Exception raised for errors in the function arguments.

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

def compile_vocabulary(words, vocabulary_size):
    # compile vocabulary, with a UNK entry at index 0 for the words left out
    fd_words = nltk.FreqDist(words)
    frequents = fd_words.most_common(VOCABULARY_SIZE)
    count_UNK = len(fd_words) - len(frequents)
    frequents = [( 'UNK', count_UNK )] + frequents
    # compile the dictionary and reverse dictionary
    vocabulary, word_counts = zip(*frequents)
    dictionary = {}
    reverse_dictionary = {}
    for iWord, word in enumerate(vocabulary):
        dictionary[word] = iWord
        reverse_dictionary[iWord] = word
    return fd_words, vocabulary, dictionary, reverse_dictionary

def get_review_data(review_words, dictionary, VECTOR_LEN=None):
    # compile review data as the indices of the words into the vocabulary
    review_data = []
    if VECTOR_LEN == None:
        max_len = max(len(review) for review in review_words)
        len_review_vector = ((max_len + 9) // 10) * 10  # multiple of 10
    else:
        len_review_vector = VECTOR_LEN
    for review in review_words:
        words_in_review = [0] * len_review_vector
        for i, word in enumerate(review):
            words_in_review[i] = dictionary.get(word, 0)
        review_data.append(words_in_review)
    return review_data

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
    review_words = [ review.split() for review, rating in reviews ]
    review_labels = [int(rating) for review, rating in reviews ]
    words = [ word for words_in_sent in review_words for word in words_in_sent ]
    # compile the vocabulary and the review data as word indices
    fd_words, vocabulary, dictionary, reverse_dictionary = \
        compile_vocabulary(words, vocabulary_size=VOCABULARY_SIZE)
    review_data = get_review_data(review_words, dictionary)
    # return the results
    return words, review_words, review_data, review_labels, \
        fd_words, vocabulary, dictionary, reverse_dictionary

def load_test_set(text, dictionary, VECTOR_LEN):
    """
    Get the words and vocabulary from the reviews in a test sest.
    The test set has one review sentence per line and does not have any ratings.
    """
    # separate the review sentences
    re_test_reviews = re.compile(r'^(.+)$', re.MULTILINE)
    reviews = re_test_reviews.findall(text)
    # tokenize the review sentences and compile a list of words
    review_words = [ review.split() for review in reviews ]
    words = [ word for words_in_sent in review_words for word in words_in_sent ]
    review_data = get_review_data(review_words, dictionary, VECTOR_LEN)
    # return the test set reviews
    return reviews, words, review_words, review_data

def write_test_set_with_ratings(outDir, outFilename, reviews, labels):
    assert(len(reviews) == len(labels))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outPath = os.path.join(outDir, outFilename + "_" + timestamp + ".txt")
    with open(outPath, 'w') as f:
        for iReview, review in enumerate(reviews):
            rating = labels[iReview]
            f.write("%s|%d\n" % (review, rating))

# ------------------------------------------------------------------------
# Word vector embeddings ---
# ------------------------------------------------------------------------

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
    Compile a dictionary of vector index -> word, for the words in the reviews.
    Compile a reverse dictionary: vector index -> word
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

# ------------------------------------------------------------------------
# POS tagging ---
# ------------------------------------------------------------------------

# tagset from nltk.help.upenn_tagset(), plus:
#   '#', which occurs in the training data
#   'UNK', in case the tagger returns something not in this tag set
TAGSET = {
        'UNK', '$', "''", '(', ')', ',', '--', '.', ':', 'CC', 'CD', \
        'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', \
        'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', \
        'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', \
        'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', \
        '``', '#'
    }
tags_dictionary = { list(TAGSET)[i] : i for i in range(len(TAGSET)) }
reverse_tags_dictionary = { i: list(TAGSET)[i] for i in range(len(TAGSET)) }

def get_pos_tags_reviews(text, HAS_RATINGS=True, VECTOR_LEN=None):
    """
    Get a vector of POS tags for each review sentence.
    Return -
        reviews_tag_vectors - [ [ tag index, tag index, ... ] ... ]
    """
    PAD = -1
    if (HAS_RATINGS):
        re_reviews = re.compile(r'^(.+)\|(.+)$', re.MULTILINE)
    else:
        re_reviews = re.compile(r'^(.+)$', re.MULTILINE)
    reviews_w_caps = re_reviews.findall(text)
    if (HAS_RATINGS):
        review_words_w_caps = [ review.split() for review, rating in reviews_w_caps ]
    else:
        review_words_w_caps = [ review.split() for review in reviews_w_caps ]
    review_pos_tags_w_caps = [ nltk.pos_tag(s) for s in review_words_w_caps ]
    _reviews_tag_vectors = [ [ tags_dictionary.get(tag, 0) for word, tag in s ] \
        for s in review_pos_tags_w_caps ]
    if VECTOR_LEN == None:
        max_len = max(len(s) for s in _reviews_tag_vectors)
        vector_len = ((max_len + 9) // 10) * 10
    else:
        vector_len = VECTOR_LEN
    reviews_tag_vectors = [ s + [PAD] * (vector_len - len(s))\
        for s in _reviews_tag_vectors ]
    return np.array(reviews_tag_vectors)

# ------------------------------------------------------------------------
# Sentiment Lexicons ---
# ------------------------------------------------------------------------

PATH_PITT = "data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"

rx_pitt_pos = r'word1=(.+) pos1=.+ stemmed1=.* .*polarity=positive.*$'
rx_pitt_neg = r'word1=(.+) pos1=.+ stemmed1=.* .*polarity=negative.*$'
re_positive = re.compile(rx_pitt_pos, re.MULTILINE)
re_negative = re.compile(rx_pitt_neg, re.MULTILINE)

def load_lexicon_u_pitt(filePath):
    """
    Load the subjectivity lexicon from U Pittsburgh.
    """
    with open(filePath) as f:
        pitt_text = f.read()
    pitt_pos = re_positive.findall(pitt_text)
    pitt_neg = re_negative.findall(pitt_text)
    return set(pitt_pos), set(pitt_neg)

PATH_UIC_POS = "data/cs_uic_edu-liu_bing/positive-words.txt"
PATH_UIC_NEG = "data/cs_uic_edu-liu_bing/negative-words.txt"

rx_uic = r'^([^;].+)$'
re_uic = re.compile(rx_uic, re.MULTILINE)

def load_words_uic(filePath):
    """
    Load the opinion lexicon from Ming Liu, U Illinois at Chicago,
    either the positive or the negative words.
    """
    with open(filePath) as f:
        uic_text = f.read()
    uic_words = re_uic.findall(uic_text)
    return set(uic_words)

def load_lexicon_uic(posPath, negPath):
    """
    Load the opinion lexicon from Ming Liu, U Illinois at Chicago.
    """
    uic_pos = load_words_uic(posPath)
    uic_neg = load_words_uic(negPath)
    return set(uic_pos), set(uic_neg)

def sentiment_words(pitt_pos, pitt_neg, uic_pos, uic_neg):
    """
    Merge the U Pittsburg and U Illinois sentiment words,
    removing any words classified positive in one and negative in the other.
    """
    positive_words = [ w for w in pitt_pos if w not in uic_neg ]
    positive_words += [ w for w in uic_pos if w not in pitt_neg ]
    negative_words = [ w for w in pitt_neg if w not in uic_pos ]
    negative_words += [ w for w in uic_neg if w not in pitt_pos ]
    return set(positive_words), set(negative_words)

def sentiment_vectors(review_words, positive_words, negative_words):
    """
    Get a sentiment vector for each sentence, a vector containing:
        -0.5 if word is negative
         0.0 if word is neither negative nor positive, or there is no word
         0.5 if word is positive
    """
    review_sentiment_vectors = []
    for review in review_words:
        rsvs = [ 0.0 ] * 60
        for i, word in enumerate(review):
            if word in positive_words:
                rsvs[i] = 0.5
            elif word in negative_words:
                rsvs[i] = -0.5
        review_sentiment_vectors.append(rsvs)
    return review_sentiment_vectors

def load_sentiment_vectors(review_words):
    """
    Load words from U Pitt and U Illinois lexicons,
    then create sentiment word vectors.
    """
    pitt_pos, pitt_neg = load_lexicon_u_pitt(PATH_PITT)
    uic_pos, uic_neg = load_lexicon_uic(PATH_UIC_POS, PATH_UIC_NEG)
    positive_words, negative_words = \
        sentiment_words(pitt_pos, pitt_neg, uic_pos, uic_neg)
    review_sentiment_vectors = \
        sentiment_vectors(review_words, positive_words, negative_words)
    return np.array(review_sentiment_vectors)

# ------------------------------------------------------------------------
# Hot Vectors ---
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

def split_data(review_data, review_labels, number_of_trials):
    """
    Spit the data and labels into a training set, and a validation set.
    Shuffle the reviews, then select (N-1)/N for N training trials and
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

    # Select a portion of the data/labels for training, the remainder for validation
    N = number_of_trials
    train_count = (N-1) * (len(review_data) // N)
    train_data = shuffled_data[:train_count]
    train_labels = shuffled_labels[:train_count]
    val_data = shuffled_data[train_count:]
    val_labels = shuffled_labels[train_count:]

    return shuffled_indices, train_data, train_labels, val_data, val_labels

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

def plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
        heading, subheading, plotName='tests/p3_tf_MLP_test_plot_'):

    figure, axis_1 = plt.subplots()

    plt.suptitle(heading, size=12)
    plt.title(subheading, size=10)

    # Plot loss for both training and validation
    axis_1.plot(np_train_loss, 'r--')
    axis_1.plot(np_val_loss, 'b--')
    axis_1.set_xlabel('Epoch')
    axis_1.set_ylabel('Avg Loss')
    axis_1.legend(['Training Loss', 'Validation Loss'], loc='upper left')

    # Plot accuracy for both training and validation
    axis_2 = axis_1.twinx()
    axis_2.plot(np_train_acc, 'r')
    axis_2.plot(np_val_acc, 'b')
    axis_2.set_ylabel('Avg Acc')
    axis_2.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')

    figure.subplots_adjust(top=0.9, right=0.9)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(plotName + timestamp + '.png')

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

def test_plot():
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

    input_type = 'aws'
    num_h1_units = 60
    h1_activation = h2_activation = 'relu'
    num_epochs_per_trial = 20

    heading = "Keras MLP: %s:Lin, %d:%s, 10:%s, 5:Softmax; epochs=%d" % \
        (input_type, num_h1_units, h1_activation, h2_activation, num_epochs_per_trial)
    subheading = "validation accuracy: %7.4f %7.4f %7.4f" % \
        (val_acc_min, val_acc_mean, val_acc_max)

    plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
        heading, subheading, plotName='tests/p3_utils_test_')

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    text = get_text_from_file(PATH_TRAIN)
    print("Read %d bytes from '%s' as text" % (len(text), PATH_TRAIN))
    print("Text begins : '%s'" % text[:30])
    print("Text ends : '%s'" % text[-30:])

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
        split_data(count_hots, review_labels, 10)
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
        split_training_data_for_cross_validation(review_data, review_labels, 10)
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

    test_plot()

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Reading embeddings ...")
    embeddings = load_embeddings(WORD_VECTORS_FILE, fd_words)
    print("Length embeddings: %d" % len(embeddings))
    for word, vector in list(embeddings.items())[:10]:
        print("%20s : %s" % (word, vector[:4]))
    for word, vector in list(embeddings.items())[-10:]:
        print("%20s : %s" % (word, vector[:4]))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Reading embeddings with gensim ...")
    vectors = load_embeddings_gensim()
    print("Length embeddings: %d" % len(vectors.vocab))
    for word, vector in list(vectors.vocab.items())[:10]:
        print("%20s : %s" % (word, vector.index))
    for word, vector in list(vectors.vocab.items())[-10:]:
        print("%20s : %s" % (word, vector.index))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Generating embeddings vocabulary")
    wv_vocabulary, wv_dictionary, wv_reverse_dictionary = \
        get_embeddings_vocabulary(vectors, fd_words)
    print("Count words in embeddings vocabulary: %d" % len(wv_vocabulary))
    print("--- %s .. %s" % (wv_vocabulary[:4], wv_vocabulary[-4:]))
    print("Length word->index dictionary: %d" % len(wv_dictionary))
    print("Length index->word dictionary: %d" % len(wv_reverse_dictionary))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Generating review sentence word vectors")
    wv_review_data, wv_review_vectors, wv_review_sentence_average_vectors \
        = get_embeddings(vectors, review_words)
    print("Shape of review data: %s" % \
        str(np.array(wv_review_data).shape))
    print("  %s" % str([(wv_review_data[i][:3], wv_review_data[i][-3:]) for i in range(3)]))
    print("Shape of review vectors: %s" % \
        str(np.array(wv_review_vectors).shape))
    for i in range(3):
        print("  %s .. %s" % (wv_review_vectors[i][1][:3], wv_review_vectors[i][-2][:3]))
    print("Shape of review avg sentence vectors: %s" % \
        str(np.array(wv_review_sentence_average_vectors).shape))
    for i in range(3):
        print("  %s" % wv_review_sentence_average_vectors[i][:3])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Reading sentiment lexicons ...")
    review_sentiment_vectors = load_sentiment_vectors(review_words)
    print("Length sentiment vectors: %d" % len(review_sentiment_vectors))
    rsv_count_nonzero = len([len(rsv) for rsv in review_sentiment_vectors \
        if len([x for x in rsv if x != 0.0]) != 0 ])
    print("Count non-zero sentiment vectors: %d" % rsv_count_nonzero)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Reading test data ...")
    text = get_text_from_file(PATH_TEST)
    print("Read %d bytes from '%s' as text" % (len(text), PATH_TEST))
    print("Text begins : '%s'" % text[:30])
    print("Text ends : '%s'" % text[-30:])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Loading test reviews and data ...")
    test_reviews, test_words, test_review_words, test_review_data = \
        load_test_set(text, dictionary)
    print("Number of reviews: %d" % len(test_reviews))
    assert(len(test_review_words) == len(test_reviews))
    assert(len(test_review_data) == len(test_reviews))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Generating review sentence word vectors")
    wv_test_review_data, wv_test_review_vectors, wv_test_review_sentence_average_vectors \
        = get_embeddings(vectors, test_review_words)
    print("Shape of review data: %s" % \
        str(np.array(wv_test_review_data).shape))
    print("  %s" % str([(wv_test_review_data[i][:3], wv_test_review_data[i][-3:]) for i in range(3)]))
    print("Shape of review vectors: %s" % \
        str(np.array(wv_test_review_vectors).shape))
    for i in range(3):
        print("  %s .. %s" % (wv_test_review_vectors[i][1][:3], wv_test_review_vectors[i][-2][:3]))
    print("Shape of review avg sentence vectors: %s" % \
        str(np.array(wv_review_sentence_average_vectors).shape))
    for i in range(3):
        print("  %s" % wv_review_sentence_average_vectors[i][:3])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    print("Writing mock results ...")
    test_labels_random = np.random.randint(5, size=len(test_reviews))
    outDir = "tests"
    outFilename = "p3_utils_mock_test_labels"
    write_test_set_with_ratings(outDir, outFilename, test_reviews, test_labels_random)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
