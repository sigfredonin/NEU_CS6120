"""
p3_utils.py
Utilities to support the solution of Assigment 2 Problem 3, Sentiment Analysis

Sig Nin
October 26, 2018
"""

import re
import math
from nltk import FreqDist

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

def assemble_data(review_data, review_labels, \
        training_count, sets_count, batch_size):
    """
    Spit the data and labels into training/eval sets, and test set.
    Both the data and the labels are to be split.
        Training/eval:  training_count
        Test:           len(data) - training count
    The training data is to be further subdivided into sets for n-way
    cross-validation.
        Set:    set_size = training_count / sets_count
                training_count % sets_count == 0
    Each set will be further subdivided into batches.
        Batch:  batch_count = set_size / batch_size
                set_size % batch_size == 0
    For problem 3,
        len(review_data) == len(review_labels) == 9484 reviews
        sets_count == 10
        if training__count == 8000, then
            len(test_data) == 9484 - 8000 == 1484 reviews
            reviews_per_set == 8000 / 10 == 800 reviews
        if batch_size == 80 reviews per batch, then
            batch_count_per_set == 800 / 80 == 10 batches of 80 reviews
    Return:
        training_data = [
                          [
                            [ [ batch_data ], [ batch_labels] ]   # batch 1
                            [ [ batch_data ], [ batch_labels] ]   # batch 2
                            ... ] # set 1
                          [
                            [ [ batch_data ], [ batch_labels] ]   # batch 1
                            [ [ batch_data ], [ batch_labels] ]   # batch 2
                            ... ] # set 2
                          ...
                        ]
        test_data = [ [ data ], [ labels ] ]    # just one batch
    """

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
    print(" count_hots 1..10 ".center(50, '-'))
    for i in range(10):
        indices = [ iX for iX, c in enumerate(one_hots[i]) if c > 0 ]
        print("%4d: %s" % (i+1, indices))
    print(" count_hots 101..110 ".center(50, '-'))
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
