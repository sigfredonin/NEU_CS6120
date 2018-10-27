"""
p3_utils.py
Utilities to support the solution of Assigment 2 Problem 3, Sentiment Analysis

Sig Nin
October 26, 2018
"""

import re
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
        review_labels: ratings from each review
            [ r_1, r_2, ... ], where r_i is a decimal number in { 0 .. 4 }
        vocabulary: the VOCABULARY_SIZE most frequent wordS,
            in order of decreasing frequency
        dictionary: a mapping word -> index in the vocabulary
        reverse_dictionary: a mapping index -> word
        review_data: review words encoded as indices into the vocabulary
    """
    # collect words and ratings from the reviews
    re_reviews = re.compile(r'^(.+\.)\|(.+)$', re.MULTILINE)
    reviews = re_reviews.findall(text)
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
    for review in review_words:
        words_in_review = []
        for word in review:
            words_in_review.append(dictionary.get(word, 0))
        review_data.append(words_in_review)
    # return the results
    return words, review_words, review_data, review_labels, \
        vocabulary, dictionary, reverse_dictionary

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
        len(review_data) == len(review_labels) == 8886 reviews
        sets_count == 10
        if training__count == 6000, then
            len(test_data) == 8886 - 6000 == 2886 reviews
            reviews_per_set == 6000 / 10 == 600 reviews
        if batch_size == 60 reviews per batch, then
            batch_count_per_set == 600 / 60 == 10 batches of 60 reviews
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
