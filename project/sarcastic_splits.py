"""
Sarcastic / Non-sarcastic Tweets
N-gram Count Features
Cross-Validation Splits

Replaces n-gram count features with corresponding features
computed for just the tweets in a cross-validation split.

Sig Nin
06 Dec 2018
"""

import numpy as np

from sarcastic_ngrams import SARCASTIC
from sarcastic_ngrams import NON_SARCASTIC
from sarcastic_ngrams import get_tweet_words
from sarcastic_ngrams import find_ngrams_in_tweets
from sarcastic_ngrams import sarcastic_set_factory as SSF

# n-grams processing
NUM_MOST_COMMON_NGRAMS = 20000
ssf = SSF(NUM_MOST_COMMON_NGRAMS)

# replace original n-gram features with those computed for the split
def get_features(data, labels, tweets, sarcastic_freqs, non_sarcastic_freqs):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    count_sarcastic_freq_unigrams, count_sarcastic_freq_bigrams, \
        count_non_sarcastic_freq_unigrams, count_non_sarcastic_freq_bigrams = \
        ssf.get_ngram_counts(words_in_tweets, bigrams_in_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    _su, _sb, _uu, _ub, rc, pc, ss = zip(*list(data))
    _data = list(zip(\
        count_sarcastic_freq_unigrams, count_sarcastic_freq_bigrams, \
        count_non_sarcastic_freq_unigrams, count_non_sarcastic_freq_bigrams, \
        rc, pc, ss ))
    return np.array(_data), labels

# get common ngrams for this split and recompute features based on them
def get_features_for_split(train_data, train_labels, train_tweets, \
        val_data, val_labels, val_tweets):
    sarcastic_tweets, non_sarcastic_tweets = \
        ssf.separate_sarcastic_by_labels(list(train_tweets), list(train_labels))
    sarcastic_freqs, non_sarcastic_freqs = \
        ssf.get_freq_unigram_and_bigram_sets(\
            sarcastic_tweets, non_sarcastic_tweets)
    np_train_data, np_train_labels = \
        get_features(train_data, train_labels, train_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    np_val_data, np_val_labels = \
        get_features(val_data, val_labels, val_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    return np_train_data, np_train_labels, np_val_data, np_val_labels
