"""
Sarcastic / Non-sarcastic Tweets Common N-Grams

Sig Nin
06 Dec 2018
"""

import nltk

################################### N-Grams ####################################

def get_tweet_words(tweets):
    tweet_words = [ nltk.word_tokenize(tweet) for tweet in tweets ]
    return tweet_words

def get_tweet_words_lowercase(tweets):
    tweet_words = [ word.lower()                    for word in
                        nltk.word_tokenize(tweet)   for tweet in tweets ]
    return tweet_words

def get_tweet_words_in_sents(tweets):
    tweet_sentences = [ [nltk.word_tokenize(sentence) for sentence in
                           nltk.sent_tokenize(tweet)] for tweet in tweets ]
    return tweet_sentences

def get_tweet_words_in_sents_lowercase(tweets):
    tweet_sentences = [ [ [word.lower()                    for word in
                             nltk.word_tokenize(sentence)] for sentence in
                             nltk.sent_tokenize(tweet)]    for tweet in tweets ]
    return tweet_sentences

def get_ngrams(n, tokens):
    return [tuple(tokens[i:i+n]) for i in range (len(tokens)-(n-1))]

def find_ngrams_in_tweets(n, tokenized_tweets):
    ngrams = []
    for tokens in tokenized_tweets:
        tweet_ngrams = get_ngrams(n, tokens)
        ngrams.append(tweet_ngrams)
    return ngrams

#
# Tweet labels
SARCASTIC = 1
NON_SARCASTIC = 0

class sarcastic_set_factory:
    """
    Create sarcastic and non-sarcastic common words and n-grams sets.
    """

    def __init__(self, num_most_common_ngrams=20000):
        self.NUM_MOST_COMMON_NGRAMS = num_most_common_ngrams

    def get_ngram_frequencies(self, ngrams_in_tweets):
        ngrams = [ ngram for tweet in ngrams_in_tweets for ngram in tweet]
        fd_ngrams = nltk.FreqDist(ngrams)
        return fd_ngrams

    def remove_common_ngrams(self, ngrams_in_sarcastic_tweets, ngrams_in_non_sarcastic_tweets):
        fd_sarcastic = self.get_ngram_frequencies(ngrams_in_sarcastic_tweets)
        fd_non_sarcastic = self.get_ngram_frequencies(ngrams_in_non_sarcastic_tweets)
        fd_just_sarcastic = fd_sarcastic - fd_non_sarcastic
        fd_just_non_sarcastic = fd_non_sarcastic - fd_sarcastic
        return fd_just_sarcastic, fd_just_non_sarcastic

    def get_most_common_ngrams_set(self, fd_ngrams):
        freq = fd_ngrams.most_common(self.NUM_MOST_COMMON_NGRAMS)
        freq_ngrams, freq_counts = zip(*freq)
        return set(freq_ngrams)

    def get_freq_ngram_sets(self, ngrams_in_sarcastic_tweets, ngrams_in_non_sarcastic_tweets):
        fd_just_sarcastic_ngrams, fd_just_non_sarcastic_ngrams = \
            self.remove_common_ngrams(ngrams_in_sarcastic_tweets, ngrams_in_non_sarcastic_tweets)
        sarcastic_set = self.get_most_common_ngrams_set(fd_just_sarcastic_ngrams)
        non_sarcastic_set = self.get_most_common_ngrams_set(fd_just_non_sarcastic_ngrams)
        return sarcastic_set, non_sarcastic_set

    def get_freq_unigram_and_bigram_sets(self, training_sarcastic_tweets, training_non_sarcastic_tweets):
        # unigrams
        unigrams_in_sarcastic_tweets = [ nltk.word_tokenize(tweet) for tweet in training_sarcastic_tweets ]
        unigrams_in_non_sarcastic_tweets = [ nltk.word_tokenize(tweet) for tweet in training_non_sarcastic_tweets ]
        sarcastic_unigrams_set, non_sarcastic_unigrams_set = \
            self.get_freq_ngram_sets(unigrams_in_sarcastic_tweets, unigrams_in_non_sarcastic_tweets)
        # bigrams
        bigrams_in_sarcastic_tweets = find_ngrams_in_tweets(2, unigrams_in_sarcastic_tweets)
        bigrams_in_non_sarcastic_tweets = find_ngrams_in_tweets(2, unigrams_in_non_sarcastic_tweets)
        sarcastic_bigrams_set, non_sarcastic_bigrams_set = \
            self.get_freq_ngram_sets(bigrams_in_sarcastic_tweets, bigrams_in_non_sarcastic_tweets)
        # combined sets of most frequent unigrams and bigrams
        sarcastic_set = sarcastic_unigrams_set.union(sarcastic_bigrams_set)
        non_sarcastic_set = sarcastic_unigrams_set.union( non_sarcastic_bigrams_set)
        return sarcastic_set, non_sarcastic_set

    def separate_sarcastic_by_labels(self, tweets, labels):
        sarcastic_tweets = []
        non_sarcastic_tweets = []
        for i, tweet in enumerate(tweets):
            label = labels[i]
            if label == SARCASTIC:
                sarcastic_tweets.append(tweet)
            else:
                non_sarcastic_tweets.append(tweet)
        return sarcastic_tweets, non_sarcastic_tweets
