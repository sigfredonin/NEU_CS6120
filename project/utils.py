"""
Sarcastic / Non-sarcastic Tweets
Features for sarcastic tweets classifiers.

Load tweets and compute features:
- counts of most-common unigrams and bigrams in sarcastic and non-sarcastic
- average synset sentiment score
- percent capitals
- count repeated characters

Sig Nin
06 Dec 2018
"""

import sys
import re
import nltk
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet
from nltk.corpus.reader.wordnet import WordNetError

from datetime import datetime

from sarcastic_ngrams import SARCASTIC
from sarcastic_ngrams import NON_SARCASTIC
from sarcastic_ngrams import get_tweet_words
from sarcastic_ngrams import find_ngrams_in_tweets
from sarcastic_ngrams import sarcastic_set_factory as SSF

################################## Parameters ##################################

STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = { ',', '.', '?', '!', ';', ':', '..', '...' }

USE_FULL_TRAIN = True
TRAIN_SIZE = 20000       # when USE_FULL_TRAIN = False
TUNE = False             # Cross-validate if True, else train then predict on test

COUNT_SARCASTIC_TRAINING_TWEETS = 20000
COUNT_NON_SARCASTIC_TRAINING_TWEETS = 100000

# n-grams processing
NUM_MOST_COMMON_NGRAMS = 25000
ssf = SSF(NUM_MOST_COMMON_NGRAMS)

# Synsets processing
MIN_SENSES = 3
MAX_SENSES = 12
REMOVE_COMMON_NGRAMS = True
REMOVE_STOPWORDS = False
REMOVE_PUNCTUATION = False
SS_PUNCTUATION = PUNCTUATION - { '?', '!' }

################################## Load Data ###################################

# removes blank lines, replaces \n with space, removes duplicate spaces
def process_whitespace(token_str):
    no_new_line = re.sub(r'\n', " ", token_str)
    no_dup_spaces = re.sub(r'  +', " ", no_new_line)
    return no_dup_spaces

def load_data():
    posproc = np.load('posproc.npy')
    negproc = np.load('negproc.npy')

    sarcastic_tweets = []
    non_sarcastic_tweets = []

    for tweet in posproc:
        sarcastic_tweets.append((tweet.decode('utf-8'), SARCASTIC))

    for tweet in negproc:
        non_sarcastic_tweets.append((tweet.decode('utf-8'), NON_SARCASTIC))

    print('num sarcastic tweets: ' + str(len(sarcastic_tweets)))
    print('num non sarcastic tweets: ' + str(len(non_sarcastic_tweets)))

    return sarcastic_tweets, non_sarcastic_tweets

# Separate training and testing data
def get_data(sarcastic_tweets, non_sarcastic_tweets):
    training_sarcastic_tweets = sarcastic_tweets[0:COUNT_SARCASTIC_TRAINING_TWEETS]
    testing_sarcastic_tweets = sarcastic_tweets[COUNT_SARCASTIC_TRAINING_TWEETS:]

    training_non_sarcastic_tweets = non_sarcastic_tweets[0:COUNT_NON_SARCASTIC_TRAINING_TWEETS]
    testing_non_sarcastic_tweets = non_sarcastic_tweets[COUNT_NON_SARCASTIC_TRAINING_TWEETS:]

    labeled_train_tweets = training_sarcastic_tweets + training_non_sarcastic_tweets
    labeled_test_tweets = testing_sarcastic_tweets + testing_non_sarcastic_tweets

    train_tweets, train_labels = zip(*labeled_train_tweets)
    test_tweets, test_labels = zip(*labeled_test_tweets)

    return train_tweets, train_labels, test_tweets, test_labels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tweet Vectors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# find all words with frequency less than threshold
def find_freq(tokens, THRESHOLD):
    fdist = nltk.FreqDist(tokens)
    freq = [word for word in fdist if fdist[word] > THRESHOLD]
    print("frequent token count:", len(freq))
    return freq

# create a dictionary of all the words in the vocabulary to an index
def create_vocab_dict(freq_tokens):
    tokens = ['UNK'] + freq_tokens
    vocab_dict = {}
    for i, token in enumerate(tokens):
        vocab_dict[token] = i
    return vocab_dict

# creates a list of one hot vectors from tweets
def one_hot_vector_tweets(grams_in_tweets, vocab_dict):
    one_hot_vectors = []
    for tweet in grams_in_tweets:
        one_hot = [0] * len(vocab_dict)
        for token in tweet:
            one_hot[vocab_dict.get(token, 0)] = 1
        one_hot_vectors.append(one_hot)
    return one_hot_vectors

# creates a list of index vectors from tweets
LEN_INDEX_VECTORS = 128
def index_vector_tweets(grams_in_tweets, vocab_dict):
    index_vectors = []
    for tweet in grams_in_tweets:
        vector = [0] * LEN_INDEX_VECTORS
        for i, token in enumerate(tweet):
            if i < len(vector):
                vector[i] = vocab_dict.get(token, 0)
        index_vectors.append(vector)
    return index_vectors

######################## Unigram and Bigram Features #########################

# converts tokens into a list of word index vectors using an existing dictionary
def ngrams_to_indices(ngrams_in_tweets, vocab_dict):
    index_vectors = index_vector_tweets(ngrams_in_tweets, vocab_dict)
    return index_vectors

# Compute training vocabulary
def get_training_vocabulary(words_in_tweets, bigrams_in_tweets):
    words = [ word for tweet in words_in_tweets for word in tweet ]
    word_dict = create_vocab_dict(words)
    bigrams = [ bigram for tweet in bigrams_in_tweets for bigram in tweet ]
    bigram_dict = create_vocab_dict(bigrams)
    return word_dict, bigram_dict

############# Repeated Characters and Capitalized Words Features ###############

def _get_repeated_character_count_tweet(tweet):
    repeated_character_count = 0
    characters = ['null_1', 'null_2', 'null_3']
    repeated_characters = False
    for character in tweet:
        characters.pop(0)
        characters.append(character)
        if characters[0] == characters[1] and characters[1] == characters[2]:
            repeated_characters = True
            break
    if repeated_characters:
        repeated_character_count += 1
    return repeated_character_count

def get_repeated_character_count_tweets(tweets):
    repeated_character_counts = []
    for tweet in tweets:
        repeated_character_counts.append(_get_repeated_character_count_tweet(tweet))
    return repeated_character_counts

def get_percent_caps(tweet):
    num_caps = 0
    for letter in tweet:
        if letter.isupper():
            num_caps += 1
    percent_caps = num_caps / len(tweet)
    adjusted_percent_caps = math.ceil(percent_caps * 100)
    return adjusted_percent_caps

def get_percent_caps_tweets(tweets):
    caps = {}
    percents = []
    for tweet in tweets:
        percent = get_percent_caps(tweet)
        count = caps.get(percent) or 0
        count += 1
        caps[percent] = count
        percents.append(percent)
    return percents

########################## Sentiment Score Features ###########################

def convert_tag(tag):
    if tag.startswith('NN'):
        return 'n'
    elif tag.startswith('VB'):
        return 'v'
    elif tag.startswith('JJ'):
        return 'a'
    elif tag.startswith('RB'):
        return 'r'
    else:
        return ''

SENTI_TAGS = { 'NN':'n', 'VB':'v', 'JJ':'a', 'RB':'r' }
SENTI_NETS = [ '.%02d' % i for i in range(1, MAX_SENSES+1) ]

def get_word_tag_senti_synset(word_tag_sense, DEBUG=False):
    try:
        synset = sentiwordnet.senti_synset(word_tag_sense)
    except WordNetError:
        synset = None
    except:
        print("Unexpected error getting synset:", sys.exc_info()[0])
    if DEBUG:
        print("--- %s : %s" % (word_tag_sense, synset))
    return synset

def get_word_tag_senti_score(word_tag, senti_scores_dict, VERBOSE=False):
    senti_score = None
    num_exceptions = 0
    if word_tag in senti_scores_dict:
        senti_score = senti_scores_dict[word_tag]
    else:
        synsets = [ get_word_tag_senti_synset(word_tag + n) for n in SENTI_NETS ]
        synsets_found = [ s for s in synsets if s != None ]
        num_exceptions += len(SENTI_NETS) - len(synsets_found)
        if len(synsets_found) >= MIN_SENSES:
            senti_score_pos = np.average([ s.pos_score() for s in synsets_found ])
            senti_score_neg = np.average([ s.neg_score() for s in synsets_found])
            senti_score = (senti_score_pos - senti_score_neg)
            senti_scores_dict[word_tag] = senti_score
            if VERBOSE:
                for i, n in enumerate(SENTI_NETS):
                    print("%s.%s: %s" % (word_tag, n, synsets[i]))
                print("Averages: POS %f NEG %f DIFF %f" % \
                    (senti_score_pos, senti_score_neg, senti_score))
    return senti_score, num_exceptions

def get_senti_score(tweet, senti_scores_dict, DEBUG=False, VERBOSE=False):
    tokens = nltk.word_tokenize(tweet)
    tagged = nltk.pos_tag(tokens)
    senti_tagged = [ w + '.' + SENTI_TAGS[t[:2]] for w, t in tagged if t[:2] in SENTI_TAGS ]

    avg_senti_score = 0
    num_senti_words = 0
    num_exceptions = 0

    for word_tag in senti_tagged:
        senti_score, word_tag_exceptions =  get_word_tag_senti_score(word_tag, senti_scores_dict)
        num_exceptions += word_tag_exceptions
        if senti_score != None:
            avg_senti_score += senti_score
            num_senti_words += 1

    if num_senti_words > 0:
        avg_senti_score /= num_senti_words

    if avg_senti_score >= 0:
        adjusted_score = math.ceil(avg_senti_score * 100)
    else:
        adjusted_score = math.floor(avg_senti_score * 100)

    return adjusted_score, num_senti_words, num_exceptions

def get_sentiments_tweets(tweets, senti_scores_dict):
    DEBUG = False
    sentiments = {}
    scores = []
    total_words_scored = 0
    total_exceptions = 0
    tweets_with_scored_words = 0
    tweets_with_exceptions = 0
    print("Scoring sentiment in %d tweets ..." % len(tweets))
    for i, tweet in enumerate(tweets):
        score, num_words_scored, num_exceptions = get_senti_score(tweet, senti_scores_dict, DEBUG)
        count = sentiments.get(score) or 0
        count += 1
        sentiments[score] = count
        scores.append(score)
        # monitoring ...
        total_words_scored += num_words_scored
        total_exceptions += num_exceptions
        tweets_with_scored_words += 1 if num_words_scored > 0 else 0
        tweets_with_exceptions += 1 if num_exceptions > 0 else 0
        if (i+1) % 100 == 0:
#           print("%d %d %s" % (i, score, tweet))
            print(".", end='', flush=True)
            DEBUG = True
        else:
            DEBUG = False
        if (i+1) % 5000 == 0:
            print()
    print((nltk.FreqDist(sentiments)).most_common(20))
    print("Tweets with scored words: %d; total words scored: %d" % \
        (tweets_with_scored_words, total_words_scored))
    print("Tweets with exceptions: %d; total exceptions: %d" % \
        (tweets_with_exceptions, total_exceptions))
    print("Total word/tags with scores: %d" % len(senti_scores_dict))
    return scores

############################## Assemble Features ##############################

senti_scores_dict = {}  # word_tag : senti_score = pos - neg

def assemble_features(tweets, words_in_tweets, bigrams_in_tweets, \
        sarcastic_freqs, non_sarcastic_freqs):
    count_sarcastic_freq_unigrams, count_sarcastic_freq_bigrams, \
        count_non_sarcastic_freq_unigrams, count_non_sarcastic_freq_bigrams = \
        ssf.get_ngram_counts(words_in_tweets, bigrams_in_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    count_ngrams = list(zip(count_sarcastic_freq_unigrams, count_sarcastic_freq_bigrams, \
        count_non_sarcastic_freq_unigrams, count_non_sarcastic_freq_bigrams))
    repeated_character_counts = get_repeated_character_count_tweets(tweets)
    percent_caps = get_percent_caps_tweets(tweets)
    sentiment_scores = get_sentiments_tweets(tweets, senti_scores_dict)

    features = []
    for i, ncs in enumerate(count_ngrams):
        nc = list(ncs)
        rc = [ repeated_character_counts[i] ]
        pc = [ percent_caps[i] ]
        ss = [ sentiment_scores[i] ]
        feature_vector = nc + rc + pc + ss
        features.append(feature_vector)

    return features

# Assemble the features for training or test tweets
def get_features_train_tweets(tweets, labels):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    sarcastic_tweets, non_sarcastic_tweets = \
        ssf.separate_sarcastic_by_labels(tweets, labels)
    sarcastic_freqs, non_sarcastic_freqs = \
        ssf.get_freq_unigram_and_bigram_sets(\
            sarcastic_tweets, non_sarcastic_tweets)
    features = assemble_features(tweets, words_in_tweets, bigrams_in_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    return np.array(features), sarcastic_freqs, non_sarcastic_freqs

def get_features_test_tweets(tweets, sarcastic_freqs, non_sarcastic_freqs):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    features = assemble_features(tweets, words_in_tweets, bigrams_in_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    return np.array(features)

if __name__ == '__main__':

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    # Display hyper-parameters for this run ...
    print(" Hyper-parameters ".center(80, '-'))
    print("USE_FULL_TRAIN: %s, TRAIN_SIZE: %d" % (USE_FULL_TRAIN, TRAIN_SIZE))
    print("Full training set size - sarcastic: %d, non-sarcastic: %d" % \
        (COUNT_SARCASTIC_TRAINING_TWEETS, COUNT_NON_SARCASTIC_TRAINING_TWEETS))
    print("Senti-scores - # most common n-grams: %d, remove common n-grams: %s" % \
        (NUM_MOST_COMMON_NGRAMS, REMOVE_COMMON_NGRAMS))
    print("Senti-scores - min senses: %d, max senses: %d" % \
        (MIN_SENSES, MAX_SENSES))
    print("Senti-scores - remove stopwords: %s remove punctuation: %s" % \
        (REMOVE_STOPWORDS, REMOVE_PUNCTUATION))
    print("Senti-scores - punctuation: %s" % ' '.join(SS_PUNCTUATION))
    print('-'*80)

    sarcastic_tweets, non_sarcastic_tweets = load_data()
    train_tweets, train_labels, test_tweets, test_labels = \
        get_data(sarcastic_tweets, non_sarcastic_tweets)

    assert(len(train_tweets) + len(test_tweets) == \
           len(sarcastic_tweets) + len(non_sarcastic_tweets))
    assert(len(train_tweets) == len(train_labels))
    assert(len(test_tweets) == len(test_labels))


    if USE_FULL_TRAIN:
        _train_tweets = train_tweets
        _train_labels = train_labels
        _test_tweets = test_tweets
        _test_labels = test_labels
    else:
    # abbreviate the tweets for testing ...
        TRAIN_SIZE_HALF = TRAIN_SIZE // 2
        _train_tweets = train_tweets[:TRAIN_SIZE_HALF] + train_tweets[-TRAIN_SIZE_HALF:]
        _train_labels = train_labels[:TRAIN_SIZE_HALF] + train_labels[-TRAIN_SIZE_HALF:]
        _test_tweets = test_tweets[:TRAIN_SIZE_HALF] + test_tweets[-TRAIN_SIZE_HALF:]
        _test_labels = test_labels[:TRAIN_SIZE_HALF] + test_labels[-TRAIN_SIZE_HALF:]

    np_train_tweets = np.array(_train_tweets)
    np_train_features, sarcastic_freqs, non_sarcastic_freqs = \
        get_features_train_tweets(_train_tweets, _train_labels)
    np_test_features = \
        get_features_test_tweets(_test_tweets, sarcastic_freqs, non_sarcastic_freqs)
    np_train_labels = np.array(_train_labels)
    np_test_labels = np.array(_test_labels)

    print("Training features:\n%s" % np_train_features[:2])
    print("Training labels:\n%s" % np_train_labels[:10])
    print("Testing features:\n%s" % np_test_features[:2])
    print("Testing labels:\n%s" % np_test_labels[:10])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    import max_ent as me
    print(" Max Entropy ".center(80, '+'))
    if TUNE:
        print("Ten-fold cross-validate ...")
        me.cross_validate_lr(np_train_features, np_train_labels, np_train_tweets)
    else:
        print("Train and predict ...")
        mse, pearson, f_score = me.train_and_validate_lr( \
            np_train_features, np_train_labels, np_test_features, np_test_labels)
        print("... MSE: %f PEARSON: %s F-SCORE: %f"  % (mse, pearson, f_score))

    import svm
    print(" Support Vector Machine ".center(80, '+'))
    if TUNE:
        print("Ten-fold cross-validate ...")
        svm.cross_validate_svm(np_train_features, np_train_labels, np_train_tweets)
    else:
        print("Train and predict ...")
        mse, pearson, f_score = svm.train_and_validate_svm( \
            np_train_features, np_train_labels, np_test_features, np_test_labels)
        print("... MSE: %f PEARSON: %s F-SCORE: %f"  % (mse, pearson, f_score))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
