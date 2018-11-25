import re
import nltk
import math
import numpy as np
from nltk.corpus import sentiwordnet
from nltk.corpus import stopwords

from datetime import datetime

################################## Load Data ###################################

COUNT_SARCASTIC_TRAINING_TWEETS = 20000
COUNT_NON_SARCASTIC_TRAINING_TWEETS = 100000

SARCASTIC = 1
NON_SARCASTIC = 0

NUM_MOST_COMMON_NGRAMS = 50

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

# Create sarcastic and non-sarcastic common words dictionaries
def get_ngram_frequencies(ngrams_in_tweets):
    ngrams = [ ngram for tweet in ngrams_in_tweets for ngram in tweets]
    fd_ngrams = nltk.FreqDist(ngrams)
    return fd_ngrams

def remove_common_ngrams(ngrams_in_sarcastic_tweets, ngrams_in_non_sarcastic_tweets):
    fd_sarcastic = get_ngram_frequencies(ngrams_in_sarcastic_tweets)
    fd_non_sarcastic = get_ngram_frequencies(ngrams_in_non_sarcastic_tweets)
    fd_just_sarcastic = fd_sarcastic - fd_non_sarcastic
    fd_just_non_sarcastic = fd_non_sarcastic - fd_sarcastic
    return fd_just_sarcastic, fd_just_non_sarcastic

def get_most_common_ngrams_set(fd_ngrams):
    freq = fd_ngrams.most_common(NUM_MOST_COMMON_NGRAMS)
    freq_ngrams, freq_counts = zip(*freq)
    return set(freq_ngrams)

def get_freq_ngram_sets(ngrams_in_sarcastic_tweets, ngrams_in_non_sarcastic_tweets):
    fd_just_sarcastic_ngrams, fd_just_non_sarcastic_ngrams = \
        remove_common_ngrams(ngrams_in_sarcastic_tweets, ngrams_in_non_sarcastic_tweets)
    sarcastic_set = get_most_common_ngrams_set(fd_just_sarcastic_ngrams)
    non_sarcastic_set = get_most_common_ngrams_set(fd_just_non_sarcastic_ngrams)
    return sarcastic_set, non_sarcastic_set

def get_freq_unigram_and_bigram_sets(training_sarcastic_tweets, training_non_sarcastic_tweets):
    # unigrams
    unigrams_in_sarcastic_tweets = [ nltk.word_tokenize(tweet) for tweet in training_sarcastic_tweets ]
    unigrams_in_non_sarcastic_tweets = [ nltk.word_tokenize(tweet) for tweet in training_non_sarcastic_tweets ]
    sarcastic_unigrams_set, non_sarcastic_unigrams_set = \
        get_freq_ngram_sets(unigrams_in_sarcastic_tweets, unigrams_in_non_sarcastic_tweets)
    # bigrams
    bigrams_in_sarcastic_tweets = find_ngrams_in_tweets(n, unigrams_in_sarcastic_tweets)
    bigrams_in_non_sarcastic_tweets = find_ngrams_in_tweets(n, unigrams_in_non_sarcastic_tweets)
    sarcastic_bigrams_set, non_sarcastic_bigrams_set = \
        get_freq_ngram_sets(bigrams_in_sarcastic_tweets, bigrams_in_non_sarcastic_tweets)
    # combined sets of most frequent unigrams and bigrams
    sarcastic_set = sarcastic_unigrams_set + sarcastic_bigrams_set
    non_sarcastic_set = sarcastic_unigrams_set + non_sarcastic_bigrams_set
    return sarcastic_set, non_sarcastic_set

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

    sarcastic_freqs, non_sarcastic_freqs = \
        get_freq_unigram_and_bigram_sets(training_sarcastic_tweets, training_non_sarcastic_tweets)

    return train_tweets, train_labels, test_tweets, test_labels,
        sarcastic_freqs, non_sarcastic_freqs

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


STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = { ',', '.', '?', '!', ';', ':' }

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

# Compute count of ngrams in each tweet that are in the given set
def get_count_ngrams_in_set(ngrams_in_tweets, freq_set):
    count = 0
    for tokens in ngrams_in_tweets:
        for token in tokens:
            if token in freq_set:
                count += 1
    return w_count

def get_ngram_counts(words_in_tweets, bigrams_in_tweets, \
    sarcastic_freqs, non_sarcastic_freqs):
    count_sarcastic_freq_unigrams = \
        get_count_ngrams_in_set(words_in_tweets, sarcastic_freqs)
    count_sarcastic_freq_bigrams = \
        get_count_ngrams_in_set(bigrams_in_tweets, sarcastic_freqs)
    count_non_sarcastic_freq_unigrams = \
        get_count_ngrams_in_set(words_in_tweets, non_sarcastic_freqs)
    count_non_sarcastic_freq_bigrams = \
        get_count_ngrams_in_set(bigrams_in_tweets, non_sarcastic_freqs)
    return [ count_sarcastic_freq_unigrams, count_sarcastic_freq_bigrams, \
        count_non_sarcastic_freq_unigrams, count_non_sarcastic_freq_bigrams ]

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

def get_senti_score(sentence):
    token = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(token)

    avg_senti_score = 0
    num_senti_words = 0

    for pair in tagged:
        word = pair[0]
        tag = convert_tag(pair[1])
        if tag != '':
            senti_word_1 = word + '.' + tag + '.01'
            senti_word_2 = word + '.' + tag + '.02'
            senti_word_3 = word + '.' + tag + '.03'
            try:
                senti_score_1 = sentiwordnet.senti_synset(senti_word_1)
                senti_score_2 = sentiwordnet.senti_synset(senti_word_2)
                senti_score_3 = sentiwordnet.senti_synset(senti_word_3)
                senti_score_pos = (senti_score_1.pos_score() + senti_score_2.pos_score() + senti_score_3.pos_score()) / 3
                senti_score_neg = (senti_score_1.neg_score() + senti_score_2.neg_score() + senti_score_3.neg_score()) / 3
                avg_senti_score += (senti_score_pos - senti_score_neg)
                num_senti_words += 1
            except:
                avg_senti_score += 0

    if num_senti_words > 0:
        avg_senti_score /= num_senti_words

    if avg_senti_score >= 0:
        adjusted_score = math.ceil(avg_senti_score * 100)
    else:
        adjusted_score = math.floor(avg_senti_score * 100)

    return adjusted_score

def get_sentiments_tweets(tweets):
    sentiments = {}
    scores = []
    print("Scoring sentiment in %d tweets ..." % len(tweets))
    for i, tweet in enumerate(tweets):
        score = get_senti_score(tweet)
        count = sentiments.get(score) or 0
        count += 1
        sentiments[score] = count
        scores.append(score)
        if (i+1) % 100 == 0:
#           print("%d %d %s" % (i, score, tweet))
            print(".", end='')
        if (i+1) % 1000 == 0:
            print()
    print((nltk.FreqDist(sentiments)).most_common(20))
    return scores

############################## Assemble Features ##############################

def assemble_features(tweets, words_in_tweets, bigrams_in_tweets, \
        sarcastic_freqs, non_sarcastic_freqs):
    count_sarcastic_freq_unigrams, count_sarcastic_freq_bigrams, \
        count_non_sarcastic_freq_unigrams, count_non_sarcastic_freq_bigrams = \
        get_ngram_counts(words_in_tweets, bigrams_in_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    repeated_character_counts = get_repeated_character_count_tweets(tweets)
    percent_caps = get_percent_caps_tweets(tweets)
    sentiment_scores = get_sentiments_tweets(tweets)

    features = []
    for i, suc in enumerate(count_sarcastic_freq_unigrams):
        sbc = count_sarcastic_freq_bigrams[i]
        nuc = count_non_sarcastic_freq_unigrams[i]
        nbc = count_non_sarcastic_freq_bigrams[i]
        rc = [ repeated_character_counts[i] ]
        pc = [ percent_caps[i] ]
        ss = [ sentiment_scores[i] ]
        feature_vector = suc + sbc + nuc + nbc + rc + pc + ss
        features.append(feature_vector)

    return features

# Assemble the features for training or test tweets
def get_features_tweets(tweets, sarcastic_freqs, non_sarcastic_freqs):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    features = assemble_features(tweets, words_in_tweets, bigrams_in_tweets, \
            sarcastic_freqs, non_sarcastic_freqs)
    return np.array(features)

if __name__ == '__main__':

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    sarcastic_tweets, non_sarcastic_tweets = load_data()
    train_tweets, train_labels, test_tweets, test_labels, \
        sarcastic_freqs, non_sarcastic_freqs = \
        get_data(sarcastic_tweets, non_sarcastic_tweets)

    assert(len(train_tweets) + len(test_tweets) == \
           len(sarcastic_tweets) + len(non_sarcastic_tweets))
    assert(len(train_tweets) == len(train_labels))
    assert(len(test_tweets) == len(test_labels))

    # abbreviate the tweets for testing ...
    _train_tweets = train_tweets[:2000] + train_tweets[-2000:]
    _train_labels = train_labels[:2000] + train_labels[-2000:]
    _test_tweets = test_tweets[:2000] + test_tweets[-2000:]
    _test_labels = test_labels[:2000] + test_labels[-2000:]

    np_train_features, word_dict, bigram_dict = \
        get_train_features_tweets(_train_tweets, sarcastic_freqs, non_sarcastic_freqs)
    np_test_features = \
        get_test_features_tweets(_test_tweets, sarcastic_freqs, non_sarcastic_freqs)
    np_train_labels = np.array(_train_labels)
    np_test_labels = np.array(_test_labels)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    import svm
    print("Train and predict ...")
    mse, pearson = svm.train_and_validate_svm( \
        np_train_features, np_train_labels, np_test_features, np_test_labels)
    print("... MSE: %f PEARSON: %s"  % (mse, pearson))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
