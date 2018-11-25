import re
import nltk
import numpy as np
from nltk.corpus import stopwords

################################## Load Data ###################################

COUNT_SARCASTIC_TRAINING_TWEETS = 20000
COUNT_NON_SARCASTIC_TRAINING_TWEETS = 100000

SARCASTIC = 1
NON_SARCASTIC = 0

# removes blank lines, replaces \n with space, removes duplicate spaces
def process_whitespace(token_str):
    no_new_line = re.sub(r'\n', " ", token_str)
    no_dup_spaces = re.sub(r'  +', " ", no_new_line)
    return no_dup_spaces

def load_data():
    posproc = numpy.load('posproc.npy')
    negproc = numpy.load('negproc.npy')

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

    return(train_tweets, train_labels, test_tweets, test_labelss)

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

def ngrams(n, tokens):
    return [tuple(tokens[i:i+n]) for i in range (len(tokens)-(n-1))]

def find_ngrams_in_tweets(n, tokenized_tweets):
    ngrams = []
    for tokens in tokenized_tweets:
        tweet_ngrams = ngrams(n, tokens)
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

def get_repeated_character_count_tweets(tweets):
    repeated_character_count = 0
    for tweet in tweets:
        repeated_character_count += _get_repeated_character_count_tweet(tweet)
    return repeated_character_count

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
    for tweet in tweets:
        percent = get_percent_caps(tweet)
        count = caps.get(percent) or 0
        count += 1
        caps[percent] = count
    return caps

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
    print("Scoring sentiment in %d tweets ..." % len(tweets))
    for i, tweet in enumerate(tweets):
        score = get_senti_score(tweet)
        count = sentiments.get(score) or 0
        count += 1
        sentiments[score] = count
        if i+1 % 100 == 0:
            print(".", end='')
        if i+1 % 1000 == 0:
            print()
    print()
    return sentiments

############################## Assemble Features ##############################

# Assemble the features for training tweets
def get_train_features_tweets(tweets):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    word_dict, bigram_dict = get_training_vocabulary(words_in_tweets, bigrams_in_tweets)
    index_vectors_unigrams = ngrams_to_indices(words_in_tweets, word_dict)
    index_vectors_bigrams = ngrams_to_indices(bigrams_in_tweets, bigram_dict)
    repeated_character_counts = get_repeated_character_count_tweets(tweets)
    percent_caps = get_percent_caps_tweets(tweets)
    sentiment_scores = get_sentiments_tweets(tweets)

    features = []
    for i, uv in enumerate(index_vectors_unigrams):
        bv = index_vectors_bigrams[i]
        rc = [ repeated_character_counts[i] ]
        pc = [ percent_caps[i] ]
        ss = [ sentiment_scores[i] ]
        feature_vector = uv + bv + rc + pc + ss
        feature.append(feature_vector)

    return np.array(features), word_dict, bigram_dict

# Assemble the features for test tweets
def get_test_features_tweets(tweets, word_dict, bigram_dict):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    index_vectors_unigrams = ngrams_to_indices(words_in_tweets, word_dict)
    index_vectors_bigrams = ngrams_to_indices(bigrams_in_tweets, bigram_dict)
    repeated_character_counts = get_repeated_character_count_tweets(tweets)
    percent_caps = get_percent_caps_tweets(tweets)
    sentiment_scores = get_sentiments_tweets(tweets)

    features = []
    for i, uv in enumerate(index_vectors_unigrams):
        bv = index_vectors_bigrams[i]
        rc = [ repeated_character_counts[i] ]
        pc = [ percent_caps[i] ]
        ss = [ sentiment_scores[i] ]
        feature_vector = uv + bv + rc + pc + ss
        feature.append(feature_vector)

    return np.array(features)
