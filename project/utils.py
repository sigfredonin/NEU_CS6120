import re
import nltk
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

############################## Assemble Features ##############################

# Assemble the features for tweets
def get_train_features_tweets(tweets, word_dict, bigram_dict):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    word_dict, bigram_dict = get_training_vocabulary(words_in_tweets, bigrams_in_tweets)
    index_vectors_unigrams = ngrams_to_indices(words_in_tweets, word_dict)
    index_vectors_bigrams = ngrams_to_indices(bigrams_in_tweets, bigram_dict)


    repetitive_unigram_counts = summary_repetive_ngram_counts(1, tokenized_tweets)
    repetitive_bigram_counts = summary_repetive_ngram_counts(2, tokenized_tweets)
    min_Flesch_scores = summary_min_Flesch_scores(summaries)
    return zip_features(repetitive_unigram_counts, repetitive_bigram_counts, \
                        min_Flesch_scores)
