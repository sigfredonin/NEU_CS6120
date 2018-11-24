import numpy
import nltk
import math
from nltk.corpus import sentiwordnet

COUNT_SARCASTIC_TRAINING_TWEETS = 20000
COUNT_NON_SARCASTIC_TRAINING_TWEETS = 100000

# load data
def load_data():
    posproc = numpy.load('posproc.npy')
    negproc = numpy.load('negproc.npy')

    sarcastic_tweets = []
    non_sarcastic_tweets = []

    for tweet in posproc:
        sarcastic_tweets.append(tweet.decode('utf-8'))

    for tweet in negproc:
        non_sarcastic_tweets.append(tweet.decode('utf-8'))

    print('num sarcastic tweets: ' + str(len(sarcastic_tweets)))
    print('num non sarcastic tweets: ' + str(len(non_sarcastic_tweets)))

    return sarcastic_tweets, non_sarcastic_tweets

# Separate training and testing data
def get_data(sarcastic_tweets, non_sarcastic_tweets):
    training_sarcastic_tweets = sarcastic_tweets[0:COUNT_SARCASTIC_TRAINING_TWEETS]
    testing_sarcastic_tweets = sarcastic_tweets[COUNT_SARCASTIC_TRAINING_TWEETS:]

    training_non_sarcastic_tweets = non_sarcastic_tweets[0:COUNT_NON_SARCASTIC_TRAINING_TWEETS]
    testing_non_sarcastic_tweets = non_sarcastic_tweets[COUNT_NON_SARCASTIC_TRAINING_TWEETS:]

    return(training_sarcastic_tweets, training_non_sarcastic_tweets, \
        testing_sarcastic_tweets, testing_non_sarcastic_tweets)

# --- Training ---

# - Unigrams -
def _get_unigram_counts(tweets, unique_unigram_count, other_unigram_counts):
    unigram_counts = {}
    total_unigram_count = 0

    for tweet in tweets:
        tokenized_tweet = nltk.word_tokenize(tweet)
        for token in tokenized_tweet:
            count = unigram_counts.get(token) or 0
            if count == 0 and other_unigram_counts.get(token) is None:
                unique_unigram_count += 1
            unigram_counts[token] = count + 1
            total_unigram_count += 1

    return unigram_counts, total_unigram_count, unique_unigram_count

def get_unigram_counts(training_sarcastic_tweets, testing_non_sarcastic_tweets):
    sarcastic_unigram_counts = {}
    non_sarcastic_unigram_counts = {}

    total_sarcastic_unigram_count = 0
    total_non_sarcastic_unigram_count = 0
    unique_unigram_count = 0

    # get unigram counts for sarcastic tweets
    sarcastic_unigram_counts, total_sarcastic_unigram_count, unique_unigram_count = \
        _get_unigram_counts(training_sarcastic_tweets, unique_unigram_count, non_sarcastic_unigram_counts)

    # get unigram counts for non sarcastic tweets
    non_sarcastic_unigram_counts, total_non_sarcastic_unigram_count, unique_unigram_count = \
        _get_unigram_counts(training_non_sarcastic_tweets, unique_unigram_count, sarcastic_unigram_counts)

    print('')
    print('unique unigram count: ' + str(unique_unigram_count))
    print('total sarcastic unigram count: ' + str(total_sarcastic_unigram_count))
    print('total non sarcastic unigram count: ' + str(total_non_sarcastic_unigram_count))

    return (sarcastic_unigram_counts, total_sarcastic_unigram_count, unique_unigram_count), \
           (non_sarcastic_unigram_counts, total_non_sarcastic_unigram_count, unique_unigram_count)

# - Bigrams -
def _get_bigram_counts(tweets, unique_bigram_count, other_bigram_counts):
    bigram_counts = {}
    total_bigram_count = 0

    for tweet in tweets:
        tokenized_tweet = nltk.word_tokenize(tweet)
        tokenized_tweet.append('<end>')
        bigram_array = ['', '<start>']

        for word in tokenized_tweet:
            bigram_array.pop(0)
            bigram_array.append(word)
            bigram = ' '.join(bigram_array)
            count = bigram_counts.get(bigram) or 0
            if count == 0 and other_bigram_counts.get(bigram) is None:
                unique_bigram_count += 1
            bigram_counts[bigram] = count + 1
            total_bigram_count += 1

    return bigram_counts, total_bigram_count, unique_bigram_count

def get_bigram_counts(training_sarcastic_tweets, testing_non_sarcastic_tweets):
    sarcastic_bigram_counts = {}
    non_sarcastic_bigram_counts = {}

    total_sarcastic_bigram_count = 0
    total_non_sarcastic_bigram_count = 0
    unique_bigram_count = 0

    # get bigram counts for sarcastic tweets
    sarcastic_bigram_counts, total_sarcastic_bigram_count, unique_bigram_count = \
        _get_bigram_counts(training_sarcastic_tweets, unique_bigram_count, non_sarcastic_bigram_counts)

    # get bigram counts for non sarcastic tweets
    non_sarcastic_bigram_counts, total_non_sarcastic_bigram_count, unique_bigram_count = \
        _get_bigram_counts(training_non_sarcastic_tweets, unique_bigram_count, sarcastic_bigram_counts)

    print('')
    print('unique bigram count: ' + str(unique_bigram_count))
    print('total sarcastic bigram count: ' + str(total_sarcastic_bigram_count))
    print('total non sarcastic bigram count: ' + str(total_non_sarcastic_bigram_count))

    return (sarcastic_bigram_counts, total_sarcastic_bigram_count, unique_bigram_count), \
           (non_sarcastic_bigram_counts, total_non_sarcastic_bigram_count, unique_bigram_count)

# - Repeated Characters -
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

def _get_repeated_character_count(tweets):
    repeated_character_count = 0
    for tweet in tweets:
        repeated_character_count += _get_repeated_character_count_tweet(tweet)
    return repeated_character_count

def get_repeated_character_counts(training_sarcastic_tweets, testing_non_sarcastic_tweets):
    sarcastic_repeated_character_count = 0
    non_sarcastic_repeated_character_count = 0

    # get repeated character count for sarcastic tweets
    sarcastic_repeated_character_count = \
        _get_repeated_character_count(training_sarcastic_tweets)

    # get repeated character count for non sarcastic tweets
    non_sarcastic_repeated_character_count = \
        _get_repeated_character_count(training_non_sarcastic_tweets)

    print('')
    print('sarcastic repeated character count: ' + str(sarcastic_repeated_character_count))
    print('non sarcastic repeated character count: ' + str(non_sarcastic_repeated_character_count))

    return sarcastic_repeated_character_count, non_sarcastic_repeated_character_count

# - Sentiment -
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

def _get_sentiments(tweets):
    sentiments = {}
    for tweet in training_sarcastic_tweets:
        score = get_senti_score(tweet)
        count = sentiments.get(score) or 0
        count += 1
        sentiments[score] = count
    return sentiments

def get_sentiments(training_sarcastic_tweets, testing_non_sarcastic_tweets):
    sarcastic_sentiments = _get_sentiments(training_sarcastic_tweets)
    non_sarcastic_sentiments = _get_sentiments(training_sarcastic_tweets)
    return sarcastic_sentiments, non_sarcastic_sentiments

# - Capitalization -
def get_percent_caps(tweet):
    num_caps = 0
    for letter in tweet:
        if letter.isupper():
            num_caps += 1
    percent_caps = num_caps / len(tweet)
    adjusted_percent_caps = math.ceil(percent_caps * 100)
    return adjusted_percent_caps

def _get_caps(tweets):
    caps = {}
    for tweet in tweets:
        percent = get_percent_caps(tweet)
        count = caps.get(percent) or 0
        count += 1
        caps[percent] = count
    return caps

def get_caps(training_sarcastic_tweets, testing_non_sarcastic_tweets):
    sarcastic_caps = _get_caps(training_sarcastic_tweets)
    non_sarcastic_caps = _get_caps(training_non_sarcastic_tweets)
    return sarcastic_caps, non_sarcastic_caps

# --- Testing ---
def _get_bigrams(tokenized_tweet):
    bigrams = []
    tokenized_tweet.append('<end>')
    bigram_array = ['', '<start>']
    for word in tokenized_tweet:
        bigram_array.pop(0)
        bigram_array.append(word)
        bigram = ' '.join(bigram_array)
        bigrams.append(bigram)
    return bigrams

def _get_smoothed_probability_product(tokenized_tweet, counts, total, unique):
    prob = 1.0
    for word in tokenized_tweet:
        w_count = (counts.get(word) or 0) + 1
        w_prob = w_count / (total + unique)
        prob *= w_prob
    return prob

# combine sarcastic and non-sarcastic tweets into one testing set
sarcastic_tweets, non_sarcastic_tweets = load_data()
training_sarcastic_tweets, training_non_sarcastic_tweets, \
    testing_sarcastic_tweets, testing_non_sarcastic_tweets = \
    get_data(sarcastic_tweets, non_sarcastic_tweets)

testing_tweets = testing_sarcastic_tweets
testing_tweets.extend(testing_non_sarcastic_tweets)

# results matrix (true positive, false positive, false negative, true negative)
results = {
    'tp': 0, 'fp': 0,
    'fn': 0, 'tn': 0
}

# determine result of each tweet in testing set
for tweet in testing_tweets:
    tokenized_tweet = nltk.word_tokenize(tweet)

    # unigram testing
    (sarcastic_unigram_counts, total_sarcastic_unigram_count, unique_unigram_count), \
           (non_sarcastic_unigram_counts, total_non_sarcastic_unigram_count, unique_unigram_count) = \
           get_unigram_counts(training_sarcastic_tweets, testing_non_sarcastic_tweets)
    sarcastic_prob = \
        _get_smoothed_probability_product(tokenized_tweet, \
            sarcastic_unigram_counts, total_sarcastic_unigram_count, unique_unigram_count)
    non_sarcastic_prob = \
        _get_smoothed_probability_product(tokenized_tweet, \
            non_sarcastic_unigram_counts, total_non_sarcastic_unigram_count, unique_unigram_count)

    # bigram testing
    bigrams = _get_bigrams(tokenized_tweet)
    (sarcastic_bigram_counts, total_sarcastic_bigram_count, unique_bigram_count), \
           (non_sarcastic_bigram_counts, total_non_sarcastic_bigram_count, unique_bigram_count) = \
           get_bigram_counts(training_sarcastic_tweets, testing_non_sarcastic_tweets)
    sarcastic_prob *= \
        _get_smoothed_probability_product(tokenized_tweet, \
            sarcastic_bigram_counts, total_sarcastic_bigram_count, unique_bigram_count)
    non_sarcastic_prob *= \
        _get_smoothed_probability_product(tokenized_tweet, \
            non_sarcastic_bigram_counts, total_non_sarcastic_bigram_count, unique_bigram_count)

    # repeated character testing
    sarcastic_repeated_character_count, non_sarcastic_repeated_character_count = \
           get_repeated_character_counts(training_sarcastic_tweets, testing_non_sarcastic_tweets)
    repeated_characters_count = _get_repeated_character_count(tweet)
    if repeated_characters_count > 0:
        sarcastic_prob *= (sarcastic_repeated_character_count / COUNT_SARCASTIC_TRAINING_TWEETS)
        non_sarcastic_prob *= (non_sarcastic_repeated_character_count / COUNT_NON_SARCASTIC_TRAINING_TWEETS)
    else:
        sarcastic_prob *= ((COUNT_SARCASTIC_TRAINING_TWEETS - sarcastic_repeated_character_count) / COUNT_SARCASTIC_TRAINING_TWEETS)
        non_sarcastic_prob *= ((COUNT_NON_SARCASTIC_TRAINING_TWEETS - non_sarcastic_repeated_character_count) / COUNT_NON_SARCASTIC_TRAINING_TWEETS)

    # sentiment testing
    senti_score = get_senti_score(tweet)
    sarcastic_sentiments, non_sarcastic_sentiments = \
        get_sentiments(training_sarcastic_tweets, testing_non_sarcastic_tweets)
    sarcastic_senti_count = (sarcastic_sentiments.get(senti_score) or 0) + 1
    non_sarcastic_senti_count = (non_sarcastic_sentiments.get(senti_score) or 0) + 1

    sarcastic_prob *= (sarcastic_senti_count / COUNT_SARCASTIC_TRAINING_TWEETS)
    non_sarcastic_prob *= (non_sarcastic_senti_count / COUNT_NON_SARCASTIC_TRAINING_TWEETS)

    # capitalization testing
    percent = get_percent_caps(tweet)
    sarcastic_caps, non_sarcastic_caps = \
        get_caps(training_sarcastic_tweets, testing_non_sarcastic_tweets)
    sarcastic_caps_count = (sarcastic_caps.get(percent) or 0) + 1
    non_sarcastic_caps_count = (non_sarcastic_caps.get(percent) or 0) + 1

    sarcastic_prob *= (sarcastic_caps_count / COUNT_SARCASTIC_TRAINING_TWEETS)
    non_sarcastic_prob *= (non_sarcastic_caps_count / COUNT_NON_SARCASTIC_TRAINING_TWEETS)


    # results
    result = 's'
    if non_sarcastic_prob > sarcastic_prob:
        result = 'ns'

    if result == 's':
        if tweet in sarcastic_tweets:
            results['tp'] = results.get('tp') + 1
        else:
            results['fp'] = results.get('fp') + 1
    else:
        if tweet in sarcastic_tweets:
            results['fn'] = results.get('fn') + 1
        else:
            results['tn'] = results.get('tn') + 1


precision = results.get('tp') / (results.get('tp') + results.get('fp'))
recall = results.get('tp') / (results.get('tp') + results.get('fn'))
f_score = (2 * precision * recall) / (precision + recall)


print('')
print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('f-score: ' + str(f_score))
