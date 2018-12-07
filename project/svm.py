import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

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

def train_and_validate_svm(train_data, train_labels, test_data, test_labels, \
                           kernel='rbf', C=1.0, gamma='scale', verbose=False, class_weight='balanced'):
    print("training svm with kernel='%s', C=%s, gamma='%s', class_weight='%s'" % (kernel, C, gamma, class_weight))
    svm = SVC(kernel=kernel, C=C, gamma=gamma, verbose=verbose, \
              class_weight=class_weight, cache_size=1000)
    svm.fit(train_data, train_labels)
    predicted_labels = svm.predict(test_data)
    print('test', test_labels[:10], test_labels[-10:])
    print('pred', predicted_labels[:10], predicted_labels[-10:])
    mse = mean_squared_error(test_labels, predicted_labels)
    print('mean squared error:', mse)
    pearson = pearsonr(test_labels, predicted_labels)
    print('pearson coefficient:', pearson)
    f_score = f1_score(test_labels, predicted_labels)
    print('f-score:', f_score)
    return mse, pearson, f_score

def cross_validate_svm(data, labels, tweets, \
                       kernel='linear', C=1.0, gamma='scale', verbose=False, class_weight='balanced'):
    print('cross-validating svm...')
    num_cross_validation_trials = 10
    kfold = KFold(num_cross_validation_trials, True, 1)

    mses = []
    pearsons = []
    f_scores = []
    for trial_index, (train, val) in enumerate(kfold.split(data)):
        print((" Trial %d of %d" % (trial_index+1, num_cross_validation_trials)).center(80, '-'))

        _data, _labels, _val_data, _val_labels = \
            get_features_for_split(data[train], labels[train], tweets[train],
                data[val], labels[val], tweets[val])

        mse, (pearson_r, pearson_p), f_score = \
            train_and_validate_svm(_data, _labels, _val_data, _val_labels, \
                                kernel, C, gamma, verbose, class_weight)
        mses.append(mse)
        pearsons.append(pearson_r)
        f_scores.append(f_score)
    print_results_of_trials(mses, pearsons, f_scores)
    return mses, pearsons, f_scores

def print_results_of_trials(mses, pearsons, f_scores):
    print_metric_results_of_trials(mses, "Mean Squared Error")
    print_metric_results_of_trials(pearsons, "Pearson Coefficients")
    print_metric_results_of_trials(f_scores, "F-Scores")

def print_metric_results_of_trials(metrics_over_trials, metric_name):
    np_metrics = np.array(metrics_over_trials)

    min_metric = np_metrics.min()
    avg_metric = np.average(np_metrics)
    max_metric = np_metrics.max()

    print(("> %s over all trials <" % metric_name).center(80, '='))
    print("%s min:  %7.4f" % (metric_name, min_metric))
    print("%s mean: %7.4f" % (metric_name, avg_metric))
    print("%s max:  %7.4f" % (metric_name, max_metric))
    print(80*'=')
