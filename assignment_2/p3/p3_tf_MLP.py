"""
CS 6120 Natural Language Processing
Northeastern University
Fall 2018

Assignment 2 Problem 3
Sentiment Analysis

Input file: a3_p3_train_data.text

Sig Nin
October 29, 2018
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import p3_utils

def mlp_model(input_shape, num_h1_units, h1_activation='relu', h2_activation='relu', \
              num_classes=5, dropout_rate=0.0, input_dropout_rate=0.0):
    """
    Creates a TF Keras Multi-Layer Perceptron model,
    with an input layer, two hidden layers, and an output layer.
    """
    input_dim = input_shape[0]
    print("--- Model ---")
    print("Input    : %d x %d" % (1, input_dim))
    print("Layer h1 : %d x %d %s" % (input_dim, num_h1_units, h1_activation))
    print("Layer h2 : %d x %d %s" % (num_h1_units, 10, h2_activation))
    print("Output   : %d x %d SOFTMAX" % (10, 5))
    print("----------")
    model = models.Sequential()
    # input layer
    model.add(Dropout(rate=input_dropout_rate, input_shape=input_shape))
    # h1: hidden layer 1
    model.add(Dense(units=num_h1_units, activation=h1_activation))
    model.add(Dropout(rate=dropout_rate))
    # h2: hidden layer 2
    model.add(Dense(units=10, activation=h2_activation))
    model.add(Dropout(rate=dropout_rate))
    # output layer
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def mlp_train(model, train_data, epochs=10):
    """
    Train the Multi-Layer Perceptron.
    Inputs:
        train_data - training data and labels, evaluation data and labels
    """
    (train_data, train_labels), (val_data, val_labels) = train_data

    if len(val_data) == 0 or len(val_labels) == 0:
        validation_data = None
    else:
        validation_data = (val_data, val_labels)

    loss = 'sparse_categorical_crossentropy'
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    history = model.fit(train_data, train_labels, \
        validation_data = validation_data, \
        epochs=epochs, batch_size=32)

    return history

def get_data(input_type, num_cross_validation_trials):
    """
    Load the data and select the correct format.
    Return -

    """
    # Load the training text
    train_text = p3_utils.get_text_from_file(p3_utils.PATH_TRAIN)
    print("Read %d bytes from '%s' as text" % \
        (len(train_text), p3_utils.PATH_TRAIN))
    print("Text begins : '%s'" % train_text[:30])

    # Load the test text
    test_text = p3_utils.get_text_from_file(p3_utils.PATH_TEST)
    print("Read %d bytes from '%s' as text" % \
        (len(test_text), p3_utils.PATH_TEST))
    print("Text begins : '%s'" % test_text[:30])

    # Compile reviews and ratings, with word index encoding
    # ... for the training data ...
    words, review_words, review_data, review_labels, \
        fd_words, vocabulary, dictionary, reverse_dictionary = \
        p3_utils.get_words_and_ratings(train_text)
    vocabulary_size = len(vocabulary)
    print("Training vocabulary size: %d" % vocabulary_size)
    print("Number of training reviews: %d" % len(review_words))
    print("Length of review_data vectors: %d" % len(review_data[0]))

    # Compile reviews, with word index encoding
    #... for the test data ...
    test_reviews, test_words, test_review_words, test_review_data = \
        p3_utils.load_test_set(test_text, dictionary, VECTOR_LEN=len(review_data[0]))
    print("Number of test reviews: %d" % len(test_review_words))

    # Load word vectors if going to use them
    if input_type == 'awv' or input_type == 'awv+sv' or input_type == 'awv+pos':
        vectors = p3_utils.load_embeddings_gensim()
        wv_review_data, wv_review_vectors, vw_review_sentence_average_vectors \
            = p3_utils.get_embeddings(vectors, review_words)
        wv_test_data, wv_test_vectors, vw_test_sentence_average_vectors \
            = p3_utils.get_embeddings(vectors, test_review_words)

    # Get POS vectors if going to use them
    if input_type == 'pos' or input_type == 'awv+pos':
        review_pos_vectors = p3_utils.get_pos_tags_reviews(train_text)
        test_pos_vectors = p3_utils.get_pos_tags_reviews(test_text, \
            HAS_RATINGS=False, VECTOR_LEN=len(review_pos_vectors[0]))

    # Prepare combined word vectors and pos vectors if going to use them
    if input_type == 'awv+pos':
        # ... for training data ...
        review_sentence_awv_pos = []
        for i in range(len(review_words)):
            rsv = np.concatenate([vw_review_sentence_average_vectors[i], \
                                  review_pos_vectors[i]])
            review_sentence_awv_pos.append(rsv)
        # ... for test data ...
        test_sentence_awv_pos = []
        for i in range(len(test_review_words)):
            rsv = np.concatenate([vw_test_sentence_average_vectors[i], \
                                  test_pos_vectors[i]])
            test_sentence_awv_pos.append(rsv)

    # Load sentiment vectors if going to use them
    if input_type == 'rsv' or input_type == 'awv+sv':
        review_sentiment_vectors = p3_utils.load_sentiment_vectors(review_words)
        test_sentiment_vectors = p3_utils.load_sentiment_vectors(test_review_words)

    # Prepare combined word vectors and sentiment vectors if going to use them
    if input_type == 'awv+sv':
        # ... for training data ...
        review_sentence_awv_sv = []
        for i in range(len(review_words)):
            rsv = np.concatenate([vw_review_sentence_average_vectors[i], \
                                  review_sentiment_vectors[i]])
            review_sentence_awv_sv.append(rsv)
        # ... for test data ...
        test_sentence_awv_sv = []
        for i in range(len(test_review_words)):
            rsv = np.concatenate([vw_test_sentence_average_vectors[i], \
                                  test_sentiment_vectors[i]])
            test_sentence_awv_sv.append(rsv)

    # Select the input type
    if input_type == 'one hot':
        train_data = p3_utils.one_hot_vectors(review_data, vocabulary_size)
        test_data = p3_utils.one_hot_vectors(test_review_data, vocabulary_size)
        print("Count one-hot vectors: %d" % len(train_data))
    elif input_type == 'count-hot':
        train_data = p3_utils.count_vectors(review_data, vocabulary_size)
        test_data = p3_utils.count_vectors(test_review_data, vocabulary_size)
        print("Count count vectors: %d" % len(train_data))
    elif input_type == 'td-idf hot':
        train_data = p3_utils.tdIdf_vectors(review_data, vocabulary_size)
        test_data = p3_utils.tdIdf_vectors(test_review_data, vocabulary_size)
        print("Count tdIdf vectors: %d" % len(train_data))
    elif input_type == 'word index':
        train_data = review_data
        test_data = test_review_data
        print("Count word index vectors: %d" % len(train_data))
    elif input_type == 'awv':
        train_data = vw_review_sentence_average_vectors
        test_data = vw_test_sentence_average_vectors
        print("Count embedding vectors: %d" % len(train_data))
    elif input_type == 'rsv':
        train_data = review_sentiment_vectors
        test_data = test_sentiment_vectors
        print("Count sentiment vectors: %d" % len(train_data))
    elif input_type == 'awv+rsv':
        train_data = review_sentence_awv_sv
        test_data = test_sentence_awv_sv
        print("Count embedding + sentiment vectors: %d" % len(train_data))
    elif input_type == 'pos':
        train_data = review_pos_vectors
        test_data = test_pos_vectors
        print("Count POS vectors: %d" % len(train_data))
    elif input_type == 'awv+pos':
        train_data = review_sentence_awv_pos
        test_data = test_sentence_awv_pos
        print("Count embedding + POS vectors: %d" % len(train_data))
    else:
        raise InvalidArgumentException("Unrecognized data input type: %s" % input_type)

    assert(len(train_data) == len(review_words))
    assert(len(train_data) == len(review_labels))
    assert(len(test_data) == len(test_review_words))

    shuffle_indices, xval_sets = \
        p3_utils.split_training_data_for_cross_validation(train_data, review_labels, \
            num_cross_validation_trials)

    return xval_sets, test_reviews, test_data

def run_one_trial(train_data, train_labels, val_data, val_labels, num_epochs_per_trial, \
        num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate):

    np_train_data = np.array(train_data)
    np_train_labels = np.array(train_labels)
    np_val_data = np.array(val_data)
    np_val_labels = np.array(val_labels)
    train_data = ((np_train_data, np_train_labels), (np_val_data, np_val_labels))

    print("Train data shape:   %s" % str(np_train_data.shape))
    print("Train labels shape: %s" % str(np_train_data.shape))
    print("Test data shape:    %s" % str(np_train_data.shape))
    print("Test labels shape:  %s" % str(np_train_data.shape))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = mlp_model(input_shape=np_train_data.shape[1:], \
                      num_h1_units=num_h1_units, h1_activation=h1_activation, \
                      h2_activation=h2_activation, dropout_rate=h1_h2_dropout_rate)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    history = mlp_train(model, train_data, epochs=num_epochs_per_trial)

    return model, history

def run_trials(xval_sets, num_cross_validation_trials, num_epochs_per_trial, \
        num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate):

    scores = []
    for iTrial in range(num_cross_validation_trials):

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        print((" Trial %d of %d" % (iTrial+1, num_cross_validation_trials)).center(80, '-'))

        (train_data, train_labels), (val_data, val_labels) = \
            p3_utils.assemble_cross_validation_data(xval_sets, iTrial)

        model, history = \
            run_one_trial(train_data, train_labels, val_data, val_labels, num_epochs_per_trial, \
                num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate)

        scores.append(history)

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

    return scores

def print_results_of_trials(scores, trial_parameters):

    input_type, num_h1_units, h1_activation, h2_activation, num_epochs_per_trial = trial_parameters

    print(" Trial Results ".center(80, '='))
    for iTrial, history in enumerate(scores):
        print("Trial %d of %d ---" % (iTrial+1, len(scores)))
        for iEpoch in range(num_epochs_per_trial):
            print("%4d: " % (iEpoch+1), end='')
            for history_key in history.history:
                print(" %s: %7.4f" % (history_key, history.history[history_key][iEpoch]), end='')
            print()

    # Collect trial results: loss, accuracy for training and evaluation
    train_loss = []
    train_acc  = []
    val_loss = []
    val_acc =  []
    for iTrial, history in enumerate(scores):
        train_loss.append(history.history["loss"])
        train_acc.append(history.history["acc"])
        if "val_loss" in history.history:
            val_loss.append(history.history["val_loss"])
            val_acc.append(history.history["val_acc"])

    # Compute means over all trials
    np_train_loss = np.array(train_loss).mean(axis=0)
    np_train_acc = np.array(train_acc).mean(axis=0)
    if len(val_loss) > 0:
        np_val_loss = np.array(val_loss).mean(axis=0)
        np_val_acc = np.array(val_acc).mean(axis=0)
    else:
        np_val_loss = np_val_acc = np.array([])

    # Compute overall min, max, mean for validation accuracy
    if len(val_loss) > 0:
        np_val_acc_finals = np.array(val_acc)[:,-1] # last value from each trial
        val_acc_min  = np_val_acc_finals.min()
        val_acc_mean = np_val_acc_finals.mean()
        val_acc_max  = np_val_acc_finals.max()
        print()
        print("> Validation Accuracy over all trials <".center(80, '='))
        print("validation accuracy min:  %7.4f" % val_acc_min)
        print("validation accuracy mean: %7.4f" % val_acc_mean)
        print("validation accuracy max:  %7.4f" % val_acc_max)
        print(80*'=')
    else:
        val_acc_min = val_acc_mean = val_acc_max = None

    # Plot loss and accuracy over the trials
    heading = "Keras MLP: %s:Lin, %d:%s, 10:%s, 5:Softmax; epochs=%d" % \
        (input_type, num_h1_units, h1_activation, h2_activation, num_epochs_per_trial)
    if val_acc_min != None and val_acc_mean != None and val_acc_max != None:
        subheading = "validation accuracy: %7.4f %7.4f %7.4f" % \
            (val_acc_min, val_acc_mean, val_acc_max)
    else:
        subheading = ""
    p3_utils.plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
        heading, subheading)

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    # Set parameters for this set of trials
    # input types:  'one hot', 'count-hot', 'td-idf hot', 'word index',
    #               'awv', 'rsv', 'awv+rsv', 'pos', 'awv+pos'
    input_type = 'awv'
    num_cross_validation_trials = 10
    num_epochs_per_trial = 40
    num_h1_units = 60
    h1_activation = 'relu'
    h2_activation = 'relu'
    h1_h2_dropout_rate = 0.5

    num_epochs_for_training = 20    # ... when training on full training set

    print("Input type: %s" % input_type)
    print("Number of cross-validation trials: %d" % num_cross_validation_trials)
    print("Number of epochs per trial: %d" % num_epochs_per_trial)
    print("Number of units in first hidden layer: %d" % num_h1_units)
    print("Activation function for first hidden layer: %s" % h1_activation)
    print("Activation function for second hidden layer: %s" % h2_activation)
    print("Dropout rate for hidden layers: %f" % h1_h2_dropout_rate)
    print()
    print("Number of epochs for full training set: %d" % num_epochs_for_training)
    print()

    xval_sets, test_reviews, test_data = get_data(input_type, num_cross_validation_trials)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    trial_parameters = (input_type, num_h1_units, \
        h1_activation, h2_activation, num_epochs_for_training)

    scores = run_trials(xval_sets, num_cross_validation_trials, num_epochs_per_trial, \
            num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate)
    print_results_of_trials(scores, trial_parameters)

    train_data, train_labels = p3_utils.assemble_full_training_data(xval_sets)

    model, history = \
        run_one_trial(train_data, train_labels, [], [], num_epochs_for_training, \
            num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate)
    print_results_of_trials([ history ], trial_parameters)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save("data/p3_tf_MLP_model_" + timestamp + ".h5")

    np_test_data = np.array(test_data)
    print("Test data shape: %s" % str(np_test_data.shape))
    test_labels = model.predict(np_test_data)
    test_ratings = [ np.argmax(predictions) for predictions in test_labels  ]
    print("Test labels shape: %s" % str(test_labels.shape))
    print("Test labels: %s" % test_labels[:5])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    outDir = "tests"
    outFilename = "test_data_%s_%d-%s_%d-%s_%d-%s" % \
        (input_type, num_h1_units, h1_activation, 10, h2_activation, 5, "SOFTMAX")
    p3_utils.write_test_set_with_ratings(outDir, outFilename, \
        test_reviews, test_ratings)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
