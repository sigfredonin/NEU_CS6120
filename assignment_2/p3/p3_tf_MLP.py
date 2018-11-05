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

def mlp_model(input_shape, h1_units, h1_activation='relu', h2_activation='relu', \
              num_classes=5, dropout_rate=0.0, input_dropout_rate=0.0):
    """
    Creates a TF Keras Multi-Layer Perceptron model,
    with an input layer, two hidden layers, and an output layer.
    """
    input_dim = input_shape[0]
    print("--- Model ---")
    print("Input    : %d x %d" % (1, input_dim))
    print("Layer h1 : %d x %d %s" % (input_dim, h1_units, h1_activation))
    print("Layer h2 : %d x %d %s" % (h1_units, 10, h2_activation))
    print("Output   : %d x %d SOFTMAX" % (10, 5))
    print("----------")
    model = models.Sequential()
    # input layer
    model.add(Dropout(rate=input_dropout_rate, input_shape=input_shape))
    # h1: hidden layer 1
    model.add(Dense(units=h1_units, activation=h1_activation))
    model.add(Dropout(rate=dropout_rate))
    # h2: hidden layer 2
    model.add(Dense(units=10, activation=h2_activation))
    model.add(Dropout(rate=dropout_rate))
    # output layer
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def mlp_train(model, data, epochs=10):
    """
    Train the Multi-Layer Perceptron.
    Inputs:
        data - training data and labels, evaluation data and labels
    """
    (train_data, train_labels), (val_data, val_labels) = data

    loss = 'sparse_categorical_crossentropy'
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    history = model.fit(train_data, train_labels, \
        validation_data = (val_data, val_labels), \
        epochs=epochs, batch_size=32)

    return history

def train_and_eval(model, data):
    """
    Run a training and validation cycle for given training/evaluation set
    of data and labels.
    """
    return score

def get_data(filePath, input_type, num_cross_validation_trials):
    """
    Load the data and select the correct format.
    Return -

    """
    # Load the text
    text = p3_utils.get_text_from_file(filePath)
    print("Read %d bytes from '%s' as text" % (len(text), filePath))
    print("Text begins : '%s'" % text[:30])

    # Compile reviews and ratings, with word index encoding
    words, review_words, review_data, review_labels, \
        fd_words, vocabulary, dictionary, reverse_dictionary = \
        p3_utils.get_words_and_ratings(text)
    vocabulary_size = len(vocabulary)
    print("Vocabulary size: %d" % vocabulary_size)
    print("Number of reviews: %d" % len(review_words))

    # Load word vectors if going to use them
    if input_type == 'avg embedding':
        vectors, wv_dictionary, wv_reverse_dictionary, \
            wv_review_data, wv_review_vectors, vw_review_sentence_average_vectors \
            = p3_utils.load_embeddings_gensim(fd_words, review_words)

    # Select the input type
    if input_type == 'one hot':
        data = p3_utils.one_hot_vectors(review_data, vocabulary_size)
        print("Count one-hot vectors: %d" % len(data))
    elif input_type == 'count-hot':
        data = p3_utils.count_vectors(review_data, vocabulary_size)
        print("Count count vectors: %d" % len(data))
    elif input_type == 'td-idf hot':
        data = p3_utils.tdIdf_vectors(review_data, vocabulary_size)
        print("Count tdIdf vectors: %d" % len(data))
    elif input_type == 'word index':
        data = review_data
        print("Count word index vectors: %d" % len(data))
    elif input_type == 'avg embedding':
        data = vw_review_sentence_average_vectors
        print("Count embedding vectors: %d" % len(data))
    else:
        raise InvalidArgumentException("Unrecognized data input type: %s" % input_type)

    shuffle_indices, xval_sets = \
        p3_utils.split_training_data_for_cross_validation(data, review_labels, \
            num_cross_validation_trials)

    return xval_sets

def run_trials(xval_sets, num_cross_validation_trials, num_epochs_per_trial, \
        num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate):

    scores = []
    for iTrial in range(num_cross_validation_trials):

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        print((" Trial %d of %d" % (iTrial+1, num_cross_validation_trials)).center(80, '-'))

        (train_data, train_labels), (val_data, val_labels) = \
            p3_utils.assemble_cross_validation_data(xval_sets, iTrial)

        np_train_data = np.array(train_data)
        np_train_labels = np.array(train_labels)
        np_val_data = np.array(val_data)
        np_val_labels = np.array(val_labels)
        data = ((np_train_data, np_train_labels), (np_val_data, np_val_labels))

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        print(">>>>> np_train_data.shape:", np_train_data.shape)

        model = mlp_model(input_shape=np_train_data.shape[1:], \
                          h1_units=num_h1_units, h1_activation=h1_activation, \
                          h2_activation=h2_activation, dropout_rate=h1_h2_dropout_rate)

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        history = mlp_train(model, data, epochs=num_epochs_per_trial)
        scores.append(history)

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

    return scores

def output_results_of_trials(scores, num_epochs_per_trial):

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
        val_loss.append(history.history["val_loss"])
        val_acc.append(history.history["val_acc"])

    # Compute means over all trials
    np_train_loss = np.array(train_loss).mean(axis=0)
    np_train_acc = np.array(train_acc).mean(axis=0)
    np_val_loss = np.array(val_loss).mean(axis=0)
    np_val_acc = np.array(val_acc).mean(axis=0)

    # Compute overall min, max, mean for validation accuracy
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

    # Plot loss and accuracy over the trials
    p3_utils.plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
        val_acc_min, val_acc_mean, val_acc_max, \
        input_type=input_type, h1_units=num_h1_units, \
        h1_f=h1_activation, h2_f=h2_activation, \
        epochs=num_epochs_per_trial)

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    # Set parameters for this set of trials
    input_type = 'avg embedding'
    num_cross_validation_trials = 10
    num_epochs_per_trial = 20
    num_h1_units = 10
    h1_activation = 'relu'
    h2_activation = 'relu'
    h1_h2_dropout_rate = 0.5

    filePath = "data/a2_p3_train_data.txt"
    xval_sets = get_data(filePath, input_type, num_cross_validation_trials)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    scores = run_trials(xval_sets, num_cross_validation_trials, num_epochs_per_trial, \
            num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate)
    output_results_of_trials(scores, num_epochs_per_trial)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
