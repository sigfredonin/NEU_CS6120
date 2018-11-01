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
import matplotlib.pyplot as plt

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import p3_utils

def mlp_model(input_shape, h1_units, \
              num_classes=5, dropout_rate=0.0, input_dropout_rate=0.0):
    """
    Creates a TF Keras Multi-Layer Perceptron model,
    with an input layer, two hidden layers, and an output layer.
    """
    input_dim = input_shape[0]
    print("--- Model ---")
    print("Input    : %d x %d" % (1, input_dim))
    print("Layer h1 : %d x %d RELU" % (input_dim, h1_units))
    print("Layer h2 : %d x %d RELU" % (h1_units, 10))
    print("Output   : %d x %d SOFTMAX" % (10, 5))
    print("----------")
    model = models.Sequential()
    # input layer
    model.add(Dropout(rate=input_dropout_rate, input_shape=input_shape))
    # h1: hidden layer 1
    model.add(Dense(units=h1_units, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    # h2: hidden layer 2
    model.add(Dense(units=10, activation='relu'))
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

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    filePath = "data/a2_p3_train_data.txt"
    text = p3_utils.get_text_from_file(filePath)
    print("Read %d bytes from '%s' as text" % (len(text), filePath))
    print("Text begins : '%s'" % text[:30])

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    words, review_words, review_data, review_labels, \
        fd_words, vocabulary, dictionary, reverse_dictionary = \
        p3_utils.get_words_and_ratings(text)
    vocabulary_size = len(vocabulary)
    print("Vocabulary size: %d" % vocabulary_size)
    print("Number of reviews: %d" % len(review_words))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    one_hots = p3_utils.one_hot_vectors(review_data, vocabulary_size)
    print("Count count vectors: %d" % len(one_hots))

    count_hots = p3_utils.count_vectors(review_data, vocabulary_size)
    print("Count count vectors: %d" % len(count_hots))

    tdIdf_hots = p3_utils.tdIdf_vectors(review_data, vocabulary_size)
    print("Count tdIdf vectors: %d" % len(tdIdf_hots))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    shuffle_indices, xval_sets = \
        p3_utils.split_training_data_for_cross_validation(tdIdf_hots, review_labels)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    num_cross_validation_trials = len(xval_sets)
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

        model = mlp_model(input_shape=np_train_data.shape[1:],
                          h1_units=60, dropout_rate=0.5)

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        num_epochs_per_trial = 20
        history = mlp_train(model, data, epochs=num_epochs_per_trial)
        scores.append(history)

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

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
    bp_train_acc = np.array(train_acc).mean(axis=0)
    np_val_loss = np.array(val_loss).mean(axis=0)
    bp_val_acc = np.array(val_acc).mean(axis=0)

    # Plot loss and accuracy over the trials
    plt.figure(1)
    plt.plot(np_train_loss, 'r--')
    plt.plot(bp_train_acc, 'r')
    plt.plot(np_val_loss, 'b--')
    plt.plot(bp_val_acc, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss / Avg Acc')
    plt.legend(['Training Loss', 'Training Accuracy', \
        'Validation Loss', 'Validation Accuracy'], loc='upper left')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('tests/p3_tf_MLP_test' + timestamp + '.png')

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
