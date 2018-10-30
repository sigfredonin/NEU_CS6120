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

def mlp_model(input_shape, h1_units, \
              num_classes=5, dropout_rate=0.0, input_dropout_rate=0.0):
    """
    Creates a TF Keras Multi-Layer Perceptron model,
    with an input layer, two hidden layers, and an output layer.
    """
    input_dim = input_shape[0]
    print("--- Model ---")
    print("Input    : %d x %d" % (1, input_dim))
    print("Layer h1 : %d x %d" % (input_dim, h1_units))
    print("Layer h2 : %d x %d" % (h1_units, 10))
    print("Output   : %d x %d" % (10, 5))
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

def mlp_train(model, data):
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
        epochs=10, batch_size=32)

    score = model.evaluate(val_data, val_labels, batch_size=32)

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

    count_hots = p3_utils.count_vectors(review_data, vocabulary_size)
    print("Count count vectors: %d" % len(count_hots))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    shuffle_indices, train_data, train_labels, val_data, val_labels = \
        p3_utils.split_data(count_hots, review_labels)

    np_train_data = np.array(train_data)
    np_train_labels = np.array(train_labels)
    np_val_data = np.array(val_data)
    np_val_labels = np.array(val_labels)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    input_width = np_train_data.shape[1]
    model = mlp_model(input_shape=np_train_data.shape[1:],
                      h1_units=120, dropout_rate=0.5)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    data = ((np_train_data, np_train_labels), (np_val_data, np_val_labels))
    score = mlp_train(model, data)
    print("Scores with validation data ---")
    for i, item in enumerate(score):
        print("%s: %s" % (model.metrics_names[i], str(score[i])))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
