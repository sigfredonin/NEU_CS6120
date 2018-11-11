"""
CS 6120 Natural Language Processing
Northeastern University
Fall 2018

Assignment 2 Problem 4
Summary Evaluation

Input files: a2_p4_train_set.csv, a2_p4_test_set.csv

Sig Nin
November 10, 2018
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import p4_utils

def mlp_model(input_shape, h1_units, h1_activation='relu', h2_activation='relu', \
              output_activation='sigmoid', dropout_rate=0.0, input_dropout_rate=0.0):
    """
    Creates a TF Keras Multi-Layer Perceptron model,
    with an input layer, two hidden layers, and an output layer.
    """
    input_dim = input_shape[0]
    print("--- Model ---")
    print("Input    : %d x %d" % (1, input_dim))
    print("Layer h1 : %d x %d %s" % (input_dim, h1_units, h1_activation))
    print("Layer h2 : %d x %d %s" % (h1_units, 10, h2_activation))
    print("Output   : %d x %d %s" % (10, 1, output_activation))
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
    model.add(Dense(units=1, activation=output_activation))
    return model

def mlp_train(model, data, epochs=10):
    """
    Train the Multi-Layer Perceptron.
    Inputs:
        data - training data and labels, evaluation data and labels
    """
    (train_data, train_labels), (val_data, val_labels) = data

    if len(val_data) == 0 or len(val_labels) == 0:
        validation_data = None
    else:
        validation_data = (val_data, val_labels)

    loss = 'mean_squared_error'
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]

    history = model.fit(train_data, train_labels, \
        validation_data = validation_data, \
        epochs=epochs, batch_size=32)

    return history

def get_data():
    """
    Load the datasets, training and testing.
    Load the Google News word embedding vectors.
    Preprocess the data to calculate the feature vectors.
    """
    ds_train = p4_utils.load_summary_training_data()
    ds_test  = p4_utils.load_summary_test_data()

    train_summmaries, train_non_redundancies, train_fluencies = ds_train
    test_summmaries, test_non_redundancies, test_fluencies = ds_test

    v = vectors = p4_utils.load_embeddings_gensim()

    train_features = [ p4_utils.get_non_redundancy_features(v, s) \
                       for s in train_summaries ]
    test_features  = [ p4_utils.get_non_redundancy_features(v, s) \
                       for s in test_summaries ]
