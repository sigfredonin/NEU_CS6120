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

from datetime import datetime
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import p4_utils

def get_data(output_type, num_cross_validation_trials):
    """
    Load the datasets, training and testing.
    Load the Google News word embedding vectors.
    Preprocess the data to calculate the feature vectors.
    """
    train_dataset = p4_utils.load_summary_training_data()
    test_dataset  = p4_utils.load_summary_test_data()

    train_summaries, train_non_redundancies, train_fluencies = train_dataset
    test_summaries, test_non_redundancies, test_fluencies = test_dataset

    assert(len(train_summaries) == len(train_non_redundancies))
    assert(len(train_summaries) == len(train_fluencies))
    assert(len(test_summaries) == len(test_non_redundancies))
    assert(len(test_summaries) == len(test_fluencies))

    v = vectors = p4_utils.load_embeddings_gensim()

    train_features = [ p4_utils.get_non_redundancy_features(v, s) \
                       for s in train_summaries ]
    test_features  = [ p4_utils.get_non_redundancy_features(v, s) \
                       for s in test_summaries ]

    assert(len(train_summaries) == len(train_features))
    assert(len(test_summaries) == len(test_features))

    data = train_features
    if output_type == 'nonrep':
        labels = train_non_redundancies
    elif output_type == 'fluency':
        labels = train_fluencies

    shuffle_indices, xval_sets = \
        p4_utils.split_training_data_for_cross_validation(data, labels, \
            num_cross_validation_trials)

    return xval_sets, v, train_dataset, train_features, test_dataset, test_features

def mlp_model(input_shape, h1_units, h1_activation='relu', h2_activation='relu', \
              output_activation='tanh', dropout_rate=0.0, input_dropout_rate=0.0):
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

    # TODO: Add model.evaluate()

    return history

def run_one_trial(train_data, train_labels, val_data, val_labels, num_epochs_per_trial, \
        num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate):

    np_train_data = np.array(train_data)
    np_train_labels = np.array(train_labels)
    np_val_data = np.array(val_data)
    np_val_labels = np.array(val_labels)
    data = ((np_train_data, np_train_labels), (np_val_data, np_val_labels))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = mlp_model(input_shape=np_train_data.shape[1:], \
                      h1_units=num_h1_units, h1_activation=h1_activation, \
                      h2_activation=h2_activation, dropout_rate=h1_h2_dropout_rate)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    history = mlp_train(model, data, epochs=num_epochs_per_trial)

    return model, history

def run_trials(xval_sets, num_cross_validation_trials, num_epochs_per_trial, \
        num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate):

    scores = []
    for iTrial in range(num_cross_validation_trials):

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        print((" Trial %d of %d" % (iTrial+1, num_cross_validation_trials)).center(80, '-'))

        (train_data, train_labels), (val_data, val_labels) = \
            p4_utils.assemble_cross_validation_data(xval_sets, iTrial)

        model, history = \
            run_one_trial(train_data, train_labels, val_data, val_labels, num_epochs_per_trial, \
                num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate)

        scores.append(history)

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

    return scores

def save_results_of_trials(scores, trial_parameters, trial_ID, timestamp):

    input_type, num_h1_units, h1_activation, h2_activation, num_epochs_per_trial = trial_parameters

    outFilePath = "tests/p4_tf_model_trials" + trial_ID + timestamp
    with open(outFilePath, 'w') as f:
        f.write(" Trial Results ".center(80, '=')+"\n")
        for iTrial, history in enumerate(scores):
            f.write("Trial %d of %d ---\n" % (iTrial+1, len(scores)))
            for iEpoch in range(num_epochs_per_trial):
                f.write("%4d: " % (iEpoch+1))
                for history_key in history.history:
                    f.write(" %s: %7.4f" % (history_key, history.history[history_key][iEpoch]))
                f.write("\n")

        # Collect trial results: loss, mean squared error for training and evaluation
        train_loss = []
        train_mse  = []
        val_loss = []
        val_mse =  []
        for iTrial, history in enumerate(scores):
            train_loss.append(history.history["loss"])
            train_mse.append(history.history["mean_squared_error"])
            if "val_loss" in history.history:
                val_loss.append(history.history["val_loss"])
                val_mse.append(history.history["val_mean_squared_error"])

        # Compute means over all trials
        np_train_loss = np.array(train_loss).mean(axis=0)
        np_train_mse = np.array(train_mse).mean(axis=0)
        if len(val_loss) > 0:
            np_val_loss = np.array(val_loss).mean(axis=0)
            np_val_mse = np.array(val_mse).mean(axis=0)
        else:
            np_val_loss = np_val_mse = np.array([])

        # Compute overall min, max, mean for validation mean squared error
        if len(val_loss) > 0:
            np_val_mse_finals = np.array(val_mse)[:,-1] # last value from each trial
            val_mse_min  = np_val_mse_finals.min()
            val_mse_mean = np_val_mse_finals.mean()
            val_mse_max  = np_val_mse_finals.max()
            f.write("\n")
            f.write("> Validation mean squared error over all trials <".center(80, '=')+"\n")
            f.write("validation mean squared error min:  %7.4f\n" % val_mse_min)
            f.write("validation mean squared error mean: %7.4f\n" % val_mse_mean)
            f.write("validation mean squared error max:  %7.4f\n" % val_mse_max)
            f.write(80*'='+"\n")
        else:
            val_mse_min = val_mse_mean = val_mse_max = None

    # Plot loss and mean squared error over the trials
    heading = "Keras MLP: %s:Lin, %d:%s, 10:%s, 5:Softmax; epochs=%d" % \
        (input_type, num_h1_units, h1_activation, h2_activation, num_epochs_per_trial)
    if val_acc_min != None and val_acc_mean != None and val_acc_max != None:
        subheading = "validation accuracy: %7.4f %7.4f %7.4f" % \
            (val_acc_min, val_acc_mean, val_acc_max)
    else:
        subheading = ""
    p4_utils.plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
        heading, subheading)

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    # Set parameters for this set of trials
    output_type = 'nonrep'
    num_cross_validation_trials = 10
    num_epochs_per_trial = 40
    num_h1_units = 10
    h1_activation = 'relu'
    h2_activation = 'relu'
    h1_h2_dropout_rate = 0.5

    num_epochs_for_training = 20    # ... when training on full training set

    xval_sets, v, train_dataset, train_features, test_dataset, test_features = \
        get_data(output_type, num_cross_validation_trials)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    scores = run_trials(xval_sets, num_cross_validation_trials, num_epochs_per_trial, \
            num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_parameters = (input_type, num_h1_units, \
        h1_activation, h2_activation, num_epochs_per_trial)
    trial_ID = "_%s_%d-%s_10-%s_1-tanh_epochs_%s_" % trial_parameters
    save_results_of_trials(scores, trial_parameters, trial_IID, timestamp)

    train_data, train_labels = p4_utils.assemble_full_training_data(xval_sets)

    model, history = \
        run_one_trial(train_data, train_labels, [], [], num_epochs_for_training, \
            num_h1_units, h1_activation, h2_activation, h1_h2_dropout_rate)

    trial_parameters = (input_type, num_h1_units, \
        h1_activation, h2_activation, num_epochs_for_training)
    trial_ID = "_%s_%d-%s_10-%s_1-tanh_epochs_%s_" % trial_parameters
    save_results_of_trials([ history ], trial_parameters, trial_ID, timestamp)

    model.save("data/p4_tf_MLP_model_" + trial_ID + timestamp + ".h5")

    test_summaries, test_non_redundancies, test_fluencies = test_dataset
    if output_type == 'nonrep':
        test_labels = test_non_redundancies
    elif output_type == 'fluency':
        test_labels = test_fluencies

    np_test_data = np.array(test_features)
    print("Test data shape: %s" % str(np_test_data.shape))
    test_predictions = model.predict(np_test_data)
    predicted_test_labels = [ np.argmax(predictions) for predictions in test_predictions  ]
    fd_predicted_test_labels = nltk.FreqDist(predicted_test_labels)
    counts_predicted_test_labels = [ fd_predicted_test_labels[r] for r in range(5)]
    test_corr = linregress(test_labels, predicted_test_labels)
    print("Linear regression analysis, labels vs. predicted labels")
    print(train_corr)
    r, p = pearsonr(test_labels, predicted_test_labels)
    print("Pearson Correlation r-value and p-value: %7.4f, %7.4f" % (r, p))
    mse = mean_squared_error(test_labels, predicted_test_labels)
    print("Mean Squared Error: %7.4f" % mse)
    print("Predicted test ratings distribution: %s" % counts_predicted_test_labels)
    for i, prediction in enumerate(test_predictions[:5]):
        print("%4d %d %s %s" % (i, np.argmax(predictions), prediction, test_reviews[i]))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
