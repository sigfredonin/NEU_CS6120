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
import nltk

from datetime import datetime
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import p4_utils

def get_data(trial_parameters):
    """
    Load the datasets, training and testing.
    Load the Google News word embedding vectors.
    Preprocess the data to calculate the feature vectors.
    """
    output_type, num_cross_validation_trials, num_epochs_per_trial = trial_parameters

    train_dataset = p4_utils.load_summary_training_data()
    test_dataset  = p4_utils.load_summary_test_data()

    train_summaries, train_non_redundancies, train_fluencies = train_dataset
    test_summaries, test_non_redundancies, test_fluencies = test_dataset

    assert(len(train_summaries) == len(train_non_redundancies))
    assert(len(train_summaries) == len(train_fluencies))
    assert(len(test_summaries) == len(test_non_redundancies))
    assert(len(test_summaries) == len(test_fluencies))


    if output_type == 'nonrep' or output_type == 'nonrepQ':
        v = vectors = p4_utils.load_embeddings_gensim()
        train_features = p4_utils.get_non_redundancy_features(v, train_summaries)
        test_features  = p4_utils.get_non_redundancy_features(v, test_summaries)
    elif output_type == 'fluency' or output_type == 'fluencyQ':
        v = None
        train_features = p4_utils.get_fluency_features(train_summaries)
        test_features  = p4_utils.get_fluency_features(test_summaries)

    assert(len(train_summaries) == len(train_features))
    assert(len(test_summaries) == len(test_features))

    data = train_features
    if output_type == 'nonrep':
        labels = train_non_redundancies
    elif output_type == 'fluency':
        labels = train_fluencies
    elif output_type == 'nonrepQ':
        labels = p4_utils.get_class_labels(train_non_redundancies)
    elif output_type == 'fluencyQ':
        labels = p4_utils.get_class_labels(train_fluencies)

    shuffle_indices, xval_sets = \
        p4_utils.split_training_data_for_cross_validation(data, labels, \
            num_cross_validation_trials)

    return xval_sets, v, train_dataset, train_features, test_dataset, test_features

def mlp_model(input_shape, model_parameters):
    """
    Creates a TF Keras Multi-Layer Perceptron model,
    with an input layer, two hidden layers, and an output layer.
    """
    input_dim = input_shape[0]
    num_h1_units, h1_activation, \
        num_h2_units, h2_activation, \
        num_output_units, output_activation, \
        input_dropout_rate, h1_h2_dropout_rate = model_parameters
    print("--- Model ---")
    print("Input    : %d x %d" % (1, input_dim))
    print("Layer h1 : %d x %d %s" % (input_dim, num_h1_units, h1_activation))
    print("Layer h2 : %d x %d %s" % (num_h1_units, num_h2_units, h2_activation))
    print("Output   : %d x %d %s" % (num_h2_units, num_output_units, output_activation))
    print("----------")
    model = models.Sequential()
    # input layer
    model.add(Dropout(rate=input_dropout_rate, input_shape=input_shape))
    # h1: hidden layer 1
    model.add(Dense(units=num_h1_units, activation=h1_activation))
    model.add(Dropout(rate=h1_h2_dropout_rate))
    # h2: hidden layer 2
    model.add(Dense(units=num_h2_units, activation=h2_activation))
    model.add(Dropout(rate=h1_h2_dropout_rate))
    # output layer
    model.add(Dense(units=num_output_units, activation=output_activation))

    model.summary()

    return model

def mlp_train(model, data, trial_parameters):
    """
    Train the Multi-Layer Perceptron.
    Inputs:
        data - training data and labels, evaluation data and labels
    """
    (train_data, train_labels), (val_data, val_labels) = data
    output_type, num_cross_validation_trials, num_epochs_per_trial = trial_parameters

    if len(val_data) == 0 or len(val_labels) == 0:
        validation_data = None
    else:
        validation_data = (val_data, val_labels)

    print("Training   - data: %s labels: %s" % (train_data.shape, train_labels.shape))
    print("Validation - data: %s labels: %s" % (val_data.shape, val_labels.shape))

    if output_type == 'nonrep' or output_type == 'fluency':
        loss = 'mean_squared_error'
        learning_rate = 1e-3
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        metrics = ['mse']
    elif output_type == 'nonrepQ' or output_type == 'fluencyQ':
        loss = 'sparse_categorical_crossentropy'
        learning_rate = 1e-3
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        metrics=['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]

    history = model.fit(train_data, train_labels, \
        validation_data = validation_data, \
        epochs=num_epochs_per_trial, batch_size=32)

    # TODO: Add model.evaluate()

    return history

def run_one_trial(train_data, train_labels, val_data, val_labels, \
        model_parameters, trial_parameters):

    output_type, num_cross_validation_trials, num_epochs_per_trial = trial_parameters

    np_train_data = np.array(train_data)
    np_train_labels = np.array(train_labels)
    np_val_data = np.array(val_data)
    np_val_labels = np.array(val_labels)
    data = ((np_train_data, np_train_labels), (np_val_data, np_val_labels))

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = mlp_model(np_train_data.shape[1:], model_parameters)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    history = mlp_train(model, data, trial_parameters)

    return model, history

def run_trials(xval_sets, model_parameters, trial_parameters):

    output_type, num_cross_validation_trials, num_epochs_per_trial = trial_parameters

    scores = []
    for iTrial in range(num_cross_validation_trials):

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        print((" Trial %d of %d" % (iTrial+1, num_cross_validation_trials)).center(80, '-'))

        (train_data, train_labels), (val_data, val_labels) = \
            p4_utils.assemble_cross_validation_data(xval_sets, iTrial)

        model, history = \
            run_one_trial(train_data, train_labels, val_data, val_labels, \
                model_parameters, trial_parameters)

        scores.append(history)

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

    return scores

def save_results_of_trials(scores, model_parameters, trial_parameters, trial_ID, timestamp):

    output_type, num_cross_validation_trials, num_epochs_per_trial = trial_parameters
    num_h1_units, h1_activation, \
        num_h2_units, h2_activation, \
        num_output_units, output_activation, \
        input_dropout_rate, h1_h2_dropout_rate = model_parameters

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

        # Collect trial results:
        #   loss, and accuracy or mean squared error for training and evaluation
        train_loss = []
        train_acc  = []
        train_mse  = []
        val_loss = []
        val_acc  = []
        val_mse =  []
        for iTrial, history in enumerate(scores):
            if "loss" in history.history:
                train_loss.append(history.history["loss"])
            if "acc" in history.history:
                train_acc.append(history.history["acc"])
            if "mean_squared_error" in history.history:
                train_mse.append(history.history["mean_squared_error"])
            if "val_loss" in history.history:
                val_loss.append(history.history["val_loss"])
            if "val_acc" in history.history:
                val_acc.append(history.history["val_acc"])
            if "val_mean_squared_error" in history.history:
                val_mse.append(history.history["val_mean_squared_error"])

        # Compute means over all trials
        np_train_loss = np_train_acc = np_val_mse = np.array([])
        if len(train_loss) > 0:
            np_train_loss = np.array(train_loss).mean(axis=0)
        if len(train_acc) > 0:
            np_train_acc = np.array(train_acc).mean(axis=0)
        if len(train_mse) > 0:
            np_train_mse = np.array(train_mse).mean(axis=0)

        np_val_loss = np_val_acc = np_val_mse = np.array([])
        if len(val_loss) > 0:
            np_val_loss = np.array(val_loss).mean(axis=0)
        if len(val_acc) > 0:
            np_val_acc = np.array(val_acc).mean(axis=0)
        if len(val_mse) > 0:
            np_val_mse = np.array(val_mse).mean(axis=0)

        # Compute overall min, max, mean for validation mean squared error
        if len(val_mse) > 0:
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

        # Compute overall min, max, mean for validation accuracy
        if len(val_acc) > 0:
            np_val_acc_finals = np.array(val_acc)[:,-1] # last value from each trial
            val_acc_min  = np_val_acc_finals.min()
            val_acc_mean = np_val_acc_finals.mean()
            val_acc_max  = np_val_acc_finals.max()
            f.write("\n")
            f.write("> Validation Accuracy over all trials <".center(80, '=') + "\n")
            f.write("validation accuracy min:  %7.4f\n" % val_acc_min)
            f.write("validation accuracy mean: %7.4f\n" % val_acc_mean)
            f.write("validation accuracy max:  %7.4f\n" % val_acc_max)
            f.write(80*'='+"\n")
        else:
            val_acc_min = val_acc_mean = val_acc_max = None

    # Plot loss and accuracy or mean squared error over the trials
    heading = "Keras MLP: %s:Lin, %d:%s, %d:%s, %d:%s; epochs=%d" % \
        (output_type, num_h1_units, h1_activation, num_h2_units, h2_activation, \
            num_output_units, output_activation, num_epochs_per_trial)
    if val_mse_min != None and val_mse_mean != None and val_mse_max != None:
        subheading = "validation mean squared error: %7.4f %7.4f %7.4f" % \
            (val_mse_min, val_mse_mean, val_mse_max)
    elif val_acc_min != None and val_acc_mean != None and val_acc_max != None:
        subheading = "validation accuracy: %7.4f %7.4f %7.4f" % \
            (val_acc_min, val_acc_mean, val_acc_max)
    else:
        subheading = ""
    if (len(train_mse) > 0):
        plotName='tests/p4_tf_MLP_test_loss_' + trial_ID + timestamp
        p4_utils.plot_results(np_train_loss, np_train_mse, np_val_loss, np_val_mse, \
            heading, subheading, "MSE", plotName)
    if (len(train_acc) > 0):
        plotName='tests/p4_tf_MLP_test_acc_' + trial_ID + timestamp
        p4_utils.plot_results(np_train_loss, np_train_acc, np_val_loss, np_val_acc, \
            heading, subheading, "Accuracy", plotName)

    return outFilePath

def save_model_predictions(trial_type, features, gold_labels, predicted_labels, summaries, \
        outFilePath, model_parameters, trial_parameters, trial_ID, timestamp):

    output_type, num_cross_validation_trials, num_epochs_per_trial = trial_parameters
    num_h1_units, h1_activation, \
        num_h2_units, h2_activation, \
        num_output_units, output_activation, \
        input_dropout_rate, h1_h2_dropout_rate = model_parameters

    print_heading = (" Predictions on %s Set " % trial_type).center(80, '-')

    with open(outFilePath, 'a') as f:
        f.write("%s\n\n" % print_heading)

        line = linregress(gold_labels, predicted_labels)
        f.write("Linear regression analysis, labels vs. predicted labels --\n")
        f.write("%s\n" % str(line))

        r, p = pearsonr(gold_labels, predicted_labels)
        f.write("Pearson Correlation r-value and p-value: %7.4f, %7.4f\n" % (r, p))

        mse = mean_squared_error(gold_labels, predicted_labels)
        f.write("Mean Squared Error: %7.4f\n" % mse)

        f.write("--i- --Features--- --Gold- --Pred- --Summary-----------------------------------\n")
        #           1 [3 3  0.7410]  0.0000  1.0000 UKIP has fallen 18 points behind the Tories
        for i, prediction in enumerate(predicted_labels):
            if output_type == 'nonrep' or output_type == 'nonrepQ':
                _f = "[%d %d %7.4f %d %d]" % features[i]
            elif output_type == 'fluency' or output_type == 'fluencyQ':
                _f = "[%d %d %7.4f %d]" % features[i]
            f.write("%4d %s %7.4f %7.4f %s\n" % (i, _f, gold_labels[i], prediction, summaries[i]))

    # Plot gold labels vs. predicted labels, together with least square fit line
    heading = "Keras MLP: %s:Lin, %d:%s, %d:%s, %d:%s; epochs=%d" % \
        (output_type, num_h1_units, h1_activation, num_h2_units, h2_activation, \
            num_output_units, output_activation, num_epochs_per_trial)
    subheading = "test data pearson r, p; mse: %7.4f %7.4f %7.4f" % (r, p, mse)
    plotName='tests/p4_tf_MLP_test_comp' + trial_ID + timestamp
    p4_utils.plot_compare(gold_labels, predicted_labels, line.slope, line.intercept, \
        heading=heading, subheading=subheading, plotName=plotName)

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    from datetime import datetime

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    # Set parameters for this set of trials
    #   output types -  1 tanh      : nonrep, fluency
    #                   13 softmax  : nonrepQ, fluencyQ
    output_type = 'nonrep'
    num_cross_validation_trials = 10
    num_epochs_per_trial = 60       # ... for 10-fold cross-validatin training
    num_epochs_for_training = 10    # ... for training on the full training set

    # set model parameters for this set of trials
    num_h1_units = 10
    h1_activation = 'relu'
    num_h2_units = 300
    h2_activation = 'relu'
    if output_type == 'nonrep' or output_type == 'fluency':
        num_output_units = 1
        output_activation = 'tanh'
    elif output_type == 'nonrepQ' or output_type == 'fluencyQ':
        num_output_units = 13
        output_activation = 'softmax'
    input_dropout_rate = 0.0
    h1_h2_dropout_rate = 0.5

    model_parameters = \
        num_h1_units, h1_activation, \
        num_h2_units, h2_activation, \
        num_output_units, output_activation, \
        input_dropout_rate, h1_h2_dropout_rate
    model_ID = "%d-%s_%d-%s_%d-%s_(%3.1f,%3.1f)" % model_parameters

    trial_parameters = output_type, num_cross_validation_trials, num_epochs_per_trial

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_ID = "_%s_%d-fold_%d_epochs_" % trial_parameters
    trial_ID += model_ID + '_'

    print("Trial ID: %s" % trial_ID)
    print("  output: %s, %d-fold cross-validation, %d epochs" % trial_parameters)
    print("  H1: %d %s, H2: %d, %s, OUT: %d, %s, dropout: IN=%3.1f OUT=%3.1f" % \
        model_parameters)

    xval_sets, v, train_dataset, train_features, test_dataset, test_features = \
        get_data(trial_parameters)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    scores = run_trials(xval_sets, \
        model_parameters, trial_parameters)
    save_results_of_trials(scores, \
        model_parameters, trial_parameters, trial_ID, timestamp)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    trial_parameters = output_type, num_cross_validation_trials, num_epochs_for_training
    trial_ID = "_%s_full_%d_epochs_" % (output_type, num_epochs_for_training)
    trial_ID += model_ID + '_'

    print("Trial ID: %s" % trial_ID)
    print("  output: %s, %d-fold cross-validation, %d epochs" % trial_parameters)
    print("  H1: %d %s, H2: %d, %s, OUT: %d, %s, dropout: IN=%3.1f OUT=%3.1f" % \
        model_parameters)

    train_data, train_labels = p4_utils.assemble_full_training_data(xval_sets)

    model, history = run_one_trial(train_data, train_labels, [], [], \
        model_parameters, trial_parameters)
    model.save("data/p4_tf_MLP_model_" + trial_ID + timestamp + ".h5")

    outFilePath = save_results_of_trials([ history ], \
        model_parameters, trial_parameters, trial_ID, timestamp)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    train_summaries, train_non_redundancies, train_fluencies = train_dataset
    if output_type == 'nonrep' or output_type == 'nonrepQ':
        _labels = train_non_redundancies
    elif output_type == 'fluency' or output_type == 'fluencyQ':
        _labels = train_fluencies
    train_labels = np.array(_labels)

    np_data = np.array(train_data)
    model_predictions = model.predict(np_data)
    print("Model predictions, training: %s %s" % (model_predictions.shape, model_predictions[0]))
    if output_type == 'nonrep' or output_type == 'fluency':
        predicted_labels = np.array([ predictions[0] for predictions in model_predictions ])
    elif output_type == 'nonrepQ' or output_type == 'fluencyQ':
        predicted_classes = [ np.argmax(p) for p in model_predictions ]
        print("Predicted classes: %s" % predicted_classes[:10])
        predicted_labels = np.array([ p4_utils.VALUES[c] for c in predicted_classes ])
        print("Predicted labels: %s" % predicted_labels[:10])
    save_model_predictions("Training", train_features, train_labels, predicted_labels, train_summaries, \
        outFilePath, model_parameters, trial_parameters, "_TRAIN"+trial_ID, timestamp)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    test_summaries, test_non_redundancies, test_fluencies = test_dataset
    if output_type == 'nonrep' or output_type == 'nonrepQ':
        _labels = test_non_redundancies
    elif output_type == 'fluency' or output_type == 'fluencyQ':
        _labels = test_fluencies
    test_labels = np.array(_labels)

    np_data = np.array(test_features)
    model_predictions = model.predict(np_data)
    print("Model predictions, test: %s %s" % (model_predictions.shape, model_predictions[0]))
    if output_type == 'nonrep' or output_type == 'fluency':
        predicted_labels = np.array([ predictions[0] for predictions in model_predictions ])
    elif output_type == 'nonrepQ' or output_type == 'fluencyQ':
        predicted_classes = [ np.argmax(p) for p in model_predictions ]
        print("Predicted classes: %s" % predicted_classes[:10])
        predicted_labels = np.array([ p4_utils.VALUES[c] for c in predicted_classes ])
        print("Predicted labels: %s" % predicted_labels[:10])
    save_model_predictions("Test", test_features, test_labels, predicted_labels, test_summaries, \
        outFilePath, model_parameters, trial_parameters, "_TEST"+trial_ID, timestamp)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
