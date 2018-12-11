import os
from datetime import datetime

import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.regularizers import l1

from sarcastic_splits import get_features_for_split
from mlp_metrics import Metrics

OUTPUT_PATH = "../output"
TRAIN_PATH = "../resources/train_data.txt"
TEST_PATH = "../resources/test_data.txt"
THRESHOLD = 1

# Defaults
H1U = 60        # Hidden Layer 1 number of units
H1F = 'relu'    # Hidden Layer 1 activation function
H2U = 10        # Hidden Layer 2 number of units
H2F = 'relu'    # Hidden Layer 2 activation function
NC = 1          # Number of classes
DO = 0.0        # Dropout rate

EPOCHS = 10     # Number of training epochs
TRIALS = 10     # Number of cross-validation trials

def mlp_model(input_shape, \
        num_h1_units=H1U, h1_activation=H1F, num_h2_units=H2U, h2_activation=H2F, \
        num_classes=NC, input_dropout_rate=DO, dropout_rate=DO):
    """
    Creates a TF Keras Multi-Layer Perceptron model,
    with an input layer, two hidden layers, and an output layer.
    """

    input_dim = input_shape[0]
    print("--- Model ---")
    print("Input    : %d x %d" % (1, input_dim))
    print("Layer h1 : %d x %d %s" % (input_dim, num_h1_units, h1_activation))
    print("Layer h2 : %d x %d %s" % (num_h1_units, num_h2_units, h2_activation))
    print("Output   : %d x %d SIGMOID" % (num_h2_units, num_classes))
    print("----------")

    model = models.Sequential()
    # input layer
    model.add(Dropout(rate=input_dropout_rate, input_shape=input_shape))
    # h1: hidden layer 1
    model.add(Dense(units=num_h1_units, activation=h1_activation))
    model.add(Dropout(rate=dropout_rate))
    # h2: hidden layer 2
    model.add(Dense(units=num_h2_units, activation=h2_activation))
    model.add(Dropout(rate=dropout_rate))
    # output layer
    model.add(Dense(units=num_classes, activation='sigmoid'))

    return model

def mlp_train(model, data, epochs=EPOCHS):
    """
    Train the Multi-Layer Perceptron.
    Inputs:
        data - training data and labels, evaluation data and labels
    """
    (train_data, train_labels), validation_data = data

    if validation_data != None:
        val_data, val_labels = validation_data
        metrics = Metrics(val_data, val_labels)
        callbacks = [ metrics ]
    else:
        metrics = None
        callbacks = None

    loss = 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['mse', 'accuracy'])

    history = model.fit(train_data, train_labels, \
        validation_data=validation_data, \
        epochs=epochs, batch_size=32, \
        callbacks=callbacks)

    if metrics != None:
        history.history["val_f_score"] = []
        history.history["val_pearson_r"] = []
        history.history["val_pearson_p"] = []
        for iEpoch in range(epochs):
            f_score = metrics.val_f1s[iEpoch]
            pearson_r = metrics.val_pearson_rs[iEpoch]
            pearson_p = metrics.val_pearson_ps[iEpoch]
            history.history["val_f_score"] += [ f_score ]
            history.history["val_pearson_r"] += [ pearson_r ]
            history.history["val_pearson_p"] += [ pearson_p ]

    return history, model

def run_one_trial(data, num_epochs_per_trial=EPOCHS,
        num_h1_units=H1U, h1_activation=H1F, num_h2_units=H2U, h2_activation=H2F, \
        h1_h2_dropout_rate=DO):

    (train_data, train_labels), validation_data = data

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    model = mlp_model(input_shape=train_data.shape[1:], \
                      num_h1_units=num_h1_units, h1_activation=h1_activation, \
                      num_h2_units=num_h2_units, h2_activation=h2_activation, \
                      input_dropout_rate=h1_h2_dropout_rate, \
                      dropout_rate=h1_h2_dropout_rate)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    history, model = mlp_train(model, data, epochs=num_epochs_per_trial)

    return history, model

def train_and_validate_mlp(train_data, train_labels, test_data, test_labels, \
        num_cross_validation_trials=TRIALS, num_epochs_per_trial=EPOCHS, \
        num_h1_units=H1U, h1_activation=H1F, num_h2_units=H2U, h2_activation=H2F, \
        h1_h2_dropout_rate=DO):

    trial_data = ((train_data, train_labels), None)
    history, model = run_one_trial(trial_data, \
        num_epochs_per_trial=num_epochs_per_trial, \
        num_h1_units=num_h1_units, h1_activation=h1_activation, \
        num_h2_units=num_h2_units, h2_activation=h2_activation, \
        h1_h2_dropout_rate=h1_h2_dropout_rate)

    predictions = model.predict(test_data)
    predicted_labels = np.round(np.transpose(predictions)[0]).astype(int)
    print('test', test_labels[:10], test_labels[-10:])
    print('pred', predicted_labels[:10], predicted_labels[-10:])
    mse = mean_squared_error(test_labels, predicted_labels)
    f_score = f1_score(test_labels, predicted_labels)
    pearson_r, pearson_p = pearsonr(test_labels, predicted_labels)
    return mse, (pearson_r, pearson_p), f_score

def cross_validate_mlp(data, labels, tweets, \
        num_cross_validation_trials=TRIALS, num_epochs_per_trial=EPOCHS, \
        num_h1_units=H1U, h1_activation=H1F, num_h2_units=H2U, h2_activation=H2F, \
        h1_h2_dropout_rate=DO):
    np_data = np.array(data)
    np_labels = np.array(labels)
    kfold = KFold(num_cross_validation_trials, True, 1)

    scores = []
    for trial_index, (train, val) in enumerate(kfold.split(np_data)):

        nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        print("====" + nowStr + "====")

        print((" Trial %d of %d" % (trial_index+1, num_cross_validation_trials)).center(80, '-'))

        _data, _labels, _val_data, _val_labels = \
            get_features_for_split(data[train], labels[train], tweets[train],
                data[val], labels[val], tweets[val])

        trial_data = ((_data, _labels), (_val_data, _val_labels))
        history, model = run_one_trial(trial_data, \
            num_epochs_per_trial=num_epochs_per_trial, \
            num_h1_units=num_h1_units, h1_activation=h1_activation, \
            num_h2_units=num_h2_units, h2_activation=h2_activation, \
            h1_h2_dropout_rate=h1_h2_dropout_rate)
        scores.append(history)
    print_results_of_trials(scores)

def print_results_of_trials(scores):
    # Collect trial results:
    #   loss, and accuracy or mean squared error for training and evaluation
    train_loss = []
    train_acc  = []
    train_mse  = []
    val_loss = []
    val_acc  = []
    val_mse =  []
    val_f_score = []
    val_pearson_r = []
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
        if "val_f_score" in history.history:
            val_f_score.append(history.history["val_f_score"])
        if "val_pearson_r" in history.history:
            val_pearson_r.append(history.history["val_pearson_r"])

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
    if len(val_f_score) > 0:
        np_val_f_score = np.array(val_f_score).mean(axis=0)
    if len(val_pearson_r) > 0:
        np_val_pearson_r = np.array(val_pearson_r).mean(axis=0)

    # Compute overall min, max, mean for validation mean squared error
    if len(val_mse) > 0:
        np_val_mse_finals = np.array(val_mse)[:,-1] # last value from each trial
        val_mse_min  = np_val_mse_finals.min()
        val_mse_mean = np_val_mse_finals.mean()
        val_mse_max  = np_val_mse_finals.max()
        print("> Validation mean squared error over all trials <".center(80, '='))
        print("validation mean squared error min:  %7.4f" % val_mse_min)
        print("validation mean squared error mean: %7.4f" % val_mse_mean)
        print("validation mean squared error max:  %7.4f" % val_mse_max)
        print(80*'=')
    else:
        val_mse_min = val_mse_mean = val_mse_max = None

    # Compute overall min, max, mean for validation f-score
    if len(val_f_score) > 0:
        np_val_f_score_finals = np.array(val_f_score)[:,-1] # last value from each trial
        val_f_score_min  = np_val_f_score_finals.min()
        val_f_score_mean = np_val_f_score_finals.mean()
        val_f_score_max  = np_val_f_score_finals.max()
        print("> Validation F-score over all trials <".center(80, '=') + "\n")
        print("validation f-score min:  %7.4f" % val_f_score_min)
        print("validation f-score mean: %7.4f" % val_f_score_mean)
        print("validation f-score max:  %7.4f" % val_f_score_max)
        print(80*'=')
    else:
        val_f_score_min = val_f_score_mean = val_f_score_max = None

    # Compute overall min, max, mean for validation pearson correlation
    if len(val_pearson_r) > 0:
        np_val_pearson_r_finals = np.array(val_pearson_r)[:,-1] # last value from each trial
        val_pearson_r_min  = np_val_pearson_r_finals.min()
        val_pearson_r_mean = np_val_pearson_r_finals.mean()
        val_pearson_r_max  = np_val_pearson_r_finals.max()
        print("> Validation Pearson Correlation Coefficient over all trials <".center(80, '=') + "\n")
        print("validation pearson coefficient min:  %7.4f" % val_pearson_r_min)
        print("validation pearson coefficient mean: %7.4f" % val_pearson_r_mean)
        print("validation pearson coefficient max:  %7.4f" % val_pearson_r_max)
        print(80*'=')
    else:
        val_pearson_r_min = val_pearson_r_mean = val_pearson_r_max = None

    # Compute overall min, max, mean for validation accuracy
    if len(val_acc) > 0:
        np_val_acc_finals = np.array(val_acc)[:,-1] # last value from each trial
        val_acc_min  = np_val_acc_finals.min()
        val_acc_mean = np_val_acc_finals.mean()
        val_acc_max  = np_val_acc_finals.max()
        print("> Validation Accuracy over all trials <".center(80, '=') + "\n")
        print("validation accuracy min:  %7.4f" % val_acc_min)
        print("validation accuracy mean: %7.4f" % val_acc_mean)
        print("validation accuracy max:  %7.4f" % val_acc_max)
        print(80*'=')
    else:
        val_acc_min = val_acc_mean = val_acc_max = None

def predict_and_output_ratings(model, input_vectors, tweets):
    np_input_vectors = np.array(input_vectors)
    test_labels = model.predict(np_input_vectors)
    test_ratings = [ np.argmax(predictions) for predictions in test_labels  ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUTPUT_PATH, "labels.txt" + timestamp), 'w') as f:
        for i, tweet in enumerate(tweets):
            print("%s|%d" % (tweet, test_ratings[i]))
# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

# if __name__ == '__main__':
#
#     nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
#     print("====" + nowStr + "====")
#
#     # Set parameters for this set of trials
#     num_cross_validation_trials = 10
#     num_epochs_per_trial = 10
#     num_h1_units = 5
#     h1_activation = 'relu'
#     num_h2_units = 10
#     h2_activation = 'relu'
#     h1_h2_dropout_rate = 0.5
#
#     one_hots, ratings = utils.load_data(TRAIN_PATH)
#
#     run_trials(one_hots, ratings, \
#        num_cross_validation_trials=num_cross_validation_trials, \
#        num_epochs_per_trial=num_epochs_per_trial, \
#        num_h1_units=num_h1_units, h1_activation=h1_activation, \
#        num_h2_units=num_h2_units, h2_activation=h2_activation, \
#        h1_h2_dropout_rate=h1_h2_dropout_rate)
#
