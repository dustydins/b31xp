#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing tap delay line input for b31xp
"""

import numpy as np
import pandas as pd
from tabulate import tabulate

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import preprocessing as preprocess

#  import matplotlib.pyplot as plt

###############################################################################
# GLOBALS
###############################################################################

MATFILE = './data/POF60m_PAMExp_2PAM_DR600Mbps.mat'
TAP_DELAY = 8  # 8 good
SAMPLE_S = 20000
TEST_S = 0.2
EPOCHS = 150  # 150 good
BATCH_SIZE = 32
VERBOSE = 1

###############################################################################
# PRE-PROCESSING
###############################################################################

# load data
tx, rx = preprocess.data_from_mat(MATFILE, SAMPLE_S)

raw_df = pd.DataFrame()
raw_df['tx'] = tx
raw_df['rx'] = rx

print(tabulate(raw_df[:17], headers='keys', tablefmt='psql'))
print("------------------------------------------------------------------")

# subsequence data
tx, rx = preprocess.subsequence(rx, tx, TAP_DELAY)

data_df = preprocess.summarise_data(rx, tx, TAP_DELAY)
print(tabulate(data_df[:10], headers='keys', tablefmt='psql'))

# split data
rx_train, rx_test, tx_train, tx_test = preprocess.test_split(rx, tx,
                                                             test_size=TEST_S,
                                                             random_state=42)


###############################################################################
# MODEL
###############################################################################

# TODO: create BER loss function

model = Sequential()
model.add(Dense(32, input_dim=rx_train.shape[1],
                kernel_initializer='normal',
                activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # try tanh
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
model.fit(rx_train, tx_train, epochs=EPOCHS,
          batch_size=BATCH_SIZE, validation_split=0.2, verbose=VERBOSE)

###############################################################################
# EVALUATE
###############################################################################

# TODO: tidy this up, create a visualisation module

preds = model.predict(rx_test)
preds_bb = np.array([1 if pred[0] > 0 else -1 for pred in preds])

results_df = pd.DataFrame()
results_df['binary bipolar predictions'] = preds_bb
results_df['ground truths (tx)'] = tx_test


# confusion_matrix
conf_mat = pd.crosstab(results_df['ground truths (tx)'],
                       results_df['binary bipolar predictions'],
                       rownames=['Actual'], colnames=['Predicted'])

results_df['linear predictions'] = preds

# calculate accuracy
accuracy = accuracy_score(tx_test, preds_bb)

cf_matrix = confusion_matrix(tx_test, preds_bb)


def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    """
    Plot confusion matrix
    """
    cm_df = pd.DataFrame(conf_matrix, index=['-1', '1'],
                         columns=['-1', '1'])
    plt.figure(figsize=(16, 16))
    axes = sns.heatmap(cm_df/np.sum(cm_df),
                       annot=True, cmap="Reds", cbar=False,
                       fmt='.2%')
    axes.set(ylabel='Ground Truth (Tx)', xlabel='Predictions')
    axes.set_title(title)
    plt.show()


plot_confusion_matrix(cf_matrix)

print("------------------------------------------------------------------")
print(tabulate(results_df[:10], headers='keys', tablefmt='psql'))
print("------------------------------------------------------------------")
print(model.summary())
print("------------------------------------------------------------------")
print(conf_mat)
print("------------------------------------------------------------------")
print(f"Accuracy: {accuracy}")
print("------------------------------------------------------------------")
