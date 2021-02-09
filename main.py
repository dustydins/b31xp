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

import preprocessing as preprocess

#  import matplotlib.pyplot as plt

###############################################################################
# GLOBALS
###############################################################################

MATFILE = './data/POF60m_PAMExp_2PAM_DR600Mbps.mat'
TAP_DELAY = 8
SAMPLE_S = 20000
TEST_S = 0.2
EPOCHS = 100  # 150 good
BATCH_SIZE = 50
VERBOSE = 1

###############################################################################
# PRE-PROCESSING
###############################################################################

# load data
tx, rx = preprocess.data_from_mat(MATFILE, SAMPLE_S)

# subsequence data
tx, rx = preprocess.subsequence(rx, tx, TAP_DELAY)

# split data
rx_train, rx_test, tx_train, tx_test = preprocess.test_split(rx, tx,
                                                             test_size=TEST_S,
                                                             random_state=42)


###############################################################################
# MODEL
###############################################################################

model = Sequential()
model.add(Dense(64, input_dim=rx_train.shape[1],
                kernel_initializer='normal',
                activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # try tanh

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
model.fit(rx_train, tx_train, epochs=EPOCHS,
          batch_size=BATCH_SIZE, validation_split=0.2, verbose=VERBOSE)

###############################################################################
# EVALUATE
###############################################################################

preds = model.predict(rx_test)
# convert to binary bipolar output (1.0 or -1.0)
preds_bb = np.array([1 if pred[0] > 0 else -1 for pred in preds])
results_df = pd.DataFrame()
results_df['predictions'] = preds_bb
results_df['ground_truth'] = tx_test
confusion_matrix = pd.crosstab(results_df['ground_truth'],
                               results_df['predictions'],
                               rownames=['Actual'], colnames=['Predicted'])


print("------------------------------------------------------------------")
data_df = preprocess.summarise_data(rx_test, tx_test, TAP_DELAY)
data_df["predictions"] = preds
data_df["binary_bipolar_predictions"] = preds_bb
print(tabulate(data_df[:10], headers='keys', tablefmt='psql'))
print("------------------------------------------------------------------")
print(model.summary())
print("------------------------------------------------------------------")
print(confusion_matrix)
print("------------------------------------------------------------------")
