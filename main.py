#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing tap delay line input for b31xp
"""

import numpy as np
import pandas as pd
import argparse
from tabulate import tabulate

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import accuracy_score


import preprocessing as pre

# =============================================================================
# PARSE CLI ARGS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', dest='experiment',
                    help="Define a custom experiment name",
                    type=str, default="n/a")
parser.add_argument('-v', '--verbose', dest='verbose',
                    help="Run the program in verbose mode",
                    action="store_true", default=False)
parser.add_argument('-sh', '--save-headers', dest='save_headers',
                    help="Will save column headers to csv",
                    type=bool, default=False)
parser.add_argument('-ss', '--subsequence-size', dest='subsequence_size',
                    help="Define size of subsequence",
                    type=int, default=8)
args = parser.parse_args()

# =============================================================================
# GLOBAL
# =============================================================================


MATFILE = './data/POF60m_PAMExp_2PAM_DR600Mbps.mat'
SUBSEQUENCE_SIZE = args.subsequence_size  # 8 good
SAMPLE_S = 20000
TEST_S = 0.2
EPOCHS = 100  # 150 good
BATCH_SIZE = 32
VERBOSE = args.verbose
SAVE_HEADERS = args.save_headers

if args.experiment != 'n/a':
    EXPERIMENT = args.experiment
else:
    EXPERIMENT = f"Subsequence Size: {SUBSEQUENCE_SIZE}"

# =============================================================================
# PRE-PROCESSING
# =============================================================================

# load data
tx, rx = pre.data_from_mat(MATFILE, SAMPLE_S, verbose=VERBOSE)

raw_df = pd.DataFrame()
raw_df['tx'] = tx
raw_df['rx'] = rx

if VERBOSE:
    print(tabulate(raw_df[:17], headers='keys', tablefmt='psql'))
    print("------------------------------------------------------------------")

# subsequence data
tx, rx = pre.subsequence(rx, tx, SUBSEQUENCE_SIZE, verbose=VERBOSE)

data_df = pre.summarise_data(rx, tx, SUBSEQUENCE_SIZE)

if VERBOSE:
    print(tabulate(data_df[:10], headers='keys', tablefmt='psql'))

# split data
rx_train, rx_test, tx_train, tx_test = pre.test_split(rx, tx,
                                                      test_size=TEST_S,
                                                      random_state=42)


# =============================================================================
# MODEL
# =============================================================================

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

# =============================================================================
# EVALUATE
# =============================================================================

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

# add padding to compensate for subsequencing
for _ in range(SUBSEQUENCE_SIZE):
    row = pd.DataFrame({'ground truths (tx)': [0],
                        'binary bipolar predictions': [0],
                        'linear predictions': [0.0]})
    results_df = pd.concat([row, results_df]).reset_index(drop=True)

# calculate accuracy
accuracy = accuracy_score(tx_test, preds_bb)
results_df['accuracy'] = accuracy
results_df['experiment'] = EXPERIMENT
results_df['number_params'] = model.count_params()
results_df['sequence_position'] = np.arange(results_df.shape[0])

#  vis.plot_confusion_matrix(tx_test, preds_bb)

if VERBOSE:
    print("------------------------------------------------------------------")
    print(tabulate(results_df[:10], headers='keys', tablefmt='psql'))
    print("------------------------------------------------------------------")
    print(model.summary())
    print("------------------------------------------------------------------")
    print(conf_mat)
    print("------------------------------------------------------------------")
    print(f"Accuracy: {accuracy}")
    print("------------------------------------------------------------------")

results_df.to_csv('./results/current_test.csv', mode='a',
                  index=False, header=SAVE_HEADERS)
