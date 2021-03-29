#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing tap delay line input for b31xp
"""

import os
import argparse

import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import accuracy_score

from models import compile_model
import preprocessing as pre

# suppress TensorFlow logs (Works for Linux, may need changed otherwise)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_1 = 'POF60m_PAMExp_2PAM_DR600Mbps.mat'
DATA_2 = 'PAM_OS4_3_6dbm_26022021.csv'
DATA_3 = 'UWOC27m_PAM4_63Mb_APDGain80_P7dBm.mat'
DATA_4 = 'UWOC27m_PAM4_125Mb_APDGain80_P7dBm.mat'
DATA_5 = 'UWOC27m_PAM8_94Mb_APDGain100_P10dBm.mat'
DATA_6 = 'UWOC27m_PAM8_188Mb_APDGain100_P10dBm.mat'
DATA_7 = 'UWOC27m_PAM16_125Mb_APDGain100_P13dBm.mat'
DATA_8 = 'UWOC27m_PAM16_250Mb_APDGain100_P13dBm.mat'

# =============================================================================
# PARSE CLI ARGS
# =============================================================================

# parse arguments from CLI
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--signal', dest='signal',
                    help='select type data (signal or symbol)',
                    type=str, default="symbol")
parser.add_argument('-df', '--data-set', dest='data',
                    help="Provide a different data set for training/testing",
                    type=int, default=2)
parser.add_argument('-e', '--experiment', dest='experiment',
                    help="Variable used in experiment",
                    type=str, default="n/a")
parser.add_argument('-m', '--model', dest='model',
                    help="Select a model to train",
                    type=str, default="mlp_binary")
parser.add_argument('-v', '--verbose', dest='verbose',
                    help="Run the program in verbose mode",
                    action="store_true", default=False)
parser.add_argument('-sh', '--save-headers', dest='save_headers',
                    help="Will save column headers to csv",
                    action="store_true", default=False)
parser.add_argument('-ns', '--no-save', dest='no_save',
                    help="Don't save results to file",
                    action="store_true", default=False)
parser.add_argument('-ss', '--subsequence-size', dest='subsequence_size',
                    help="Define size of subsequence",
                    type=int, default=4)
parser.add_argument('-bs', '--batch-size', dest='batch_size',
                    help="Define batch size",
                    type=int, default=32)
parser.add_argument('-lr', '--learning_rate', dest='learning_rate',
                    help="Set a different learning rate",
                    type=float, default=0.001)
parser.add_argument('-ep', '--epochs', dest='epochs',
                    help="Set a different number of epochs",
                    type=int, default=100)
parser.add_argument('-sa', '--sample', dest='sample',
                    help="Set sample size",
                    type=int, default=20000)
parser.add_argument('-a', '--architecture', nargs='+', dest='architecture',
                    help="number of nodes in each layer (list)",
                    type=int, default=0)
parser.add_argument('-dr', '--dropout_rate', nargs='+', dest='dropout_rate',
                    help="Set a different drop out rate (list)",
                    type=float, default=0.0)
args = parser.parse_args()

# =============================================================================
# GLOBALS
# =============================================================================
TYPE = args.signal.lower()
IS_SIGNAL = True if TYPE == 'signal' else False
SUBSEQUENCE_SIZE = args.subsequence_size
SAMPLE_S = args.sample
TEST_S = 0.2
BATCH_SIZE = args.batch_size
VERBOSE = args.verbose
SAVE_HEADERS = args.save_headers
NO_SAVE = args.no_save
MODEL = args.model
ARCHITECTURE = args.architecture
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
DROPOUT_RATE = args.dropout_rate

NUM_CLASSES = 4
if args.data == 1: # first data set (-1, 1)
    TX_LABELS = {'-1.0': 0, '1.0': 1}
    TX_VALUES = {'0': -1.0, '1': 1.0}
    NUM_CLASSES = 2
elif args.data == 2: # second data set (-8191, -2730, 2730, 8191)
    TX_LABELS = {'-8191': 0, '-2730': 1, '2730': 2, '8191': 3}
    TX_VALUES = {'0': -8191, '1': -2730, '2': 2730, '3': 8191}
elif args.data == 3 or args.data == 4: # third data set (PAM4)
    if IS_SIGNAL:
        TX_LABELS = {'-1.0': 0, '-0.3333333333333333': 1, '0.0': 2, '0.3333333333333333': 3, '1.0': 4}
        TX_VALUES = {'0': -1.0, '1': -0.3333, '2': 0.0, '3': 0.3333, '4': 1.0}
        NUM_CLASSES = 5
    else:
        TX_LABELS = {'-3': 0, '-1': 1, '1': 2, '3': 3}
        TX_VALUES = {'0': -3, '1': -1, '2': 1, '3': 3}
elif args.data == 5 or args.data == 6: # third data set (PAM8)
    if IS_SIGNAL:
        TX_LABELS = {'-1.0': 0, '-0.7142857142857143': 1, '-0.42857142857142855': 2, '-0.14285714285714285': 3, '0.0': 4,
                     '0.14285714285714285': 5, '0.42857142857142855': 6, '0.7142857142857143': 7, '1.0': 8}
        TX_VALUES = {'0': -1.0, '1': -0.7142857142857143, '2': -0.42857142857142855, '3': -0.14285714285714285, '4': 0.0,
                     '5': 0.14285714285714285, '6': 0.42857142857142855, '7': 0.7142857142857143, '8': 1.0}
        NUM_CLASSES = 9
    else:
        TX_LABELS = {'-7': 0, '-5': 1, '-3': 2, '-1': 3, '1': 4, '3': 5, '5': 6, '7': 7}
        TX_VALUES = {'0': -7, '1': -5, '2': -3, '3': -1, '4': 1, '5': 3, '6': 5, '7': 7}
        NUM_CLASSES = 8
elif args.data == 7 or args.data == 8: # third data set (PAM16)
    if IS_SIGNAL:
        TX_LABELS = {'-1.0': 0, '-0.8666666666666667': 1, '-0.7333333333333333': 2, '-0.6': 3,
                     '-0.4666666666666667': 4, '-0.3333333333333333': 5, '-0.2': 6, '-0.06666666666666667': 7,
                     '0.0': 8, '0.06666666666666667': 9, '0.2': 10, '0.3333333333333333': 11,
                     '0.4666666666666667': 12, '0.6': 13, '0.7333333333333333': 14, '0.8666666666666667': 15, '1.0': 16}
        TX_VALUES = {'0': -1.0, '1': -0.8666666666666667, '2': -0.7333333333333333, '3': -0.6,
                     '4': -0.4666666666666667, '5': -0.3333333333333333, '6': -0.2, '7': -0.06666666666666667,
                     '8': 0.0, '9': 0.06666666666666667, '10': 0.2, '11': 0.3333333333333333,
                     '12': 0.4666666666666667, '13': 0.6, '14': 0.7333333333333333, '15': 0.8666666666666667, '16': 1.0}
        NUM_CLASSES = 17
    else:
        TX_LABELS = {'-15': 0, '-13': 1, '-11': 2, '-9': 3,
                     '-7': 4, '-5': 5, '-3': 6, '-1': 7,
                     '1': 8, '3': 9, '5': 10, '7': 11,
                     '9': 12, '11': 13, '13': 14, '15': 15}
        TX_VALUES = {'0': -15, '1': -13, '2': -11, '3': -9,
                     '4': -7, '5': -5, '6': -3, '7': -1,
                     '8': 1, '9': 3, '10': 5, '11': 7,
                     '12': 9, '13': 11, '14': 13, '15': 15}
        NUM_CLASSES = 16


def to_label(value):
    return TX_LABELS[str(value)]


def to_value(label):
    return TX_VALUES[str(label)]


if args.data == 1:
    DATA_FILE = f"../data/{DATA_1}"
elif args.data == 2:
    DATA_FILE = f"../data/{DATA_2}"
elif args.data == 3:
    DATA_FILE = f"../data/{DATA_3}"
elif args.data == 4:
    DATA_FILE = f"../data/{DATA_4}"
elif args.data == 5:
    DATA_FILE = f"../data/{DATA_5}"
elif args.data == 6:
    DATA_FILE = f"../data/{DATA_6}"
elif args.data == 7:
    DATA_FILE = f"../data/{DATA_7}"
elif args.data == 8:
    DATA_FILE = f"../data/{DATA_8}"


IS_BINARY = False
if 'binary' in MODEL:
    IS_BINARY = True

if args.experiment != 'n/a':
    # experiment name formatted wrt experiment variable provided
    experiment_str = args.experiment.title().replace('_', ' ')
    experiment_val = eval(args.experiment)
    EXPERIMENT = f'{experiment_str}: {experiment_val}'

# =============================================================================
# PRE-PROCESSING
# =============================================================================

# load data
tx, rx = pre.load_data(DATA_FILE, SAMPLE_S, verbose=VERBOSE,
                       is_signal=IS_SIGNAL)


tx = np.array([to_label(x) for x in tx])

raw_df = pd.DataFrame()
raw_df['tx'] = tx
raw_df['rx'] = rx

if VERBOSE:
    print(tabulate(raw_df[:17], headers='keys', tablefmt='psql'))
    print("------------------------------------------------------------------")

# subsequence data
tx, rx = pre.subsequence(rx, tx, SUBSEQUENCE_SIZE, verbose=VERBOSE)

if VERBOSE:
    data_df = pre.summarise_data(rx, tx, SUBSEQUENCE_SIZE)
    print(tabulate(data_df[:10], headers='keys', tablefmt='psql'))

# split data
rx_train, rx_test, tx_train, tx_test = pre.test_split(rx, tx,
                                                      test_size=TEST_S)

# =============================================================================
# MODEL
# =============================================================================

# compile and fit model
model = compile_model(MODEL, ARCHITECTURE, LEARNING_RATE, DROPOUT_RATE, NUM_CLASSES)
model.fit(rx_train, tx_train, epochs=EPOCHS,
          batch_size=BATCH_SIZE, validation_split=0.2, verbose=VERBOSE)

# =============================================================================
# EVALUATE
# =============================================================================

# get predictions from test set
preds = model.predict(rx_test)
confidence = np.zeros(len(preds))

# save confidence
confidence = [np.max(instance) for instance in preds]
# select bit with highest probability
preds = [to_value(np.argmax(instance)) for instance in preds]
# reformat ground truths to bipolar binary
tx_test = list(np.array([to_value(x) for x in tx_test]))

preds_str = [str(x) for x in preds]
tx_test_str = [str(x) for x in tx_test]

# calculate accuracy

accuracy = accuracy_score(tx_test_str, preds_str)
# calculate bit error rate
ber = 1.0 - accuracy

# dataframe for saving to csv
results_df = pd.DataFrame()
results_df['predictions'] = preds
results_df['targets'] = tx_test
results_df['confidence'] = confidence
results_df['accuracy'] = accuracy
results_df['ber'] = ber
results_df['experiment'] = EXPERIMENT
results_df['number_params'] = model.count_params()
results_df['sequence_position'] = np.arange(results_df.shape[0])

if VERBOSE:
    print("------------------------------------------------------------------")
    print(tabulate(results_df[:10], headers='keys', tablefmt='psql'))
    print("------------------------------------------------------------------")
    print(model.summary())
    print("------------------------------------------------------------------")
    print(f"Accuracy: {accuracy:.2%}")
    print("------------------------------------------------------------------")

# save to dataframe csv
if not NO_SAVE:
    results_df.to_csv('../results/current_test.csv', mode='a',
                      index=False, header=SAVE_HEADERS)
