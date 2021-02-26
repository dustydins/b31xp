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

from models import Models
import preprocessing as pre

# suppress TensorFlow logs (Works for Linux, may need changed otherwise)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# PARSE CLI ARGS
# =============================================================================

# parse arguments from CLI
parser = argparse.ArgumentParser()
parser.add_argument('-df', '--data-filename', dest='data',
                    help="Provide a different data set for training/testing",
                    type=str, default="POF60m_PAMExp_2PAM_DR600Mbps.mat")
parser.add_argument('-e', '--experiment', dest='experiment',
                    help="Variable used in experiment",
                    type=str, default="n/a")
parser.add_argument('-m', '--model', dest='model',
                    help="Select a model to train",
                    type=str, default="mlp_linear")
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
                    type=int, default=8)
parser.add_argument('-lr', '--learning_rate', dest='learning_rate',
                    help="Set a different learning rate",
                    type=float, default=0.001)
parser.add_argument('-nh', '--num_hidden', dest='num_hidden',
                    help="Set a different number of hidden layers",
                    type=int, default=2)
parser.add_argument('-nn', '--num_nodes', dest='num_nodes',
                    help="Set a different number of nodes per hidden layer",
                    type=int, default=32)
parser.add_argument('-ep', '--epochs', dest='epochs',
                    help="Set a different number of epochs",
                    type=int, default=100)
parser.add_argument('-sa', '--sample', dest='sample',
                    help="Set sample size",
                    type=int, default=20000)
args = parser.parse_args()

# =============================================================================
# GLOBALS
# =============================================================================

DATA_FILE = f"../data/{args.data}"
SUBSEQUENCE_SIZE = args.subsequence_size
SAMPLE_S = args.sample
TEST_S = 0.2
BATCH_SIZE = 32
VERBOSE = args.verbose
SAVE_HEADERS = args.save_headers
NO_SAVE = args.no_save
MODEL = args.model
NUM_HIDDEN = args.num_hidden
NUM_NODES = args.num_nodes
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

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
tx, rx = pre.data_from_mat(DATA_FILE, SAMPLE_S, verbose=VERBOSE)

if IS_BINARY:
    tx = np.array([0 if x == -1 else 1 for x in tx])

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
                                                      test_size=TEST_S,
                                                      random_state=42)

# =============================================================================
# MODEL PREPARATION
# =============================================================================

# dataclass storing all classifier configurations used
models = Models()


def compile_model():
    """
    Returns a newly compiled version of specified model
    """
    func = getattr(models, f"compile_{MODEL}")
    return func(num_hidden=NUM_HIDDEN, num_nodes=NUM_NODES,
                learning_rate=LEARNING_RATE)


# =============================================================================
# MODEL
# =============================================================================

# compile and fit model
model = compile_model()
model.fit(rx_train, tx_train, epochs=EPOCHS,
          batch_size=BATCH_SIZE, validation_split=0.2, verbose=VERBOSE)

# =============================================================================
# EVALUATE
# =============================================================================

# get predictions from test set
preds = model.predict(rx_test)
confidence = np.zeros(len(preds))

if IS_BINARY:
    # save confidence
    confidence = [np.max(instance) for instance in preds]
    # select bit with highest probability
    preds = [np.argmax(instance) for instance in preds]
    # reformat ground truths to bipolar binary
    tx_test = np.array([1 if signal > 0 else -1 for signal in tx_test])
else:
    # flatten linear predictions
    preds = [x for pred in preds for x in pred]

# get bipolar binary predictions
preds_bb = np.array([1 if signal > 0 else -1 for signal in preds])

# calculate accuracy
accuracy = accuracy_score(tx_test, preds_bb)
# calculate bit error rate
ber = 1.0 - accuracy

# dataframe for saving to csv
results_df = pd.DataFrame()
results_df['linear predictions'] = preds
results_df['binary bipolar predictions'] = preds_bb
results_df['ground truths (tx)'] = tx_test
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
