#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing tap delay line input for b31xp
"""

import numpy as np
import pandas as pd

from progress.bar import Bar
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

#  import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#  import matplotlib.pyplot as plt

###############################################################################
# GLOBALS
###############################################################################

TAP_DELAY = 8
SAMPLE_SIZE = 20000
TEST_SIZE = 0.2
EPOCHS = 150
BATCH_SIZE = 50

###############################################################################
# PRE-PROCESSING
###############################################################################

# load mat file
mat_data = loadmat('./data/POF60m_PAMExp_2PAM_DR600Mbps.mat')

# flatten tx/rx values
tx = [value[0] for value in mat_data['PAMsymTx']]
rx = [value[0] for value in mat_data['PAMsymRx']]


# convert to numpy arrays (take only first SAMPLE_SIZE points)
tx = np.array(tx)[:SAMPLE_SIZE]
rx = np.array(rx)[:SAMPLE_SIZE]


def to_tap_delay(data):
    """
    converts 1D Numpy array into tap delay line format
    """
    new_data = np.empty((0, TAP_DELAY), float)
    with Bar('to_tap_delay progress...', max=data.shape[0]) as prog_bar:
        for idx, _ in enumerate(data):
            seq = data[(idx+1)-TAP_DELAY: idx+1]
            if seq.shape[0] != 0:
                new_data = np.append(new_data, np.array([seq]), axis=0)
            prog_bar.next()
    return new_data


rx = to_tap_delay(rx)
print(f"rx post tapped delay line modification:\n{rx[:10]}")

# remove first elements from tx
tx = np.delete(tx, [range(TAP_DELAY-1)])
print(f"tx starting from TAP_DELAYth value:\n{tx[:10]}")


# split data
rx_train, rx_test, tx_train, tx_test = train_test_split(rx, tx,
                                                        test_size=TEST_SIZE,
                                                        random_state=42)

###############################################################################
# MODEL
###############################################################################

model = Sequential()
model.add(Dense(64, input_dim=rx_train.shape[1],
                kernel_initializer='normal',
                activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

model.fit(rx_train, tx_train, epochs=EPOCHS,
          batch_size=BATCH_SIZE, validation_split=0.2)

###############################################################################
# EVALUATE
###############################################################################

preds = model.predict(rx_test)
preds = np.array([1 if pred[0] > 0 else -1 for pred in preds])


results_df = pd.DataFrame()
results_df['predictions'] = preds
results_df['ground_truth'] = tx_test
confusion_matrix = pd.crosstab(results_df['ground_truth'],
                               results_df['predictions'],
                               rownames=['Actual'], colnames=['Predicted'])
print(model.summary())
print(confusion_matrix)
