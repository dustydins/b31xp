#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handles preprocessing for rx and tx data in order to work for ML models
"""

import functools

import numpy as np
import pandas as pd

from progress.bar import Bar
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def load_data(fname, sample_size, verbose=False, is_signal=False):
    """
    Loads data from mat file into tx and rx numpy arrays
    """
    _tx = []
    _rx = []
    # load mat file
    if '.mat' in fname:
        data = loadmat(fname)
        # flatten tx/rx values
        if is_signal:
            _tx = [value[0] for value in data['PAMTxSig']]
            _rx = [value[0] for value in data['PAMRxSig']]
        else:
            _tx = [value[0] for value in data['PAMsymTx']]
            _rx = [value[0] for value in data['PAMsymRx']]
    else:
        data = pd.read_csv(fname)
        _tx = [value for value in data['targets']]
        _rx = [value for value in data['logged_ADC']]
        # convert to numpy arrays (take only first SAMPLE_SIZE points)
    _tx = np.array(_tx)
    _rx = np.array(_rx)
    if verbose:
        print("------------------------------------------------------------------")
        print(f"Raw data shape:\n\trx: {_rx.shape}\n\ttx: {_tx.shape}")
        print("------------------------------------------------------------------")
    _tx = _tx[:sample_size]
    _rx = _rx[:sample_size]
    if verbose:
        print("------------------------------------------------------------------")
        print(f"Sampled data shape:\n\trx: {_rx.shape}\n\ttx: {_tx.shape}")
        print("------------------------------------------------------------------")
    return _tx, _rx


def tracecalls(func):
    """
    Decorator to check if function has been called
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)
    wrapper.has_been_called = False
    return wrapper


@tracecalls
def subsequence(_rx, _tx, tap_delay, verbose=False):
    """
    Preprocessing for subsequencing rx data, and removing tx data accordingly
    """
    prog_bar = None
    if verbose:
        print("------------------------------------------------------------------")
        prog_bar = Bar('subsequencing in  progress...', max=_rx.shape[0])

    new_rx = np.empty((0, tap_delay), float)
    for idx, _ in enumerate(_rx):
        seq = _rx[(idx+1)-tap_delay: idx+1]
        if seq.shape[0] != 0:
            new_rx = np.append(new_rx, np.array([seq]), axis=0)
        if verbose:
            prog_bar.next()
    if verbose:
        print("\n------------------------------------------------------------------")
    # remove first elements from tx
    _tx = np.delete(_tx, [range(tap_delay-1)])
    return _tx, new_rx


def test_split(_rx, _tx, test_size):
    """
    train test split - separation of concerns
    return rx_train, rx_test, tx_train, tx_test
    """
    return train_test_split(_rx, _tx, test_size=test_size, shuffle=False)


def summarise_data(_rx, _tx, tap_delay):
    """
    Convert data to pandas dataframe for summarisation purposes
    """
    assert subsequence.has_been_called
    data_df = pd.DataFrame()
    for idx in range(tap_delay):
        data_df[f"rx{idx}"] = _rx[:, idx]
    data_df["tx"] = _tx
    return data_df
