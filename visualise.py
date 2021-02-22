#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Used for visualising test results
"""

import numpy as np
import pandas as pd

from scipy import signal

import seaborn as sns
import matplotlib.pyplot as plt

#  DATA = pd.read_csv("results.csv", index_col='Unnamed: 0')
#  print(DATA[:10])


def plot_signal(data, sample_size=10, x_offset=0.3, y_offset=0.01):
    """
    Visualise ground truth signal vs predicted signal
    """

    # resize data to sample
    ground_truth = data['ground truths (tx)'][:sample_size]
    preds_bin = data['binary bipolar predictions'][:sample_size]
    preds = data['linear predictions'][:sample_size]
    time = np.arange(sample_size)

    # add plots

    plt.axhline(y=0, color='#c2c2c2')
    plt.step(time, ground_truth, label='Ground Truth', color='#fb4934')
    plt.step(time + x_offset, preds_bin + y_offset, label='Binary Predicitions', color='#4ea699')
    plt.plot(time + x_offset, preds + y_offset, label='Linear Predicitions',
             color='#4ea699', linestyle='--')
    plt.legend(loc='upper left')

    # add meta
    plt.title('Predicted Vs Ground Truth Tx Values')
    plt.xlabel('Sequence Position')
    plt.ylabel('Signal Value')

    ax = plt.axes()
    #  ax.set_facecolor("#7c6f64")

    plt.show()


#  plot_signal(DATA, sample_size=150)
