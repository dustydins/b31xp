#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Used for visualising test results
"""

import numpy as np
import pandas as pd

from scipy import signal

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#  DATA = pd.read_csv("results_good.csv", index_col='Unnamed: 0')

DATA_GOOD = pd.read_csv("results_good.csv", index_col='Unnamed: 0')
DATA_GOOD['experiment'] = 'Subsequencing - 8 Inputs'
DATA_BAD = pd.read_csv("results_bad.csv", index_col='Unnamed: 0')
DATA_BAD['experiment'] = 'No Subsequencing'
DATA =  pd.concat([DATA_BAD, DATA_GOOD], sort=False)

# colours
c_correct = '#2a9d8f50'
c_incorrect = '#e76f5150'
c_axhline = '#c2c2c2'
c_ground_truth = '#14213d'
c_predictions = '#dc2f02'


def add_signal(data, ax, sample_size=150):
    """
    Adds a signal to the current plot
    """

    # custom palette
    palette = []
    for idx, row in data.iterrows():
        if row['ground truths (tx)'] == row['binary bipolar predictions']:
            palette.append(c_correct)
        else:
            palette.append(c_incorrect)

    # resize data to sample
    ground_truth = data['ground truths (tx)'][:sample_size]
    preds_bin = data['binary bipolar predictions'][:sample_size]
    preds = data['linear predictions'][:sample_size]
    time = np.arange(sample_size)

    # add plots
    plt.axhline(y=0, color=c_axhline)
    plt.bar(time - 0.5, preds_bin, color=palette, width=1.0)
    plt.step(time, ground_truth, label='Ground Truth Tx', color=c_ground_truth)
    plt.plot(time, preds, label='Linear Tx Predicitions',
             color=c_predictions, linestyle='--')



def plot_signal(data, sample_size=150, title='Predicted Vs Ground Truth Tx Values'):
    """
    Plot signal for each experiment
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if 'experiment' not in data:
        # if no experiment column, just plot the the data on a single chart
        add_signal(data, ax, sample_size)
    else:
        experiments = pd.unique(data['experiment'])
        subplot_number = len(experiments)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        for idx, experiment in enumerate(experiments):
            ax_experiment = fig.add_subplot(subplot_number, 1, idx+1)
            ax_experiment.set_title(experiment)
            experiment_data = data[(data['experiment'] == experiment)]
            add_signal(experiment_data, ax_experiment, sample_size=sample_size)

    # add legend
    correct_patch = mpatches.Patch(color=c_correct, label='Correct Binary Tx Prediction')
    incorrect_patch = mpatches.Patch(color=c_incorrect, label='Incorrect Binary Tx Prediction')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(loc='upper left', handles=handles + [correct_patch, incorrect_patch],
               facecolor='#ffffff', framealpha=1)

    # add meta
    ax.set_title(title, fontsize='xx-large',
                  pad=40)
    ax.set_xlabel('Sequence Position', fontsize='x-large')
    ax.set_ylabel('Signal Value', fontsize='x-large')
    plt.show()


plot_signal(DATA, sample_size=200, title='Predicted Vs Ground Truth Tx Values for Different Subsequence Sizes')
