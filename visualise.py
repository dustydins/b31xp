#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Used for visualising test results
"""

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# colours
C_CORRECT = '#2a9d8f50'
C_INCORRECT = '#e76f5150'
C_AXHLINE = '#c2c2c2'
C_GROUND_TRUTH = '#14213d'
C_PREDICTIONS = '#dc2f02'


def plot_confusion_matrix(ground_truth, preds, title="Confusion Matrix"):
    """
    Plot confusion matrix
    """
    conf_matrix = confusion_matrix(ground_truth, preds)
    cm_df = pd.DataFrame(conf_matrix, index=['-1', '1'],
                         columns=['-1', '1'])
    plt.figure(figsize=(16, 16))
    axes = sns.heatmap(cm_df/np.sum(cm_df),
                       annot=True, cmap="Reds", cbar=False,
                       fmt='.2%')
    axes.set(ylabel='Ground Truth (Tx)', xlabel='Predictions')
    axes.set_title(title)
    plt.show()


def add_signal(data, sample_size=150, is_linear=True):
    """
    Adds a signal to the current plot
    """

    # custom palette
    palette = []
    for _, row in data.iterrows():
        if row['ground truths (tx)'] == row['binary bipolar predictions']:
            palette.append(C_CORRECT)
        else:
            palette.append(C_INCORRECT)

    # resize data to sample
    ground_truth = data['ground truths (tx)'][:sample_size]
    preds_bin = data['binary bipolar predictions'][:sample_size:]
    preds = data['linear predictions'][:sample_size]
    time = np.arange(sample_size)

    # add plots
    plt.axhline(y=0, color=C_AXHLINE)
    plt.bar(time - 0.5, preds_bin, color=palette, width=1.0)
    plt.step(time, ground_truth, label='Ground Truth Tx', color=C_GROUND_TRUTH)
    if is_linear:
        plt.plot(time, preds, label='Linear Tx Predicitions (Mean)',
                 color=C_PREDICTIONS, linestyle='--')


def plot_signal(data, sample_size=150, title='Truth Vs Predictions',
                is_linear=True):
    """
    Plot signal for each experiment
    """
    fig = plt.figure()
    _ax = fig.add_subplot(111)
    axes = []

    if 'experiment' not in data:
        # if no experiment column, just plot the the data on a single chart
        add_signal(data, sample_size, is_linear=is_linear)
    else:
        # plot each experiment
        experiments = pd.unique(data['experiment'])
        subplot_number = len(experiments)

        # disable container axes ticks
        _ax.spines['top'].set_color('none')
        _ax.spines['bottom'].set_color('none')
        _ax.spines['left'].set_color('none')
        _ax.spines['right'].set_color('none')
        _ax.tick_params(labelcolor='w', top=False, bottom=False,
                        left=False, right=False)

        # add experiment subplot
        for idx, experiment in enumerate(experiments):
            ax_experiment = fig.add_subplot(subplot_number, 1, idx+1)
            experiment_data = data[(data['experiment'] == experiment)]
            acc = pd.unique(experiment_data['accuracy'])[0]
            ber = pd.unique(experiment_data['ber'])[0]
            param = pd.unique(experiment_data['number_params'])[0]
            ex_title = f"{experiment} | Accuracy (Mean): {acc:.2%} | Bit Error Rate (Mean): {ber:.2} | #Params: {param}"
            ax_experiment.set_title(ex_title)
            add_signal(experiment_data, sample_size=sample_size,
                       is_linear=is_linear)
            axes.append(ax_experiment)

    # add legend
    correct_patch = mpatches.Patch(color=C_CORRECT,
                                   label='Correct Binary Tx Prediction (Mode)')
    incorrect_patch = mpatches.Patch(color=C_INCORRECT,
                                     label='Incorrect Binary Tx Prediction (Mode)')
    handles, _ = plt.gca().get_legend_handles_labels()
    fig.legend(handles=handles + [correct_patch, incorrect_patch],
               facecolor='#ffffff', framealpha=1)

    # add meta
    _ax.set_title(title, fontsize='xx-large',
                  pad=40)
    _ax.set_xlabel('Sequence Position', fontsize='x-large')
    _ax.set_ylabel('Signal Value', fontsize='x-large')

    # disable all but last axes ticks
    for ax_experiment in axes[:-1]:
        plt.setp(ax_experiment.get_xticklabels(), visible=False)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
