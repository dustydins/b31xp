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
import matplotlib.ticker as ticker

# colours
C_CORRECT = '#78bc6190'
C_INCORRECT = '#d64933'
C_AXHLINE = '#2e353230'
C_AXHLINE_CENTRE = '#2e353270'
C_GROUND_TRUTH = '#2e3532'
#  C_PREDICTIONS = '#dc2f0295'
C_CONFIDENCE = '#3c91e690'
C_CONFIDENCE_TEXT = '#3c91e6'

#  C_CORRECT = '#2a9d8f80'
#  C_INCORRECT = '#e76f51'
#  C_AXHLINE = '#c2c2c250'
#  C_GROUND_TRUTH = '#14213d'
#  C_PREDICTIONS = '#dc2f0295'
#  C_CONFIDENCE = '#4002dc90'

Y_TICKS = [-12286.5, -8191, -2730, 0, 2730, 8191, 12286.5]
TICK_LABEL_S = 8
CONFIDENCE_MIN = 4461
CONFIDENCE_MAX = 6461


def add_signal(data, sample_size=150, y_ticks=Y_TICKS):
    """
    Adds a signal to the current plot
    """

    # custom palette
    palette = []
    for _, row in data.iterrows():
        if row['targets'] == row['predictions']:
            palette.append(C_CORRECT)
        else:
            palette.append(C_INCORRECT)

    # resize data to sample
    ground_truth = data['targets'][:sample_size]
    preds_bin = data['predictions'][:sample_size:]
    time = np.arange(sample_size)

    # plot grid lines
    for tick in y_ticks[1:-1]:
        if tick != 0:
            plt.axhline(y=tick, color=C_AXHLINE, linestyle='--', linewidth=1)
    plt.axhline(y=0, color=C_AXHLINE_CENTRE, linestyle='-')


    # plot target and predicted signals
    plt.bar(time - 0.5, preds_bin, color=palette, width=1.0)
    plt.step(time, ground_truth, label='Ground Truth Tx', color=C_GROUND_TRUTH,
             linewidth=2)


def add_confidence(data, sample_size=150):
    """
    Adds confidence to the current plot
    """

    # resize data to sample
    time = np.arange(sample_size)

    # scale confidence
    confidence = data['confidence'][:sample_size]
    #  confidence = np.interp(confidence, (confidence.min(),
                                        #  confidence.max()), (CONFIDENCE_MIN,
                                                            #  CONFIDENCE_MAX))
    # plot confidence
    plt.plot(time, confidence, label='Confidence (Mean)',
             color=C_CONFIDENCE, linestyle='-', linewidth=1.5)


def plot_signal(data, sample_size=150, title='Truth Vs Predictions'):
    """
    Plot signal for each experiment
    """
    fig = plt.figure()
    _ax = fig.add_subplot(111)
    axes = []
    experiments = pd.unique(data['experiment'])
    handles = []

    if len(experiments) < 2:
        # if no experiment column, just plot the the data on a single chart
        # set y ticks
        y_ticks = list(pd.unique(data['targets']))
        y_ticks.sort()
        y_ticks.insert(0, y_ticks[0]*1.5)
        y_ticks.append(y_ticks[-1]*1.5)
        y_ticks_labels = y_ticks.copy()
        y_ticks_labels[0] = ''
        y_ticks_labels[-1] = ''
        add_signal(data, sample_size, y_ticks=y_ticks)
    else:
        # plot each experiment
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

            # set y ticks
            y_ticks = list(pd.unique(experiment_data['targets']))
            y_ticks.sort()
            y_ticks.insert(0, y_ticks[0]*1.5)
            y_ticks.append(y_ticks[-1]*1.5)
            y_ticks_labels = y_ticks.copy()
            y_ticks_labels[0] = ''
            y_ticks_labels[-1] = ''
            add_signal(experiment_data, sample_size=sample_size, y_ticks=y_ticks)

            
            #  print(y_ticks)
            #  print(y_ticks_labels)

            ax_experiment.set_yticks(y_ticks)
            ax_experiment.set_yticklabels(y_ticks_labels)
            ax_experiment.yaxis.set_major_locator(ticker.FixedLocator(y_ticks[1:-1]))

            if idx == 0:
                h, _ = plt.gca().get_legend_handles_labels()
                handles.append(h[0])

            ax_r = fig.add_subplot(subplot_number, 1, idx+1, sharex=ax_experiment, frameon=False)
            add_confidence(experiment_data, sample_size=sample_size)
            ax_r.yaxis.tick_right()
            ax_r.set_yticks([-0.25, 0.0, 0.5, 1.0, 1.25])
            ax_r.set_yticklabels(['', 0.0, 0.5, 1.0, ''], fontsize=TICK_LABEL_S,
                                 color=C_CONFIDENCE_TEXT)
            plt.setp(ax_r.get_xticklabels(), visible=False)

            if idx == 0:
                h, _ = plt.gca().get_legend_handles_labels()
                handles.append(h[0])
            
            axes.append(ax_experiment)

    # add legend
    correct_patch = mpatches.Patch(color=C_CORRECT,
                                   label='Correct Tx prediction (Mode)')
    incorrect_patch = mpatches.Patch(color=C_INCORRECT,
                                     label='Incorrect Tx prediction (Mode)')
    fig.legend(handles=handles + [correct_patch, incorrect_patch],
               facecolor='#ffffff', framealpha=1)

    # add meta
    _ax.set_title(title, fontsize='xx-large',
                  pad=40)
    _ax.set_xlabel('Sample', fontsize='x-large', labelpad=30)
    _ax.set_ylabel('Value', fontsize='x-large', labelpad=30)
    

    # disable all but last axes ticks
    for ax_experiment in axes[:-1]:
        plt.setp(ax_experiment.get_xticklabels(), visible=False)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
