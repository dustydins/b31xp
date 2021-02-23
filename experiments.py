#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sandbox for playing with experiment results
"""

import pandas as pd
from scipy import stats
import visualise as vis

#  DATA = pd.read_csv("./results/subsequence_multi.csv")

#  DATA = DATA.groupby(['experiment', 'sequence_position']).agg({
    #  'ground truths (tx)': [lambda x: tuple(stats.mode(x)[0])[0]],
    #  'binary bipolar predictions': [lambda x: tuple(stats.mode(x)[0])[0]],
    #  'linear predictions': ['mean'],
    #  'accuracy': ['mean'],
    #  'number_params': [lambda x: tuple(stats.mode(x)[0])[0]]})

#  DATA.columns = [col[0] for col in DATA.columns.values]

#  print(DATA)
#  DATA.to_csv('./results/dataframe_test.csv', index=True)

DATA = pd.read_csv("./results/dataframe_test.csv")


TITLE = '''Predicted Vs Ground Truth Tx Valuse \
for Different Subsequencing Sizes'''

vis.plot_signal(DATA, sample_size=100, title=TITLE)
