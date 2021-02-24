#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sandbox for playing with experiment results
"""

import argparse
import pandas as pd
from scipy import stats
import visualise as vis

# =============================================================================
# PARSE CLI ARGS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', dest='filename',
                    help="File to read/write to (in results directory)",
                    type=str, default="n/a")
parser.add_argument('-t', '--title', dest='title',
                    help="Define a custom title",
                    type=str, default="Predicted Vs Ground Truth Tx Values")
parser.add_argument('-a', '--aggregate', dest='aggregate',
                    help="Aggregate results",
                    action="store_true", default=False)
parser.add_argument('-l', '--linear', dest='linear',
                    help="Results include linear predictions",
                    action="store_true", default=False)
parser.add_argument('-ss', '--sample-size', dest='sample_size',
                    help="Define size of sample to visualise",
                    type=int, default=200)
args = parser.parse_args()

# =============================================================================
# GLOBAL
# =============================================================================

FILENAME = args.filename
TITLE = args.title
AGGREGATE = args.aggregate
IS_LINEAR = args.linear
SAMPLE_SIZE = args.sample_size


def aggregate_results():
    """
    Groups and aggregates results by sequence position and
    experiment name
    """
    data = pd.read_csv("./results/current_test.csv")

    data = data.groupby(['experiment', 'sequence_position']).agg({
        'ground truths (tx)': [lambda x: tuple(stats.mode(x)[0])[0]],
        'binary bipolar predictions': [lambda x: tuple(stats.mode(x)[0])[0]],
        'linear predictions': ['mean'],
        'accuracy': ['mean'],
        'ber': ['mean'],
        'number_params': [lambda x: tuple(stats.mode(x)[0])[0]]})
    data.columns = [col[0] for col in DATA.columns.values]
    data.to_csv(f'./results/{FILENAME}.csv', index=True)


if AGGREGATE:
    aggregate_results()

#  DATA = pd.read_csv("./results/dataframe_test.csv")
DATA = pd.read_csv(f"./results/{FILENAME}.csv")

vis.plot_signal(DATA, sample_size=SAMPLE_SIZE,
                title=TITLE, is_linear=IS_LINEAR)
