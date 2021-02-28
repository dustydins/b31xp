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
                    type=str, default="")
parser.add_argument('-v', '--visualise', dest='visualise',
                    help="Visualise results",
                    action="store_true", default=False)
parser.add_argument('-a', '--aggregate', dest='aggregate',
                    help="Aggregate results",
                    action="store_true", default=False)
parser.add_argument('-ss', '--sample-size', dest='sample_size',
                    help="Define size of sample to visualise",
                    type=int, default=200)
args = parser.parse_args()

# =============================================================================
# GLOBAL
# =============================================================================

FILENAME = args.filename
TITLE = f"Predicted Vs Ground Truth Tx Values - {args.title}"
AGGREGATE = args.aggregate
SAMPLE_SIZE = args.sample_size
VISUALISE = args.visualise


def aggregate_results():
    """
    Groups and aggregates results by sequence position and
    experiment name
    """
    data = pd.read_csv("../results/current_test.csv", low_memory=False)
    print(data)

    mode = [lambda x: tuple(stats.mode(x)[0])[0]]
    data = data.groupby(['experiment', 'sequence_position']).agg({
        'targets': mode,
        'predictions': mode,
        'accuracy': ['mean'],
        'ber': ['mean'],
        'confidence': ['mean'],
        'number_params': mode})
    data.columns = [col[0] for col in data.columns.values]
    data.to_csv(f'../results/{FILENAME}.csv', index=True)


if AGGREGATE:
    aggregate_results()

if VISUALISE:
    DATA = pd.read_csv(f"../results/{FILENAME}.csv")
    vis.plot_signal(DATA, sample_size=SAMPLE_SIZE,
                    title=TITLE)
