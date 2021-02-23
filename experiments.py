#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import visualise as vis

DATA = pd.read_csv("./results/current_test.csv")
title = 'Predicted Vs Ground Truth Tx Valuse for Different Subsequencing Sizes'
vis.plot_signal(DATA, sample_size=300, title='Predicted Vs Ground Truth Tx Values for Different Subsequence Sizes')
