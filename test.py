#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.io import loadmat

data = loadmat('./TestEqualiser/POF60m_PAMExp_2PAM_DR600Mbps.mat')

print("TX")
[print(x) for x in data['PAMsymTx'][:10]]
print("RX")
[print(x) for x in data['PAMsymRx'][:10]]
