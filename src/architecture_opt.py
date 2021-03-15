#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import pprint
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

import preprocessing as pre

DATA_FILE = '/home/ald/Documents/Heriot-Watt/MULTIDISC/b31xp_new/data/PAM_OS4_3_6dbm_26022021.csv'
SAMPLE_S = 20000
TEST_S = 0.2
SUBSEQUENCE_SIZE = 4
VERBOSE = False


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    TX_LABELS = {'-8191': 0, '-2730': 1, '2730': 2, '8191': 3}
    TX_VALUES = {'0': -8191, '1': -2730, '2': 2730, '3': 8191}
    
    def to_label(value):
        return TX_LABELS[str(value)]
    
    
    def to_value(label):
        return TX_VALUES[str(label)]

    tx, rx = pre.load_data('../data/PAM_OS4_3_6dbm_26022021.csv', 20000, verbose=False)
    
    tx = np.array([to_label(x) for x in tx])
    
    # subsequence data
    tx, rx = pre.subsequence(rx, tx, 4, verbose=False)
    
    
    # split data
    x_train, x_test, y_train, y_test = pre.test_split(rx, tx,
                                                      test_size=0.2,
                                                      random_state=42)

    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense({{choice([4, 6, 8, 10, 12, 14, 16])}}, input_shape=(4,)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense({{choice([4, 6, 8, 10, 12, 14, 16])}}))
    model.add(Activation({{choice(['relu', 'sigmoid', 'tanh'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense({{choice([4, 6, 8, 10, 12, 14, 16])}}))
    model.add(Activation({{choice(['relu', 'sigmoid', 'tanh'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
              batch_size={{choice([16, 32, 64, 128])}},
              epochs=50,
              verbose=2,
              validation_split=0.2)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          eval_space=True,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    pp = pprint.PrettyPrinter(indent=4)
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    pp.pprint(best_run)
