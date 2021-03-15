#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Description: Collection of classifiers and methods to compile them
"""

from dataclasses import dataclass

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.optimizers import Adam
from keras.backend import set_floatx

set_floatx('float64')


def compile_model(model, architecture, learning_rate, dropout_rate):
    """
    Returns a newly compiled version of specified model
    """
    models = Models()
    func = getattr(models, f"compile_{model}")
    return func(architecture=architecture,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate)


@dataclass
class Models:
    """
    Data - stores data sets
    """

    _mlp_binary = None
    _mlp_hyperas = None

    # =================================================================
    # BINARY MLP - mlp_binary
    # =================================================================

    def compile_mlp_binary(self, architecture=[0],
                           learning_rate=0.001,
                           dropout_rate=None):
        """
        MLP_BINARY - construct a binary multi layer perceptron
        """
        optimiser = Adam(lr=learning_rate)
        #  optimiser = Adam()

        self._mlp_binary = Sequential()
        for idx, num_nodes in enumerate(architecture):
            self._mlp_binary.add(Dense(num_nodes, activation='relu'))
            if isinstance(dropout_rate, list):
                self._mlp_binary.add(Dropout(dropout_rate[idx],
                                             noise_shape=None, seed=None))
        self._mlp_binary.add(Dense(4, activation='softmax'))
        self._mlp_binary.compile(optimizer=optimiser,
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])
        return self._mlp_binary

    # =================================================================
    # HYPERAS MLP - mlp_hyperas
    # =================================================================

    def compile_mlp_hyperas(self, architecture=[0],
                            learning_rate=0.001,
                            dropout_rate=0.0):
        """
        MLP_BINARY - construct a binary multi layer perceptron
        """

        self._mlp_hyperas = Sequential()
        self._mlp_hyperas.add(Dense(8, input_shape=(4,)))
        self._mlp_hyperas.add(Activation('sigmoid'))
        self._mlp_hyperas.add(Dropout(0.03323327852409652))

        self._mlp_hyperas.add(Dense(10))
        self._mlp_hyperas.add(Activation('sigmoid'))
        self._mlp_hyperas.add(Dropout(0.0886198698550964))

        self._mlp_hyperas.add(Dense(8))
        self._mlp_hyperas.add(Activation('relu'))
        self._mlp_hyperas.add(Dropout(0.2330896882313117))

        self._mlp_hyperas.add(Dense(4, activation='softmax'))

        self._mlp_hyperas.compile(optimizer='rmsprop',
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])
        return self._mlp_hyperas
