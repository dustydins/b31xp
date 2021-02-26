#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Description: Collection of classifiers and methods to compile them
"""

from dataclasses import dataclass

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.backend import set_floatx

set_floatx('float64')


def compile_model(model, architecture, learning_rate):
    """
    Returns a newly compiled version of specified model
    """
    models = Models()
    func = getattr(models, f"compile_{model}")
    return func(architecture=architecture,
                learning_rate=learning_rate)


@dataclass
class Models:
    """
    Data - stores data sets
    """

    _mlp_linear = None
    _mlp_binary = None

    # =================================================================
    # LINEAR MLP - mlp_linear
    # =================================================================

    def compile_mlp_linear(self, architecture=[0],
                           learning_rate=0.001):
        """
        MLP_LINEAR - construct a linear multi layer perceptron
        """
        optimiser = Adam(lr=learning_rate)

        self._mlp_linear = Sequential()
        for num_nodes in architecture:
            self._mlp_linear.add(Dense(num_nodes, activation='relu'))
        self._mlp_linear.add(Dense(1, activation='linear'))
        self._mlp_linear.compile(optimizer=optimiser,
                                 loss='mse',
                                 metrics=['accuracy'])
        return self._mlp_linear

    # =================================================================
    # BINARY MLP - mlp_binary
    # =================================================================

    def compile_mlp_binary(self, architecture=[0],
                           learning_rate=0.001):
        """
        MLP_BINARY - construct a binary multi layer perceptron
        """
        optimiser = Adam(lr=learning_rate)

        self._mlp_binary = Sequential()
        for num_nodes in architecture:
            self._mlp_binary.add(Dense(num_nodes, activation='relu'))
        self._mlp_binary.add(Dense(2, activation='softmax'))
        self._mlp_binary.compile(optimizer=optimiser,
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])
        return self._mlp_binary
