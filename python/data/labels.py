# ==============================================================================
# labels.py
#
# Helper functions for dealing with class labels
#
# Reference:
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# ==============================================================================

import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def integer_encode(in_labels):
    """Encode class labels as integers"""
    encoder = LabelEncoder()
    encoder.fit(in_labels)
    encoded_out = encoder.transform(in_labels)
    return encoded_out


def one_hot_encode(in_labels):
    """One hot encode class labels"""
    encoded_out = integer_encode(in_labels)
    one_hot = np_utils.to_categorical(encoded_out)
    return one_hot


