"""
Implement various feature extraction techniques using tf and sklearn
"""
import os
import functools
import pathlib

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.feature_column import (
    numeric_column, bucketized_column,
    categorical_column_with_vocabulary_list, embedding_column,
    crossed_column, indicator_column)
from tensorflow.keras.layers import DenseFeatures


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32)
                            for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        return features, labels


def categorical2onehot(unique_feats, categorical_feats):
    categorical_columns = []
    for feature in categorical_feats:
        vocab = unique_feats[feature]
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    return categorical_columns


def categorical2embedding(unique_feats, categorical_feats, categorical_feats_len):
    categorical_columns = []
    for feature, length in zip(categorical_feats, categorical_feats_len):
        vocab = unique_feats[feature]
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(
            tf.feature_column.embedding_column(cat_col, length))
    return categorical_columns


def normalization(train_data, NUMERIC_FEATURES):

    def normalize_numeric_data(data, mean, std):
        return (data-mean)/std

    desc = train_data[NUMERIC_FEATURES].describe()
    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])
    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
    numeric_column = numeric_column(
        'numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]
    return numeric_columns
