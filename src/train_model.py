import tensorflow as tf

from datetime import datetime
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, LearningRateScheduler, TensorBoard,
    EarlyStopping)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential


def get_dense_two_layer_net(preprocessing_layer):
    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=Adam(),
        metrics=['accuracy', 'AUC'])
    return model
