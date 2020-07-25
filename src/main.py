import os
import pathlib

import numpy as np
import pandas as pd

from tensorflow.keras.layers import DenseFeatures
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from load_data import load_csv, train_valid_test_datasets, show_batch
from features import (PackNumericFeatures, categorical2onehot,
                      categorical2embedding, normalization)
from utils import get_unique
from train_model import get_dense_two_layer_net

# load data and create Dataset obj
train_fileName = '../inputs/train.csv'
test_fileName = '../inputs/test.csv'

batch_size = 128  # 32

train_data, test_data = load_csv(train_fileName, test_fileName)
train_data.pop("id")
test_data.pop("id")
train_dataset, valid_dataset, test_dataset = train_valid_test_datasets(
    train_data, test_data, valid_size=0.2,
    batch_size=batch_size)
train_size = int(train_data.shape[0]*0.8)
valid_size = int(train_data.shape[0]*0.2)
print(train_data.shape, test_data.shape)
print(train_dataset.element_spec)

numeric_features = ['month', 'day']
train_dataset = train_dataset.map(PackNumericFeatures(numeric_features))
valid_dataset = valid_dataset.map(PackNumericFeatures(numeric_features))

show_batch(train_dataset)

unique_elems = get_unique(train_data)

categorical_feats_onehot = ['bin_3', 'bin_4'] + [
    'nom_' + str(i) for i in range(5)] + [
        'ord_' + str(i) for i in range(1, 6)]
categorical_feats_embeddings = ['nom_' + str(i) for i in range(5, 10)]
categorical_feats_len_embeddings = [
    len(unique_elems[feat_name])//20 for feat_name in categorical_feats_embeddings
]

onehot_feats = categorical2onehot(unique_elems, categorical_feats_onehot)
embedding_feats = categorical2embedding(
    unique_elems, categorical_feats_embeddings,
    categorical_feats_len_embeddings)

preprocessing_layer = DenseFeatures(onehot_feats + embedding_feats)

print(onehot_feats)
print(embedding_feats)

model = get_dense_two_layer_net(preprocessing_layer)
callbacks = [TensorBoard(log_dir="./logs"),
             ModelCheckpoint(filepath='./models/two_layer',
                             save_weights_only=True,
                             monitor='val_acc',
                             mode='max',
                             save_best_only=True)]
history = model.fit(train_dataset, validation_data=valid_dataset,
                    steps_per_epoch=train_size//batch_size,
                    validation_steps=valid_size//batch_size,
                    callbacks=callbacks)

test_preds = model.evaluate(test_dataset)
print(test_preds)
