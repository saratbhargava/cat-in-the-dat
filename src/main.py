import os
import pathlib

import numpy as np
import pandas as pd

from load_data import load_csv, train_valid_test_datasets

# load data and create Dataset obj
train_fileName = '../inputs/train.csv'
test_fileName = '../inputs/test.csv'

batch_size = 5  # 32

train_data, test_data = load_csv(train_fileName, test_fileName)
train_dataset, valid_dataset, test_dataset = train_valid_test_datasets(
    train_data, test_data, valid_size=0.2,
    batch_size=batch_size)

print(train_data.shape, test_data.shape)
print(train_dataset.element_spec)
