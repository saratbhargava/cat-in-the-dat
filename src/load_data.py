import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import DenseFeatures
from sklearn.model_selection import train_test_split


def load_csv(train_fileName, test_fileName=None, test_size=None):
    train_data = pd.read_csv(train_fileName)
    if test_fileName:
        test_data = pd.read_csv(test_fileName)
    else:
        if not test_size:
            test_size = 0.1
        train_data, test_data = train_test_split(
            train_data, test_size=test_size)
    return train_data, test_data


def df_to_dataset(dataframe, target_name='target', shuffle=True,
                  is_test=False, batch_size=32):
    dataframe = dataframe.copy()
    if not is_test:
        labels = dataframe.pop(target_name)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=dataframe.shape[0])
    ds = ds.batch(batch_size, True)
    return ds


def train_valid_split(train_data, valid_size=0.1):
    train_data, valid_data = train_test_split(
        train_data, test_size=valid_size)
    return train_data, valid_data


def train_valid_test_datasets(train_data, test_data, valid_size=0.1, batch_size=32):
    train_data, valid_data = train_valid_split(
        train_data, valid_size)
    train_dataset = df_to_dataset(train_data, batch_size=batch_size)
    valid_dataset = df_to_dataset(valid_data, batch_size=batch_size)
    test_dataset = df_to_dataset(
        test_data, is_test=True, batch_size=batch_size)
    return train_dataset, valid_dataset, test_dataset


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))
