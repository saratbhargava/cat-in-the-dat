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
    if is_test:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
        ds = ds.batch(batch_size, False)
    else:
        labels = dataframe.pop(target_name)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=dataframe.shape[0])
        ds = ds.batch(batch_size, True)
    return ds


def train_valid_split(train_data, valid_size=0.1):
    train_data, valid_data = train_test_split(
        train_data, test_size=valid_size)
    return train_data, valid_data


def train_valid_test_datasets(train_data, test_data, valid_size=0.1,
                              batch_size=32, test_shuffle=False):
    train_data, valid_data = train_valid_split(
        train_data, valid_size)
    if type(train_data) is pd.DataFrame:
        train_dataset = df_to_dataset(train_data, batch_size=batch_size)
        valid_dataset = df_to_dataset(valid_data, batch_size=batch_size)
        test_dataset = df_to_dataset(
            test_data, is_test=True, batch_size=batch_size,
            shuffle=test_shuffle)

    return train_dataset, valid_dataset, test_dataset


def get_dataset(file_path, batch_size=32, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))
