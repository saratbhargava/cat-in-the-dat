import tensorflow as tf

from datetime import datetime
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, LearningRateScheduler, TensorBoard,
    EarlyStopping)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorboard.plugins.hparams import api as hp


def get_dense_two_layer_net(preprocessing_layer, decay_rate=1e-3):
    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=l2(decay_rate)),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=l2(decay_rate)),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=Adam(),
        metrics=['accuracy', 'AUC'])
    return model


def get_logistic_regression(preprocessing_layer):
    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=Adam(),
        metrics=['accuracy', 'AUC'])
    return model


class train_dense_two_layer_net_hparam:
    # A bad way to use classes, can use funcs

    def __init__(self, train_data, valid_data, preprocessing_layer,
                 HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_EPOCHS,
                 exp_name='exp1'):

        self.HP_NUM_UNITS = HP_NUM_UNITS
        self.HP_DROPOUT = HP_DROPOUT
        self.HP_OPTIMIZER = HP_OPTIMIZER
        self.HP_EPOCHS = HP_EPOCHS
        self.preprocessing_layer = preprocessing_layer
        self.train_data = train_data
        self.valid_data = valid_data

        session_num = 0
        for num_units in HP_NUM_UNITS.domain.values:
            for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                for optimizer in HP_OPTIMIZER.domain.values:
                    for epochs in HP_EPOCHS.domain.values:
                        hparams = {
                            HP_NUM_UNITS: num_units,
                            HP_DROPOUT: dropout_rate,
                            HP_OPTIMIZER: optimizer,
                            HP_EPOCHS: epochs
                        }
                        session_name = f'num_units_{num_units}-dropout_{dropout_rate}-lr_{optimizer}-epochs_{epochs}'
                        print(f'Session number: {session_num}')
                        print({h.name: hparams[h] for h in hparams})
                        self.__helper__(
                            hparams,
                            f'logs/{exp_name}/' + session_name,
                            f'models/{exp_name}/' + session_name)
                        session_num += 1

    def __helper__(self, hparams, logs, models_dir):
        model = Sequential(
            [
                self.preprocessing_layer,
                Dense(hparams[self.HP_NUM_UNITS], activation='relu'),
                Dropout(hparams[self.HP_DROPOUT]),
                Dense(hparams[self.HP_NUM_UNITS], activation='relu'),
                Dense(1, activation='sigmoid')
            ]
        )
        model.compile(
            optimizer=RMSprop(lr=hparams[self.HP_OPTIMIZER]),
            loss=BinaryCrossentropy(),
            metrics=['accuracy', 'AUC'])
        model_checkpoint = ModelCheckpoint(
            filepath=f'{models_dir}/checkpoint',
            save_best_only=True, verbose=1, monitor='val_accuracy')
        log_dir = logs + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        hp_callback = hp.KerasCallback(logs, hparams)
        callbacks = [model_checkpoint, tensorboard_callback,
                     hp_callback]
        history = model.fit(self.train_data, validation_data=self.valid_data,
                            epochs=hparams[self.HP_EPOCHS], callbacks=callbacks)
        with tf.summary.create_file_writer(logs).as_default():
            hp.hparams(hparams)
            tf.summary.scalar(
                METRIC_ACCURACY, history.history['val_accuracy'][-1], step=1)
        model.save(f"{models_dir}/final_epoch_{history.history['val_accuracy'][-1]}_" +
                   datetime.now().strftime("%Y%m%d-%H%M%S"))
