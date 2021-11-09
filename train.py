import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Accuracy, Recall, Precision
from model import get_model
from load_data import config, get_tis_data_split, get_ptb_data_split, DataGenerator
from generate_class_weights import generate_class_weights

train_config = config["training"]
classification_type = config['classification_type']


def train_model(train_config):
    optimizer = Adam(train_config["lr"])
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=train_config["lr"] / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau

                               min_delta=0.00001)]

    # load data generator
    train_df_tis, val_df_tis, _, y_train_labels_tis, y_val_labels_tis, _ = get_tis_data_split(classification_type)
    train_df_ptb, val_df_ptb, _, y_train_labels_ptb, y_val_labels_ptb, _ = get_ptb_data_split(classification_type)

    train_df = pd.concat([train_df_tis, train_df_ptb], axis=0, ignore_index=True, sort=False)
    y_train_labels = np.concatenate([y_train_labels_tis, y_train_labels_ptb], axis=0)
    train_gen = DataGenerator(train_df, y_train_labels, source='both', batch_size=train_config['batch_size'])

    val_df = pd.concat([val_df_tis, val_df_ptb], axis=0, ignore_index=True, sort=False)
    y_val_labels = np.concatenate([y_val_labels_tis, y_val_labels_ptb], axis=0)
    val_gen = DataGenerator(val_df, y_val_labels, source='both', batch_size=train_config['batch_size'])

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    activation_func = 'softmax' if classification_type == 'multi-class' else 'sigmoid'
    multi_class = True if classification_type == 'multi-class' else False
    class_weights = generate_class_weights(y_train_labels, multi_class=multi_class, one_hot_encoded=True)

    model = get_model(len(config["labels"]), activation_func)
    model.compile(loss=train_config["loss"], optimizer=optimizer, metrics=[Accuracy(), Recall(), Precision()])
    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./{}_backup_model_last.hdf5'.format(classification_type)),
                  ModelCheckpoint('./{}_backup_model_best.hdf5'.format(classification_type), save_best_only=True)]
    # Train neural network
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=train_config["epochs"],
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=1)
    # Save final result
    model.save("./{}_final_model.hdf5".format(classification_type))

train_model(train_config)
