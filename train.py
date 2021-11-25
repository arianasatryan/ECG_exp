import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Accuracy, Recall, Precision
from model import get_model
from load_data import config, get_tis_data_split, get_ptb_data_split, DataGenerator
from generate_class_weights import generate_class_weights

train_config = config["training_config"]
classification_type = config["classification_type"]
data_source = config['data_config']['source']

data_split_func = {
    'tis': get_tis_data_split,
    'ptb': get_ptb_data_split
}


def train_model():
    optimizer = Adam(train_config["lr"])
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=train_config["lr"] / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau

                               min_delta=0.00001)]

    # load data generator
    if data_source != 'both':
        train_df, val_df, _, y_train_labels, y_val_labels, _ = data_split_func[data_source](classification_type)

    else:
        train_df_tis, val_df_tis, _, y_train_labels_tis, y_val_labels_tis, _ = data_split_func['tis'](classification_type)
        train_df_ptb, val_df_ptb, _, y_train_labels_ptb, y_val_labels_ptb, _ = data_split_func['ptb'](classification_type)

        train_df = pd.concat([train_df_tis, train_df_ptb], axis=0, ignore_index=True, sort=False)
        y_train_labels = np.concatenate([y_train_labels_tis, y_train_labels_ptb], axis=0)

        val_df = pd.concat([val_df_tis, val_df_ptb], axis=0, ignore_index=True, sort=False)
        y_val_labels = np.concatenate([y_val_labels_tis, y_val_labels_ptb], axis=0)

    train_gen = DataGenerator(x_df=train_df, y_labels=y_train_labels, source=data_source)
    val_gen = DataGenerator(x_df=val_df, y_labels=y_val_labels, source=data_source)

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    activation_func = 'softmax' if classification_type == 'multi-class' else 'sigmoid'
    multi_class = True if classification_type == 'multi-class' else False
    class_weights = generate_class_weights(y_train_labels, multi_class=multi_class, one_hot_encoded=True)

    model = get_model(len(config["labels"]), activation_func)
    model.compile(loss=train_config["loss"], optimizer=optimizer, metrics=[Accuracy(), Recall(), Precision()])

    # Create log
    if not os.path.exists('./results'):
        os.makedirs('./results')

    callbacks += [TensorBoard(log_dir='./results/logs', write_graph=False),
                  CSVLogger('./results/training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./results/backup_model_last.hdf5'),
                  ModelCheckpoint('./results/backup_model_best.hdf5', save_best_only=True)]
    
    # Train neural network
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=train_config["epochs"],
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=1)
    # Save final result
    model.save(f'{config["path_to_model"]}')
    save_model_info()


def save_model_info():
    from datetime import datetime
    model_info = {'saved_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                  'classification_type': config['classification_type'],
                  'data_config': {'source': config['data_config']['source'],
                                  'sampling_rate': config['data_config']['sampling_rate'],
                                  'points': config['data_config']['points'],
                                  'pad_mode': config['data_config']['pad_mode'],
                                  'norm_by': config['data_config']['norm_by']}}
    
    with open(os.path.dirname(os.path.abspath(config["path_to_model"])) + '/model_info.json', 'w+') as fout:
        json.dump(model_info, fout, indent=4, default=str)

train_model()
