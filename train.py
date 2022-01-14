import os
import json
import re
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Recall, Precision
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping)
from tensorflow.keras.models import load_model
from load_data import config, get_tis_data_split, get_ptb_data_split, DataGenerator
from generate_class_weights import generate_class_weights
from model import get_model
from sklearn.preprocessing import MultiLabelBinarizer

SEED = config["SEED"]
label_names = config['labels']
path_to_model = config['path_to_model']
train_config = config["training_config"]
classification_type = config["classification_type"]
data_source = config['data_config']['source']

data_split_func = {
    'tis': get_tis_data_split,
    'ptb': get_ptb_data_split
}

train_details = {
    'multi-label': {'loss': 'binary_crossentropy', 'activation': 'sigmoid'},
    'multi-class': {'loss': 'categorical_crossentropy', 'activation': 'softmax'},
    'binary': {'loss': 'binary_crossentropy', 'activation': 'sigmoid'}
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
        train_df, val_df, test_df = data_split_func[data_source](classification_type)

    else:
        train_df_ptb, val_df_ptb, _ = data_split_func['ptb'](classification_type)
        train_df_tis, val_df_tis, _ = data_split_func['tis'](classification_type)

        train_df = pd.concat([train_df_tis, train_df_ptb], axis=0, ignore_index=True, sort=False)
        val_df = pd.concat([val_df_tis, val_df_ptb], axis=0, ignore_index=True, sort=False)

    print(train_df.labels.value_counts(), '\n\n', val_df.labels.value_counts())
    train_gen, val_gen = DataGenerator(x_df=train_df), DataGenerator(x_df=val_df)

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)

    loss, activation_func = train_details[classification_type]['loss'], train_details[classification_type]['activation']
    print(loss, activation_func)
    class_weights = get_class_weights(train_df)

    # use this to train from the beginning
    model = get_model(len(label_names), activation_func)
    model.compile(loss=loss, optimizer=optimizer, metrics=[Accuracy(), Recall(), Precision()])

    # use this to continue training
    # model = load_model(path_to_model, compile=True)

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
                        initial_epoch=0,  # If you are continuing an interrupted section change here
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=1)
    # Save final result
    model.save(f'{path_to_model}', include_optimizer=True)
    save_model_info()


def save_model_info():
    from datetime import datetime
    model_info = config
    model_info['saved_time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    model_name = re.sub(r'\..+', '', os.path.basename(path_to_model))
    with open(os.path.dirname(os.path.abspath(path_to_model)) + '/zinfo_{}.json'.format(model_name), 'w+') as fout:
        json.dump(model_info, fout, indent=4, default=str)


def get_class_weights(train_df):
    if classification_type == 'multi-class':
        train_df = train_df[train_df['labels'].apply(lambda x: len(x) == 1)]
    elif classification_type == 'multi-label':
        train_df = train_df[train_df['labels'].apply(lambda x: len(x) >= 1)]
    y_train_labels = MultiLabelBinarizer(classes=config['labels']).fit_transform(train_df.labels)
    return generate_class_weights(class_series=y_train_labels, one_hot_encoded=True,
                                  multi_class=True if classification_type != 'multi-label' else False)


train_model()