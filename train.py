from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Accuracy, Recall, Precision

from model import get_model
from load_data import train_val_test_split, config

train_config = config["training"]
classification_type = 'multi-label'


def train_generator(X_train, y_train):
    batch_size = train_config["batch_size"]
    for j in range(0, train_config["epochs"]):
        i = 0
        while i < X_train.shape[0]:
            if i + batch_size < X_train.shape[0]:
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
            else:
                X_batch = X_train[i:]
                y_batch = y_train[i:]
            i += batch_size
            yield X_batch, y_batch


def val_generator(X_val, y_val):
    batch_size = train_config["batch_size"]
    for j in range(0, train_config["epochs"]):
        i = 0
        while i < X_val.shape[0]:
            if i + batch_size < X_val.shape[0]:
                X_batch = X_val[i:i + batch_size]
                y_batch = y_val[i:i + batch_size]
            else:
                X_batch = X_val[i:]
                y_batch = y_val[i:]
            i += batch_size
            yield X_batch, y_batch


def train_model(train_config):
    optimizer = Adam(train_config["lr"])
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=train_config["lr"] / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    #add data spliting
    X_train, X_val,  X_test, y_train, y_val, y_test = train_val_test_split()

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(len(train_config["labels"]))
    model.compile(loss=train_config["loss"], optimizer=optimizer, metrics=[Accuracy(), Recall(), Precision()])
    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./{}_backup_model_last.hdf5'.format(classification_type)),
                  ModelCheckpoint('./{}_backup_model_best.hdf5'.format(classification_type), save_best_only=True)]
    # Train neural network
    history = model.fit(train_generator(X_train, y_train),
                        steps_per_epoch=X_train.shape[0]//train_config["batch_size"],
                        validation_data=val_generator(X_val, y_val),
                        epochs=train_config["epochs"],
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        verbose=1)
    # Save final result
    model.save("./{}_final_model.hdf5".format(classification_type))