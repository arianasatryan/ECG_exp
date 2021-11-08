from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Accuracy, Recall, Precision
from model import get_model
from load_data import config, get_data_generators

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
    train_gen, val_gen, _, class_weights = get_data_generators(classification_type=classification_type,
                                                               return_weights=True, data_source='both',
                                                               batch_size=32, needed_length=5000, pad_mode='constant')

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    activation_func = 'softmax' if classification_type == 'multi-class' else 'sigmoid'

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
