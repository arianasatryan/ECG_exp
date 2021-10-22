from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision, SensitivityAtSpecificity
from model import get_model
import tensorflow as tf
from load_data import train_val_test_split, config

label = config['label']

if __name__ == "__main__":
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    #add data spliting
    X_train, X_val,  X_test, y_train, y_val, y_test = train_val_test_split()

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(7)
    model.compile(loss=loss, optimizer=opt, metrics=[BinaryAccuracy(), Recall(), Precision()])
    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./{}backup_model_last.hdf5'.format(label)),
                  ModelCheckpoint('./{}backup_model_best.hdf5'.format(label), save_best_only=True)]
    # Train neural network
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=70,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        verbose=1)
    # Save final result
    model.save("./models/{}_final_model.hdf5".format(label))