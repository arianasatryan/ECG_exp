from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from load_data import train_val_test_split, config

path_to_model = config["path_to_model"]

loss = 'categorical_crossentropy'
lr = 0.001
opt = Adam(lr)


def predict():
    X_train, X_val,  X_test, y_train, y_val, y_test = train_val_test_split()

    model = load_model(path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_pred = model.predict(X_test)
    return y_pred
