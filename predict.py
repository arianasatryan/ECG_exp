from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
                            classification_report, confusion_matrix
import numpy as np
import json

from load_data import train_val_test_split, config

path_to_multi_class_model = config["path_to_multi_class_model"]
path_to_multi_label_model = config["path_to_multi_label_model"]

multi_class_loss = 'categorical_crossentropy'
multi_label_loss = 'binary_crossentropy'
lr = 0.001
opt = Adam(lr)


def multi_class_predict():
    X_train, X_val,  X_test, y_train, y_val, y_test = train_val_test_split()

    model = load_model(path_to_multi_class_model, compile=False)
    model.compile(loss=multi_class_loss, optimizer=Adam())

    y_test_labels = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    metrics = get_metrics(y_test_labels, y_pred_labels)
    with open('./quality.json', 'w')as f:
        json.dump(metrics, f, indent=4)

    return y_pred


def multi_label_predict():
    X_train, X_val,  X_test, y_train, y_val, y_test = train_val_test_split()

    model = load_model(path_to_multi_label_model, compile=False)
    model.compile(loss=multi_label_loss, optimizer=Adam())

    y_pred = model.predict(X_test)
    y_pred_labels = preprocessing(y_pred)

    metrics = get_metrics(y_test, y_pred_labels)
    with open('./multi_label_quality.json', 'w')as f:
         json.dump(metrics, f, indent=4)

    return y_pred


def get_metrics(y_test, y_pred):
    return {'accuracy': accuracy_score(y_test, y_pred).round(3),
            'recall_macro': recall_score(y_test, y_pred, average="macro").round(3),
            'recall_micro': recall_score(y_test, y_pred, average="micro").round(3),
            'f1-score_macro': f1_score(y_test, y_pred, average="macro").round(3),
            'f1-score_micro': f1_score(y_test, y_pred, average="micro").round(3),
            'precision_macro': precision_score(y_test, y_pred, average="macro").round(3),
            'precision_micro': precision_score(y_test, y_pred, average="micro").round(3)}


def preprocessing(y_pred):
    y_pred_labels = []
    for pred in y_pred:
        pred_label = []
        for sample in pred:
            label = 1 if sample >= 0.5 else 0
            pred_label.append(label)
        y_pred_labels.append(pred_label)
    return np.array(y_pred_labels)

