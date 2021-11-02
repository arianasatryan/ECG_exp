from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
                            classification_report, confusion_matrix
import numpy as np

import json
import pickle

from load_data import train_val_test_split, config, get_tis_data

path_to_model = config["path_to_model"]

loss = 'categorical_crossentropy'
lr = 0.001
opt = Adam(lr)


def predict():
    #X_train, X_val,  X_test, y_train, y_val, y_test = train_val_test_split()
    X_test, y_test = get_tis_data(classification='multi-class')

    model = load_model(path_to_model, compile=False)
    
    y_pred = model.predict(X_test, batch_size=16)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    metrics = get_metrics(y_test_labels, y_pred_labels)
    with open('./quality.json', 'w')as f:
        json.dump(metrics, f, indent=4)
    return y_pred


def get_metrics(y_test, y_pred):
    label_names = config['labels']
    conf_matrix = confusion_matrix(y_test, y_pred)
    with open('./confusion_matrix.pickle', 'wb')as f:
        pickle.dump(confusion_matrix, f)
    with open('./metrics_per_label.txt', 'w')as f:
        f.write(classification_report(y_test, y_pred, target_names=label_names))
    return {'accuracy': accuracy_score(y_test, y_pred).round(3),
            'recall_macro': recall_score(y_test, y_pred, average="macro").round(3),
            'recall_micro': recall_score(y_test, y_pred, average="micro").round(3),
            'f1-score_macro': f1_score(y_test, y_pred, average="macro").round(3),
            'f1-score_micro': f1_score(y_test, y_pred, average="micro").round(3),
            'precision_macro': precision_score(y_test, y_pred, average="macro").round(3),
            'precision_micro': precision_score(y_test, y_pred, average="micro").round(3)}

