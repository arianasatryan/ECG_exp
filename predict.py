from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
    classification_report, confusion_matrix
import numpy as np
import json
import pickle
from load_data import config, get_tis_data_split, get_ptb_data_split, DataGenerator

classification_type = config['classification_type']
experiment_name = '{}_ptb8_tis2'.format(classification_type)


def predict():
    model_path = config["path_to_multi_class_model"] if classification_type == 'multi-class' \
        else config["path_to_multi_label_model"]
    _, _, test_df, _, _, y_test_labels = get_tis_data_split(classification_type)
    test_gen = DataGenerator(test_df, y_test_labels, source='tis', batch_size=config['training']['batch_size'])

    model = load_model(model_path, compile=False)
    y_pred = model.predict(test_gen)
    y_test_labels = np.argmax(y_test_labels, axis=1) if classification_type == 'multi-class' else y_test_labels
    y_pred_labels = np.argmax(y_pred, axis=1) if classification_type == 'multi-class' else preprocessing(y_pred)

    metrics = get_metrics(y_test_labels, y_pred_labels, classification_type)
    with open('./{}_quality.json'.format(experiment_name), 'w') as f:
        json.dump(metrics, f, indent=4)

    return y_pred


def get_metrics(y_test, y_pred, classification='multi-class'):
    label_names = config['labels']
    if classification == 'multi-class':
        conf_matrix = confusion_matrix(y_test, y_pred)
        with open('./{}_confusion_matrix.pickle'.format(experiment_name), 'wb')as f:
            pickle.dump(conf_matrix, f)
        with open('./{}_metrics_per_label.txt'.format(experiment_name), 'w')as f:
            f.write(classification_report(y_test, y_pred, target_names=label_names))

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

