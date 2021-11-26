import os
import json
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from load_data import config, get_tis_data_split, get_ptb_data_split, DataGenerator
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
    classification_report, confusion_matrix


train_config = config["training_config"]
classification_type = config["classification_type"]
data_source = config['data_config']['source']
model_path = config['path_to_model']
experiment_name = config['test_experiment_name']


data_split_func = {
    'tis': get_tis_data_split,
    'ptb': get_ptb_data_split
}


def predict():
    # load data generator
    if data_source != 'both':
        _, _, test_df, _, _, y_test_labels = data_split_func[data_source](classification_type)
    else:
        _, _, test_df_tis, _, _, y_test_labels_tis = data_split_func['tis'](
            classification_type)
        _, _, test_df_ptb, _, _, y_test_labels_ptb = data_split_func['ptb'](
            classification_type)

        test_df = pd.concat([test_df_tis, test_df_ptb], axis=0, ignore_index=True, sort=False)
        y_test_labels = np.concatenate([y_test_labels_tis, y_test_labels_ptb], axis=0)

    test_gen = DataGenerator(x_df=test_df, y_labels=y_test_labels, source=data_source)

    # load model and predict on test generator
    model = load_model(model_path, compile=False)
    y_pred = model.predict(test_gen)

    # get labels as there might be artefact records which were discarded
    y_test_labels = test_gen.get_labels()

    # compute model's quality and save results
    if not os.path.exists('./results/quality/'):
        os.makedirs('./results/quality/')

    y_test_labels = np.argmax(y_test_labels, axis=1) if classification_type == 'multi-class' else y_test_labels
    y_pred_labels = np.argmax(y_pred, axis=1) if classification_type == 'multi-class' else preprocessing(y_pred)

    metrics = get_metrics(y_test_labels, y_pred_labels, classification_type)
    with open('./results/quality/{}_quality.json'.format(experiment_name), 'w') as f:
        json.dump(metrics, f, indent=4)

    return y_pred


def get_metrics(y_test, y_pred, classification='multi-class'):

    label_names = config['labels']
    if classification == 'multi-class':
        conf_matrix = confusion_matrix(y_test, y_pred)
        with open('./results/quality/{}_confusion_matrix.pickle'.format(experiment_name), 'wb')as f:
            pickle.dump(conf_matrix, f)
        with open('./results/quality/{}_metrics_per_label.txt'.format(experiment_name), 'w')as f:
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
