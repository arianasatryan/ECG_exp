import os
import json
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from load_data import config, get_tis_data_split, get_ptb_data_split, DataGenerator, load_info_df
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix


SEED = config["SEED"]
label_names = config['labels']
path_to_model = config['path_to_model']
train_config = config["training_config"]
classification_type = config["classification_type"]
data_source = config['data_config']['source']
experiment_name = config['test_experiment_name']


data_split_func = {
    'tis': get_tis_data_split,
    'ptb': get_ptb_data_split
}


def predict():
    # load data generator
    if data_source != 'both':
        _, _, test_df = data_split_func[data_source](classification_type, 0.2, 0.2)
        #test_df, _, _ = load_info_df(classification_type, 'test')

    print(test_df.shape)
    print(test_df.labels.value_counts())
    test_gen = DataGenerator(x_df=test_df, source=data_source)

    # load model and predict on test generator
    model = load_model(path_to_model, compile=False)
    y_pred = model.predict(test_gen)

    # get labels as there might be artefact records which were discarded
    y_test_labels = test_gen.get_labels()

    # compute model's quality and save results
    if not os.path.exists('./results/new/'):
        os.makedirs('./results/new/')

    y_test_labels = np.argmax(y_test_labels, axis=1) if classification_type != 'multi-label' else y_test_labels
    y_pred = np.argmax(y_pred, axis=1) if classification_type != 'multi-label' else preprocessing(y_pred)

    metrics = get_metrics(y_test_labels, y_pred, classification_type)
    with open('./results/new/{}_quality.json'.format(experiment_name), 'w') as f:
        json.dump(metrics, f, indent=4)

    return y_pred


def get_metrics(y_test, y_pred, classification='multi-class'):
    if classification != 'multi-label':
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        f = sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="coolwarm")
        f.figure.savefig('./results/new/{}_confusion_matrix.png'.format(experiment_name))
    with open('./results/new/{}_metrics_per_label.txt'.format(experiment_name), 'w')as f:
        f.write(classification_report(y_test, y_pred, target_names=label_names))

    if classification == 'binary':
        return {
                'accuracy': accuracy_score(y_test, y_pred).round(3),
                'recall': recall_score(y_test, y_pred, average="binary", pos_label=1).round(3),
                'precision': precision_score(y_test, y_pred, average="binary", pos_label=1).round(3),
                'f1-score': f1_score(y_test, y_pred, average="binary", pos_label=1).round(3),
                }

    return {
        'accuracy': accuracy_score(y_test, y_pred).round(3),
        'recall_macro': recall_score(y_test, y_pred, average="macro").round(3),
        'recall_micro': recall_score(y_test, y_pred, average="micro").round(3),
        'precision_macro': precision_score(y_test, y_pred, average="macro").round(3),
        'precision_micro': precision_score(y_test, y_pred, average="micro").round(3),
        'f1-score_macro': f1_score(y_test, y_pred, average="macro").round(3),
        'f1-score_micro': f1_score(y_test, y_pred, average="micro").round(3)
    }


def preprocessing(y_pred):
    y_pred_labels = []
    for pred in y_pred:
        pred_label = []
        for sample in pred:
            label = 1 if sample >= 0.5 else 0
            pred_label.append(label)
        y_pred_labels.append(pred_label)
    return np.array(y_pred_labels)

predict()