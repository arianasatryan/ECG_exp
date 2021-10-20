import pandas as pd
import numpy as np
import wfdb
import ast
import json
import math
import random

with open('config.json', 'r')as fin:
    config = json.load(fin)

SEED = 1


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def filter_labels(x):
    return [key for key in x.keys() if key in config['labels']]


def discard_other_labels(source_file, source='ptb'):
    if source == 'ptb':
        # load and convert annotation data
        Y = pd.read_csv(source_file, index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        # filter needed labels
        Y['labels'] = Y.scp_codes.apply(filter_labels)
        Y['labels'] = pd.Series([item for sublist in list(Y['labels']) for item in sublist])
        Y = Y[Y['labels'].str.len() != 0]
        Y = Y[Y['labels'].notna()]
        # adding source label '0'
        Y['source'] = 0
    return Y


def save_per_label(Y, collected_datasets_path):
    for label in set(Y['labels']):
        sub_df = Y[Y['labels'] == label]
        sub_df.to_csv(collected_datasets_path + '{}_subdataset.csv'.format(label))


def get_train_test_split(label):
    trusted_folds = [9, 10]
    Y = pd.read_csv(config['collected_datasets_path'] + '{}_subdataset.csv'.format(label))
    trusted_df = Y[Y['strat_fold'].isin(trusted_folds)]
    not_trusted_df = Y[~Y['strat_fold'].isin(trusted_folds)]

    needed_test_size = math.floor(len(Y) * 0.2)

    if needed_test_size < len(trusted_df):
        random.seed(SEED)
        test_indexes = random.sample(range(len(trusted_df)), needed_test_size)
        remaining_indexes = list(set(list(range(len(trusted_df)))) - set(test_indexes))

        y_test = trusted_df.iloc[test_indexes, :]
        y_train = pd.concat([trusted_df.iloc[remaining_indexes, :], not_trusted_df])

    else:
        random.seed(SEED)
        test_indexes = random.sample(range(len(not_trusted_df)), needed_test_size - len(trusted_df))
        remaining_indexes = list(set(list(range(len(not_trusted_df)))) - set(test_indexes))

        y_test = pd.concat([trusted_df, not_trusted_df.iloc[test_indexes, :]])
        y_train = not_trusted_df.iloc[remaining_indexes, :]

    # adding SR examples
    sr_df = pd.read_csv(config['collected_datasets_path'] + 'SR_subdataset.csv')
    trusted_sr_df = sr_df[sr_df['strat_fold'].isin(trusted_folds)]
    random.seed(SEED)
    test_indexes = random.sample(range(len(trusted_sr_df)), y_test.shape[0])
    remaining_indexes = list(set(list(range(len(sr_df)))) - set(test_indexes))
    random.seed(SEED)
    train_indexes = random.sample(remaining_indexes, y_train.shape[0])

    y_test = pd.concat([y_test, sr_df.iloc[test_indexes, :]])
    y_train = pd.concat([y_train, sr_df.iloc[train_indexes, :]])

    x_train = load_raw_data(y_train, config['sampling_rate'], config['ptb_path'])
    x_test = load_raw_data(y_test, config['sampling_rate'], config['ptb_path'])

    y_train["labels"].replace({"SR": 0, label: 1}, inplace=True)
    y_test["labels"].replace({"SR": 0, label: 1}, inplace=True)

    return x_train, x_test, y_train["labels"], y_test["labels"]


"""
# check number of instances
stat = json.loads(Y['labels'].value_counts().to_json())
with open(path + 'sub_ptbxl_stat.json', 'w') as fout:
    json.dump(stat, fout, indent=4)

# Load raw signal data
# i.e. for 'CRBBB' label
our_label_df = pd.read_csv(config['collected_datasets_path']+'CRBBB_subdataset.csv', index_col='ecg_id')
X = load_raw_data(our_label_df, config['sampling_rate'], config['ptb_path'])

# Separately save scv files for each label
labeled_df = discard_other_labels(config['ptb_path']+'ptbxl_database.csv', source='ptb')
save_per_label(labeled_df, config['collected_datasets_path'])

# Get splits for given label
# i.e. for 'CRBBB' label
X_train, X_test, y_train, y_test = get_train_test_split('CRBBB')
"""

