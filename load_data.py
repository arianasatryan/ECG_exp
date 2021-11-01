import pandas as pd
import numpy as np
import wfdb
import ast
import json
import math
import os
from sklearn.preprocessing import MultiLabelBinarizer

with open('config.json', 'r')as fin:
    config = json.load(fin)

SEED = 1


def load_raw_data_ptb(df, path):
    if config['sampling_rate'] == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    elif config['sampling_rate'] == 500:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def load_raw_data_tis(df, path, needed_length=5000, pad_mode='constant'):
    # considering that all of the files in df are sampled as config['sampling_rate']
    files = [path + file for file in list(df.filename)]
    X = []
    for file in files:
        row_data = []
        file_df = pd.read_csv(file, skiprows=10, sep=',')
        file_df.columns = file_df.columns.str.strip()
        for lead in config['leads_order']:
            if len(file_df[lead]) >= needed_length:
                # truncating
                lead_data = np.array(file_df[lead][:needed_length])
            else:
                # padding
                lead_data = np.array(file_df[lead])
                lead_data = np.pad(lead_data, (0, 5000 - lead_data.shape[0]), mode=pad_mode)
            row_data.append(lead_data)
        row_data = np.stack(row_data, axis=0)
        row_data = np.transpose(row_data)
        X.append(row_data)
    return np.array(X)


def filter_labels(x):
    return [key for key in x.keys() if key in config['labels']]


def train_val_test_split(test_portion=0.2, val_portion=0.2):
    Y = pd.read_csv(config['ptb_path'] + 'ptbxl_database.csv')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # selecting config labels
    Y['labels'] = Y.scp_codes.apply(filter_labels)
    Y = Y[Y['labels'].apply(lambda x: len(x) != 0)]

    trusted_folds = [9, 10]
    trusted_df = Y[Y['strat_fold'].isin(trusted_folds)]
    not_trusted_df = Y[~Y['strat_fold'].isin(trusted_folds)]
    needed_test_size = math.floor(len(Y) * test_portion)

    # splitting train and test
    indices = np.random.RandomState(seed=SEED).permutation(trusted_df.shape[0])
    test_idx, training_idx = indices[:needed_test_size], indices[needed_test_size:]
    y_test = trusted_df.iloc[test_idx, :]
    y_train = pd.concat([trusted_df.iloc[training_idx, :], not_trusted_df])

    x_train = load_raw_data_ptb(y_train, config['sampling_rate'], config['ptb_path'])
    x_test = load_raw_data_ptb(y_test, config['sampling_rate'], config['ptb_path'])

    # splitting train and validation
    indices = np.random.RandomState(seed=SEED).permutation(x_train.shape[0])
    needed_val_size = math.floor(len(x_train) * val_portion)
    val_idx, training_idx = indices[:needed_val_size], indices[needed_val_size:]
    x_train, x_val = x_train[training_idx, :], x_train[val_idx, :]
    y_train, y_val = y_train.iloc[training_idx], y_train.iloc[val_idx]


    # one-hot encoding labels
    mlb = MultiLabelBinarizer(classes=config['labels'])
    y_train_labels = mlb.fit_transform(y_train.labels)
    y_val_labels = mlb.fit_transform(y_val.labels)
    y_test_labels = mlb.fit_transform(y_test.labels)

    return x_train, x_val, x_test, y_train_labels, y_val_labels, y_test_labels


def map_labels(labels, source='tis'):
    # if source='tis' the label is tis_code, else the label is ptb label
    map_df = pd.read_csv(config['pathology_mapping_file'])
    map_df['ТИС'] = map_df['ТИС'].apply(lambda x: str(x).split(','))
    if source == 'tis':
        mapped = map_df[map_df['ТИС'].apply(lambda x: len(set(labels).intersection(x))) != 0]['PTB-XL']
        mapped = mapped.dropna()
        mapped_labels = list(mapped)
    else:
        mapped_labels = []
        mapped = map_df[map_df['PTB-XL'].isin(labels)]['ТИС']
        mapped = mapped.dropna()
        for row in mapped:
            mapped_labels.extend(list(row))
    return mapped_labels


def save_filtered():
    # filter ptb
    Y = pd.read_csv(config['ptb_path'] + 'ptbxl_database.csv')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y['labels'] = Y.scp_codes.apply(lambda x: [key for key in x.keys() if key in config['labels']])
    Y = Y[Y['labels'].apply(lambda x: len(x) != 0)]
    Y.to_csv('./ptbxl_database.csv', index=False)

    # filter tis
    files = [file for file in os.listdir(config['tis_path'])if file.endswith('.csv')]
    i = 1
    info = []
    for file in files:
        print(i)
        file_df = pd.read_csv(config['tis_path'] + file, nrows=5, sep=':')
        if int(file_df.iloc[0][1]) == config['sampling_rate']:
            tis_codes = ast.literal_eval(file_df.iloc[4][1].strip())
            ptb_labels = map_labels(tis_codes, source='tis')
            ptb_labels = [key for key in ptb_labels if key in config['labels']]
            if ptb_labels:
                info.append({'filename': file, 'ptb_labels': ptb_labels, 'tis_codes': tis_codes})
        i += 1
    Y = pd.DataFrame(info)
    Y.to_csv('./tis_database.csv', index=False)


save_filtered()





