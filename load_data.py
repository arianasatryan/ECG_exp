
import pandas as pd
import numpy as np
import wfdb
import ast
import json
import math
import os
import zipfile
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from math import ceil
from sklearn.utils import shuffle
from generate_class_weights import generate_class_weights


with open('config.json', 'r')as fin:
    config = json.load(fin)

SEED = 1


def get_data_generators(classification_type, data_source='both', return_weights=True,
                        batch_size=32, needed_length=5000, pad_mode='constant'):
    if data_source == 'tis':
        train_df, val_df, test_df, y_train_labels, y_val_labels, y_test_labels = get_tis_data_split(classification_type)
    elif data_source == 'ptb':
        train_df, val_df, test_df, y_train_labels, y_val_labels, y_test_labels = get_ptb_data_split(classification_type)
    elif data_source == 'both':
        train_df_tis, val_df_tis, test_df_tis, y_train_labels_tis, y_val_labels_tis, y_test_labels_tis = \
            get_tis_data_split(classification_type)
        train_df_ptb, val_df_ptb, test_df_ptb, y_train_labels_ptb, y_val_labels_ptb, y_test_labels_ptb = \
            get_ptb_data_split(classification_type)

        train_df = pd.concat([train_df_tis, train_df_ptb], axis=0, ignore_index=True)
        y_train_labels = np.concatenate([y_train_labels_tis, y_train_labels_ptb], axis=0)

        val_df = pd.concat([val_df_tis, val_df_ptb], axis=0, ignore_index=True)
        y_val_labels = np.concatenate([y_val_labels_tis, y_val_labels_ptb], axis=0)

        test_df = pd.concat([test_df_tis, test_df_ptb], axis=0, ignore_index=True)
        y_test_labels = np.concatenate([y_test_labels_tis, y_test_labels_ptb], axis=0)

    train_gen = DataGenerator(train_df, y_train_labels, data_source, batch_size, needed_length, pad_mode)
    val_gen = DataGenerator(val_df, y_val_labels,  data_source, batch_size, needed_length, pad_mode)
    test_gen = DataGenerator(test_df, y_test_labels,  data_source, batch_size, needed_length, pad_mode)

    if not return_weights:
        return train_gen, val_gen, test_gen

    multi_class = True if classification_type == 'multi-class' else False
    class_weights = generate_class_weights(y_train_labels, multi_class=multi_class, one_hot_encoded=True)
    return train_gen, val_gen, test_gen, class_weights


class DataGenerator(Sequence):

    def __init__(self, x_df, y_labels, source, batch_size=32, needed_length=5000, pad_mode='constant'):
        self.batch_size = batch_size
        self.source = source
        self.needed_length = needed_length
        self.pad_mode = pad_mode
        if self.source == 'both':
            self.x, self.y = shuffle(x_df, y_labels, random_state=SEED)
        else:
            self.x, self.y = x_df, y_labels

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        end = min(self.x.shape[0], (idx + 1) * self.batch_size)
        if self.source == 'tis':
            x_batch = load_raw_data_tis(self.x[idx * self.batch_size:end], self.needed_length, self.pad_mode)
        elif self.source == 'ptb':
            x_batch = load_raw_data_ptb(self.x[idx * self.batch_size:end], self.needed_length, self.pad_mode)
        elif self.source == 'both':
            ptb_rows = self.x[idx * self.batch_size:end][self.x["tis_codes"].isnull()]
            tis_rows = self.x[idx * self.batch_size:end][~self.x["tis_codes"].isnull()]
            ptb_raw_data = load_raw_data_ptb(ptb_rows, self.needed_length, self.pad_mode)
            tis_raw_data = load_raw_data_tis(tis_rows, self.needed_length, self.pad_mode)
            x_batch = np.concatenate([ptb_raw_data, tis_raw_data], axis=0)
        return x_batch, self.y[idx * self.batch_size:end]


def load_raw_data_ptb(df, needed_length=5000, pad_mode='constant'):
    path = config['ptb_path']
    if config['sampling_rate'] == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    elif config['sampling_rate'] == 500:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]

    if len(data[0][0]) >= needed_length:
        # truncating
        data = np.array([signal[:needed_length] for signal, meta in data])
    else:
        # padding
        data = np.array([np.pad(signal, (0, needed_length - len(data[0][0])), mode=pad_mode) for signal, meta in data])
    return data


def load_raw_data_tis(df, needed_length=5000, pad_mode='constant'):
    # considering that all of the files in df are sampled as config['sampling_rate']
    zf = zipfile.ZipFile(config['tis_path'])
    files = ['technion_ecg_data/'+file for file in list(df.filename)]
    X = []
    for file in files:
        row_data = []
        file_df = pd.read_csv(zf.open(file), skiprows=10, sep=',')
        file_df.columns = file_df.columns.str.strip()
        for lead in config['leads_order']:
            if len(file_df[lead]) >= needed_length:
                # truncating
                lead_data = np.array(file_df[lead][:needed_length])
            else:
                # padding
                lead_data = np.array(file_df[lead])
                lead_data = np.pad(lead_data, (0, needed_length - lead_data.shape[0]), mode=pad_mode)
            row_data.append(lead_data)
        row_data = np.stack(row_data, axis=0)
        row_data = np.transpose(row_data)
        X.append(row_data)
    return np.array(X)


def get_tis_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    df = pd.read_csv('./tis_database.csv')
    df.ptb_labels = df.ptb_labels.apply(lambda x: ast.literal_eval(x))
    if classification == 'multi-class':
        df = df[df['ptb_labels'].apply(lambda x:len(x) == 1)]
    elif classification == 'multi-label':
        df = df[df['ptb_labels'].apply(lambda x:len(x) >= 1)]

    train_df, test_df = train_test_split(df, test_size=test_portion, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=val_portion, random_state=SEED)

    # one-hot encoding labels
    mlb = MultiLabelBinarizer(classes=config['labels'])
    y_train_labels = mlb.fit_transform(train_df.ptb_labels)
    y_val_labels = mlb.fit_transform(val_df.ptb_labels)
    y_test_labels = mlb.fit_transform(test_df.ptb_labels)

    return train_df, val_df, test_df, y_train_labels, y_val_labels, y_test_labels


def get_ptb_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    df = pd.read_csv('./ptbxl_database.csv')
    df.labels = df.labels.apply(lambda x: ast.literal_eval(x))
    if classification == 'multi-class':
        df = df[df['labels'].apply(lambda x: len(x) == 1)]
    elif classification == 'multi-label':
        df = df[df['labels'].apply(lambda x: len(x) >= 1)]

    trusted_folds = [9, 10]
    trusted_df = df[df['strat_fold'].isin(trusted_folds)]
    not_trusted_df = df[~df['strat_fold'].isin(trusted_folds)]
    needed_test_size = math.floor(len(df) * test_portion)

    # splitting train and test
    indices = np.random.RandomState(seed=SEED).permutation(trusted_df.shape[0])
    test_idx, training_idx = indices[:needed_test_size], indices[needed_test_size:]
    test_df = trusted_df.iloc[test_idx, :]
    train_df = pd.concat([trusted_df.iloc[training_idx, :], not_trusted_df])

    # splitting train and validation
    indices = np.random.RandomState(seed=SEED).permutation(train_df.shape[0])
    needed_val_size = math.floor(train_df.shape[0] * val_portion)
    val_idx, training_idx = indices[:needed_val_size], indices[needed_val_size:]
    train_df, val_df = train_df.iloc[training_idx], train_df.iloc[val_idx]

    # one-hot encoding labels
    mlb = MultiLabelBinarizer(classes=config['labels'])
    y_train_labels = mlb.fit_transform(train_df.labels)
    y_val_labels = mlb.fit_transform(val_df.labels)
    y_test_labels = mlb.fit_transform(test_df.labels)

    return train_df, val_df, test_df, y_train_labels, y_val_labels, y_test_labels


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
        file_df = pd.read_csv(config['tis_path'] + file, nrows=5, sep=':')
        if int(file_df.iloc[0][1]) == config['sampling_rate']:
            tis_codes = ast.literal_eval(file_df.iloc[4][1].strip())
            ptb_labels = map_labels(tis_codes, source='tis')
            ptb_labels = [key for key in ptb_labels if key in config['labels']]
            if ptb_labels:
                info.append({'filename': file, 'ptb_labels': ptb_labels, 'tis_codes': tis_codes})
    Y = pd.DataFrame(info)
    Y.to_csv('./tis_database.csv', index=False)





