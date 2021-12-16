import os
import json
import math
import ast
import wfdb
import zipfile
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

with open('config.json', 'r') as fin:
    config = json.load(fin)

data_config = config['data_config']
training_config = config['training_config']

SEED = 1


class DataGenerator(Sequence):

    def __init__(self, x_df, source,
                 classification_type=config['classification_type'],
                 batch_size=training_config['batch_size'],
                 needed_length=data_config['points'],
                 pad_mode=data_config['pad_mode'],
                 norm_by=data_config['norm_by']):

        self.batch_size = batch_size
        self.source = source
        self.needed_length = needed_length
        self.pad_mode = pad_mode
        self.norm_by = norm_by

        self.x = x_df
        if self.source == 'both':
            self.x = shuffle(x_df, random_state=SEED)
        if classification_type == 'multi-class':
            self.x = self.x[self.x['labels'].apply(lambda x: len(x) == 1)]
        elif classification_type == 'multi-label':
            self.x = self.x[self.x['labels'].apply(lambda x: len(x) >= 1)]
        self.x = self.x.reset_index()

        self.is_artefact = np.array([True] * self.x.shape[0], dtype=bool)
        self.mlb = MultiLabelBinarizer(classes=config['labels'])
        self.y = []

    def get_labels(self):
        return np.array(self.y[self.batch_size:])

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        end = min(self.x.shape[0], (idx + 1) * self.batch_size)
        df_batch = self.x[idx * self.batch_size:end]

        if self.source == 'ptb':
            records, labels = load_raw_data_ptb(df_batch, needed_length=self.needed_length,
                                                pad_mode=self.pad_mode, norm_by=False)
        elif self.source == 'tis':
            records, labels = load_raw_data_tis(df_batch, needed_length=self.needed_length,
                                                pad_mode=self.pad_mode, norm_by=self.norm_by)
        else:
            ptb_df = df_batch[df_batch['tis_codes'].isnull()]
            tis_df = df_batch[~df_batch['tis_codes'].isnull()]
            ptb_records, ptb_labels = load_raw_data_ptb(ptb_df, needed_length=self.needed_length,
                                                        pad_mode=self.pad_mode, norm_by=False)
            tis_records, tis_labels = load_raw_data_tis(tis_df, needed_length=self.needed_length,
                                                        pad_mode=self.pad_mode, norm_by=self.norm_by)
            records = np.concatenate((ptb_records, tis_records))
            labels = np.concatenate((ptb_labels, tis_labels))

        x_batch = []
        y_batch = []
        for record, label in zip(records, labels):
            if not artefact_check(record):
                x_batch.append(record)
                y_batch.append(label)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        y_batch = self.mlb.fit_transform(y_batch)
        self.y.extend(y_batch)
        return x_batch, y_batch


def normalize(norm_by, arr, lead_number):
    if norm_by == 'global_min_max':
        source_lead_min_max_config = config['data_config']['normalization_params']['global_min_max']['tis_lead_min_max']

        if config['used_norm'] == 'new_global':
            (lead_min, lead_max) = source_lead_min_max_config[lead_number]
            X_std = (arr - lead_min) / (lead_max - lead_min)

        if config['used_norm'] == 'new_local':
            X_std = (arr - min(arr)) / (max(arr) - min(arr))

        target_lead_min_max_config = config['data_config']['normalization_params']['global_min_max']['ptb_lead_min_max']
        (lead_min, lead_max) = target_lead_min_max_config[lead_number]
        X_scaled = X_std * (lead_max - lead_min) + lead_min
        return X_scaled


def artefact_check(record):
    record = record.transpose()
    for lead in record:
        if len(set(lead)) == 1:
            return True
    return False


def load_raw_data_ptb(df, needed_length, pad_mode, norm_by):
    path = config['ptb_path']
    if data_config['sampling_rate'] == 100:
        data = [wfdb.rdsamp(path + f)[0] for f in df.filename_lr]
    elif data_config['sampling_rate'] == 500:
        data = [wfdb.rdsamp(path + f)[0] for f in df.filename_hr]

    data = np.array(data).transpose(0, 2, 1)
    len_ = min(len(data[0][0]), needed_length)

    if norm_by:
        # normalization
        normalized_data = []
        for ecg in data:
            norm_ecg = []
            for i, lead_data in enumerate(ecg):
                norm_ecg.append(normalize(norm_by, lead_data[:len_], i))
            normalized_data.append(norm_ecg)
        data = np.array(normalized_data)

    data = np.array(data).transpose(0, 2, 1)

    if len_ < needed_length:
        # padding
        npad = [(0, 0)] * data[0].ndim
        npad[0] = (0, needed_length - len_)
        data = np.array([np.pad(ecg_rec, pad_width=npad, mode=pad_mode) for ecg_rec in data])
    return data, df.labels


def load_raw_data_tis(df, needed_length, pad_mode, norm_by):
    # considering that all the files in df are sampled as config['sampling_rate']
    zf = zipfile.ZipFile(config['tis_path'])
    files = ['technion_ecg_data/'+file for file in list(df.filename) if file.endswith('.csv')]
    X = []
    for i in range(len(files)):
        row_data = []
        file_df = pd.read_csv(zf.open(files[i]), skiprows=10, sep=',')
        file_df.columns = file_df.columns.str.strip()
        for j, lead in enumerate(file_df.columns):
            len_ = min(len(file_df[lead]), needed_length)
            lead_data = np.array(file_df[lead][:len_])
            if norm_by:
                # normalization
                lead_data = normalize(norm_by, lead_data, j)
            if len_ < needed_length:
                # padding
                lead_data = np.pad(lead_data, (0, needed_length - lead_data.shape[0]), mode=pad_mode)
            row_data.append(lead_data)
        row_data = np.stack(row_data, axis=0)
        row_data = np.transpose(row_data)
        X.append(row_data)
    return np.array(X), df.labels


def get_tis_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    df = pd.read_csv('./tis_database.csv')
    if classification == 'binary':
        df.labels = pd.Series([[label] for label in df.labels])
    else:
        df.labels = df.labels.apply(lambda x: ast.literal_eval(x))
    if classification == 'multi-class':
        df = df[df['labels'].apply(lambda x: len(x) == 1)]
    elif classification == 'multi-label':
        df = df[df['labels'].apply(lambda x: len(x) >= 1)]

    train_df, test_df = train_test_split(df, test_size=test_portion, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=val_portion, random_state=SEED)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def get_ptb_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    df = pd.read_csv('./ptbxl_database.csv')
    if classification == 'binary':
        df.labels = pd.Series([[label] for label in df.labels])
    else:
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

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df