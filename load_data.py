import os
import json
import math
import ast
import wfdb
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from statistics import mean

with open('config.json', 'r') as fin:
    config = json.load(fin)

data_config = config['data_config']
training_config = config['training_config']
data_source_df = data_config['source_df']

SEED = config["SEED"]


def load_info_df(classification=config['classification_type'], train_or_test='train'):
    print(data_config['source'], train_or_test)
    df = pd.read_csv(data_source_df[data_config['source']][train_or_test])
    df = df[df['sampling_rate'] == data_config['sampling_rate']]

    if classification == 'binary':
        df.labels = df.labels.apply(lambda x: [x])
    else:
        df = df.drop('labels', axis=1)
        df = df.rename(columns={"multilabels": "labels"})
        df.labels = df.labels.apply(lambda x: ast.literal_eval(x))
    if classification == 'multi-class':
        df = df[df['labels'].apply(lambda x: len(x) == 1)]
    elif classification == 'multi-label':
        df = df[df['labels'].apply(lambda x: len(x) >= 1)]
    return df


class DataGenerator(Sequence):

    def __init__(self, x_df, source=data_config['source'],
                 batch_size=training_config['batch_size'],
                 needed_length=data_config['points'],
                 pad_mode=data_config['pad_mode'],
                 scale_by=data_config['scale_by'],
                 resample=data_config['resample'],
                 sampling_rate=data_config['sampling_rate']):

        self.batch_size = batch_size
        self.source = source
        self.needed_length = needed_length
        self.pad_mode = pad_mode
        self.scale_by = scale_by
        self.sampling_rate = sampling_rate
        self.resample = resample

        self.x = x_df
        self.x = self.x.reset_index(drop=True)
    
        if self.source == 'both':
            self.x = self.x.sample(frac=1, random_state=SEED).reset_index(drop=True)

        self.mlb = MultiLabelBinarizer(classes=config['labels'])
        self.y = []

    def get_labels(self):
        return np.array(self.y[self.batch_size:])

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def _load_raw_data_ptb(self, df_batch):
        path = config['ptb_path']
        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(path + f)[0] for f in df_batch.filename]
        elif self.sampling_rate == 500:
            data = [wfdb.rdsamp(path + f)[0] for f in df_batch.filename]

        data = np.array(data).transpose(0, 2, 1)
        X, Y = [], []
        for record, labels, sampling_rate in zip(data, df_batch.labels):
            preprocessed_record, is_artefact = self._preprocess(record, self.sampling_rate, 'ptb')
            if not is_artefact:
                X.append(preprocessed_record)
                Y.append(labels)
        return np.array(X), np.array(Y)

    def _load_raw_data_tis(self, df_batch):
        # considering that all the files in df are sampled as config['sampling_rate']
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        zf = zipfile.ZipFile(config['tis_path'])
        folder = [name for name in zf.namelist() if name.endswith('/')][0]
        X, Y = [], []
        for filename, labels, sampling_rate in zip(list(df_batch.filename),
                                                   list(df_batch.labels),
                                                   list(df_batch.sampling_rate)):
            df = pd.read_csv(zf.open(folder+filename), sep='\n', header=None)
            ecg_rec_start = df[df.columns[0]].apply(lambda x: all(lead in x for lead in leads))
            ecg_rec_start = ecg_rec_start[ecg_rec_start].index.tolist()[0]

            record = pd.read_csv(zf.open(folder+filename), skiprows=ecg_rec_start, sep=',').transpose().to_numpy()
            preprocessed_record, is_artefact = self._preprocess(record, sampling_rate, 'tis')
            if not is_artefact:
                X.append(preprocessed_record)
                Y.append(labels)
        return np.stack(X), np.asarray(Y)

    def _scale(self, arr, lead_number, source):
        target = 'ptb' if source == 'tis' else 'tis'
        if self.scale_by == 'global_min_max':
            source_lead_min_max_config = data_config['scale_params']['global_min_max'][f'{source}_lead_min_max']
            (lead_min, lead_max) = source_lead_min_max_config[lead_number]
            X_std = (arr - lead_min) / (lead_max - lead_min)

            target_lead_min_max_config = data_config['scale_params']['global_min_max'][f'{target}_lead_min_max']
            (lead_min, lead_max) = target_lead_min_max_config[lead_number]
            X_scaled = X_std * (lead_max - lead_min) + lead_min
        return X_scaled

    def _preprocess(self, ecg_record, sampling_rate, source):
        is_artefact = False
        row_data = []
        for j, lead_data in enumerate(ecg_record):
            if len(set(lead_data.shape)) == 0:
                # check artefact
                is_artefact = True
                break

            if self.resample and sampling_rate != data_config['sampling_rate']:
                # resampling
                step = int(sampling_rate / data_config['sampling_rate'])
                lead_data = [int(mean(lead_data[i: i + step])) for i in range(0, len(lead_data) - 1, step)]

            if self.scale_by:
                # normalization
                lead_data = self._scale(lead_data, j, source)

            len_ = min(len(lead_data), self.needed_length)
            lead_data = lead_data[:len_]
            if len(lead_data) < self.needed_length:
                # padding
                lead_data = np.pad(lead_data, (0, self.needed_length - lead_data.shape[0]), mode=self.pad_mode)

            row_data.append(lead_data)

        if not is_artefact:
            row_data = np.stack(row_data, axis=0)
            row_data = np.transpose(row_data)
        return row_data, is_artefact

    def __getitem__(self, idx):
        end = min(self.x.shape[0], (idx + 1) * self.batch_size)
        df_batch = self.x[idx * self.batch_size:end]

        if self.source == 'ptb':
            X, Y = self._load_raw_data_ptb(df_batch)

        elif self.source == 'tis':
            X, Y = self._load_raw_data_tis(df_batch)
            
        elif self.source == 'both':
            X, Y = self._load_raw_data_tis(df_batch[df_batch['source'] == 'tis'])
            X1, Y1 = self._load_raw_data_ptb(df_batch[df_batch['source'] == 'ptb'])
            X.extend(X1)
            Y.extend(Y1)

        Y = self.mlb.fit_transform(Y)
        self.y.extend(Y)
        return X, Y


def get_tis_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    train_df = load_info_df(classification)
    val_df, test_df = None, None

    if test_portion is not None:
        train_df, test_df = train_test_split(train_df, test_size=test_portion, random_state=SEED)
        test_df = test_df.reset_index(drop=True)

    if val_portion is not None:
        train_df, val_df = train_test_split(train_df, test_size=val_portion, random_state=SEED)
        val_df = val_df.reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    return train_df, val_df, test_df


def get_ptb_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    df = load_info_df(classification)
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
