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


with open('config.json', 'r')as fin:
    config = json.load(fin)
    
data_config = config['data_config']
training_config = config['training_config']
    
SEED = 1


class DataGenerator(Sequence):

    def __init__(self, x_df, y_labels, source, 
                 batch_size=training_config['batch_size'], 
                 needed_length=data_config['points'],
                 pad_mode=data_config['pad_mode'], 
                 norm_by=data_config['norm_by']):
        self.x, self.y = x_df, y_labels
        self.is_artefact = np.array([True] * self.x.shape[0], dtype=bool)
        self.batch_size = batch_size
        self.source = source
        self.needed_length = needed_length
        self.pad_mode = pad_mode
        self.norm_by = norm_by
        if self.source == 'both':
            self.x, self.y = shuffle(x_df, y_labels, random_state=SEED)

    def get_labels(self):
        return np.array([label for label, is_artefact in zip(self.y, self.is_artefact) if not is_artefact])

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        end = min(self.x.shape[0], (idx + 1) * self.batch_size)
        df_batch = self.x[idx * self.batch_size:end]

        if self.source == 'ptb':
            ptb_records = load_raw_data_ptb(df_batch, needed_length=self.needed_length, pad_mode=self.pad_mode,
                                            norm_by=self.norm_by)
            # discarding artefact records
            is_artefact = [np.isnan(np.min(record)) for record in ptb_records]
            order = [list(df_batch.index)[i] for i in range(len(df_batch.index)) if not is_artefact[i]]
            x_batch = np.array([ptb_records[i] for i in range(len(ptb_records)) if not is_artefact[i]])
            self.is_artefact[order] = False
            y_batch = self.y[order]

        elif self.source == 'tis':
            tis_records = load_raw_data_tis(df_batch, needed_length=self.needed_length, pad_mode=self.pad_mode,
                                            norm_by=self.norm_by)
            # discarding artefact records
            is_artefact = [artefact_check(record) for record in tis_records]
            order = [list(df_batch.index)[i] for i in range(len(df_batch.index)) if not is_artefact[i]]
            x_batch = np.array([tis_records[i] for i in range(len(tis_records)) if not is_artefact[i]])
            self.is_artefact[order] = False
            y_batch = self.y[order]

        elif self.source == 'both':
            ptb_df = df_batch[df_batch['tis_codes'].isnull()]
            tis_df = df_batch[~df_batch['tis_codes'].isnull()]
            ptb_records = load_raw_data_ptb(ptb_df, needed_length=self.needed_length,
                                            pad_mode=self.pad_mode, norm_by=self.norm_by)
            tis_records = load_raw_data_tis(tis_df, needed_length=self.needed_length,
                                            pad_mode=self.pad_mode, norm_by=self.norm_by)

            # discarding artefact records
            is_artefact = [artefact_check(record) for record in tis_records]
            tis_records = np.array([tis_records[i] for i in range(len(tis_records)) if not is_artefact[i]])
            tis_indx = [list(tis_df.index)[i] for i in range(len(tis_df.index)) if not is_artefact[i]]

            is_artefact = [artefact_check(record) for record in ptb_records]
            ptb_records = np.array([ptb_records[i] for i in range(len(ptb_records)) if not is_artefact[i]])
            ptb_indx = [list(ptb_df.index)[i] for i in range(len(ptb_df.index)) if not is_artefact[i]]

            x_batch = np.concatenate([ptb_records, tis_records], axis=0)
            order = ptb_indx + tis_indx
            self.is_artefact[order] = False
            y_batch = self.y[order]
        return x_batch, y_batch


def normalize(norm_by, arr, lead_number, source_lead_min_max_config):
    if norm_by == 'global_min_max':
        (lead_min, lead_max) = source_lead_min_max_config[lead_number]
        return (arr-lead_min)/(lead_max-lead_min)


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
                norm_ecg.append(normalize(norm_by, lead_data[:len_], i,
                                          data_config['normalization_params'][norm_by]['ptb_lead_min_max']))
            normalized_data.append(norm_ecg)
        data = np.array(normalized_data)

    data = np.array(data).transpose(0, 2, 1)

    if len_ < needed_length:
        # padding
        npad = [(0, 0)] * data[0].ndim
        npad[0] = (0, needed_length - len_)
        data = np.array([np.pad(ecg_rec, pad_width=npad, mode=pad_mode) for ecg_rec in data])
    return data


def load_raw_data_tis(df, needed_length, pad_mode, norm_by):
    # considering that all of the files in df are sampled as config['sampling_rate']
    zf = zipfile.ZipFile(config['tis_path'])
    files = ['technion_ecg_data/' + file for file in list(df.filename)]
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
                lead_data = normalize(norm_by, lead_data, j, 
                                      data_config['normalization_params'][norm_by]['tis_lead_min_max'])
            if len_ < needed_length:
                # padding
                lead_data = np.pad(lead_data, (0, needed_length - lead_data.shape[0]), mode=pad_mode)
            row_data.append(lead_data)
        row_data = np.stack(row_data, axis=0)
        row_data = np.transpose(row_data)
        X.append(row_data)
    return np.array(X)


def get_tis_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    df = pd.read_csv('./tis_database.csv')
    if classification == 'binary':
        df.labels = pd.Series([[label] for label in df.labels])
    else:
        df.labels = df.labels.apply(lambda x: ast.literal_eval(x))
    if classification == 'multi-class':
        df = df[df['labels'].apply(lambda x:len(x) == 1)]
    elif classification == 'multi-label':
        df = df[df['labels'].apply(lambda x:len(x) >= 1)]

    train_df, test_df = train_test_split(df, test_size=test_portion, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=val_portion, random_state=SEED)

    # one-hot encoding labels
    mlb = MultiLabelBinarizer(classes=config['labels'])
    y_train_labels = mlb.fit_transform(train_df.labels)
    y_val_labels = mlb.fit_transform(val_df.labels)
    y_test_labels = mlb.fit_transform(test_df.labels)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df, y_train_labels, y_val_labels, y_test_labels


def get_ptb_data_split(classification='multi-class', test_portion=0.2, val_portion=0.2):
    df = pd.read_csv('./ptbxl_database.csv')
    if classification == 'binary':
        df.labels = pd.Series([[label]for label in df.labels])
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

    # one-hot encoding labels
    mlb = MultiLabelBinarizer(classes=config['labels'])
    y_train_labels = mlb.fit_transform(train_df.labels)
    y_val_labels = mlb.fit_transform(val_df.labels)
    y_test_labels = mlb.fit_transform(test_df.labels)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
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
    Y['source'] = 'ptb'
    Y.to_csv('./ptbxl_database.csv', index=False)

    # filter tis
    files = [file for file in os.listdir(config['tis_path'])if file.endswith('.csv')]
    info = []
    for file in files:
        file_df = pd.read_csv(config['tis_path'] + file, nrows=5, sep=':')
        if int(file_df.iloc[0][1]) == config['sampling_rate']:
            tis_codes = ast.literal_eval(file_df.iloc[4][1].strip())
            ptb_labels = map_labels(tis_codes, source='tis')
            ptb_labels = [key for key in ptb_labels if key in config['labels']]
            if ptb_labels:
                info.append({'filename': file, 'labels': ptb_labels, 'tis_codes': tis_codes, 'source': 'tis'})
    Y = pd.DataFrame(info)
    Y.to_csv('./tis_database.csv', index=False)