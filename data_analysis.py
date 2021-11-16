import pandas as pd
import numpy as np
import wfdb
import zipfile
import json
from sklearn.preprocessing import RobustScaler
from load_data import get_tis_data_split, get_ptb_data_split, config


def get_scaled_lead_min_max_values_for_tis(df):
    zf = zipfile.ZipFile(config['tis_path'])
    files = ['technion_ecg_data/' + file for file in list(df.filename)]

    X_max = []
    X_min = []
    for i in range(len(files)):
        print(i)
        example_max = []
        example_min = []
        file_df = pd.read_csv(zf.open(files[i]), skiprows=10, sep=',')
        file_df.columns = file_df.columns.str.strip()
        for lead in config['leads_order']:
            lead_data = list(file_df[lead])
            if len(set(lead_data)) > 1:
                example_max.append(max(lead_data))
                example_min.append(min(lead_data))
        if len(example_max) == 12:
            X_max.append(example_max)
            X_min.append(example_min)

    max_tr = RobustScaler().fit(X_max)
    min_tr = RobustScaler().fit(X_min)
    leads_min_max = [(min_tr.center_[i] - min_tr.scale_[i], max_tr.center_[i] + max_tr.scale_[i]) for i in range(12)]
    return leads_min_max


def get_scaled_lead_min_max_values_for_ptb(df):
    path = config['ptb_path']
    if config['sampling_rate'] == 100:
        data = [wfdb.rdsamp(path + f)[0] for f in df.filename_lr]
    elif config['sampling_rate'] == 500:
        data = [wfdb.rdsamp(path + f)[0] for f in df.filename_hr]

    data = np.array(data).transpose(0, 2, 1)
    X_max = []
    X_min = []
    for ecg in data:
        example_max = []
        example_min = []
        for lead_data in ecg:
            if len(set(lead_data)) > 1:
                example_max.append(max(lead_data))
                example_min.append(min(lead_data))
        if len(example_max) == 12:
            X_max.append(example_max)
            X_min.append(example_min)

    max_tr = RobustScaler().fit(X_max)
    min_tr = RobustScaler().fit(X_min)
    leads_min_max = [(min_tr.center_[i] - min_tr.scale_[i], max_tr.center_[i] + max_tr.scale_[i]) for i in range(12)]
    return leads_min_max


"""
train_df, _, _, _, _, _ = get_tis_data_split()
lead_min_max = get_scaled_lead_min_max_values_for_tis(train_df)
with open('tis_min_max.json', 'w+')as fin:
    json.dump(lead_min_max, fin)
"""
"""
train_df, _, _, _, _, _ = get_ptb_data_split()
lead_min_max = get_scaled_lead_min_max_values_for_ptb(train_df)
with open('ptb_min_max.json', 'w+')as fin:
    json.dump(lead_min_max, fin)
"""


