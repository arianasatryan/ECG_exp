import pandas as pd
import numpy as np
import wfdb
import ast
import json

with open('config.json', 'r')as fin:
    config = json.load(fin)


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def filter_labels(x):
    return [key for key in x.keys() if key in config['labels']]


def save_collected(save_path, df):
    df.to_csv(save_path)


# load and convert annotation data
Y = pd.read_csv(config['path']+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# filter needed labels
Y['labels'] = Y.scp_codes.apply(filter_labels)
Y['labels'] = pd.Series([item for sublist in list(Y['labels']) for item in sublist])
Y = Y[Y['labels'].str.len() != 0]
Y = Y[Y['labels'].notna()]

# Load raw signal data
X = load_raw_data(Y, config['sampling_rate'], config['path'])

"""
# save sub-database 
Y.to_csv(config['collected_datasets_path'] + 'sub_ptbxl_database.csv')

# check number of instances
stat = json.loads(Y['labels'].value_counts().to_json())
with open(path + 'sub_ptbxl_stat.json', 'w') as fout:
    json.dump(stat, fout, indent=4)
"""





