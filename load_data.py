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


labeled_df = discard_other_labels(config['ptb_path']+'ptbxl_database.csv', source='ptb')
save_per_label(labeled_df, config['collected_datasets_path'])


"""
# check number of instances
stat = json.loads(Y['labels'].value_counts().to_json())
with open(path + 'sub_ptbxl_stat.json', 'w') as fout:
    json.dump(stat, fout, indent=4)
"""

"""
# Load raw signal data
# i.e. for 'IRBBB' label load_from_path=config['collected_datasets_path'] + 'IRBBB_subdataset.csv'
X = load_raw_data(our_label_df, config['sampling_rate'], load_from_path)
"""

