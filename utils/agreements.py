import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import combinations

states_path = 'path/to/ECG-states.csv'
reports_path = 'path/to/reports.xls'

all_labels = list(set(list(pd.read_csv(states_path)['Код в SCP'])))
mlt = MultiLabelBinarizer(classes=all_labels)
labels_order = {label: i for i, label in enumerate(all_labels)}


def get_pair_report(reports_df, username1, username2):
    pair_report = reports_df[reports_df['username'].isin([username1, username2])]
    pair_report = pair_report[pair_report.duplicated(subset=['electrocardiogram'], keep=False)]
    return pair_report


def get_agreement(username1, username2, labels=None):
    labels = labels if labels else all_labels
    agreement_per_label = {}
    pair_report = get_pair_report(reports, username1, username2)
    pair_report = pair_report.sort_values(by='electrocardiogram')

    user1_ecg_one_hot_labels = mlt.fit_transform(list(pair_report[pair_report['username'] == username1].diagnoses))
    user2_ecg_one_hot_labels = mlt.fit_transform(list(pair_report[pair_report['username'] == username2].diagnoses))

    for label in labels:
        order = labels_order[label]
        user1_label_decisions = [one_hot_labels[order] for one_hot_labels in user1_ecg_one_hot_labels]
        user2_label_decisions = [one_hot_labels[order] for one_hot_labels in user2_ecg_one_hot_labels]

        support_df = pair_report[pair_report['diagnoses'].apply(lambda x: label in x)]
        support_df = support_df.drop_duplicates(subset='electrocardiogram', keep="last")
        support = len(support_df)

        if support == 0:
            agreement_per_label[label] = None
        else:
            agreement_per_label[label] = cohen_kappa_score(user1_label_decisions, user2_label_decisions), support
    return agreement_per_label


reports = pd.read_excel(reports_path)
reports = reports[reports['username'] != 'user_g1']
reports['diagnoses'] = reports['diagnoses'].apply(lambda x: x.split(','))
reports['createdAt'] = pd.to_datetime(reports.createdAt)
reports = reports.sort_values(by='createdAt')
reports = reports.drop_duplicates(subset=['electrocardiogram', 'username'], keep='last')

user_pairs = list(combinations(list(reports.username.unique()), 2))
user_pair_agreements = {}
for user1, user2 in user_pairs:
    diagnosis_report = {}
    agreement_per_label = get_agreement(user1, user2)
    user_pair_agreements['_'.join([user1, user2])] = agreement_per_label

rep_df = pd.DataFrame.from_dict(user_pair_agreements)
rep_df = rep_df.dropna(how='all', axis=0)
rep_df = rep_df.dropna(how='all', axis=1)
rep_df = rep_df.sort_values(by='_'.join(user_pairs[0]), ascending=False)
rep_df.to_csv("cohen's_kappa_scores.csv")



