# coding=gbk
import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import re
import jieba
import random
import json
import sklearn.metrics
import seaborn as sns
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_curve
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler
from torchsummary import summary
import pandas as pd


# ��ȡJSON�ļ�����,����json��ʽ
def read_data(file_path):
    with open(file_path, 'r', encoding='gbk') as f:
        file_data = json.load(f)
    return file_data


class LogisticRegression(nn.Module):
    def __init__(self, d_prob=0.5):
        super(LogisticRegression, self).__init__()
        self.features = nn.Linear(5004, 1)
        self.sigmoid = nn.Sigmoid()
        nn.Dropout(d_prob)

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


# �ж�һ���Ƿ������֣������ж�һ���ǲ��Ǿ��ӿ�ͷ
def has_digit(line):
    for char in line:
        if char.isdigit():
            return True
    return False


# ���ļ����ж�ȡ����
def read_data_from_folder(data_dir):
    data = []
    total_files = len(os.listdir(data_dir))
    files_read = 0
    id1 = []
    p_front = []
    p_behind = []
    a_front = []
    a_behind = []
    a_in = []
    word = []
    se = []
    with open("1998.txt", 'r', encoding='gbk') as f:
        text_lines = f.readlines()
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            files_read += 1
            file_path = os.path.join(data_dir, file_name)
            json_data = read_data(file_path)
            # ����json
            sentence_id = json_data["pronoun"]["id"]
            # id.append(sentence_id)
            pronoun_front = json_data["pronoun"]['indexFront']
            # p_front.append(pronoun_front)
            pronoun_behind = json_data["pronoun"]["indexBehind"]
            # p_behind.append(pronoun_behind)
            antecedent_front = json_data["0"]["indexFront"]
            # a_front.append(antecedent_front)
            antecedent_behind = json_data["0"]["indexBehind"]
            # a_behind.append(antecedent_behind)
            # ��Ӧ����

            sentence = ''
            f = 0
            for line in text_lines:
                # ��鵱ǰ���Ƿ���������Ϣ
                if sentence_id in line:
                    f = 1
                    sentence += ''.join(line)
                elif has_digit(line):
                    f = 0
                if f == 1 and not (sentence_id in line):
                    sentence += ''.join(line)
            words = sentence.strip().split(' ')

            words.pop(0)
            for i in words:
                if '' in words:
                    words.remove('')
            words = words[:pronoun_behind]

            for i in range(len(words)):
                id1.append(sentence_id)
                if i == pronoun_front:
                    p_front.append(1)
                else:
                    p_front.append(0)
                if i == pronoun_behind:
                    p_behind.append(1)
                else:
                    p_behind.append(0)
                if i == antecedent_front:
                    a_front.append(1)
                else:
                    a_front.append(0)
                if i == antecedent_behind:
                    a_behind.append(1)
                else:
                    a_behind.append(0)
                if i > antecedent_front and i < antecedent_behind:
                    a_in.append(1)
                else:
                    a_in.append(0)
                word.append(words[i])
                se.append(i)
        id2 = pd.DataFrame(id1)
        id2['id'] = id1
        id2['p_fron'] = p_front
        id2['p-be'] = p_behind
        id2['a_fron'] = a_front
        id2['a_be'] = a_behind
        id2['a_in'] = a_in
        id2['word'] = word
        id2['se'] = se

    return id2


if __name__ == "__main__":
    # ��ȡ�ı�����
    f = open('1998.txt', encoding='gbk')
    sen_words = f.readlines()
    f.close()

    # ��ϴ���ݲ����зִ�
    temp_words = deepcopy(sen_words)
    for i in temp_words:
        if i == '\n':
            sen_words.remove(i)

    # �Ƴ������ŵ�
    sen_wordsnew = []
    for i in sen_words:
        str = re.sub('[a-zA-Z0-9��!"#$&\'()*+-.<=>@�����{}����[\\]^_`{|}~\s]+', "", i)
        sen_wordsnew.append(str)

        # Resegmentation
    dic3 = []
    for sentence in sen_wordsnew:
        seg_list2 = jieba.cut(sentence, cut_all=False)
        cut_str = ",".join(seg_list2)
        dic3.append(cut_str)

    seg_sen = []
    seg_sen = [sentence.split() for sentence in sen_words]
    key = []
    for i in seg_sen:
        for j in i:
            m = re.findall(r'[\u4e00-\u9fff](?:(?!/nrg|/nrf|/rr).)*', j)
            key.append(m)

    key_words = []
    for i in key:
        for j in i:
            key_words.append(j)
    # print(key_words)

    # ��ʼ���ʴ�ģ��
    keywords = []
    counts = nltk.FreqDist(key_words)
    item = list(counts.items())
    item.sort(key=lambda x: x[1], reverse=True)
    for sen in item[:5000]:
        keywords.append(sen[0])
    print("keywords")
    print(keywords)
    # Bag of Words (based on word frequency)
    bow_feature = CountVectorizer(vocabulary=keywords)
    wordsfeature = bow_feature.fit_transform(dic3).toarray()
    # print(wordsfeature)
    # ת��Ϊ DataFrame
    feature_df = pd.DataFrame(wordsfeature)

    # �����������б�
    nrf_features = []
    # ���� sen_words �е�ÿ���ʻ�
    for word in sen_words:
        # ���ʻ�ĺ�׺
        if word.endswith('/nrf'):
            nrf_features.append(1)  # ����ʻ�ĺ�׺������������1��ӵ������б�
        else:
            nrf_features.append(0)  # �������0

    # ���������б���ӵ� DataFrame ��
    feature_df['nrf_feature'] = nrf_features

    nrg_features = []
    # ���� sen_words �е�ÿ���ʻ�
    for word in sen_words:
        # ���ʻ�ĺ�׺
        if word.endswith('/nrg'):
            nrg_features.append(1)  # ����ʻ�ĺ�׺������������1��ӵ������б�
        else:
            nrg_features.append(0)  # �������0

    # ���������б���ӵ� DataFrame ��
    feature_df['nrg_feature'] = nrg_features
    rr_features = []
    # ���� sen_words �е�ÿ���ʻ�
    for word in sen_words:
        # ���ʻ�ĺ�׺
        if word.endswith('/rr'):
            rr_features.append(1)  # ����ʻ�ĺ�׺������������1��ӵ������б�
        else:
            rr_features.append(0)  # �������0

    # ���������б���ӵ� DataFrame ��
    feature_df['rr_feature'] = rr_features

    # ��ȡѵ��������֤���Ͳ��Լ�����

    id_t = read_data_from_folder("train")

    id_v = read_data_from_folder("validation")

    id_e = read_data_from_folder("test")

    final_df = pd.merge(feature_df, id_t, how='inner', left_index=True, right_index=True)
    print(final_df)
    validation_df = pd.merge(feature_df, id_v, how='inner', left_index=True, right_index=True)
    test_df = pd.merge(feature_df, id_e, how='inner', left_index=True, right_index=True)

    #
    #
    # final_df.columns = final_df.columns.astype(str)
    # print("final_df")
    print(final_df.columns)

    # # Event 1
    # event1list = list(final_df['a_fron'])
    # Oversampling = RandomOverSampler(random_state=0)
    #
    # X1_resampled, Y1_resampled = Oversampling.fit_resample(final_df.iloc[:, 0:5000], event1list)
    # X1_resampled['a_fron'] = Y1_resampled
    # testevent1list = list(validation_df['a_fron'])
    # Oversampling = RandomOverSampler(random_state=0)
    # X1_testresampled, Y1_testresampled = Oversampling.fit_resample(validation_df.iloc[:, 0:5000], testevent1list)
    # X1_testresampled['a_fron'] = Y1_testresampled
    #
    # # Event 2
    # event2list = list(final_df['a_be'])
    # Oversampling = RandomOverSampler(random_state=0)
    # X2_resampled, Y2_resampled = Oversampling.fit_resample(final_df.iloc[:, 0:5000], event2list)
    # X2_resampled['a_be'] = Y2_resampled
    #
    # testevent2list = list(validation_df['a_be'])
    # Oversampling = RandomOverSampler(random_state=0)
    # X2_testresampled, Y2_testresampled = Oversampling.fit_resample(validation_df.iloc[:, 0:5000], testevent2list)
    # X2_testresampled['a_be'] = Y2_testresampled
    # fig, axes = plt.subplots(2, 2, figsize=(40, 20))
    # axes = axes.ravel()
    #
    # sns.set(font_scale=1.5)
    # sns.distplot(X1_resampled['event1'], axlabel='Labels for Train_Event1', color='darkred', ax=axes[0])
    # sns.distplot(X1_testresampled['event1'], axlabel='Labels for Validation_Event1', color='red', ax=axes[1])
    # sns.distplot(X2_resampled['event2'], axlabel='Labels for Train_Event2', color='darkgreen', ax=axes[2])
    # sns.distplot(X2_testresampled['event2'], axlabel='Labels for Validation_Event2', color='olivedrab', ax=axes[3])
    #
    # # Event 1
    # X_train1 = torch.tensor(X1_resampled.iloc[:, 0:5000].values.astype('float32'))
    # Y_train1 = torch.tensor(X1_resampled.iloc[:, 5000:5001].values.astype('float32'))
    # X_test1 = torch.tensor(X1_testresampled.iloc[:, 0:5000].values.astype('float32'))
    # Y_test1 = torch.tensor(X1_testresampled.iloc[:, 5000:5001].values.astype('float32'))
    # # Event 2
    # X_train2 = torch.tensor(X2_resampled.iloc[:, 0:5000].values.astype('float32'))
    # Y_train2 = torch.tensor(X2_resampled.iloc[:, 5000:5001].values.astype('float32'))
    # X_test2 = torch.tensor(X2_testresampled.iloc[:, 0:5000].values.astype('float32'))
    # Y_test2 = torch.tensor(X2_testresampled.iloc[:, 5000:5001].values.astype('float32'))

    # Event 1
    selected_columns = final_df.iloc[:, [i for i in range(1, 5003)] + [5005, 5006]]
    X_train1 = torch.tensor(selected_columns.values.astype('float32'))
    Y_train1 = torch.tensor(final_df.iloc[:, -5:-4].values.astype('float32'))
    selected2_columns = test_df.iloc[:, [i for i in range(1, 5003)] + [5005, 5006]]
    X_test1 = torch.tensor(selected2_columns.values.astype('float32'))
    Y_test1 = torch.tensor(test_df.iloc[:, -5:-4].values.astype('float32'))
    # Event 2
    selected3_columns = final_df.iloc[:, [i for i in range(1, 5003)] + [5005, 5006]]
    X_train2 = torch.tensor(selected3_columns.values.astype('float32'))
    Y_train2 = torch.tensor(final_df.iloc[:, -4:-3].values.astype('float32'))
    selected4_columns = test_df.iloc[:, [i for i in range(1, 5003)] + [5005, 5006]]
    X_test2 = torch.tensor(selected4_columns.values.astype('float32'))
    Y_test2 = torch.tensor(test_df.iloc[:, -4:-3].values.astype('float32'))
    # Event 3
    selected5_columns = final_df.iloc[:, [i for i in range(1, 5003)] + [5005, 5006]]
    X_train3 = torch.tensor(selected5_columns.values.astype('float32'))
    Y_train3 = torch.tensor(final_df.iloc[:, -3:-2].values.astype('float32'))
    selected6_columns = test_df.iloc[:, [i for i in range(1, 5003)] + [5005, 5006]]
    X_test3 = torch.tensor(selected6_columns.values.astype('float32'))
    Y_test3 = torch.tensor(test_df.iloc[:, -3:-2].values.astype('float32'))

    # Event 1 - a_fron
    logistic_a_fron = LogisticRegression()
    criteria_a_fron = nn.BCELoss()
    optimizer_a_fron = torch.optim.Adam(logistic_a_fron.parameters(), lr=0.01, weight_decay=0.001)

    # Event 2 - a_be
    logistic_a_be = LogisticRegression()
    criteria_a_be = nn.BCELoss()
    optimizer_a_be = torch.optim.Adam(logistic_a_be.parameters(), lr=0.01, weight_decay=0.001)

    # Event 3 - a_in
    logistic_a_in = LogisticRegression()
    criteria_a_in = nn.BCELoss()
    optimizer_a_in = torch.optim.Adam(logistic_a_be.parameters(), lr=0.01, weight_decay=0.001)

    # Train Process
    loss_train_a_fron = []
    loss_train_a_be = []
    loss_train_a_in = []

    accuracy_event1_train_a_fron = []
    accuracy_event1_train_a_be = []
    accuracy_event1_train_a_in = []

    for iteration in range(1000):
        # Forward Propagation
        y_pred_a_fron = logistic_a_fron(X_train1)
        y_pred_a_be = logistic_a_be(X_train2)
        y_pred_a_in = logistic_a_in(X_train3)

        # Loss Calculating
        loss_a_fron = criteria_a_fron(y_pred_a_fron, Y_train1)
        loss_a_be = criteria_a_be(y_pred_a_be, Y_train2)
        loss_a_in = criteria_a_in(y_pred_a_in, Y_train3)

        loss_train_a_fron.append(loss_a_fron.item())
        loss_train_a_be.append(loss_a_be.item())
        loss_train_a_in.append(loss_a_in.item())

        # Accuracy Calculating
        f1_measure_train_a_fron = sklearn.metrics.f1_score(Y_train1.ge(0.5).float(), y_pred_a_fron.ge(0.5).float())
        f1_measure_train_a_be = sklearn.metrics.f1_score(Y_train2.ge(0.5).float(), y_pred_a_be.ge(0.5).float())
        f1_measure_train_a_in = sklearn.metrics.f1_score(Y_train3.ge(0.5).float(), y_pred_a_in.ge(0.5).float())

        accuracy_event1_train_a_fron.append(f1_measure_train_a_fron)
        accuracy_event1_train_a_be.append(f1_measure_train_a_be)
        accuracy_event1_train_a_in.append(f1_measure_train_a_in)

        print(
            'Epoch: {}, Loss_a_fron: {:.4f}, Loss_a_be: {:.4f}, F1-measure-train_a_fron: {:.4f}, F1-measure-train_a_be: {:.4f}, F1-measure-train_a_in: {:.4f}'.format(
                iteration, loss_a_fron.item(), loss_a_be.item(), f1_measure_train_a_fron, f1_measure_train_a_be,
                f1_measure_train_a_in))

        # Backward Propagation - for a_fron
        optimizer_a_fron.zero_grad()
        loss_a_fron.backward()
        optimizer_a_fron.step()

        # Backward Propagation - for a_be
        optimizer_a_be.zero_grad()
        loss_a_be.backward()
        optimizer_a_be.step()

    # ���� final_df �����ԭʼ DataFrame
    # ���� y_pred_a_fron, y_pred_a_in, y_pred_a_be �����Ԥ��ֵ�б������

    # ȷ��Ԥ��ֵ�ĳ����� final_df ��������ƥ��
    # assert len(y_pred_a_fron) == len(final_df)
    assert len(y_pred_a_in) == len(final_df)
    # assert len(y_pred_a_be) == len(final_df)
    #
    # # ��Ԥ��ֵת��Ϊ Pandas Series��������ǻ����ǵĻ���
    # y_pred_a_fron_series = pd.Series(y_pred_a_fron)
    # y_pred_a_in_series = pd.Series(y_pred_a_in)
    # y_pred_a_be_series = pd.Series(y_pred_a_be)
    y_pred_a_in_numpy = y_pred_a_in.detach().numpy()
    # ��Ԥ��ֵ��Ϊ������ӵ� final_df ��
    # final_df['y_pred_a_fron'] = y_pred_a_fron
    final_df['y_pred_a_in'] = y_pred_a_in_numpy
    # final_df['y_pred_a_be'] = y_pred_a_be

    # ���� final_df ������Ԥ��ֵ��Ϊ����
    print(final_df.head())  # ��ӡǰ�����Բ鿴���

    # �� id �� final_df ���з���
    grouped = final_df.groupby('id')

    # ����ÿ�� id ��
    for id_group, group_data in grouped:
        # # ɸѡ�� y_pred_a_fron Ϊ 1 ����
        # fron_ones = group_data[group_data['y_pred_a_fron'] == 1]
        # if not fron_ones.empty:
        #     print(f"id: {id_group}, y_pred_a_fron: 1")
        #     print(fron_ones['word'].values)  # ��ӡ word ��ֵ

        # ɸѡ�� y_pred_a_in Ϊ 1 ����
        in_ones = group_data[group_data['y_pred_a_in'] >= 0.5]
        if not in_ones.empty:
            print(f"id: {id_group}, y_pred_a_in: 1")
            print(in_ones['word'].values)  # ��ӡ word ��ֵ

        # # ɸѡ�� y_pred_a_be Ϊ 1 ����
        # be_ones = group_data[group_data['y_pred_a_be'] == 1]
        # if not be_ones.empty:
        #     print(f"id: {id_group}, y_pred_a_be: 1")
        #     print(be_ones['word'].values)  # ��ӡ word ��ֵ
        #
        # # ���Ը�����Ҫ��ӻ��з��������ָ������������
        # print("\n")
    # # ��ȡ���в�ͬ�ľ���ID
    # unique_ids = final_df['id'].unique()
    #
    # for sent_id in unique_ids:
    #     # ��ȡ��ǰ���ӵ�����
    #     sent_data = final_df[final_df['id'] == sent_id]
    #     # ɸѡ 'a_fron' ���� 1 ���У�����ӡ 'word' �е�ֵ
    #     fron_ones = sent_data[sent_data['a_fron'] == 1]
    #     print("Rows with 'a_fron' equal to 1:")
    #     print(fron_ones['word'])
    #
    #     # ɸѡ 'a_in' ���� 1 ���У�����ӡ 'word' �е�ֵ
    #     in_ones = sent_data[sent_data['a_in'] == 1]
    #     print("Rows with 'a_in' equal to 1:")
    #     print(in_ones['word'])
    #
    #     # ɸѡ 'a_be' ���� 1 ���У�����ӡ 'word' �е�ֵ
    #     be_ones = sent_data[sent_data['a_be'] == 1]
    #     print("Rows with 'a_be' equal to 1:")
    #     print(be_ones['word'])

        # sentence = ''
        # f = 0
        # for line in sen_words:
        #     # ��鵱ǰ���Ƿ���������Ϣ
        #     if sent_id in line:
        #         f = 1
        #         sentence += ''.join(line)
        #     elif has_digit(line):
        #         f = 0
        #     if f == 1 and not (sent_id in line):
        #         sentence += ''.join(line)
        # words = sentence.strip().split(' ')
        # print(words[0])
        # words.pop(0)
        # for i in words:
        #     if '' in words:
        #         words.remove('')

        # �ҵ�'a_fron'��'a_be'��λ������
        # a_fron_idx = [i for i, word in enumerate(words) if sent_data.iloc[i]['a_fron'] == 1]
        # a_be_idx = [i for i, word in enumerate(words) if sent_data.iloc[i]['a_be'] == 1]
        #
        # if a_fron_idx and a_be_idx:
        #     # ���'a_fron'��'a_be'������
        #     a_fron_word = words[a_fron_idx[0]]
        #     a_be_word = words[a_be_idx[0]]
        #
        #     print(f"����ID: {sent_id}")
        #     print(f"'a_fron'����: {a_fron_word}")
        #     print(f"'a_be'����: {a_be_word}")
        #
        #     # ��ӡ'a_fron'��'a_be'֮��Ĵ���
        #     middle_words = words[a_fron_idx[0] + 1:a_be_idx[0]]
        #     print(f"�м����: {' '.join(middle_words)}")
        #     print()