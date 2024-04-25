# coding=gbk
import os
import torch.optim as optim

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F

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
from scipy.sparse import vstack

from scipy.sparse import csr_matrix


def read_data(file_path):
    with open(file_path, 'r', encoding='gbk') as f:
        file_data = json.load(f)
    return file_data

# �ж�һ���Ƿ������֣������ж�һ���ǲ��Ǿ��ӿ�ͷ
def has_digit(line):
    for char in line:
        if char.isdigit():
            return True
    return False

def read_data_from_folder(data_dir):
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
                if words[i].endswith("/wd"):
                    continue
             
                id1.append(sentence_id)
                
                p_front.append(pronoun_front)
                
                
                p_behind.append(pronoun_behind)
                if i == antecedent_front:
                    a_front.append(1)
                else:
                    a_front.append(0)
                if i == antecedent_behind:
                    a_behind.append(1)
                else:
                    a_behind.append(0)
                if i >= antecedent_front and i <= antecedent_behind:
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

def cidai(id_t, feature_df):
    print(1)
    id_t_with_one_hot = id_t.copy()
    word_counts = id_t['word'].value_counts()
    
    # ����һ���յ�DataFrame���洢ƥ�����
    match_df_list = []

    # ��feature_df��ѡ��ǰ5000��
    selected_features = feature_df.iloc[:, :5000]

    # ����selected_features��ÿһ��
    for column in selected_features.columns:
        # ��ƥ�䵽���б��Ϊ1�����洢���б���
        match_df_list.append((id_t['word'] == column).astype(int))

    # ������ƥ�����һ������ӵ�DataFrame�У�������feature_df���б�ǩ
    match_df = pd.DataFrame(np.column_stack(match_df_list), columns=selected_features.columns, index=id_t.index)

    # ��ƥ������ϲ���id_t_with_one_hot��
    id_t_with_one_hot = pd.concat([id_t_with_one_hot.iloc[:, 1:], match_df], axis=1)

    # ����������һ��ƥ�����
    id_t_with_one_hot = id_t_with_one_hot[id_t_with_one_hot[selected_features.columns].sum(axis=1) > 0]

    print(id_t_with_one_hot)
    return id_t_with_one_hot


def oversample_data(data, target_column):
    X = data.iloc[:, [0, 1, 2, 5] + [i for i in range(8, 5011)]]
    y = data[target_column]
    oversampler = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


# ��ȡ���ݵ�����������
def data_generator(data_frame, chunk_size):
    total_rows = len(data_frame)
    start_idx = 0
    while start_idx < total_rows:
        yield data_frame.iloc[start_idx:start_idx + chunk_size]  # ����Ŀ���� 'a_fron'
        start_idx += chunk_size


def main3(train_data, chunk_size):
    for chunk_train in data_generator(train_data, chunk_size):
        # ��ѵ�������й�����
        X_train_resampled, y_train_resampled = oversample_data(chunk_train, 'a_in')

    return X_train_resampled, y_train_resampled


def find_continuous_indices(data, threshold=0.5):
    continuous_indices = []
    start_index = None
    for i, value in enumerate(data):
        if value >= threshold:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                continuous_indices.append((start_index, i - 1))
                start_index = None
    if start_index is not None:
        continuous_indices.append((start_index, len(data) - 1))
    return continuous_indices


def count_overlap(list1, list2):
    count = 0
    for start1, end1 in list1:
        for start2, end2 in list2:
            if start1 == start2 and end1 ==end2:  # �ж��������������Ƿ����ص�
                count += 1
    return count





class ImprovedMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(ImprovedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x



if __name__ == "__main__":
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
    

    seg_sen = []
    seg_sen = [sentence.split() for sentence in sen_words]
    key = []
    for i in seg_sen:
        for j in i:
        # ���ø���ǰհ�ų�ָ����׺�Ĵ�
            m = re.findall(r'[\u4e00-\u9fff](?:(?!/t|/wd|/wu|/wj|/d|/p|/ul|/vl|��|/f|\vl).)*$', j)
        # ��� m ��Ϊ�գ�˵�� j ����������������ӵ� key ��
            if m:
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
    
    sentence = []  # ��ʼ��һ�����б����洢����
    current_sentence = ""  # ��ʼ��һ�����ַ������洢��ǰ���ڹ����ľ���

    for line in sen_words:
        if has_digit(line):
            # �����ǰ�а������֣��ҵ�ǰ���Ӳ�Ϊ�գ����ȱ��浱ǰ����
            if current_sentence:
                sentence.append(current_sentence)
                # ���õ�ǰ���Ӳ���ʼ�µľ���
                current_sentence = ''.join(line)
            else:
                # �����ǰ�в��������֣���׷�ӵ���ǰ������
                current_sentence += ''.join(line)

                # ����Ƿ���δ����ĵ�ǰ���ӣ����б�����һ��Ԫ��֮������ݣ�
    if current_sentence:
        sentence.append(current_sentence)
    
        # # Bag of Words (based on word frequency)
    bow_feature = CountVectorizer(vocabulary=keywords)
    wordsfeature = bow_feature.fit_transform(sentence).toarray()

    feature_names = bow_feature.get_feature_names_out()  
    # ת��Ϊ DataFrame
    feature_df = pd.DataFrame(wordsfeature, columns=feature_names)
    print(feature_df)

    # ��ȡѵ��������֤���Ͳ��Լ�����

    id_t = read_data_from_folder("train")
    print(id_t)
    id_v = read_data_from_folder("validation")

    id_e = read_data_from_folder("test")
    # ��� word ���е��ַ���ֵ�Ƿ��� "\rr" Ϊ��׺���������µı�ǩ��
    

    id_t['\rr'] = id_t['word'].str.endswith('/rr').astype(int)
    id_t['\nrg'] = id_t['word'].str.endswith('/nrg').astype(int)
    id_t['\nrf'] = id_t['word'].str.endswith('/nrf').astype(int)
    # ��ӡ�޸ĺ�� DataFrame
    print(id_t)
    id_e['\rr'] = id_t['word'].str.endswith('/rr').astype(int)
    id_v['\rr'] = id_t['word'].str.endswith('/rr').astype(int)
    id_e['\nrg'] = id_e['word'].str.endswith('/nrg').astype(int)
    id_e['\nrf'] = id_e['word'].str.endswith('/nrf').astype(int)
    id_v['\nrg'] = id_v['word'].str.endswith('/nrg').astype(int)
    id_v['\nrf'] = id_v['word'].str.endswith('/nrf').astype(int)

    id_t_with_one_hot = cidai(id_t, feature_df)
    id_e_with_one_hot = cidai(id_e, feature_df)
    id_v_with_one_hot = cidai(id_v, feature_df)

    chunk_size = 10000
    X_train_resampled3, y_train_resampled3 = main3(id_t_with_one_hot, chunk_size)
    X_vali_resampled3, y_vali_resampled3 = main3(id_v_with_one_hot, chunk_size)
    print(X_vali_resampled3)

    # ��ʼ�� Y_train_1_
    Y_train_1_ = []
    Y_vali_1_ = []
    y_trainpred_ = []
    y_valipred_ = []
    X_train_resampled3['a_in'] = y_train_resampled3
    X_vali_resampled3['a_in'] = y_vali_resampled3

    print(X_train_resampled3)

    X_train_values3 = X_train_resampled3.iloc[:, [1, 2] + [i for i in range(4, 5007)]].values.astype('float32')
    X_vali_values3 = X_vali_resampled3.iloc[:, [1, 2] + [i for i in range(4, 5007)]].values.astype('float32')
    X_train3 = torch.tensor(X_train_values3)

    X_vali3 = torch.tensor(X_vali_values3)
    # ��NumPy����ת��ΪPyTorch����
    Y_train3 = torch.tensor(y_train_resampled3.values.astype('float32'))
    Y_vali3 = torch.tensor(y_vali_resampled3.values.astype('float32'))

    class_counts9 = id_e_with_one_hot.groupby('id').size().reset_index(name='Count').shape[0]
    print(class_counts9)
    # ģ�ͳ�ʼ��
    input_size = 5005
    hidden_size1 = 128
    hidden_size2 = 64
    improved_model = ImprovedMLP(input_size, hidden_size1, hidden_size2)

    # ������ʧ�������Ż���
    criteria = nn.BCELoss()
    optimizer = optim.Adam(improved_model.parameters(), lr=0.001, weight_decay=0.001)
    # Train Process
    loss_train = []
    loss_val = []
    accuracy_event1_train = []
    accuracy_event1_val = []
    for iteration in range(1000):
        # Forward Propagation
        y_pred = improved_model(X_train3)
        y_valipred = improved_model(X_vali3)
        # Loss Calculating
        loss1 = criteria(y_pred.squeeze(), Y_train3)
        loss2 = criteria(y_valipred.squeeze(), Y_vali3)
        print('iteration:{}'.format(iteration))
        print('loss_train:{}'.format(loss1.item()))
        print('loss_val:{}'.format(loss2.item()))
        loss_train.append(loss1.item())  # ����ÿ�ε�������ʧֵ
        loss_val.append(loss2.item())  # ����ÿ�ε�������ʧֵ

        # Accuracy Calculating
        f1_measure_train = sklearn.metrics.f1_score(Y_train3.ge(0.5).float(), y_pred.ge(0.5).float())
        f1_measure_val = sklearn.metrics.f1_score(Y_vali3.ge(0.5).float(), y_valipred.ge(0.5).float())
        if iteration % 10 == 0:
            print('F1-measure-validation:{}'.format(f1_measure_val))
            print('F1-measure-validation:{}'.format(f1_measure_train))

        accuracy_event1_train.append(f1_measure_train)
        accuracy_event1_val.append(f1_measure_val)

        # Backward Propagation
        optimizer.zero_grad()
        loss1.mean().backward()  # ʹ��ƽ����ʧ���з��򴫲�

        optimizer.step()



#     #
#     batch_size = 6400
#     total_batches1 = 100000 // batch_size
#     total_batches2 = 33485 // batch_size
#
#     # ������ʧ�������Ż���
#     criteria = nn.MSELoss()
#     optimizer = optim.Adam(improved_model.parameters(), lr=0.01)
#
#     # Train Process
#     loss_train_ = []
#     loss_val_ = []
#     class_counts1 = id_t_with_one_hot.groupby('id').size().reset_index(name='Count').shape[0]
#     print(class_counts1)
#     class_counts2 = id_v_with_one_hot.groupby('id').size().reset_index(name='Count').shape[0]
#     print(class_counts2)
#     class_counts3 = id_e_with_one_hot.groupby('id').size().reset_index(name='Count').shape[0]
#     print(class_counts3)
#
#
#     for iteration_ in range(1000):
#     # �𲽽���Ԥ��
#         total_loss_train = torch.tensor(0.0, requires_grad=True)  # ��ʼ��ѵ������ʧΪ0�������������ۼ���ʧ
#         total_loss_valid = torch.tensor(0.0, requires_grad=True)  # ��ʼ����֤����ʧΪ0�������������ۼ���ʧ
#         count_total1 = 0
#         count_total2 = 0
#         for i in range(total_batches1):
#             start_idx = i * batch_size
#             end_idx = (i + 1) * batch_size
#
#         # ��ȡ��ǰѵ�����ε�����
#             X_train_batch = id_t_with_one_hot.iloc[start_idx:end_idx, [2, 3] + [i for i in range(8, 5011)]]
#             Y_train_batch = id_t_with_one_hot.iloc[start_idx:end_idx, [5]]
#
#             X_train_chunk = torch.tensor(X_train_batch.values.astype('float32'))
#             Y_train_chunk = torch.tensor(Y_train_batch.values.astype('float32')).squeeze()
#
#         # ǰ�򴫲�
#             y_pred_train = improved_model(X_train_chunk).squeeze()
#         # ����ѵ������ʧ
#             loss_train = criteria(y_pred_train.ge(0.5).float(), Y_train_chunk)
#             total_loss_train = total_loss_train.add(loss_train)
#
#     # ����֤���Ͻ�������
#         for i in range(total_batches2):
#             start_idx = i * batch_size
#             end_idx = (i + 1) * batch_size
#
#         # ��ȡ��ǰ��֤���ε�����
#             X_valid_batch = id_v_with_one_hot.iloc[start_idx:end_idx, [2, 3] + [i for i in range(8, 5011)]]
#             Y_valid_batch = id_v_with_one_hot.iloc[start_idx:end_idx, [5]]
#
#             X_valid_chunk = torch.tensor(X_valid_batch.values.astype('float32'))
#             Y_valid_chunk = torch.tensor(Y_valid_batch.values.astype('float32')).squeeze()
#
#         # ǰ�򴫲�
#             y_pred_valid = improved_model(X_valid_chunk).squeeze()
#         # ������֤����ʧ
#             loss_valid = criteria(y_pred_valid.ge(0.5).float(), Y_valid_chunk)
#             total_loss_valid = total_loss_valid.add(loss_valid)
#
#
#
#             continuous_ones_indices_valid = find_continuous_indices(y_pred_valid)
#             continuous_ones_indices_valid_label = find_continuous_indices(Y_valid_chunk)
#
# # �������Ϊ1�Ĵʵ��������������
#             overlap_count = count_overlap(continuous_ones_indices_valid, continuous_ones_indices_valid_label)
#             count_total2 = count_total2 + overlap_count
# # ������
#             print("�������������б���ص����ָ���Ϊ:", overlap_count)
#     # ���ѵ������ʧ����֤����ʧ
#         print('epoch:{}'.format(iteration_))
#         print('loss_train:{}'.format(total_loss_train.item() / total_batches1))  # ����ƽ��ѵ������ʧ
#         print('loss_valid:{}'.format(total_loss_valid.item() / total_batches2))  # ����ƽ����֤����ʧ
#
#
#         precition2 = count_total2 / class_counts2
#         print('precition on validation set:', precition2)
#
#     # ���򴫲��������Ż�
#         optimizer.zero_grad()
#         total_loss_train.backward()
#         optimizer.step()
#
#     # ��̬����ѧϰ��
#         if iteration_ <= 300:
#             lr_ = 0.01
#         elif 300 < iteration_ <= 500:
#             lr_ = 0.001
#         else:
#             lr_ = 0.0001
#
#         for param_group_ in optimizer.param_groups:
#             param_group_['lr'] = lr_


    X_finaltest1 = torch.tensor(
        id_e_with_one_hot.iloc[:, [1, 2] + [i for i in range(8, 5011)]].values.astype('float32'))
    Y_finaltest1 = torch.tensor(id_e_with_one_hot['a_in'].values.astype('float32'))

    y_final_pred = improved_model(X_finaltest1)
    count_total2 = 0
    continuous_ones_indices_test = find_continuous_indices(y_final_pred)
    continuous_ones_indices_test_label = find_continuous_indices(Y_finaltest1)
    # �������Ϊ1�Ĵʵ��������������
    overlap_count = count_overlap(continuous_ones_indices_test, continuous_ones_indices_test_label)
    count_total2 = count_total2 + overlap_count

    precition2 = count_total2 / class_counts9
    print('precition on validation set:', precition2)
# Model Evaluation
