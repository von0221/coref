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
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, accuracy_score
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


# 判断一行是否有数字，用于判断一行是不是句子开头
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
            # 读入json
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
            # 对应句子

            sentence = ''
            f = 0
            for line in text_lines:
                # 检查当前行是否包含序号信息
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

                a_front.append(antecedent_front)
                a_behind.append(antecedent_behind)
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

    # 创建一个空的DataFrame来存储匹配情况
    match_df_list = []

    # 从feature_df中选择前5000列
    selected_features = feature_df.iloc[:, :5000]

    # 遍历selected_features的每一列
    for column in selected_features.columns:
        # 将匹配到的行标记为1，并存储到列表中
        match_df_list.append((id_t['word'] == column).astype(int))

    # 将所有匹配情况一次性添加到DataFrame中，并保留feature_df的列标签
    match_df = pd.DataFrame(np.column_stack(match_df_list), columns=selected_features.columns, index=id_t.index)

    # 将匹配情况合并到id_t_with_one_hot中
    id_t_with_one_hot = pd.concat([id_t_with_one_hot.iloc[:, 1:], match_df], axis=1)

    # # 保留至少有一个匹配的行
    # id_t_with_one_hot = id_t_with_one_hot[id_t_with_one_hot[selected_features.columns].sum(axis=1) > 0]

    print(id_t_with_one_hot)
    return id_t_with_one_hot


def oversample_data(data, target_column):
    X = data.iloc[:, [0, 1, 2] + [i for i in range(8, 5012)]]
    y = data[target_column]
    oversampler = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


# 读取数据的生成器函数
def data_generator(data_frame, chunk_size):
    total_rows = len(data_frame)
    start_idx = 0
    while start_idx < total_rows:
        yield data_frame.iloc[start_idx:start_idx + chunk_size]  # 包括目标列 'a_fron'
        start_idx += chunk_size


def main3(train_data, chunk_size):
    for chunk_train in data_generator(train_data, chunk_size):
        # 对训练集进行过采样
        X_train_resampled, y_train_resampled = oversample_data(chunk_train, 'a_in')

    return X_train_resampled, y_train_resampled


def find_continuous_indices(data, threshold):
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
            if start1 == start2 and end1 == end2:  # 判断两个连续部分是否有重叠
                count += 1
                break
    return count


def return_overlap(list1, list2):
    continuous_indices1 = []

    for start1, end1 in list1:
        for start2, end2 in list2:
            if start1 == start2 and end1 == end2:  # 判断两个连续部分是否有重叠
                continuous_indices1.append((start1, end1))
                break
    return continuous_indices1


import torch.nn.functional as F


class ImprovedMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(ImprovedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


if __name__ == "__main__":
    f = open('1998.txt', encoding='gbk')
    sen_words = f.readlines()
    f.close()

    # 清洗数据并进行分词
    temp_words = deepcopy(sen_words)
    for i in temp_words:
        if i == '\n':
            sen_words.remove(i)



        # Resegmentation

    seg_sen = []
    seg_sen = [sentence.split() for sentence in sen_words]
    key = []
    for i in seg_sen:
        for j in i:
            # 利用负向前瞻排除指定后缀的词
            m = re.findall(r'[\u4e00-\u9fff](?:(?!/t|/wd|/wu|/wj|/d|/p|/ul|/vl|不|/f|/vl|/vt|/vi).)*$', j)
            # 如果 m 不为空，说明 j 符合条件，将其添加到 key 中
            if m:
                key.append(m)

    key_words = []
    for i in key:
        for j in i:
            key_words.append(j)
    # print(key_words)

    # 初始化词袋模型
    keywords = []
    counts = nltk.FreqDist(key_words)
    item = list(counts.items())
    item.sort(key=lambda x: x[1], reverse=True)
    for sen in item[:5000]:
        keywords.append(sen[0])

    sentence = []  # 初始化一个空列表来存储句子
    current_sentence = ""  # 初始化一个空字符串来存储当前正在构建的句子

    for line in sen_words:
        if has_digit(line):
            # 如果当前行包含数字，且当前句子不为空，则先保存当前句子
            if current_sentence:
                sentence.append(current_sentence)
                # 重置当前句子并开始新的句子
                current_sentence = ''.join(line)
            else:
                # 如果当前行不包含数字，则追加到当前句子中
                current_sentence += ''.join(line)

                # 检查是否还有未保存的当前句子（即列表的最后一个元素之后的内容）
    if current_sentence:
        sentence.append(current_sentence)

        # # Bag of Words (based on word frequency)
    bow_feature = CountVectorizer(vocabulary=keywords)
    wordsfeature = bow_feature.fit_transform(sentence).toarray()

    feature_names = bow_feature.get_feature_names_out()
    # 转换为 DataFrame
    feature_df = pd.DataFrame(wordsfeature, columns=feature_names)
    print(feature_df)

    # 读取训练集、验证集和测试集数据

    id_t = read_data_from_folder("train")
    print(id_t)
    id_v = read_data_from_folder("validation")

    id_e = read_data_from_folder("test")
    # 检查 word 列中的字符串值是否以 "\rr" 为后缀，并创建新的标签列

    id_t['\rr'] = id_t['word'].str.endswith('/rr').astype(int)
    id_t['\nrg'] = id_t['word'].str.endswith('/nrg').astype(int)
    id_t['\nrf'] = id_t['word'].str.endswith('/nrf').astype(int)
    id_t['\n'] = id_t['word'].str.endswith('/n').astype(int)
    # 打印修改后的 DataFrame
    print(id_t)
    id_e['\rr'] = id_e['word'].str.endswith('/rr').astype(int)
    id_v['\rr'] = id_v['word'].str.endswith('/rr').astype(int)
    id_e['\nrg'] = id_e['word'].str.endswith('/nrg').astype(int)
    id_e['\nrf'] = id_e['word'].str.endswith('/nrf').astype(int)
    id_v['\nrg'] = id_v['word'].str.endswith('/nrg').astype(int)
    id_v['\nrf'] = id_v['word'].str.endswith('/nrf').astype(int)
    id_e['\n'] = id_e['word'].str.endswith('/n').astype(int)
    id_v['\n'] = id_v['word'].str.endswith('/n').astype(int)

    id_t_with_one_hot = cidai(id_t, feature_df)
    id_e_with_one_hot = cidai(id_e, feature_df)
    id_v_with_one_hot = cidai(id_v, feature_df)

    chunk_size = 10000
    X_train_resampled3, y_train_resampled3 = main3(id_t_with_one_hot, chunk_size)
    X_vali_resampled3, y_vali_resampled3 = main3(id_v_with_one_hot, chunk_size)
    print(X_vali_resampled3)

    # 初始化 Y_train_1_
    Y_train_1_ = []
    Y_vali_1_ = []
    y_trainpred_ = []
    y_valipred_ = []
    X_train_resampled3['a_in'] = y_train_resampled3
    X_vali_resampled3['a_in'] = y_vali_resampled3

    print(X_train_resampled3)

    X_train_values3 = X_train_resampled3.iloc[:, [i for i in range(1, 5007)]].values.astype('float32')
    X_vali_values3 = X_vali_resampled3.iloc[:, [i for i in range(1, 5007)]].values.astype('float32')
    X_train3 = torch.tensor(X_train_values3)

    X_vali3 = torch.tensor(X_vali_values3)
    # 将NumPy数组转换为PyTorch张量
    Y_train3 = torch.tensor(y_train_resampled3.values.astype('float32'))
    Y_vali3 = torch.tensor(y_vali_resampled3.values.astype('float32'))

    class_counts9 = X_vali_resampled3.groupby('id').size().reset_index(name='Count').shape[0]
    print(class_counts9)
    # 模型初始化
    # 模型初始化
    input_size = 5006
    hidden_size1 = 256
    hidden_size2 = 128
    hidden_size3 = 64
    improved_model = ImprovedMLP(input_size, hidden_size1, hidden_size2, hidden_size3)
    # weights = torch.tensor([1, 100])
    # criteria = nn.CrossEntropyLoss(weight=weights)

    # # 定义损失函数和优化器
    criteria = nn.BCELoss()
    optimizer = optim.Adam(improved_model.parameters(), lr=0.001, weight_decay=0.001)

    if torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
        improved_model.cuda()
        X_train3 = X_train3.cuda()
        Y_train3 = Y_train3.cuda()
        X_vali3 = X_vali3.cuda()
        Y_vali3 = Y_vali3.cuda()
        print("Using CUDA for training.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU for training.")

    # Train Process
    loss_train = []
    loss_val = []
    accuracy_event1_train = []
    accuracy_event1_val = []
    for iteration in range(400):
        # Forward Propagation
        y_pred = improved_model(X_train3)

        # Loss Calculating
        loss1 = criteria(y_pred.squeeze(), Y_train3)

        # Data preparation for validation
        X_finalvali1_list = []
        Y_finalvali1_list = []
        batch_size = 10000
        total_batches = len(id_v_with_one_hot) // batch_size
        class_counts2 = 0

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(id_v_with_one_hot))
            X_batch = id_v_with_one_hot.iloc[start_idx:end_idx,
                      [1, 2] + [i for i in range(8, 5012)]].values.astype('float32')
            Y_batch = id_v_with_one_hot['a_in'].iloc[start_idx:end_idx].values.astype('float32')

            X_finalvali1_list.append(X_batch)
            Y_finalvali1_list.append(Y_batch)
            class_counts2 += \
            id_v_with_one_hot.iloc[start_idx:end_idx].groupby('id').size().reset_index(name='Count').shape[0]

        X_finalvali1 = torch.tensor(np.concatenate(X_finalvali1_list, axis=0))
        Y_finalvali1 = torch.tensor(np.concatenate(Y_finalvali1_list, axis=0))

        if torch.cuda.is_available():
            X_finalvali1 = X_finalvali1.cuda()
            Y_finalvali1 = Y_finalvali1.cuda()
            print("Using CUDA for training.")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU for training.")

        # Forward propagation for validation
        y_valipred = improved_model(X_finalvali1)

        # Loss calculation for validation
        loss2 = criteria(y_valipred.squeeze(), Y_finalvali1)

        print('iteration:{}'.format(iteration))
        print('loss_train:{}'.format(loss1.item()))
        print('loss_val:{}'.format(loss2.item()))
        if iteration % 10 == 0:
            # F1 Score calculation
            f1 = f1_score(Y_train3.cpu().detach().numpy(), y_pred.cpu().detach().numpy().round())
            f11 = f1_score(Y_finalvali1.cpu().detach().numpy(), y_valipred.cpu().detach().numpy().round())

            # Accuracy calculation
            accuracy = accuracy_score(Y_finalvali1.cpu().detach().numpy(), y_valipred.cpu().detach().numpy().round())
            accuracy1 = accuracy_score(Y_train3.cpu().detach().numpy(), y_pred.cpu().detach().numpy().round())

            print('F1 Score in train:{}'.format(f1))
            print('F1 Score in val:{}'.format(f11))
            print('Accuracy in train:{}'.format(accuracy1))
            print('Accuracy in val:{}'.format(accuracy))

        loss_train.append(loss1.item())
        loss_val.append(loss2.item())


        # Backward Propagation
        optimizer.zero_grad()
        loss1.mean().backward()  # 使用平均损失进行反向传播

        optimizer.step()

    # 初始化空列表来存储所有的特征和标签
    X_finaltest1_list = []
    Y_finaltest1_list = []

    # 设置每次追加的数据批次大小
    batch_size = 10000

    # 计算总共需要追加多少次批次
    total_batches = len(id_e_with_one_hot) // batch_size
    class_counts1 = 0
    # 迭代读取数据并追加到列表中
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(id_e_with_one_hot))  # 避免超出索引范围
        X_batch = id_e_with_one_hot.iloc[start_idx:end_idx, [1, 2] + [i for i in range(8, 5012)]].values.astype(
            'float32')
        Y_batch = id_e_with_one_hot['a_in'].iloc[start_idx:end_idx].values.astype('float32')

        # 将批次数据追加到列表中
        X_finaltest1_list.append(X_batch)
        Y_finaltest1_list.append(Y_batch)
        class_counts1 += id_e_with_one_hot.iloc[start_idx:end_idx].groupby('id').ngroups

    # 将列表转换为张量
    X_finaltest1 = torch.tensor(np.concatenate(X_finaltest1_list, axis=0))
    Y_finaltest1 = torch.tensor(np.concatenate(Y_finaltest1_list, axis=0))
    if torch.cuda.is_available():
        X_finaltest1 = X_finaltest1.cuda()
        Y_finaltest1 = Y_finaltest1.cuda()
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU for training.")
    y_final_pred = improved_model(X_finaltest1)
    loss_test = criteria(y_final_pred.squeeze(), Y_finaltest1)
    print()
    print("开始测试部分")
    print('loss_test:{}'.format(loss_test.item()))
    f2 = f1_score(Y_finaltest1.cpu().detach().numpy(), y_final_pred.cpu().detach().numpy().round())
    print('F1 Score in test:{}'.format(f2))
    accuracy2 = accuracy_score(Y_finaltest1.cpu().detach().numpy(), y_final_pred.cpu().detach().numpy().round())
    print('Accuracy in test:{}'.format(accuracy2))

    a_in_list = id_e_with_one_hot['a_fron'].tolist()
    a_be_list = id_e_with_one_hot['a_be'].tolist()

    # 将两个列表组合成列表对的形式存储
    label_pairs = list(zip(a_in_list, a_be_list))
    # 输出预测值连续>0.5的词的连续行数的序号,二分类中>0.5便视为负例
    continuous_ones_indices_test = find_continuous_indices(y_final_pred, 0.5)
    # 输出真实值连续为1的词的连续行数的序号
    continuous_ones_indices_test_label = find_continuous_indices(Y_finaltest1, 0.99)

    print("真实序列：")
    print(continuous_ones_indices_test_label)
    print("预测序列：")
    print(continuous_ones_indices_test)


    # 计算两个序列完全相等的序列个数，即模型预测正确的个数，也就是TP
    overlap_count = count_overlap(continuous_ones_indices_test, continuous_ones_indices_test_label)

    #test中总共有970个句子
    num_rows = 970

    # 找到 预测值和真实值 中值都为 1 的行的索引

    resultlines = return_overlap(continuous_ones_indices_test, continuous_ones_indices_test_label)
    count = 0
    listflag = [0] * len(id_e_with_one_hot['id'].unique())
    x = id_e_with_one_hot['id'].unique()


    id_index = 0
    for start, end in resultlines:
        # 提取开始与结尾之间的行
        selected_rows = id_e_with_one_hot.iloc[start:end + 1]  # 加1是因为结束索引是不包含在内的
        first_row_id = selected_rows.iloc[0]['id']
        first_index = x.tolist().index(first_row_id )

        if listflag[first_index] == 0:
            count += 1
            for idx, row in selected_rows.iterrows():
                # 打印每一行的 id 列和 word 列
                id_value = row['id']
                id_index = x.tolist().index(id_value)
                listflag[id_index] = 1
                print("ID: {}, Word: {}, NO:{}".format(row['id'], row['word'], row['se']))

    precition3 = count / num_rows
    print('precition on test set:', precition3)
    # 预测错误的负例数=真实的正例数-预测正确的正例数
    FN = len(continuous_ones_indices_test_label) - overlap_count
    # recall=TP/(TP+FN)
    recall = overlap_count / (overlap_count + FN)
    print('recall on test set:', recall)