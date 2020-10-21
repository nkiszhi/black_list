# -*-coding:utf-8 -*-
"""Train and test LSTM.MI classifier"""

import numpy as np
import os
import random
import csv
import collections
import math
import pandas as pd
import string
import tld
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, classification_report,accuracy_score, f1_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_binary_model(max_features, maxlen):
    """Build LSTM model for two-class classification"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop')

    return model


def create_class_weight(labels_dict, mu):
    """Create weight based on the number of sld name in the dataset"""
    labels_dict = dict(labels_dict)
    keys = labels_dict.keys()
    total = labels_dict[1] + labels_dict[0]
    class_weight = dict()
    for key in keys:
        score = math.pow(total/float(labels_dict[key]), mu)
        class_weight[key] = score

    return class_weight


def get_name(url):
    """
    用python自带库进行域名提取
    :param url: url
    :return: 二级域名，顶级域名
    """
    url = url.strip(string.punctuation)
    try:
        SLD = tld.get_tld(url, as_object=True, fix_protocol=True).domain
    except Exception as e:
        na_list = url.split(".")
        print("------------------------------{}".format(e))
        print("========={}".format(na_list))
        SLD = na_list[-2]
    return str(SLD)


def get_dataset(black_path, white_path):
    """
    合并black和white
    分层抽样8：2形成train,test
    """
    df = pd.read_csv(black_path, names=["domain_name", "label"])
    df["domain_name"] = df["domain_name"].apply(get_name)
    print(df.head(2))
    print("------black shape----{}".format(len(df["domain_name"].tolist())))

    df_white=pd.read_csv(white_path, names=["domain_name", "label"])
    df_white["domain_name"] = df_white["domain_name"].apply(get_name)
    print("------white shape----{}".format(len(df_white["domain_name"].tolist())))

    df = df.append(df_white)
    sld_ls = df['domain_name'].tolist()
    print("------sld shape----{}".format(len(sld_ls)))
    label_ls = df['label'].tolist()
    return sld_ls, label_ls


def get_dataset_pinjie(black_path, white_path):
    """
    合并black和white
    分层抽样8：2形成train,test
    拼接
    """
    df = pd.read_csv(black_path, names=["domain_name", "label"])
    df['domain_name'] = df['domain_name'].apply(data_pro)
    print("------black shape----{}".format(len(df["domain_name"].tolist())))

    df_white = pd.read_csv(white_path, names=["domain_name", "label"])
    df_white['domain_name'] = df_white['domain_name'].apply(data_pro)
    print("------white shape----{}".format(len(df_white["domain_name"].tolist())))

    df = df.append(df_white)
    sld_ls = df['domain_name'].tolist()
    label_ls = df['label'].tolist()

    print("------sld shape----{}".format(len(sld_ls)))
    return sld_ls, label_ls


def data_pro(url):
    """
    预处理字符串
    :param url:
    :return:
    """
    url = url.strip().strip('.')
    url = url.split('/')[0]
    url = url.split('?')[0]
    url = url.split('=')[0]
    dn_list = url.split('.')
    for i in reversed(dn_list):
        if i in tld_list:
            dn_list.remove(i)
        elif i == 'www':
            dn_list.remove(i)
        else:
            continue
    short_url = ''.join(dn_list)
    short_url = short_url.replace('[', '').replace(']', '')
    short_url = short_url.lower()
    return short_url


def run(max_epoch=20, nfolds=10, batch_size=128):
    """
    Run train/test on logistic regression model
    """
    global maxlen
    # sld, label = get_dataset(DGA_dataset_path, NDGA_dataset_path)
    sld, label = get_dataset_pinjie(DGA_dataset_path, NDGA_dataset_path)
    
    print("------{}={}".format("valid_chars", valid_chars))
    maxlen = np.max([len(x) for x in sld])
    print("------maxfeatures is :{}".format(max_features))
    print("------maxlen is :{}".format(maxlen))

    X = [[valid_chars[y] for y in x] for x in sld]
    X = sequence.pad_sequences(X, maxlen=maxlen)
    y = np.array(label)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=4)
    for train, test in sss.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    
        print("---train:{}---test:{}----y_train:{}----y_test:{}".format(len(X_train), len(X_test), len(y_train), len(y_test)))

        #shuffle
        np.random.seed(4)  # 1024
        index = np.arange(len(X_train))
        np.random.shuffle(index)
        X_train = np.array(X_train)[index]
        y_train = np.array(y_train)[index]

        model = build_binary_model(max_features, maxlen)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
        for train, test in sss1.split(X_train, y_train):
            X_train, X_holdout, y_train, y_holdout = X_train[train], X_train[test], y_train[train], y_train[test]   # holdout验证集

        labels_dict = collections.Counter(y_train)
        class_weight = create_class_weight(labels_dict, 0.3)
        print('----class weight:{}'.format(class_weight))
        #20
        best_acc = 0.0
        best_model = None
        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, class_weight=class_weight)
            # model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)

            t_probs = model.predict_proba(X_holdout)
            t_result = [0 if x <= 0.5 else 1 for x in t_probs]
            t_acc = accuracy_score(y_holdout, t_result)
            print("epoch:{}--------val acc:{}---------best_acc:{}".format(ep, t_acc, best_acc))
            if t_acc > best_acc:
                best_model = model
                best_acc = t_acc

        model_json = best_model.to_json()
        # 模型的权重保存在HDF5中
        # 模型的结构保存在JSON文件或者YAML文件中
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)
            model.save_weights(model_h5_path)
        print("Saved two-class model to disk")

        # test
        y_pred = best_model.predict_proba(X_test, batch_size=128, verbose=1)
        y_result = [0 if x <= 0.5 else 1 for x in y_pred]
        #Calculate the final result
        score = f1_score(y_test, y_result, average="macro")
        precision = precision_score(y_test, y_result, average="macro")
        recall = recall_score(y_test, y_result, average="macro")
        report = classification_report(y_test, y_result, digits=4)
        acc = accuracy_score(y_test, y_result)
        # classifaction_report_csv(report, precision, recall, score)
        print('\n clasification report:\n', report)
        print('F1 score:', score)
        print('Recall:', recall)
        print('Precision:', precision)
        print('Acc:', acc)
             
        # # roc
        # fpr, tpr, thresholds = roc_curve(y_test, y_result)
        # auc_score = auc(fpr, tpr)
        # print("AUC:{}".format(auc_score))
        # plt.figure()
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc_score))
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title('roc_curve')
        # plt.legend(loc='best')
        # plt.savefig(fig_path)
        # # plt.show()
        break


def predict(url):
    """

    """
    # global valid_chars
    global maxlen
    # global max_features
    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()
    new_model = model_from_json(model_json)
    new_model.load_weights(model_h5_path)
    # 编译模型
    new_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    sld = get_name(url)
    sld = sld.replace('[', '').replace(']', '')
    print("sld-----{}".format(sld))

    sld_int = [[valid_chars[y] for y in x] for x in [sld]]
    print(sld_int)
    sld_int = sequence.pad_sequences(sld_int, maxlen=maxlen)
    sld_np = np.array(sld_int)
    scores = new_model.predict(sld_np)
    label = new_model.predict_classes(sld_np)
    print(label[0][0], scores[0][0])


def predict_train(score_path):
    """
    针对一群数据进行预测
    :return:
    """
    # global valid_chars
    global maxlen
    sld, label = get_dataset_pinjie(DGA_dataset_path, NDGA_dataset_path)
    X = [[valid_chars[y] for y in x] for x in sld]
    X = sequence.pad_sequences(X, maxlen=maxlen)
    X = np.array(X)
    # global max_features
    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()
    new_model = model_from_json(model_json)
    new_model.load_weights(model_h5_path)
    new_model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    scores = new_model.predict(X)
    scores_ls = [x[0] for x in scores]
    tmp_df = pd.DataFrame({'score': scores_ls})
    tmp_df = tmp_df.sort_values(by='score', ascending=False)  # 降序
    tmp_df.to_csv(score_path+'\lstm_score_rank.csv', index=False, header=None)


def predict_pinjie(url):
    """
    单个数据预测
    """
    # global valid_chars
    global maxlen
    # global max_features
    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()
    new_model = model_from_json(model_json)
    new_model.load_weights(model_h5_path)
    # 编译模型
    new_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    short_url = data_pro(url)
    print("sld-----{}".format(short_url))
    sld_int = [[valid_chars[y] for y in x] for x in [short_url]]
    sld_int = sequence.pad_sequences(sld_int, maxlen=maxlen)
    sld_np = np.array(sld_int)
    scores = new_model.predict(sld_np)
    score = scores[0][0]
    p_value = cal_p(score)
    if score > 0.5:
        label = 1
    else:
        label = 0
    print(label, score, p_value)
    return label, score, p_value


def cal_p(s):
    """
    计算p_value, 二分查找
    :param s: float
    :return:
    """
    global score_l
    flag = 0  # score偏da的对应的
    for i in range(len(score_l)):
        if score_l[i] <= 0.5000000000000000:
            flag = i - 1
            break
    print("flag:{}".format(flag))
    if s > score_l[0]:
        return 1.0
    if s < score_l[-1]:
        return 0.0
    if s == score_l[flag]:
        return 1 / ((flag + 1) * 1.0)

    high_index = len(score_l)
    low_index = 0
    while low_index < high_index:
        mid = int((low_index + high_index) / 2)
        if s > score_l[mid]:
            high_index = mid - 1
        elif s == score_l[mid]:
            if s > 0.5:
                return (flag - mid + 1) / ((flag + 1) * 1.0)
            else:
                return (len(score_l) - mid) / ((len(score_l) - flag - 1) * 1.0)
        else:
            low_index = mid + 1
    if s > 0.5:
        # print(low_index, (flag - low_index), ((flag + 1) * 1.0))
        return (flag - low_index) / ((flag + 1) * 1.0)
    else:
        # print(low_index, len(score_l) - low_index, (len(score_l) - flag - 1) * 1.0)
        return (len(score_l) - low_index) / ((len(score_l) - flag - 1) * 1.0)


def predict_txt(data_path):
    """
    对txt脚本中的一群进行预测
    :param data_path:
    :return:
    """
    global maxlen
    # global max_features
    # with open(model_json_path, "r") as json_file:
    #     model_json = json_file.read()
    # new_model = model_from_json(model_json)
    # new_model.load_weights(model_h5_path)
    # # 编译模型
    # new_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    # label_l = []
    # s_l = []
    # p_l = []
    with open(data_path, 'r', encoding='utf8') as f:
        for url in f.readlines():
            predict_pinjie(url)

            # short_url = data_pro(url)
            # # print(short_url)
            # sld_int = [[valid_chars[y] for y in x] for x in [short_url]]
            # sld_int = sequence.pad_sequences(sld_int, maxlen=maxlen)
            # sld_np = np.array(sld_int)
            # scores = new_model.predict(sld_np)
            # p_value = cal_p(scores[0][0])
            # if scores > 0.5:
            #     label = 1
            # else:
            #     label = 0
            # label_l.append(label)
            # s_l.append(scores[0][0])
            # p_l.append(p_value)


if __name__ == "__main__":
    # DGA_dataset_path = r"Z:\APT\data\sample\sample_black.csv"
    DGA_dataset_path = r"Z:\APT\data\sample\phishing.csv"
    NDGA_dataset_path = r"Z:\APT\data\sample\sample_white.csv"
    model_json_path = r"Z:\APT\model\pinjie\LSTM_model.json"
    model_h5_path = r"Z:\APT\model\pinjie\LSTM_model.h5"
    # # fig_path = r"Z:\APT\fig\roc_lstm.svg"
    # score_path = r"Z:\APT\data\score"
    score_path = r"data\lstm_score_rank.csv"
    tld_path = r'Z:\APT\data\tld.txt'
    tld_list = []
    with open(tld_path, 'r', encoding='utf8') as f:
        for i in f.readlines():
            tld_list.append(i.strip()[1:])

    valid_chars = {'q': 17, '0': 27, 'x': 24, 'd': 4, 'l': 12, 'm': 13, 'v': 22, 'n': 14, 'c': 3, 'g': 7, '7': 34, 'u': 21, '5': 32, 'p': 16, 'h': 8, 'b': 2, '6': 33, '-': 38, 'z': 26, '3': 30, 'f': 6, 't': 20, 'j': 10, '1': 28, '4': 31, 's': 19, 'o': 15, 'w': 23, '9': 36, 'r': 18, 'i': 9, 'e': 5, 'y': 25, 'a': 1, '.': 37, '2': 29, '_': 39, '8': 35, 'k': 11}
    maxlen = 178
    max_features = 40
    # run()
    # predict_train(score_path)

    score_list = []
    score_df = pd.read_csv(score_path, names=['score'])
    score_df = score_df.sort_values(by='score', ascending=False)  # 降序
    score_df.to_csv('data\lstm_score_rank.csv', index=False, header=None)
    score_l = score_df['score'].tolist()

    predict_txt(r'data\3.txt')
    # a = predict_pinjie("www.cnblogs.com")
    # predict_pinjie("amvbylrohzowzfrsbqjpmvsivf-dot-gleowayel400503.uc.r.appspot.com")
    # predict_pinjie("qq.com")
    # predict_pinjie("www.qq.com")
    # # predict_pinjie("baidu.com")
    # # predict_pinjie("jqcros.com")
    # predict_pinjie("maill.czec.com.cn.accountvalidation.verifay728gh4dgy6378et6.com.cdaxpropsvc.net")
    # # predict_pinjie("maill.cgwic.com.accountvalidation.verifay765hgy87.com.cdaxpropsvc.net")
    # predict_pinjie("maill.sasac.gov.cn.accountverify.validation8u6453.jsbch876452.nxjkgdg096574.fghe5392.ncdjkbfkj873e65.nckjdbcj86hty1.cdjcksdcuh57hgy43.njkd8766532.njfg73452.kdjsdkj7564.jdchjsdy.rthfgyert231.winmanagerservice[.]net.")
    # predict_pinjie("webmail.mofcom.gov.cn.accountverify.validation8u2904.jsbchkufd546.nxjkgdgfhh345s.fghese4.ncdjkbfkjh244e.nckjdbcj86hty1.cdjcksdcuh57hgy43.njkd75894t5.njfg87543.kdjsdkj7564.jdchjsdy.rthfgyerty33.wangluojiumingjingli.org")
    # predict_pinjie("webmail.avic.com.accountverify.validation8u7329.jsbchk82056.nxjkgdgf34523.fghe5103.ncdjkbfkjh5674e.nckjdbcj86hty1.cdjcksdcuh57hgy43.njkd75894t5.njfg87543.kdjsdkj7564.jdchjsdy.rthfgyerty86.wangluojiumingjingli.org")
    # predict_pinjie("count.mail.163.com.thecheak.com")
