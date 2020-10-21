"""Train and test LSTM classifier"""
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, classification_report,accuracy_score, f1_score
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
        try:
            SLD = na_list[-2]
            print("error==={}==={}".format(url, SLD))
        except:
            SLD = na_list[-1]
            print("error==={}==={}".format(url, SLD))
    return str(SLD)


def get_dataset(black_dataset_path, white_dataset_path):
    """
    get train test dataset
    """
    bla_df = pd.read_csv(black_dataset_path, names=["domain_name", "label"])
    bla_df = bla_df.sample(n=660471, axis=0, random_state=23)
    bla_df["domain_name"] = bla_df["domain_name"].apply(get_name)
    wh_df = pd.read_csv(white_dataset_path, names=["domain_name", "label"])
    wh_df["domain_name"] = wh_df["domain_name"].apply(get_name)
    df = bla_df.append(wh_df)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=23)

    sld_ls = df_train["domain_name"].to_list()
    label_ls = df_train["label"].to_list()

    test_sld_ls = df_test["domain_name"].to_list()
    test_label_ls = df_test["label"].to_list()
    print("get_dateset done")
    return sld_ls, label_ls, test_sld_ls, test_label_ls


def run(max_epoch=20, nfolds=10, batch_size=128):
    """
    Run train/test on logistic regression model
    """
    global maxlen
    #Read data to process
    X_train, y_train, X_test, y_test = get_dataset(black_dataset_path, white_dataset_path)
    X = list(set(X_train+X_test))

    # Generate a dictionary of valid characters
    print("------{}={}".format("valid_chars", valid_chars))
    maxlen = np.max([len(x) for x in X])
    print("------maxfeatures is :{}".format(max_features))
    print("------maxlen is :{}".format(maxlen))

    # Convert characters to int and pad
    X_train = [[valid_chars[y] for y in x] for x in X_train]
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = [[valid_chars[y] for y in x] for x in X_test]
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    # Convert labels to 0-1 for binary class
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # shuffle
    np.random.seed(4)  # 1024
    index = np.arange(len(X_train))
    np.random.shuffle(index)
    X_train = np.array(X_train)[index]
    y_train = np.array(y_train)[index]

    index2 = np.arange(len(X_test))
    np.random.shuffle(index2)
    X_test = np.array(X_test)[index2]
    y_test = np.array(y_test)[index2]

    #Build the model for two-class classification stage
    model = build_binary_model(max_features, maxlen)

    print ("Training the model for two-class classification stage...")
    print("------train size:{}   {}".format(X_train.size, y_train.size))
    print("------test size:{}   {}".format(X_test.size, y_test.size))
    # sss1 = StratifiedShuffleSplit(n_splits=1, random_state=0)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
    for train, test in sss1.split(X_train, y_train):
        X_train, X_holdout, y_train, y_holdout = X_train[train], X_train[test], y_train[train], y_train[test]   # holdout验证集

    #Create weight for two-class classification stage
    labels_dict = collections.Counter(y_train)
    class_weight = create_class_weight(labels_dict, 0.3)
    print('----class weight:{}'.format(class_weight))
    #20
    best_acc = 0.0
    best_model = None
    for ep in range(max_epoch):
        # model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, class_weight=class_weight)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1, class_weight=class_weight)
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
    if sld == '':
        print("{} {} {}".format(url, 0, 0.0))
        return url, 0, 0.0
    sld_int = [[valid_chars[y] for y in x] for x in [sld]]
    sld_int = sequence.pad_sequences(sld_int, maxlen=maxlen)
    sld_np = np.array(sld_int)
    scores = new_model.predict(sld_np)
    label = new_model.predict_classes(sld_np)
    print("{} {} {}".format(url, label[0][0], scores[0][0]))
    return url, label[0][0], scores[0][0]


def predict_txt():
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

    url_ls = []
    with open("/home/liying/mh/DGA/mh_domain_name/MH_dname.txt", 'r', encoding='utf8') as f:
        for line in f.readlines():
            url_ls.append(line.strip())


    sld_ls = [get_name(url) for url in url_ls]
    print(sld_ls[5])
    sld_int = [[valid_chars[y] for y in x] for x in sld_ls]
    sld_int = sequence.pad_sequences(sld_int, maxlen=maxlen)
    sld_np = np.array(sld_int)
    scores = new_model.predict(sld_np)
    scores = [s[0] for s in scores]
    label = new_model.predict_classes(sld_np)
    label = [l[0] for l in label]
    pre_df = pd.DataFrame({"domain_name": url_ls, "predict": scores, "label": label})
    pre_df.to_csv("/home/liying/mh/DGA/mh_domain_name/MH_dname_lstm.txt", index=None, header=False)


if __name__ == "__main__":
    black_dataset_path = "/home/liying/mh/DGA/data/black/black.csv"
    white_dataset_path = "/home/liying/mh/DGA/data/white/white.csv"
    model_json_path = "./model/LSTM_model.json"
    model_h5_path = "./model/LSTM_model.h5"
    valid_chars = {'q': 17, '0': 27, 'x': 24, 'd': 4, 'l': 12, 'm': 13, 'v': 22, 'n': 14, 'c': 3, 'g': 7, '7': 34, 'u': 21, '5': 32, 'p': 16, 'h': 8, 'b': 2, '6': 33, '-': 38, 'z': 26, '3': 30, 'f': 6, 't': 20, 'j': 10, '1': 28, '4': 31, 's': 19, 'o': 15, 'w': 23, '9': 36, 'r': 18, 'i': 9, 'e': 5, 'y': 25, 'a': 1, '.': 37, '2': 29, '_': 39, '8': 35, 'k': 11}
    maxlen = 63
    max_features = 40
    # run()
    # predict_txt()
    while True:
        predict(input("请输入："))
    predict("csdn_csdn.com")

