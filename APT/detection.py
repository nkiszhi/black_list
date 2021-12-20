# -*- coding: utf-8 -*-
"""
Created on 2020/9/14 9:36

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

from feature_extraction import wash_tld, phishing_get_feature


# 二分类随机森林(B_RF)算法
class RF_classifier:

    def __init__(self):
        self.RF_clf = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                             random_state=23, n_jobs=-1, max_features=20)
        self.standardScaler = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        BRF算法训练数据
        :param model_folder: 模型存储文件夹
        :param train_feature_add: 训练数据路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______RF Training_______")
        self.RF_clf.fit(x_train, y_train)
        mal_scores = np.array(self.RF_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/RF_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.RF_clf, open("{}/RF_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_folder: 模型存储文件夹
        :return:
        """
        self.RF_clf = pickle.load(open("{}/RF_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/RF_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param model_folder: 模型存储文件夹
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______RF Predicting_______")
        y_predict = self.RF_clf.predict(x_test)
        print("RF accuracy: ", self.RF_clf.score(x_test, y_test))
        print("RF precision: ", precision_score(y_test, y_predict, average='macro'))
        print("RF recall: ", recall_score(y_test, y_predict, average='macro'))
        print("RF F1: ", f1_score(y_test, y_predict, average='macro'))
        print("RF TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

        plot_roc_curve(self.RF_clf, x_test, y_test)
        plt.show()

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param model_folder: 模型存储文件夹
        :param dname: 域名
        :return:
        """
        if not self.isload_:
            self.load(model_folder)
        dname = dname.replace("www.", "")
        dname = wash_tld(dname)
        if dname == "":
            label = 0
            prob = 0.0000
            p_value = 1.0000
            print("label:", label)
            print("mal_prob:", prob)
            print("p_value:", p_value)
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([phishing_get_feature(dname)]))
            label = self.RF_clf.predict(feature)
            prob = self.RF_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("label:", label[0])
            print("mal_prob:", prob[0][1])
            print("p_value:", p_value)
            return label[0], prob[0][1], p_value


# XGBoost算法
class XGBoost_classifier:

    def __init__(self):
        self.XGBoost_clf = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimator=100, silent=True,
                                         objective='binary:logistic')
        self.standardScaler = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        XGBoost算法训练数据
        :param model_folder: 模型存储文件夹
        :param train_feature_add: 训练数据路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______XGBoost Training_______")
        self.XGBoost_clf.fit(x_train, y_train)
        mal_scores = np.array(self.XGBoost_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/XGBoost_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.XGBoost_clf, open("{}/XGBoost_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_folder: 模型存储文件夹
        :return:
        """
        self.XGBoost_clf = pickle.load(open("{}/XGBoost_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/XGBoost_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param model_folder: 模型存储文件夹
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______XGBoost Predicting_______")
        y_predict = self.XGBoost_clf.predict(x_test)
        print("XGBoost accuracy: ", self.XGBoost_clf.score(x_test, y_test))
        print("XGBoost precision: ", precision_score(y_test, y_predict, average='macro'))
        print("XGBoost recall: ", recall_score(y_test, y_predict, average='macro'))
        print("XGBoost F1: ", f1_score(y_test, y_predict, average='macro'))
        print("XGBoost TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

        plot_roc_curve(self.XGBoost_clf, x_test, y_test)
        plt.show()

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param model_folder: 模型存储文件夹
        :param dname: 域名
        :return:
        """
        if not self.isload_:
            self.load(model_folder)
        dname = dname.replace("www.", "")
        dname = wash_tld(dname)
        if dname == "":
            label = 0
            prob = 0.0000
            p_value = 1.0000
            print("label:", label)
            print("mal_prob:", prob)
            print("p_value:", p_value)
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([phishing_get_feature(dname)]))
            label = self.XGBoost_clf.predict(feature)
            prob = self.XGBoost_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("label:", label[0])
            print("mal_prob:", prob[0][1])
            print("p_value:", p_value)
            return label[0], prob[0][1], p_value


def cal_pValue(score_list, key, label):
    """
    计算p_value
    :param score_list: 训练集得分列表
    :param key: 测试样本得分
    :param label: 测试样本标签
    :return: p_value, 保留四位小数
    """
    count = 0
    if label == 0:
        temp = sorted(score_list, reverse=True)
        score_list = [i for i in temp if i <= 0.5]
        left = 0
        right = len(score_list) - 1
        while left <= right:
            middle = (left+right)//2
            if key < score_list[middle]:
                left = middle + 1
            elif key > score_list[middle]:
                right = middle - 1
            else:
                count = middle + 1
                break
        count = left
    elif label == 1:
        temp = sorted(score_list, reverse=False)
        score_list = [i for i in temp if i > 0.5]
        left = 0
        right = len(score_list) - 1
        while left <= right:
            middle = (left+right)//2
            if key > score_list[middle]:
                left = middle + 1
            elif key < score_list[middle]:
                right = middle - 1
            else:
                count = middle + 1
                break
        count = left
    p_value = count/len(score_list)
    return round(p_value, 4)


if __name__ == "__main__":
    train_add = r"./data/feature/train_features.csv"
    test_add = r"./data/feature/test_features.csv"
    RF_model_add = r"./data/model/RF_model.pkl"
    XGBoost_model_add = r"./data/model/XGBoost_model.pkl"
    standard_scaler_add = r"./data/model/standardscalar.pkl"

    phishing_train_add = r"./data/feature/phishing_train_features.csv"
    phishing_test_add = r"./data/feature/phishing_test_features.csv"
    phishing_model_folder = r"./data/model/phishing"

    # RF_clf = RF_classifier()
    # RF_clf.train(phishing_model_folder, phishing_train_add)
    # RF_clf.predict(phishing_model_folder, phishing_test_add)
    # RF_clf.predict_singleDN(phishing_model_folder, "nkamg.dsalkswjgoijdslk.com")


    # XGBoost_clf = XGBoost_classifier()
    # XGBoost_clf.train(phishing_model_folder, phishing_train_add)
    # XGBoost_clf.predict(phishing_model_folder, phishing_test_add)
    # XGBoost_clf.predict_singleDN(phishing_model_folder, "urldetec.gcowsec.com")

    XGBoost_clf = XGBoost_classifier()
    mal = 0
    benigh = 0
    with open("test.txt", "r") as f:
        for line in f.readlines():
            name = line.strip()
            print("-----------------------------")
            print(name)
            label, mal_prob, p_value = XGBoost_clf.predict_singleDN(phishing_model_folder, name)
            if label:
                mal += 1
            else:
                benigh += 1
    print("-----------------------------")
    print("1:{}, 0:{}".format(mal, benigh))
    print("accu:", mal/(mal+benigh))

