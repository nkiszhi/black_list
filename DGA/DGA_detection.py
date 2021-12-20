# -*- coding: utf-8 -*-
"""
Created on 2020/8/16 12:38

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_exaction import RF_get_feature
from feature_exaction import SVM_get_feature


# 二分类随机森林(B_RF)算法
class RF_classifier:

    def __init__(self):
        self.RF_clf = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                             random_state=23, n_jobs=-1, max_features=20)
        self.standardScaler = StandardScaler()

    def train(self, train_feature_add, model_add):
        """
        BRF算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______RF Training_______")
        self.RF_clf.fit(x_train, y_train)
        pickle.dump(self.RF_clf, open(model_add, 'wb'))

    def load(self, model_add, standard_scaler_add):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :param standard_scaler_add: 归一化scaler存储路径
        :return:
        """
        self.RF_clf = pickle.load(open(model_add, 'rb'))
        self.standardScaler = pickle.load(open(standard_scaler_add, 'rb'))

    def predict(self, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
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

    def predict_singleDN(self, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
        """
        feature = self.standardScaler.transform(pd.DataFrame([RF_get_feature(dname)]))
        label = self.RF_clf.predict(feature)
        prob = self.RF_clf.predict_proba(feature)
        print("label:", label[0])
        print("mal_prob:", prob[0][1])
        return label[0], prob[0][1]


# SVM算法
class SVM_classifier:

    def __init__(self):
        self.SVM_clf = SVC(kernel='linear', probability=True, random_state=23)
        self.standardScaler = StandardScaler()

    def train(self, train_feature_add, model_add):
        """
        SVM算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______SVM Training_______")
        self.SVM_clf.fit(x_train, y_train)
        pickle.dump(self.SVM_clf, open(model_add, 'wb'))

    def load(self, model_add, standard_scaler_add):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :param standard_scaler_add: 归一化scaler存储路径
        :return:
        """
        self.SVM_clf = pickle.load(open(model_add, 'rb'))
        self.standardScaler = pickle.load(open(standard_scaler_add, 'rb'))

    def predict(self, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______SVM Predicting_______")
        y_predict = self.SVM_clf.predict(x_test)
        print("SVM accuracy: ", self.SVM_clf.score(x_test, y_test))
        print("SVM precision: ", precision_score(y_test, y_predict, average='macro'))
        print("SVM recall: ", recall_score(y_test, y_predict, average='macro'))
        print("SVM F1: ", f1_score(y_test, y_predict, average='macro'))
        print("SVM TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

        plot_roc_curve(self.SVM_clf, x_test, y_test)
        plt.show()

    def predict_singleDN(self, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
        """
        feature = self.standardScaler.transform(pd.DataFrame([SVM_get_feature(dname)]))
        label = self.SVM_clf.predict(feature)
        prob = self.SVM_clf.predict_proba(feature)
        print("label:", label[0])
        print("mal_prob:", prob[0][1])
        return label[0], prob[0][1]


# XGBoost算法
class XGBoost_classifier:

    def __init__(self):
        self.XGBoost_clf = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimator=100, silent=True,
                                         objective='binary:logistic')
        self.standardScaler = StandardScaler()

    def train(self, train_feature_add, model_add):
        """
        XGBoost算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______XGBoost Training_______")
        self.XGBoost_clf.fit(x_train, y_train)
        pickle.dump(self.XGBoost_clf, open(model_add, 'wb'))

    def load(self, model_add, standard_scaler_add):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :param standard_scaler_add: 归一化scaler存储路径
        :return:
        """
        self.XGBoost_clf = pickle.load(open(model_add, 'rb'))
        self.standardScaler = pickle.load(open(standard_scaler_add, 'rb'))

    def predict(self, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
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

    def predict_singleDN(self, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
        """
        feature = self.standardScaler.transform(pd.DataFrame([RF_get_feature(dname)]))
        label = self.XGBoost_clf.predict(feature)
        prob = self.XGBoost_clf.predict_proba(feature)
        print("label:", label[0])
        print("mal_prob:", prob[0][1])
        return label[0], prob[0][1]


if __name__ == "__main__":
    train_add = r"./data/feature/RF_train_features.csv"
    test_add = r"./data/feature/RF_test_features.csv"
    SVM_train_add = r"./data/feature/SVM_train_features.csv"
    SVM_test_add = r"./data/feature/SVM_test_features.csv"
    RF_model_add = r"./data/model/RF_model.pkl"
    RF_standard_scaler_add = r"./data/model/RF_standardscalar.pkl"
    SVM_model_add = r"./data/model/SVM_model.pkl"
    SVM_standard_scaler_add = r"./data/model/SVM_standardscalar.pkl"
    # XGBoost用的是RF特征
    XGBoost_model_add = r"./data/model/XGBoost_model.pkl"
    XGBoost_standard_scaler_add = r"./data/model/RF_standardscalar.pkl"

    # RF_clf = RF_classifier()
    # # RF_clf.train(train_add, RF_model_add)
    # RF_clf.load(RF_model_add, RF_standard_scaler_add)
    # RF_clf.predict(test_add)
    # # RF_clf.predict_singleDN("baijiahao.dsalkswjgoijdslk.com")
    # # RF_clf.predict_singleDN("baijiahao.cnblog.org")
    #
    # SVM_clf = SVM_classifier()
    # # SVM_clf.train(SVM_train_add, SVM_model_add)
    # SVM_clf.load(SVM_model_add, SVM_standard_scaler_add)
    # SVM_clf.predict(SVM_test_add)
    # # SVM_clf.predict_singleDN("baijiahao.dsalkswjgoijdslk.com")

    XGBoost_clf = XGBoost_classifier()
    # XGBoost_clf.train(SVM_train_add, XGBoost_model_add)
    XGBoost_clf.load(XGBoost_model_add, XGBoost_standard_scaler_add)
    # XGBoost_clf.predict(SVM_test_add)
    XGBoost_clf.predict_singleDN("baijiahao.dsalkswjgoijdslk.com")
