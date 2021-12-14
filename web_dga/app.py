#!/usr/bin/env python3
# -*-coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"

from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
#from flask_paginate import Pagination, get_page_args
import json
import pandas as pd
import sys
import imp
import re
import os
import string
from DGA_detection import LSTM_classifier, XGBoost_classifier, RF_classifier, SVM_classifier, KNN_classifier, GNB_classifier, LR_classifier, DT_classifier, GDBT_classifier, AdaBoost_classifier

HOST_IP = "0.0.0.0"
PORT = 5003
ROW_PER_PAGE = 20

imp.reload(sys)


app = Flask(__name__)

labels = None
contents = None
slabels = None
scontents = None
title = None
list_sha256 = []

list_info = []

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, str):
            return str(obj)


def get_page_info(list_info, offset=0, per_page=ROW_PER_PAGE):
    return list_info[offset: offset + per_page]


@app.route('/')
def show_index():
    return render_template('malware_url_query.html')


@app.route('/malware_url')
def show_malUrl():
    return render_template("malware_url_query.html")


@app.route('/malware_reuslt', methods=["POST"])
def detect_url():
    # 1. get url string
    url_str = request.form["url"].strip()
    # 2. validate string
    if url_str == '':
        return render_template("malware_url_result.html",
                           status=400, url=url_str,
                           message="域名不可为空!!")
    validate = re.match(r"^[A-Za-z0-9._\-]*$", url_str)
    if validate == None:
        return render_template("malware_url_result.html",
                               status=401, url=url_str,
                               message="域名格式不正确，域名中只能包含下划线、短横线、点、字母、数字，请输入正确域名！！")
    # # 3. detect the str with models
    KNN = KNN_clf.predict_singleDN(model_folder, url_str)
    SVM = SVM_clf.predict_singleDN(model_folder, url_str)
    GNB = GNB_clf.predict_singleDN(model_folder, url_str)
    LR = LR_clf.predict_singleDN(model_folder, url_str)
    DT = DT_clf.predict_singleDN(model_folder, url_str)
    BRF = RF_clf.predict_singleDN(model_folder, url_str)
    ADABOOST = AdaBoost_clf.predict_singleDN(model_folder, url_str)
    GDBT = GDBT_clf.predict_singleDN(model_folder, url_str)
    XGBOOST = XGBoost_clf.predict_singleDN(model_folder, url_str)
    LSTM = LSTM_clf.predict_singleDN(url_str)

    base_result = {}
    result = -1
    base_result["KNN"] = [KNN[0], format(KNN[1], '.4f'), KNN[2]]
    base_result["SVM"] = [SVM[0], format(SVM[1], '.4f'), SVM[2]]
    base_result["GNB"] = [GNB[0], format(GNB[1], '.4f'), GNB[2]]
    base_result["LR"] = [LR[0], format(LR[1], '.4f'), LR[2]]
    base_result["DT"] = [DT[0], format(DT[1], '.4f'), DT[2]]
    base_result["B-RF"] = [BRF[0], format(BRF[1], '.4f'), BRF[2]]
    base_result["ADABOOST"] = [ADABOOST[0], format(ADABOOST[1], '.4f'), ADABOOST[2]]
    base_result["GDBT"] = [GDBT[0], format(GDBT[1], '.4f'), GDBT[2]]
    base_result["XGBOOST"] = [XGBOOST[0], format(XGBOOST[1], '.4f'), XGBOOST[2]]
    base_result["LSTM"] = [LSTM[0], format(LSTM[1], '.4f'), LSTM[2]]

    ## 4.p
    KNN = KNN if KNN[2] > 0.01 else (2, KNN[1], KNN[2])
    SVM = SVM if SVM[2] > 0.01 else (2, SVM[1], SVM[2])
    GNB = GNB if GNB[2] > 0.01 else (2, GNB[1], GNB[2])
    LR = LR if LR[2] > 0.01 else (2, LR[1], LR[2])
    DT = DT if DT[2] > 0.01 else (2, DT[1], DT[2])
    BRF = BRF if BRF[2] > 0.01 else (2, BRF[1], BRF[2])
    ADABOOST = ADABOOST if ADABOOST[2] > 0.01 else (2, ADABOOST[1], ADABOOST[2])
    GDBT = GDBT if GDBT[2] > 0.01 else (2, GDBT[1], GDBT[2])
    XGBOOST = XGBOOST if XGBOOST[2] > 0.01 else (2, XGBOOST[1], XGBOOST[2])
    LSTM = LSTM if LSTM[2] > 0.01 else (2, LSTM[1], LSTM[2])
    


    base_resultT = {"KNN": KNN, "SVM": SVM, "GNB": GNB, "LR": LR, "DT": DT, "B-RF": BRF, "ADABOOST": ADABOOST, "GDBT": GDBT, "XGBOOST": XGBOOST, "LSTM": LSTM}

    ## 5.result
    rs_list = []
    for item in base_resultT:
        rs_list.append(base_resultT[item][0])
    if len(set(rs_list)) == 1:
        if LSTM[0] != 2:
            result = LSTM[0]
            return render_template("malware_url_result.html", status=200, url=url_str, base_result=base_result, result=result)
        elif LSTM[0] == 2:   # 所有模型都表现很差
            sort_result = sorted(base_resultT.items(), key=lambda base_resultT: base_resultT[1][2], reverse=True)
            if(sort_result[0][1][2] <= 0.5):
                result = 2
            else:
                result = sort_result[0][1][0]
            return render_template("malware_url_result.html", status=200, url=url_str, base_result=base_result, result=result)

    new_result = {}
    for item in base_resultT:
        if base_resultT[item][0] != 2:
            new_result[item] = base_resultT[item]
    sort_result = sorted(new_result.items(), key=lambda new_result: new_result[1][2], reverse=True)
    if(sort_result[0][1][2] <= 0.5):
        result = 2
    else:
        result = sort_result[0][1][0]
    return render_template("malware_url_result.html", status=200, url=url_str, base_result=base_result, result=result)


with open('./config.json', 'r') as load_f:
    load_dic = json.load(load_f)

model_folder = load_dic["model_floder"]
XGBoost_clf = XGBoost_classifier()
XGBoost_clf.load(model_folder)
RF_clf = RF_classifier()
RF_clf.load(model_folder)
SVM_clf = SVM_classifier()
SVM_clf.load(model_folder)
KNN_clf = KNN_classifier()
KNN_clf.load(model_folder)
GNB_clf = GNB_classifier()
GNB_clf.load(model_folder)
LR_clf = LR_classifier()
LR_clf.load(model_folder)
DT_clf = DT_classifier()
DT_clf.load(model_folder)
AdaBoost_clf = AdaBoost_classifier()
AdaBoost_clf.load(model_folder)
GDBT_clf = GDBT_classifier()
GDBT_clf.load(model_folder)
LSTM_clf = LSTM_classifier()
LSTM_clf.load(load_dic["LSTM"]["model_add"], load_dic["LSTM"]["model_weight"])

if __name__ == '__main__':
    # debug=True,
    app.run(host=HOST_IP, port=PORT, threaded=True)

