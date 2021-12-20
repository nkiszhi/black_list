# -*- coding: utf-8 -*-
"""
Created on 2020/9/13 18:11

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import re
import pickle
import math
import wordfreq
import operator
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

hmm_add = r"./static/hmm_matrix.csv"
gib_add = r"./static/gib_model.pki"
gramfile_add = r"./static/n_gram_rank_freq.txt"
private_tld_file = r"./static/private_tld.txt"
tld_add = r"./static/tld.txt"
tld_rank_add = r"./static/tld_rank.txt"
white_file_add = r"./data/sample/sample_white.csv"
black_file_add = r"./data/sample/sample_black.csv"
phishing_file_add = r"./data/sample/phishing.csv"
hmm_prob_threshold = -120
tld_list = list()
with open(tld_add, 'r', encoding='utf8') as f:
    for i in f.readlines():
        tld_list.append(i.strip().strip('.'))
accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])
abuse_tld = ['email', 'fit', 'tk', 'fail', 'run', 'rest', 'ml', 'cn', 'viajes', 'cf', 'recipes', 'gq', 'ga']

feature_dir = r"./data/feature"
model_dir = r"./data/model"

def wash_tld(dn):
    """
    将域名字符串中顶级域名去掉，剩余部分拼接成完整字符串
    :param dn: 原始域名
    :return: 拼接字符串
    """
    dn_list = dn.split('.')
    dn_list = list(set(dn_list).difference(set(tld_list)))
    namestring = "".join(dn_list)
    return namestring

def cal_rep_cart(ns):
    """
    计算字符串中重复出现的字符个数
    :param SLD: 字符串
    :return: 重复字符个数
    """
    count = Counter(i for i in ns).most_common()
    sum_n = 0
    for letter, cnt in count:
        if cnt > 1:
            sum_n += 1
    return sum_n

def cal_ent_gni_cer(SLD):
    """
    计算香农熵, Gini值, 字符错误的分类
    :param url:
    :return:
    """
    f_len = float(len(SLD))
    count = Counter(i for i in SLD).most_common()  # unigram frequency
    ent = -sum(float(j / f_len) * (math.log(float(j / f_len), 2)) for i, j in count)  # shannon entropy
    gni = 1 - sum(float(j / f_len) * float(j / f_len) for i, j in count)
    cer = 1 - max(float(j/ f_len) for i, j in count)
    return ent, gni, cer


def cal_gram_med(SLD, n):
    """
    计算字符串n元频率中位数
    :param SLD: 字符串
    :param n: n
    :return:
    """
    grams = [SLD[i:i + n] for i in range(len(SLD) - n+1)]
    fre = list()
    for s in grams:
        fre.append(wordfreq.zipf_frequency(s, 'en'))
    return np.median(fre)


def cal_hmm_prob(url):
    """
    计算成文概率, 结果越小越异常
    :param url:
    :return: 概率
    """
    hmm_dic = defaultdict(lambda: defaultdict(float))
    with open(hmm_add, 'r') as f:
        for line in f.readlines():
            key1, key2, value = line.rstrip().split('\t')  # key1 can be '' so rstrip() only
            value = float(value)
            hmm_dic[key1][key2] = value
    url = '^' + url.strip('.') + '$'
    gram2 = [url[i:i+2] for i in range(len(url)-1)]
    prob = hmm_dic[''][gram2[0]]

    for i in range(len(gram2)-1):
        prob *= hmm_dic[gram2[i]][gram2[i+1]]
    if prob < math.e ** hmm_prob_threshold:
        prob = -999
    return prob


def cal_gib(SLD):
    """
    计算gib标签
    :param SLD:
    :return: 1: 正常 0: 异常
    """
    gib_model = pickle.load(open(gib_add, 'rb'))
    mat = gib_model['mat']
    threshold = gib_model['thresh']

    log_prob = 0.0
    transition_ct = 0
    SLD = re.sub("[^a-z]", "", SLD)
    gram2 = [SLD[i:i + 2] for i in range(len(SLD) - 1)]
    for a, b in gram2:
        log_prob += mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    prob = math.exp(log_prob / (transition_ct or 1))
    return int(prob > threshold)


def load_gramdict():
    """
    加载n元排序字典
    :return: 字典
    """
    rank_dict = dict()
    with open(gramfile_add, 'r') as f:
        for line in f:
            cat, gram, freq, rank = line.strip().split(',')
            rank_dict[gram] = int(rank)
    return rank_dict


##############################钓鱼网站###############################
def phishing_get_feature(dn):
    """
    钓鱼网站特征提取
    :param url: 域名
    :return: 25维特征
    """
    paras = dn.split('.')
    ns = re.sub(r"\.", "", dn)
    namestring = re.sub(r"\.|_|-", "", dn)
    tld_rank_dict = load_tlddict()

    # 1.段数
    para_sum = len(paras)
    # 2.包含TLD数量
    tlds = list(set(paras) & set(tld_list))
    tld_sum = len(tlds)
    # 3.包含被滥用TLD数量
    abuse_sum = len(set(paras) & set(abuse_tld))
    # 4.包含恶意TLD排序均值
    tldrank = [tld_rank_dict[t] for t in tlds]
    if tldrank:
        tldrank_avg = np.mean(tldrank)
    else:
        tldrank_avg = -1
    # 5. 字符串长度
    string_len = len(namestring)
    # 6. 字符串不重复字符数
    uni_string = len(set(namestring))
    # 7. 是否以数字开头
    flag_dig = 0
    if re.match("[0-9]", namestring) != None:
        flag_dig = 1
    # 8. 特殊符号在字符串中占比
    sym = len(re.findall(r"\.|_|-", dn))/string_len
    # 9. 数字在字符串中占比
    dig = len(re.findall(r"[0-9]", namestring))/string_len
    # 10. 元音字母在字符串中占比
    vow = len(re.findall(r"a|e|i|o|u", namestring))/string_len
    # 11. 辅音字母在字符串中占比
    con = len(re.findall(r"b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z", namestring))/string_len
    # 12. 重复字符在字符串不重复字符中占比
    rep_char_ratio = cal_rep_cart(namestring)/uni_string
    # 13. 域名中连续辅音占比
    con_list = re.findall(r"[b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z]{2,}", namestring)
    con_len = [len(con) for con in con_list]
    cons_con_ratio = sum(con_len)/string_len
    # 14. 域名中连续数字占比
    dig_list = re.findall(r"[0-9]{2,}", namestring)
    dig_len = [len(dig) for dig in dig_list]
    cons_dig_ratio = sum(dig_len)/string_len
    # 15. 字符串中由'-'分割的令牌数
    tokens_sld = len(ns.split('-'))
    # 16. 字符串中数字总数
    digits_sld = len(re.findall(r"[0-9]", namestring))
    # # 14. SLD中字符的归一化熵值
    # # 15. SLD的Gini值
    # # 16. SLD中字符分类的错误
    # ent, gni, cer = cal_ent_gni_cer(namestring)


    feature = [para_sum, tld_sum, abuse_sum, tldrank_avg, string_len, uni_string, flag_dig, sym, dig, vow, con,
               rep_char_ratio, cons_con_ratio, cons_dig_ratio, tokens_sld, digits_sld]
    return feature

def phishing_feature_extraction(df):
    """
    特征提取, 归一化
    :param df:
    :return:
    """
    col = ["domain_name", "label", "para_sum", "tld_sum", "abuse_sum", "tldrank_avg", "string_len", "uni_string",
           "flag_dig", "sym", "dig", "vow", "con", "rep_char_ratio", "cons_con_ratio", "cons_dig_ratio", "tokens_sld",
           "digits_sld"]
    fea_list = list()
    for ind in df.index:
        fea = df.loc[ind].tolist()
        if ind % 1000 == 0:
            print("{}...".format(ind))
        fea.extend(phishing_get_feature(df.at[ind, 0]))
        fea_list.append(fea)
    fea_df = pd.DataFrame(fea_list, columns=col)

    return fea_df


def phishing_dataset_generation():
    """
    数据集划分,
    :return:
    """

    bk_df = pd.read_csv(phishing_file_add, header=None)
    wh_df = pd.read_csv(white_file_add, header=None)

    df = bk_df.append(wh_df, ignore_index=True)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[1], random_state=23)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("__________Generating Train Set__________")
    train_feature = phishing_feature_extraction(df_train)
    # train_feature.to_csv(r"{}\raw_train_features.csv".format(feature_dir), index=None)
    train_feature = train_feature.set_index(['domain_name', 'label'])
    standardScaler = StandardScaler()
    standardScaler.fit(train_feature.values)

    print("__________Generating Test Set__________")
    test_feature = phishing_feature_extraction(df_test)
    # test_feature.to_csv(r"{}\raw_test_features.csv".format(feature_dir), index=None)
    test_feature = test_feature.set_index(['domain_name', 'label'])

    train_feature = pd.DataFrame(standardScaler.transform(train_feature), index=train_feature.index,
                                 columns=train_feature.columns)
    train_feature = train_feature.reset_index()
    train_feature.to_csv(r"{}/phishing_train_features.csv".format(feature_dir), index=None)
    test_feature = pd.DataFrame(standardScaler.transform(test_feature), index=test_feature.index,
                                columns=test_feature.columns)
    test_feature = test_feature.reset_index()
    test_feature.to_csv(r"{}/phishing_test_features.csv".format(feature_dir), index=None)
    pickle.dump(standardScaler, open(r"{}/phishing/standardscalar.pkl".format(model_dir), 'wb'))
    return

def create_TLD_rank():
    """
    根据钓鱼网站黑名单统计顶级域名危险程度排名
    :return:
    """
    tld_sum = dict()
    for tld in tld_list:
        tld_sum[tld] = 0
    with open(phishing_file_add, 'r', encoding='utf8') as f:
        for line in f.readlines():
            paras = line.split(',')[0].split('.')
            intersec = list(set(paras) & set(tld_list))
            for t in intersec:
                tld_sum[t] += 1
    tld_sum = dict(sorted(tld_sum.items(), key=operator.itemgetter(1), reverse=1))
    df = pd.DataFrame.from_dict(tld_sum, orient='index')
    rank = [i+1 for i in range(len(tld_sum))]
    df[1] = rank
    df.to_csv(tld_rank_add, header=False, encoding='utf8')

def load_tlddict():
    """
    读取顶级域名危险程度排名
    :return:
    """
    rank_dict = dict()
    with open(tld_rank_add, 'r', encoding='utf8') as f:
        for line in f:
            tld, freq, rank = line.strip().split(',')
            rank_dict[tld] = int(rank)
    return rank_dict

if __name__ == "__main__":
    phishing_dataset_generation()




