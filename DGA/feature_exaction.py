# -*- coding: utf-8 -*-
"""
Created on 2020/8/13 10:01

@author : dengcongyi0701@163.com

Description:

"""
import tld
import re
import math
import pandas as pd
import numpy as np
import pickle
import wordfreq
import string
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dgaTLD_list = ["cf", "recipes", "email", "ml", "gq", "fit", "cn", "ga", "rest", "tk"]
hmm_add = r".\static\hmm_matrix.csv"
gib_add = r".\static\gib_model.pki"
gramfile_add = r".\static\n_gram_rank_freq.txt"
private_tld_file = r".\static\private_tld.txt"
hmm_prob_threshold = -120
white_file_add = r"M:\DGA\data\white\white.csv"
black_file_add = r"M:\DGA\data\black\black.csv"

feature_dir = r"M:\DGA\features"
model_dir = r"M:\DGA\model"

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

RF_col = ["domain_name", "label", "domain_len", "sld_len", "tld_len", "uni_domain", "uni_sld", "uni_tld", "flag_dga",
           "flag_dig", "sym", "hex", "dig", "vow", "con", "rep_char_ratio", "cons_con_ratio", "cons_dig_ratio",
           "tokens_sld", "digits_sld", "ent", "gni", "cer", "gram2_med", "gram3_med", "gram2_cmed", "gram3_cmed"]
SVM_col = ["domain_name", "label", "entropy", "f_len", "ent_flen", "vowel_ratio", "digit_ratio", "repeat_letter",
           "consec_digit", "consec_consonant", "gib_value", "hmm_log_prob", "avg_gram1_rank", "avg_gram2_rank",
           "avg_gram3_rank", "std_gram1_rank", "std_gram2_rank", "std_gram3_rank", "has_private_tld"]

def get_name(url):
    """
    用python自带库进行域名提取
    :param url: url
    :return: 二级域名，顶级域名
    """
    url = url.strip(string.punctuation)
    try:
        TLD = tld.get_tld(url, as_object=True, fix_protocol=True)
        SLD = tld.get_tld(url, as_object=True, fix_protocol=True).domain

    except Exception as e:
        na_list = url.split(".")
        TLD = na_list[-1]
        SLD = na_list[-2]
    return str(TLD), str(SLD)


def cal_rep_cart(SLD):
    """
    计算字符串中重复出现的字符个数
    :param SLD: 字符串
    :return: 重复字符个数
    """
    count = Counter(i for i in SLD).most_common()
    sum_n = 0
    for letter, cnt in count:
        if cnt > 1:
            sum_n += 1
    return sum_n


def cal_rep_letter(SLD):
    """
    计算字符串中重复出现的字母个数
    :param SLD: 字符串
    :return: 重复字母个数
    """
    count = Counter(i for i in SLD if i.isalpha()).most_common()
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


def load_gramdict_privatetld():
    """
    加载n元排序字典
    :return: 字典
    """
    rank_dict = dict()
    with open(gramfile_add, 'r') as f:
        for line in f:
            cat, gram, freq, rank = line.strip().split(',')
            rank_dict[gram] = int(rank)
    pritld_list = list()
    with open(private_tld_file, 'r') as f:
        pritld_list = set(line.strip() for line in f)
    return rank_dict, pritld_list


def RF_get_feature(url):
    """
    B-RF算法特征提取
    :param url: 域名
    :return: 25维特征
    """
    TLD, SLD = get_name(url)
    url = SLD+"."+TLD
    url_rm = re.sub(r"\.|_|-", "", url)
    TLD_rm = re.sub(r"\.|_|-", "", TLD)
    SLD_rm = re.sub(r"\.|_|-", "", SLD)

    # 1. 域名总长度
    domain_len = len(url)
    # 2. SLD长度
    sld_len = len(SLD)
    # 3. TLD长度
    tld_len = len(TLD)
    # 4. 域名不重复字符数
    uni_domain = len(set(url_rm))
    # 5. SLD不重复字符数
    uni_sld = len(set(SLD_rm))
    # 6. TLD不重复字符数
    uni_tld = len(set(TLD_rm))

    # 7. 是否包含某些恶意顶级域名 https://www.spamhaus.org/statistics/tlds/
    flag_dga = 0
    for t in dgaTLD_list:
        if t in url:
            flag_dga = 1

    # 8. 是否以数字开头
    flag_dig = 0
    if re.match("[0-9]", url) != None:
        flag_dig = 1

    # 9. 特殊符号在SLD中占比
    sym = len(re.findall(r"\.|_|-", SLD))/sld_len
    # 10. 十六进制字符在SLD中占比
    hex = len(re.findall(r"[0-9]|[a-f]", SLD))/sld_len
    # 11. 数字在SLD中占比
    dig = len(re.findall(r"[0-9]", SLD))//sld_len
    # 12. 元音字母在SLD中占比
    vow = len(re.findall(r"a|e|i|o|u", SLD))/sld_len
    # 13. 辅音字母在SLD中占比
    con = len(re.findall(r"b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z", SLD))/sld_len
    # 14. 重复字符在SLD不重复字符中占比
    rep_char_ratio = cal_rep_cart(SLD_rm)/uni_sld
    # 15. 域名中连续辅音占比
    con_list = re.findall(r"[b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z]{2,}", url)
    con_len = [len(con) for con in con_list]
    cons_con_ratio = sum(con_len)/domain_len
    # 16. 域名中连续数字占比
    dig_list = re.findall(r"[0-9]{2,}", url)
    dig_len = [len(dig) for dig in dig_list]
    cons_dig_ratio = sum(dig_len)/domain_len
    # 17. SLD中由'-'分割的令牌数
    tokens_sld = len(SLD.split('-'))
    # 18. SLD中数字总数
    digits_sld = len(re.findall(r"[0-9]", SLD))
    # 19. SLD中字符的归一化熵值
    # 20. SLD的Gini值
    # 21. SLD中字符分类的错误
    ent, gni, cer = cal_ent_gni_cer(SLD)
    # 22. SLD中2元频次的中位数
    gram2_med = cal_gram_med(SLD, 2)
    # 23. SLD中3元频次的中位数
    gram3_med = cal_gram_med(SLD, 3)
    # 24. 重复SLD中2元频次中位数
    gram2_cmed = cal_gram_med(SLD+SLD, 2)
    # 25. 重复SLD中3元频次中位数
    gram3_cmed = cal_gram_med(SLD+SLD, 3)
    # # 26. 域名的hmm成文概率
    # hmm_prob = cal_hmm_prob(url)
    # # 27. gib判断SLD是否成文
    # sld_gib = cal_gib(SLD)

    feature = [domain_len, sld_len, tld_len, uni_domain, uni_sld, uni_tld, flag_dga, flag_dig, sym, hex, dig, vow,
               con, rep_char_ratio, cons_con_ratio, cons_dig_ratio, tokens_sld, digits_sld, ent, gni, cer, gram2_med,
               gram3_med, gram2_cmed, gram3_cmed]
    return feature


def SVM_get_feature(url):
    gram_rank_dict, private_tld = load_gramdict_privatetld()
    TLD, SLD = get_name(url)

    # 17. 是否包含私人域名#
    has_private_tld = 0
    for tld in private_tld:
        if tld in url:
            has_private_tld = 1
            name_list = tld.split('.')
            TLD = name_list[-1]
            SLD = name_list[-2]

    url = SLD + "." + TLD
    # 1. 香农熵#
    entropy = cal_ent_gni_cer(TLD)[0]
    # 2. SLD长度#
    f_len = float(len(SLD))
    # 3. 香农熵与SLD长度比#
    ent_flen = entropy/f_len
    # 4. SLD中元音占比#
    vowel_ratio = len(re.findall(r"a|e|i|o|u", SLD)) / f_len
    # 5. SLD中数字占比#
    digit_ratio = len(re.findall(r"[0-9]", SLD)) / f_len
    # 6. SLD中重复字母占比#
    repeat_letter = cal_rep_letter(SLD) / f_len
    # 7. SLD连续数字占比#
    dig_list = re.findall(r"[0-9]{2,}", url)
    dig_len = [len(dig) for dig in dig_list]
    consec_digit = sum(dig_len) / f_len
    # 8. SLD连续辅音占比#
    con_list = re.findall(r"[b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z]{2,}", url)
    con_len = [len(con) for con in con_list]
    consec_consonant = sum(con_len) / f_len
    # 9. gib成文检测#
    gib_value = cal_gib(SLD)
    # 10. hmm成文检测#
    hmm_log_prob = cal_hmm_prob(SLD)

    main_domain = '$'+SLD+'$'
    gram2 = [main_domain[i:i + 2] for i in range(len(main_domain) - 1)]
    gram3 = [main_domain[i:i + 3] for i in range(len(main_domain) - 2)]
    gram1_rank = [gram_rank_dict[i] if i in gram_rank_dict else 0 for i in main_domain[1:-1]]
    gram2_rank = [gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in gram2]
    gram3_rank = [gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in gram3]

    # 11. 一元排序均值#
    avg_gram1_rank = np.mean(gram1_rank)
    # 12. 二元排序均值#
    avg_gram2_rank = np.mean(gram2_rank)
    # 13. 三元排序均值#
    avg_gram3_rank = np.mean(gram3_rank)
    # 14. 一元排序标准差#
    std_gram1_rank = np.std(gram1_rank)
    # 15. 二元排序标准差#
    std_gram2_rank = np.std(gram2_rank)
    # 16. 三元排序标准差#
    std_gram3_rank = np.std(gram3_rank)

    feature = [entropy, f_len, ent_flen, vowel_ratio, digit_ratio, repeat_letter, consec_digit, consec_consonant,
               gib_value, hmm_log_prob, avg_gram1_rank, avg_gram2_rank, avg_gram3_rank, std_gram1_rank, std_gram2_rank,
               std_gram3_rank, has_private_tld]

    return feature


def feature_extraction(df, method):
    """
    特征提取, 归一化
    :param df:
    :return:
    """
    if method == "RF":
        col = RF_col
        fea_list = list()
        for ind in df.index:
            fea = df.loc[ind].tolist()
            if ind % 1000 == 0:
                print("{}...".format(ind))
            fea.extend(RF_get_feature(df.at[ind, 0]))
            fea_list.append(fea)
        fea_df = pd.DataFrame(fea_list, columns=col)
    elif method == "SVM":
        col = SVM_col
        fea_list = list()
        for ind in df.index:
            fea = df.loc[ind].tolist()
            if ind % 1000 == 0:
                print("{}...".format(ind))
            fea.extend(SVM_get_feature(df.at[ind, 0]))
            fea_list.append(fea)
        fea_df = pd.DataFrame(fea_list, columns=col)

    return fea_df


def dataset_generation(method):
    """
    数据集集划分,
    :return:
    """

    bk_df = pd.read_csv(black_file_add, header=None)
    bk_df = bk_df.sample(n=660471, axis=0, random_state=23)
    wh_df = pd.read_csv(white_file_add, header=None)

    df = bk_df.append(wh_df, ignore_index=True)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[1], random_state=23)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("__________Generating Train Set__________")
    train_feature = feature_extraction(df_train, method)
    train_feature.to_csv(r"{}\{}_raw_train_features.csv".format(feature_dir, method), index=None)
    train_feature = train_feature.set_index(['domain_name', 'label'])
    standardScaler = StandardScaler()
    standardScaler.fit(train_feature.values)

    print("__________Generating Test Set__________")
    test_feature = feature_extraction(df_test, method)
    test_feature.to_csv(r"{}\{}_raw_test_features.csv".format(feature_dir, method), index=None)
    test_feature = test_feature.set_index(['domain_name', 'label'])

    train_feature = pd.DataFrame(standardScaler.transform(train_feature), index=train_feature.index,
                                 columns=train_feature.columns)
    train_feature = train_feature.reset_index()
    train_feature.to_csv(r"{}\{}_train_features.csv".format(feature_dir, method), index=None)
    test_feature = pd.DataFrame(standardScaler.transform(test_feature), index=test_feature.index,
                                columns=test_feature.columns)
    test_feature = test_feature.reset_index()
    test_feature.to_csv(r"{}\{}_test_features.csv".format(feature_dir, method), index=None)
    pickle.dump(standardScaler, open(r"{}\{}_standardscalar.pkl".format(model_dir, method), 'wb'))
    return

def unbalance_feature_exaction(method):
    bk_df = pd.read_csv(black_file_add, header=None)
    wh_df = pd.read_csv(white_file_add, header=None)

    # RF features
    if method == "RF":
        fea_list_bk = list()
        for ind in bk_df.index:
            fea = bk_df.loc[ind].tolist()
            if ind % 1000 == 0:
                print("{}_Black: {}...".format(method, ind))
            fea.extend(RF_get_feature(bk_df.at[ind, 0]))
            fea_list_bk.append(fea)
        fea_df_bk = pd.DataFrame(fea_list_bk, columns=RF_col)
        fea_df_bk.to_csv(r"{}\{}_RawFeatures_Black.csv".format(feature_dir, method))

        fea_list_wh = list()
        for ind in wh_df.index:
            fea = wh_df.loc[ind].tolist()
            if ind % 1000 == 0:
                print("{}_White: {}...".format(method, ind))
            fea.extend(RF_get_feature(bk_df.at[ind, 0]))
            fea_list_wh.append(fea)
        fea_df_wh = pd.DataFrame(fea_list_wh, columns=RF_col)
        fea_df_wh.to_csv(r"{}\{}_RawFeatures_White.csv".format(feature_dir, method))

    # SVM features
    elif method == "SVM":
        fea_list_bk = list()
        for ind in bk_df.index:
            fea = bk_df.loc[ind].tolist()
            if ind % 1000 == 0:
                print("{}_Black: {}...".format(method, ind))
            fea.extend(SVM_get_feature(bk_df.at[ind, 0]))
            fea_list_bk.append(fea)
        fea_df_bk = pd.DataFrame(fea_list_bk, columns=SVM_col)
        fea_df_bk.to_csv(r"{}\{}_RawFeatures_Black.csv".format(feature_dir, method))

        fea_list_wh = list()
        for ind in wh_df.index:
            fea = wh_df.loc[ind].tolist()
            if ind % 1000 == 0:
                print("{}_White: {}...".format(method, ind))
            fea.extend(SVM_get_feature(bk_df.at[ind, 0]))
            fea_list_wh.append(fea)
        fea_df_wh = pd.DataFrame(fea_list_wh, columns=SVM_col)
        fea_df_wh.to_csv(r"{}\{}_RawFeatures_White.csv".format(feature_dir, method))




if __name__ == "__main__":

    # dataset_generation("SVM")
    # unbalance_feature_exaction("SVM")

    print(get_name('webmail.mofcom.gov.cn.accountverify.validation8u2904.jsbchkufd546.nxjkgdgfhh345s.fghese4.ncdjkbfkjh244e.nckjdbcj86hty1.cdjcksdcuh57hgy43.njkd75894t5.njfg87543.kdjsdkj7564.jdchjsdy'))