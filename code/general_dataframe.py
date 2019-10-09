# coding=utf-8

"""
作者：刘旭萌
函数名：write_to_dataframe()
函数功能：读取从2019-06-27到现在的blacklist，并放入一个dataframe中，
函数返回值为blacklist_frame_general
注：
1.frame输出格式为: ",date,ip,info,reference",第一列为行号,frame为四列
2.每天的csv未去重，总体未去重
"""

import pandas as pd
import os

# 统计起始日期和终止日期
path = '../data_final'
date_list = os.listdir(path)
blacklist_frame_general = pd.DataFrame(columns=['ip', 'info', 'reference','date'])

def write(date_str):
    global blacklist_frame_general
    if date_str != None:
        a = pd.read_csv("../data_final/" + str(date_str))
        date_str = date_str[:-4]
        a['date'] = date_str
        #blacklist_frame_general = blacklist_frame_general.append(a, sort=False)
        blacklist_frame_general = blacklist_frame_general.append(a)


def write_to_dataframe():
    global date_list
    global blacklist_frame_general
    map(write, date_list)
    return blacklist_frame_general

#print(write_to_dataframe())
