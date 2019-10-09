#! /usr/bin/env python
# coding=utf-8

"""
作者：刘旭萌
函数名：get_period_change()
函数功能：分析周期为一天的恶意来源变化趋势
函数返回值为一个dataframe，第一列为date，第二列为当日出现且第二天消失的恶意来源个数
"""
from get_adjacent_change import *
import pandas as pd
import os

num_list = []
list_all = get()
in_list = list_all[0]
de_list = list_all[1]
path = '../data_final'
date_list = os.listdir(path)[:-1]
date_num = list(range(0, len(date_list)))


def change(i):
    global num_list
    frame = in_list[i].append(de_list[i + 1])
    frame1 = frame.drop_duplicates(keep='first')
    num_list.append((frame.shape[0] - frame1.shape[0]) / 2)


def get_period_change():
    global num_list
    map(change, date_num)
    period_change_frame = pd.DataFrame({'Date': date_list, 'pieces': num_list})
    return period_change_frame
