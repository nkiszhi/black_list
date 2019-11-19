#! /usr/bin/env python
# coding=utf-8

import pandas as pd 
import os

'''
作者:赵梓杰
函数:获得相邻两天的info和reference的数据增删变化,以多个dataframe或series类型返回
'''

inc_info = []
inc_ref = []
dec_info = []
dec_ref = []
path = '../data_final'
date_list = os.listdir(path)


def get_change_data(i):
    first = pd.read_csv("../data_final/" + str(date_list[i]))
    second = pd.read_csv("../data_final/" + str(date_list[i+1]))
    again = pd.merge(first,second,on=['ip','info','reference'])
    a = first.append(again)
    a = first.append(again)
    a.drop_duplicates(keep=False,inplace=True)
    decrease_info = a[u'info'].value_counts()
    decrease_ref = a[u'reference'].value_counts()
    b = second.append(again)
    b = second.append(again)
    b.drop_duplicates(keep=False,inplace=True)
    increase_info = b[u'info'].value_counts()
    increase_ref = b[u'reference'].value_counts()
    #return increase_info,increase_ref,decrease_info,decrease_ref
    inc_info.append(increase_info)
    inc_ref.append(increase_ref)
    dec_info.append(decrease_info)
    dec_ref.append(decrease_ref)

def get():
    i = len(date_list)
    index = list(range(i-1))
    map(get_change_data,index)
    return inc_info,inc_ref,dec_info,dec_ref

