#! /usr/bin/env python
# coding=utf-8
'''
作者:赵梓杰
函数名:get_blacklist_num()
函数功能:借助general_dataframe()函数中所得到的天数内的总计dataframe,在完成去重操作之后针对date列进行筛选,筛选出每天所对应的blacklist数量
注意:函数中返回的是series类型,如果想要返回dataframe类型,应该调整为return后注释部分借助groupy函数进行操作
'''

import general_dataframe

def get_blacklist_num():
    blacklist_list = general_dataframe.write_to_dataframe()
    blacklist_list.drop_duplicates(keep=False,inplace=True)
    blacklist_series = blacklist_list[u'date'].value_counts()
    #blacklist_series.iloc[[2]]
    return blacklist_series
    #blacklist_list.groupy('date')['ip'].count()
    #return blacklist_list
