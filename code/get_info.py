#! /usr/bin/env python
# coding=utf-8
'''
作者:赵梓杰
函数:借助general_dataframe获取总共的info信息,以series格式进行输出
注释是以dataframe格式进行输出
'''
import general_dataframe

def get_info():
    info_list = general_dataframe.write_to_dataframe()
    info_list.drop_duplicates(keep=False,inplace=True)
    info_series = info_list[u'info'].value_counts()
    return info_series
    #info_list.groupy('info')['ip'].count()
    #return info_list
