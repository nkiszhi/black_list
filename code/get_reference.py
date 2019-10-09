#! /usr/bin/env python
# coding=utf-8

import general_dataframe
'''
作者:赵梓杰
函数:依赖general_dataframe完成多天reference的获取,并且存贮到ref_series
注释掉的是将其以dataframe输出
'''


def get_reference():
    ref_list = general_dataframe.write_to_dataframe()
    ref_list.drop_duplicates(keep=False,inplace=True)
    ref_series = ref_list[u'reference'].value_counts()
    return ref_series
    #ref_list.groupy('reference')['ip'].count()
    #return ref_list
