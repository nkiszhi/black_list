#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
负责对这段日期reference数量的统计
功能2
'''
import pandas as pd
import general_dataframe

def get_reference():
    info_list = general_dataframe.write_to_dataframe()
    info_list.drop_duplicates()
    info_series = info_list['reference'].value_counts()
    df=pd.DataFrame({'reference':info_series.index, 'number':info_series.values})
    df.to_csv("./data/referencenum.csv",header=1,index=0)
    return df

get_reference()