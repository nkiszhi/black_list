#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
3
'''
import pandas as pd
import general_dataframe
def get_info():
    info_list = general_dataframe.write_to_dataframe()
    info_list.drop_duplicates()
    info_series = info_list['info'].value_counts()
    df=pd.DataFrame({'info':info_series.index, 'number':info_series.values})
    df.to_csv("../result/infonumber.csv",header=1,index=0)
    return df

get_info()

