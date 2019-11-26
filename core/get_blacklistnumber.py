#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
负责对每天Blacklist数量的统计
功能1
'''
import general_dataframe
import datetime
import pandas as pd
from datetime import datetime, timedelta
import os
data_dir="../trails/"
date_num =[]
file_list=[]
blacklist= pd.DataFrame(columns=['ip', 'info', 'reference'])
def write(date_str):
    if date_str != None:
        a = pd.read_csv(data_dir+ str(date_str) + ".csv")    # 收集好的blacklist的csv文件存放的路径
        date_num.append(len(a))



file_list=general_dataframe.file_name(data_dir)
file_list.sort()
map(write, file_list)

def get_frame():
    df=pd.DataFrame({"date":file_list,"number":date_num}) 
    df.to_csv("../result/blacklist_number.csv",index=None)
    return df


get_frame()
















