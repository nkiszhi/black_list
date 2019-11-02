#coding:gbk
#! /usr/bin/env python
# coding=utf-8
import datetime
import pandas as pd
from datetime import datetime, timedelta


start_date_str = "2019-10-22"               #控制统计的起始时间
end_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')[:10]
print('start date: ' + start_date_str)
print('end date: ' + end_date_str)


date_list = []
date_str = start_date_str
blacklist_frame_general = pd.DataFrame(columns=['ip', 'info', 'reference'])


def date_add(): 
    global date_str
    temp = datetime.strptime(date_str, '%Y-%m-%d')
    temp += timedelta(days=1)
    date_str = datetime.strftime(temp, '%Y-%m-%d')



while (date_str != end_date_str):
    date_lsit = date_list.append(date_str)
    date_add()
date_list.append(end_date_str)


def write(date_str):
    global blacklist_frame_general
    if date_str != None:
        a = pd.read_csv("data\\"+ str(date_str) + ".csv",names=['ip', 'info', 'reference'])     #存放收集好的csv的路径
        a['date'] = date_str
        blacklist_frame_general = blacklist_frame_general.append(a, sort=False)
        #blacklist_frame_general=pd.concat([blacklist_frame_general,a],axis=1)

def write_to_dataframe():
    global date_list
    global blacklist_frame_general
    map(write, date_list)
    print blacklist_frame_general.head()
    return blacklist_frame_general


