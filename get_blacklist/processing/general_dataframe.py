#! /usr/bin/env python
# coding=utf-8
'''
函数库
'''
import datetime
import pandas as pd
from datetime import datetime, timedelta
import os

data_dir="/home/open/get_blacklist/processing/data/"
file_list=[]
blacklist_frame_general = pd.DataFrame(columns=['ip', 'info', 'reference'])


def file_name(file_dir):
    for root,dirs,files in os.walk(file_dir):
        for i in files:  
            if os.path.splitext(i)[1] == '.csv':
                file_list.append(os.path.splitext(i)[0])
    return file_list

def main():
    file_name(data_dir)
    file_list.sort()
    blacklist_frame_general = pd.DataFrame(columns=['ip', 'info', 'reference'])


def write(date_str):
    global blacklist_frame_general
    if date_str != None:
        a = pd.read_csv(data_dir+ str(date_str) + ".csv",names=['ip', 'info', 'reference'])    
        a['date'] = date_str
        blacklist_frame_general = blacklist_frame_general.append(a)
        #blacklist_frame_general=pd.concat([blacklist_frame_general,a],axis=1)
       

def write_to_dataframe():
    global file_list
    global blacklist_frame_general    
    map(write, file_name(data_dir))
    return blacklist_frame_general


if __name__ == "__main__":
    main()

