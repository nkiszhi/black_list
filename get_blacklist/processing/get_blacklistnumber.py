# coding=utf-8

'''
负责对每天Blacklist数量的统计
功能1
'''
#import general_dataframe
import datetime
import pandas as pd
from datetime import datetime, timedelta
import os
data_dir="/home/open/get_blacklist/processing/data/"
date_num =[]
file_list=[]
def file_name(file_dir):    
    for root,dirs,files in os.walk(file_dir):
        for i in files:  
            if os.path.splitext(i)[1] == '.csv':
                file_list.append(os.path.splitext(i)[0])

def write(date_str):
    if date_str != None:
        a = pd.read_csv(data_dir+ str(date_str) + ".csv")    # 收集好的blacklist的csv文件存放的路径
        date_num.append(len(a))


file_name(data_dir)
file_list.sort()
map(write, file_list)
df=pd.DataFrame({"date":file_list,"number":date_num})
df.to_csv("blacklist_number.csv",index=None)


















