# coding=utf-8

'''
负责对每天Blacklist数量的统计,完成README中的第一个工作
'''
import general_dataframe
import datetime
import pandas as pd
from datetime import datetime, timedelta

start_date_str = "2019-10-22"                    #这里更改统计的起始日期
end_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')[:10]
print('start date: ' + start_date_str)
print('end date: ' + end_date_str)

date_list = []
date_num =[]
date_str = start_date_str

def date_add(): 
    global date_str
    temp = datetime.strptime(date_str, '%Y-%m-%d')
    temp += timedelta(days=1)
    date_str = datetime.strftime(temp, '%Y-%m-%d')
while (date_str != end_date_str):
    date_list.append(date_str)
    date_add()
date_list.append(end_date_str)

def write(date_str):
    if date_str != None:
        a = pd.read_csv("data\\"+ str(date_str) + ".csv")    # 收集好的blacklist的csv文件存放的路径
        date_num.append(len(a))

map(write, date_list)
df=pd.DataFrame({"date":date_list,"number":date_num})
df.to_csv("blacklist_number.csv",index=None)


















