# coding=utf-8
'''
作者:赵梓杰
函数:完成去重操作,减少网络等问题带来的部分时间段数据损失
'''


import pandas as pd
import time
import csv
from datetime import datetime

date_re = ''
date_re = datetime.now().strftime("%Y-%m-%d")
df1 = pd.DataFrame(columns=['ip','info','reference'])
a = pd.read_csv("../data/" + str(date_re) + '.csv')
df1 = df1.append(a)
df1 = df1.drop_duplicates()
df1 = df1.reset_index(drop=True)
df1.to_csv("../data_final/" + str(date_re) + '.csv')
