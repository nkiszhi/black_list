# coding=utf-8
'''
函数:完成去重操作,减少网络等问题带来的部分时间段数据损失
'''


import pandas as pd
import time
import csv
from datetime import datetime

date_re = ''
date_re = datetime.now().strftime("%Y-%m-%d")
a = pd.read_csv("/root/get_blacklist/datanew/" + str(date_re) + '.csv')
df1 = a.drop_duplicates()

df1.to_csv("/root/get_blacklist/datanew/" + str(date_re) + '.csv',index=None)

