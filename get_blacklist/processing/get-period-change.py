import pandas as pd
import os
'''
作者：刘旭萌
功能5
'''


path = './data/'
date_list = os.listdir(path)
date_list.sort()
date_old = []

if len(date_old) != 0 and len(date_old) != 1:
    index = date_list.index(date_old[-2] + '.csv')
    date_list = date_list[index + 1:]
dtframe = []
for i in range(0, len(date_list)):
    try:
        dtframe.append(pd.read_csv(path + date_list[i])['ip'])
    except KeyError:
        dtframe.append(pd.read_csv(path + date_list[i]).iloc[0:, 0:1])
    dtframe[i].columns=['ip']
    dtframe[i].drop_duplicates(keep='first',inplace=True)
period = pd.DataFrame()

for i in range(1, len(date_list)):
    temp = dtframe[i - 1].append(dtframe[i])
    firstday=dtframe[i-1].shape[0]
    secondday=dtframe[i].shape[0]
    temp.drop_duplicates(keep=False, inplace=True)
    common=(firstday+secondday-temp.shape[0])/2
    todayinc=secondday-common
    yesterdaydes=firstday-common
    period = period.append(pd.DataFrame({'date': [date_list[i][:-4]], 'decline': [yesterdaydes],'increase':[todayinc]}))
period.to_csv('./get_period_change.csv', sep=',', header=True, index=False)
