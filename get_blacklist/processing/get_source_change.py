import pandas as pd
import os
'''
功能4
'''
path = './data/'
date_list = os.listdir(path)
date_list.sort()
date_old = []

if len(date_old) != 0 and len(date_old) != 1:
    index = date_list.index(date_old[-2] + '.csv')
    date_list = date_list[index + 1:]
dtframe_info = []
dtframe_reference=[]
for i in range(0, len(date_list)):
    try:
        dtframe_info.append(pd.read_csv(path + date_list[i])['info'])
    except KeyError:
        dtframe_info.append(pd.read_csv(path + date_list[i]).iloc[0:, 1:2])
    try:
        dtframe_reference.append(pd.read_csv(path + date_list[i])['reference'])
    except KeyError:
        dtframe_reference.append(pd.read_csv(path + date_list[i]).iloc[0:, 2:3])

    dtframe_info[i].columns=['info']
    dtframe_info[i].drop_duplicates(keep='first',inplace=True)
    dtframe_reference[i].columns=['reference']
    dtframe_reference[i].drop_duplicates(keep='first',inplace=True)
period = pd.DataFrame()

for i in range(1, len(date_list)):
    temp_info = dtframe_info[i - 1].append(dtframe_info[i])
    temp_reference = dtframe_reference[i - 1].append(dtframe_reference[i])
    
    referencefirst=dtframe_reference[i-1].shape[0]
    referencesecond=dtframe_reference[i].shape[0]

    infofirst=dtframe_info[i-1].shape[0]
    infosecond=dtframe_info[i].shape[0]
    
    temp_info.drop_duplicates(keep=False, inplace=True)
    temp_reference.drop_duplicates(keep=False, inplace=True)

    info_common=(infofirst+infosecond-temp_info.shape[0])/2
    inc_info=infosecond-info_common
    
    reference_common=(referencefirst+referencesecond-temp_reference.shape[0])/2
    inc_reference=referencesecond-reference_common

    period = period.append(pd.DataFrame({'date': [date_list[i][:-4]], 'change-info': [inc_info],'change-reference':[inc_reference]}))
period.to_csv('./get_source_change.csv', sep=',', header=True, index=False)






