#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
6、7
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import general_dataframe
import sys
import os

data_dir="../trails/"
file_list=[]
file_list=general_dataframe.file_name(data_dir)
file_list.sort()
def get_idu_num():
    
    ip_frame=general_dataframe.write_to_dataframe()[['ip','date']]
    print(ip_frame.tail()) 
    idu_num_frame=pd.DataFrame({"date":"","ip":"","domain":"","url":""},index=[])
    ip_pattern=re.compile(r'^((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}$')
    domain_pattern=re.compile(r'^(?=^.{3,255}$)[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+$')
    iii=d=u=0
    for i in file_list:
        date_ip_frame=ip_frame[ip_frame['date'].isin([i])]
        ip_columns=date_ip_frame['ip'].tolist()
        num=len(ip_columns)
        ip_num=domain_num=0
        for ii in ip_columns:
            if ip_pattern.search(ii):
                ip_num+=1
            elif domain_pattern.search(ii):
                domain_num+=1
        url_num=num-ip_num-domain_num
        iii+=ip_num
        d+=domain_num
        u+=url_num
        everyday_frame=pd.DataFrame({'date':[i],'ip':[ip_num],'domain':[domain_num],'url':[url_num]})
        idu_num_frame=idu_num_frame.append(everyday_frame)
    total_frame=pd.DataFrame({'class':['ip','url','domain'],'number':[iii,u,d]})
    idu_num_frame.to_csv("../result/class_ratio.csv",index=None)
    total_frame.to_csv("../result/total_ratio.csv",index=None)
    return idu_num_frame,total_frame

get_idu_num()

