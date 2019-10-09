#coding=utf-8

'''
作者：孙天琦
函数名：get_idu_num()
函数功能：6-27到现在每天的ip,domain,url的数量,从而得到各自的占比
'''
import numpy as np
import pandas as pd
import re
import general_dataframe
import sys
import os

def get_idu_num():
    ip_frame=general_dataframe.write_to_dataframe()[['ip','date']]
    path = '../data_final'
    data_list = os.listdir(path)
    leng = len(data_list)
    idu_num_frame=pd.DataFrame({"number":"","date":""},index=[])  #创建一个空的df
    ip_pattern=re.compile(r'^((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}$')
    domain_pattern=re.compile(r'^(?=^.{3,255}$)[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+$')
    while leng:
        date_ip_frame=ip_frame[ip_frame['date'].isin([data_list[leng-1][:-4]])]         #提取每天的ip_frame
        ip_columns=date_ip_frame['ip'].tolist()
        num=len(ip_columns)
        ip_num=0
        domain_num=0
        for i in ip_columns:
            if ip_pattern.search(i):                #如果与ip的正则表达式匹配
                ip_num+=1
            elif domain_pattern.search(i):
                domain_num+=1
        url_num=num-ip_num-domain_num
        everyday_frame=pd.DataFrame({'number':[ip_num,domain_num,url_num],'date':[data_list[leng-1][:-4]]},index=['ip','domain','url'])
        idu_num_frame=idu_num_frame.append(everyday_frame)
        leng -= 1
    #print(idu_num_frame)
    return idu_num_frame

