#coding=utf-8
'''
作者：孙天琦
函数名：get_inc_idu(start_date_str,end_date_str)
函数功能：输入任意end可以获得这期间相邻两天内增加的ip中ip地址，domain,url的数量，进一步画出饼图
返回的是一个列表
'''
import pandas as pd
import numpy as np
import re
import os

inc_idu=[]
path = '../data_final'
data_list = os.listdir(path)

ip_pattern=re.compile(r'^((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}$')
domain_pattern=re.compile(r'^(?=^.{3,255}$)[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+$')

    
def get_change(i):
        start_date_str=data_list[i]
	end_date_str=data_list[i+1]
	first = pd.read_csv('../data_final/' + str(start_date_str))
        second = pd.read_csv('../data_final/' + str(end_date_str))
        again = pd.merge(first,second,on=['ip','info','reference'])
        increase = second.append(again)
        increase = second.append(again)
        increase.drop_duplicates(keep=False,inplace=True)
        ip_columns=increase['ip'].tolist()
	num=len(ip_columns)
        ip_num=0
        domain_num=0
        url_num=0
        for i in ip_columns:
            if ip_pattern.search(i):                #如果与ip的正则表达式匹配
                ip_num+=1
            elif domain_pattern.search(i):
                domain_num+=1
	    else:
		url_num+=1
        everyday_frame=pd.DataFrame({'number':[ip_num,domain_num,url_num],'date':[start_date_str[:-4]]},index=['ip','domain','url'])
	inc_idu.append(everyday_frame)


def get():
    i = len(data_list)
    index = list(range(i-1))
    map(get_change,index)
    return inc_idu

print(get())
