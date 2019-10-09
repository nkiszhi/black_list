#! /usr/bin/env python
import pandas as pd 
import os

inc = []
dec = []
path = '/home/data_final'
data_list = os.listdir(path)

def get_change(i): 
    global inc
    global dec
    first = pd.read_csv("/home/data_final/" + str(data_list[i]))
    second = pd.read_csv("/home/data_final/" + str(data_list[i+1]))
    again = pd.merge(first,second,on=['ip','info','reference'])
    decrease = first.append(again)
    decrease = first.append(again)
    decrease.drop_duplicates(keep=False,inplace=True)
    increase = second.append(again)
    increase = second.append(again)
    increase.drop_duplicates(keep=False,inplace=True)
    inc.append(increase)
    dec.append(decrease)
    #return increase , decrease

def get():
    global date_list
    global inc
    global dec
    i = len(data_list)
    index = list(range(i-1))
    map(get_change,index)
    return [inc,dec]

print(get())
