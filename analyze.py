#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

tmp=os.getcwd()
os.chdir(tmp+'/core')



os.system("python get_blacklistnumber.py")
print("has done get_blacklist")
os.system("python get_infonumber.py")
print("has done get_infonumber")
os.system("python get_class_ratio.py")
print("has done get_class_ratio")
os.system("python get-period-change.py")
print("has done get_period_change")
os.system("python get_referencenumber.py")
print("has done get_referencenumber")
os.system("python get_source_change.py")
print("has done get_source_change")

