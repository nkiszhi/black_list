#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

tmp=os.getcwd()
os.chdir(tmp+'/core')



os.system("python get_blacklistnumber.py")
os.system("python get_infonumber.py")
os.system("python get_class_ratio.py")
os.system("python get-period-change.py")
os.system("python get_referencenumber.py")
os.system("python get_source_change.py")
