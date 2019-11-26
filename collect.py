#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import glob
import inspect
import os
import sqlite3
import subprocess
import sys
import time
import urllib
import csv
from datetime import datetime


date_re = ''
date_re = datetime.now().strftime("%Y-%m-%d")


def write_to_csv(retval):
    tmp = retval
    for i, (j, k) in tmp.items():
        csv_write.writerow([i, j, k])


# import .py
filenames = sorted(glob.glob(os.path.join(os.getcwd()+"/feeds/", "*.py")))
filenames = [_ for _ in filenames if "__init__.py" not in _]
for i in xrange(len(filenames)):
    f = filenames[i]
    d=os.path.basename(f).split(".py")[0]
    
    module = __import__('feeds.'+d,fromlist = (d,))

     
    for name, function in inspect.getmembers(module, inspect.isfunction):
        print name
        if name == "fetch":
            print "[o] '%s'" % (module.__url__)
            sys.stdout.write("[?] progress: %d/%d (%d%%)\r" % \
                             (i, len(filenames), i * 100 / len(filenames)))
            sys.stdout.flush()
            results = function()
            if not os.path.exists("./trails/"):
                os.makedirs("./trails/")

            with open( "./trails/" + date_re + ".csv", 'a') as ff:
                csv_write = csv.writer(ff)
                write_to_csv(results)


#Deduplication
a = pd.read_csv("./trails/" + str(date_re) + '.csv')
df1 = a.drop_duplicates()
df1.to_csv("./trails/" + str(date_re) + '.csv',index=None)


