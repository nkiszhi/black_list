#!/usr/bin/python
#!/bin/bash
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

data_re = ''
data_re = datetime.now().strftime("%Y-%m-%d")


def write_to_csv(retval):
    tmp = retval
    for i, (j, k) in tmp.items():
        csv_write.writerow([i, j, k])


# import .py
filenames = sorted(glob.glob(os.path.join(os.getcwd(), "*.py")))
filenames = [_ for _ in filenames if "__init__.py" not in _]
for i in xrange(len(filenames)):
    f = filenames[i]

    module = __import__(os.path.basename(f).split(".py")[0])

    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name == "fetch":
            print "[o] '%s'" % (module.__url__)
            sys.stdout.write("[?] progress: %d/%d (%d%%)\r" % \
                             (i, len(filenames), i * 100 / len(filenames)))
            sys.stdout.flush()

            results = function()

            with open( "/root/get_blacklist/datanew/" + data_re + ".csv", 'a')as ff:
                csv_write = csv.writer(ff)
                write_to_csv(results)
