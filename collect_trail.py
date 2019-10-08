#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import os
import csv
import glob
import inspect
import subprocess
import sys
import time
import urllib
from datetime import *

from core.trailsdict import TrailsDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# to enable calling from current directory too

# trails folders
FEEDS = os.path.abspath("trails/feeds")
CUSTOM = os.path.abspath("trails/custom")
STATIC = os.path.abspath("trails/static")

# trails csv
USERS_DIR = os.getcwd()
TRAILS_FILE =os.path.join(USERS_DIR, "trails.csv")
TRAILS_FOLDER =os.path.join(USERS_DIR, "trails_data")
COLUMNS_NAME = ["trail", "info", "ref"]

def load_trails(trails_file):
    print "[i] loading trails..."
    retval = TrailsDict()
    if os.path.isfile(trails_file):
        try:
            with open(trails_file, "rb") as f:
                reader = csv.reader(f, delimiter=',', quotechar='\"')
                for row in reader:
                    if row and len(row) == 3:
                        trail, info, reference = row
                        retval[trail] = (info, reference)

        except Exception, ex:
            exit("[!] something went wrong during trails file read '%s' ('%s')" % (trails_file, ex))
    
    else:
    	print "[i] Init %s trails file." % trails_file
        trails_file = open(trails_file,"w")
        trails_file.write("trail,info,ref\n")
        trails_file.close()

    _ = len(retval)
    try:
        _ = '{0:,}'.format(_)
    except:
        pass
    print "[i] %s trails loaded" % _
    return retval

def init_sys_path():
    sys.path.append(FEEDS)
    sys.path.append(CUSTOM)
    sys.path.append(STATIC)

def update_trails():
    """
    Update trails from feeds
    """
    #check trails folder
    if not os.path.exists(TRAILS_FOLDER):os.popen("mkdir "+TRAILS_FOLDER)
    
    date_now = datetime.now().strftime('%Y-%m-%d')
    trails_file = os.path.join(TRAILS_FOLDER, date_now)
    print date_now
    trails = TrailsDict()
    trails.update(load_trails(trails_file))       #load trails
    
    list_trails = []
    
    old_csv = pd.read_csv(trails_file)
    print "[i] Collecting latest trails ..."
    filenames = sorted(glob.glob(os.path.join(FEEDS, "*.py")))
    filenames += sorted(glob.glob(os.path.join(STATIC, "*.py"))) # in static folder, __init__.py has fetch() 
    filenames += sorted(glob.glob(os.path.join(CUSTOM, "*.py")))# in custom folder, __init__.py has fetch()
    #remove __init__.py in feeds folder
    filenames = [_ for _ in filenames if "__init__.py" not in _]

    init_sys_path()

    for i in xrange(len(filenames)):
        f = filenames[i]
        try:
            module = __import__(os.path.basename(f).split(".py")[0])
        except (ImportError, SyntaxError), ex:
            print "[x] Failed: import feed file '%s' ('%s')" % (f, ex)
            continue
       
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name == "fetch":
            	try:
                    print "[o] '%s'" % (module.__url__)
                    sys.stdout.write("[?] progress: %d/%d (%d%%)\r" % \
                    (i, len(filenames), i * 100 / len(filenames)))
                    sys.stdout.flush()
                    results = function()
                    for item in results.items():
                        list_trails.append((item[0], item[1][0],item[1][1]))       
                except Exception, ex:
                    print "[x] Failed: process feed file '%s' ('%s')" % (filename, ex)
    if list_trails:
        new_csv = pd.DataFrame(list_trails, columns=COLUMNS_NAME)
        new_csv = pd.concat([new_csv,old_csv])
        new_csv.drop_duplicates(inplace=True)
        new_csv.to_csv(trails_file, index=None)
        
    print "[i] Collecting trails finished!"
    #thread = threading.Timer(3600, update_trails)
    #thread.daemon = True
    #thread.start()
def main():
    try:
        update_trails()
    except KeyboardInterrupt:
        print "\r[x] Ctrl-C pressed"

if __name__ == "__main__":
    main()
