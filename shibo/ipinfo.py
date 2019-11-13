import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

os.chdir('/home/sameen/maltrail/new')
file_chdir = os.getcwd()


filecsv_list = []
for root, dirs, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
            filecsv_list.append(file)

data = pd.DataFrame()
datas=pd.DataFrame()
data=pd.read_csv(filecsv_list[0],names=["ip","info","reference"])
del filecsv_list[0]
label_list =[]
size=[]
count=0
for csv in filecsv_list:
    data.append(pd.read_csv(csv,names=["ip","info","reference"]),ignore_index=True)
    data.drop_duplicates()

datas=data.groupby("info").size()
datas.columns=["info","number"]
print(datas)
for row in datas.iteritems():

    if row[1]>10000:
        label_list.append(row[0])
        size.append(row[1])

plt.figure(figsize=(6, 6))

plt.pie(size, labels=label_list, autopct='%1.1f%%')

plt.show()

