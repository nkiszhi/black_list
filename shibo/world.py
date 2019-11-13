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
countx=[]
county=[]
count=1
for csv in filecsv_list:
    data=pd.read_csv(csv, header=None, names=['ip', 'info','source'],sep=None, encoding='gb18030',engine='python')
    countx.append(len(data))
    county.append(count)
    count+=1
    #datas=data.groupby("info").size()

    data.drop(labels=["ip","info","source"], axis=1,inplace=True)
#datas.drop(labels=[0, 1], axis=1)
plt.barh(range(len(countx)), countx, height=0.7, color='steelblue', alpha=0.8)
plt.yticks(range(len(county)),county)
plt.xlim(min(countx)-10000,max(countx))

for x, y in enumerate(countx):
    plt.text(y + 0.2, x - 0.1, '%s' % y)
plt.show()
