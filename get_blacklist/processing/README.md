介绍

我们的项目是基于已成熟项目恶意流量监测系统maltrail中的爬取网页内容的方法，获取到maltrail采集到的网络中各个开源黑样本(包括IP、domain、URL)，利用可视化平台superset展示出我们希望展现并且有意义的数据变化。该项目分为情报收集，情报处理，情报展示三个部分，这部分是情报处理部分。


情报处理如下：


 1 每天blacklist的数量
 
 --x轴：时间
 
 --y轴：数量


2 总共reference的种类个数的占比

输入：reference的种类和对应个数


3 总共info的种类个数占比

输入：info的种类和对应个数


4 新增的reference和info种类个数占比

输入：新增reference和info的种类和个数


5 相邻两天的数据变化


--x轴：时间

--y1：相对于前一天新增ip的数量

--y2：相对于前一天减少ip的数量


6 总共的ip、domain、url的占比


--x轴：时间

--y轴：ip、domain、url的个数


7 新增内容中ip、domain和url的占比

输入：ip,domain,url的个数


8 出现周期占比

输入：特定出现周期的个数



