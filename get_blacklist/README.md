### 依赖linux系统中crontab命令完成每天数据集的获取工作

###### Buchiyexiao

- 实现目标

  借助已有的爬虫脚本完成对样本的收集工作，然后将收集到的数据整合成一个DataFrame导入一个csv文件中，为了减少由于网络等原因引起的数据不等的影响，这里在每天的三个固定时间点（可调节）进行数据的收集工作，并导入到同一个csv文件。

  在每天20:00调用一个脚本完成对当天获得的csv文件进行去重操作，以减少数据不等的影响。

- 代码结构

  - data：存储当天获得的不去重数据csv

  - data_final：存储当天去重处理后的数据csv

  - feeds：每天多次调用get_blacklist.py在一个csv内追加数据

     			 每天调用一次handle.py对不去重的csv数据进行处理并存储到data_final文件夹内

- crontab的使用

  crontab -e对crontab内容进行修改

  crontab -l查看当前内容

  每次修改后需要输入

  service cron reload

  service cron start

  ![1570616316656](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1570616316656.png)

  以上是个人crontab内容的修改，crontab前五个参数分别是日，月，周，月，年。后面是执行的命令，需要采用绝对路径