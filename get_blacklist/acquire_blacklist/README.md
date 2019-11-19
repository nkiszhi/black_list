

# acquire_blacklist

### 依赖linux系统中crontab命令完成每天数据集的获取工作



- 实现目标

  借助feeds文件夹中已有的爬虫脚本完成对样本的收集工作，然后将收集到的数据整合成一个DataFrame导入到datanew文件夹的csv文件中，为了减少由于网络等原因引起的数据不等的影响，这里在每天的六个固定时间点（可调节）进行数据的收集工作，并在每次收集之后进行去重操作。


- 代码结构

  - datanew：存储当天获得的以日期命名的csv文件,当晚的11点完成对当天csv收集的最后一次去重工作.(可调整)

  - feeds：每天多次调用get_blacklist.py在一个csv内追加数据.

     每收集一次数据,handle.py对刚追加收集好的的csv数据进行去重处理并存储到datanew文件夹内.
     为了不影响原脚本当中的相对路径问题,调用shell脚本(get_blacklist.sh和handle.sh)进入相应目录去执行get_blacklist.py脚本和handle.py脚本.


- crontab的使用

  crontab -e对crontab内容进行修改

  crontab -l查看当前内容

  每次修改后需要输入

  service cron reload

  service cron start

  编辑的crontab内容如下:
      
  
      0 4,8,12,16,20,23 * * * sh /root/get_blacklist/handle.sh
       
      0 2,6,10,14,18,21 * * * sh /root/get_blacklist/get_blacklist.sh


  以上是个人crontab内容的修改，crontab前五个参数分别是日，月，周，月，年。后面是执行的命令，需要采用绝对路径

