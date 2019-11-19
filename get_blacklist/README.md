# 收集与处理blacklist

- #### 实现目标：

借助已有项目MalTrail的爬虫脚本对blacklist进行收集，使用python的pandas库对收集到的数据进行处理并导入csv，借助superset将数据以柱状图、饼图等直观的展现出来。最终实现 数据收集—数据处理—数据展示完全自动化的目标。

- #### 结构介绍：

acquire_blacklist：借助已有的爬虫脚本完成对样本的收集工作，并将收集数据整合为DataFrame导入一个csv文件中。

processing:对收集到的样本进行数据处理，处理过的csv存放在该目录下的data文件夹里（文件夹会自动创建）

datanew：存储当天获得的以日期命名的csv文件,当晚的11点完成对当天csv收集的最后一次去重工作.(可调整) (文件夹会自动创建)







