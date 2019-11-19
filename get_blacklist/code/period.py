# coding=utf-8
""""
作者：刘旭萌
函数名 general_dataframe()
函数功能：返回为一个dataframe，内容为周期为特定天数的攻击来源的个数
"""

import general_dataframe


def get_period():
    # 取出blacklist_frame_general中的ip和date列写入time_frame中
    time_frame = general_dataframe.write_to_dataframe()[['ip', 'date']]
    # 去除一天中重复的ip
    time_frame.drop_duplicates(inplace=True)

    time_frame['period'] = 1
    # time_frame未每个攻击来源出现的天数
    times = time_frame.groupby(['ip']).sum()
    times = times.sort_values(by='period')
    # 统计出现特定天数的攻击来源个数
    time_frame_2 = times[['period']]
    time_frame_2['pieces'] = 1
    time = time_frame_2.groupby(['period']).sum()
    time = time.sort_values(by='pieces')
    return time


# print get_period()
