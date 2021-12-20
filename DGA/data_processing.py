# -*- coding: utf-8 -*-
"""
Created on 2020/8/12 17:34

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pandas as pd
import json
import tld

col = ['aa', 'dev_id', 'dst_type', 'ad', 'atypes', 'cd', 'dst_branch_id', 'nscnt', 'id', 'rd',
         'src_port', 'ts', 'src_ip', 'ra', 'src_type', 'type', 'tc', 'qtypes', 'aclasses', 'qdcnt',
         'ttls', 'answers', 'domain_type', 'ancnt', 'qclasses', 'dst_group', 'afver', 'qr', 'record_time',
         'rcode', 'arcnt', 'src_group', 'length', 'opcode', 'dst_port', 'queries', 'dst_ip', 'z',
         'src_branch_id']


def data_processing(log_add, begin_date, end_date, to_add):
    """
    处理空管局网络日志
    :param log_add: 日志存放地址
    :param begin_date: 起始时间
    :param end_date: 结束时间
    :param to_add: 处理后日志存放地址
    :return:
    """
    # begin = "2020-02-14"
    # end = "2020-02-17"
    date_range = pd.date_range(begin_date, end_date, freq='1D')
    for day in date_range:
        date = str(day.date())
        # file_add = r"M:\huadong_log_data\log\ngfw.dnsflow\ngfw.dnsflow-{}.json".format(date)
        file_add = r"{}/ngfw.dnsflow-{}.json".format(log_add, date)
        response_list = list()
        success = 0
        with open(file_add, 'r', encoding='utf-8') as f:
            i = 0
            while True:
                i += 1
                if i % 10000 == 0:
                    print("正在处理第{}行数据...".format(i))
                try:
                    line = f.readline()
                    js = json.loads(line)
                    res_dic = js["_source"]
                    if res_dic["qr"] == "0":
                        continue
                    else:
                        response_list.append(res_dic.values())
                        if res_dic["rcode"] == "0":
                            success += 1
                except Exception as e:
                    print(e)
                    print("————分析{}数据完成————".format(date))
                    break

            print("success:", success)
            df = pd.DataFrame(response_list, columns=col)
            # df.to_csv(r"M:\mh_data\dns\dns-response-flow-{}.csv".format(date))
            df.to_csv(r"{}/dns-response-flow-{}.csv".format(to_add, date))
            del(df)


def data_washing(flag, dirty_add, clean_add):
    """
    数据清洗
    :return:
    """
    white_list = list()
    with open(dirty_add, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            try:
                url = line.replace("\n", "").split(",")[1-flag]
                dn_list = url.split('.')
                max_len = max([len(dn) for dn in dn_list])
                tld_len = len(str(tld.get_tld(url, as_object=True, fix_protocol=True)))
                short_url = url.replace('.', '').replace('-', '')
                if "xn--" in url or "_" in url or short_url.isdigit():
                    continue
                if len(dn_list) > 4 or len(url) > 255 or max_len < 7 or max_len > 64 or 2*tld_len >= max_len:
                    continue
                if max_len/len(short_url) <= 0.7:
                    continue
                url = url.lower()
                url = "{}.{}".format(tld.get_tld(url, as_object=True, fix_protocol=True).domain,
                                     tld.get_tld(url, as_object=True, fix_protocol=True))
                white_list.append([url, flag])
            except Exception as e:
                print(e)
                continue
    df = pd.DataFrame(white_list)
    df = df.drop_duplicates()
    df.to_csv(clean_add, index=False, header=None)


if __name__ == "__main__":
    data_washing(1, r"./data/sample/black_dataset.csv", r"./data/sample/sample_black.csv")