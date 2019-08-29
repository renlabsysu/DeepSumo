# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li
# @Email   : 595438103@qq.com
# @File    : OnehotSumoylation.py
# @Software: PyCharm
__author__ = 'pk'
import numpy as np
import re
from multiprocessing.dummy import Pool as ThreadPool
map1={'G':'0','A':'1','L':'2','I':'3','V':'4','P':'5','F':'6','M':'7','W':'8','S':'9','Q':'10','T':'11','C':'12','N':'13','Y':'14','D':'15','E':'16','K':'17','R':'18','H':'19','O':'20'}
def multiple_replace(text,adict):
    rx = re.compile('|'.join(map(re.escape,adict)))
    # print("rx")
    # print(rx)
    def one_xlat(match):
        # print("match")
        # print(match)
        # print("adict[match.group(0)]")
        # print(adict[match.group(0)])
        return adict[match.group(0)]
    return rx.sub(one_xlat,text) #每遇到一次匹配就会调用回调函数
    #把key做成了 |分割的内容，也就是正则表达式的OR

def line_onehot(line,feature_num):
    line= list(line.strip())
    onehot_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    line_onehot_list=[]
    for aa in line:
        if aa in map1:
            index1=int(map1[aa])
            onehot_list[index1] = 1
            line_onehot_list+= onehot_list
            onehot_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            line_onehot_list += onehot_list
    return np.array(line_onehot_list)

def onehot_x(aa_line,length):
    feature_num=length*2
    # # feature_num=11
    # a0=list(feature_num*"0")
    # a1=list(feature_num*"1")
    # a2=list(feature_num*"2")
    # a3=list(feature_num*"3")
    # a4=list(feature_num*"4")
    # a5=list(feature_num*"5")
    # a6=list(feature_num*"6")
    # a7=list(feature_num*"7")
    # a8=list(feature_num*"8")
    # a9=list(feature_num*"9")
    # a10=(feature_num*"10,").strip(",").split(",")
    # a11=(feature_num*"11,").strip(",").split(",")
    # a12=(feature_num*"12,").strip(",").split(",")
    # a13=(feature_num*"13,").strip(",").split(",")
    # a14=(feature_num*"14,").strip(",").split(",")
    # a15=(feature_num*"15,").strip(",").split(",")
    # a16=(feature_num*"16,").strip(",").split(",")
    # a17=(feature_num*"17,").strip(",").split(",")
    # a18=(feature_num*"18,").strip(",").split(",")
    # a19=(feature_num*"19,").strip(",").split(",")
    # a20=(feature_num*"20,").strip(",").split(",")
    # # print(a1)
    # print(len(a1))
    # print(len(a19))
    # enc = OneHotEncoder()
    # enc.fit([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20])
    # col=len(aa_line)
    data_list=[]
    pool =  ThreadPool(40)
    # pool = ThreadPool(10)
    for line  in aa_line:
        line=line[30-int(length):30]+line[31:30+1+int(length)]
        # print(len(line))
        data_list.append(pool.apply_async(line_onehot,args=(line,feature_num)))
    results=[]
    for result in data_list:
        results.append(result.get())

    pool.close()
    pool.join()
    return np.array(results)