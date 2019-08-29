# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li and ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : PSSMBinding.py
# @Software: PyCharm
import numpy as np

PSSM_dict = {}
AA_sort_list = ["G", "A", "L", "I", "V", "P", "F", "M", "W", "S", "Q", "T", "C", "N", "Y", "D", "E", "K", "R", "H", "O"]
# f=open("/data/chenli/sumo_bing/train_data/PSSM_matrix_sumobing.txt","r")

f=open("libdata/PSSM_sumobinding_all.txt","r")
# f = open("/data/zengyanru/DeepSumo/SourceData/PosAndNegForTrain/BindingDivideJASSA/PSSM_sumobinding_all.txt", "r")

f = f.readlines()

for i in range(len(f)):
    line = f[i]
    line = line.strip("\n").strip("[").strip("]").split(",")
    # print("i")
    # print(line)
    # print(len(line))
    for j in range(len(line)):
        aa_site = line[j]
        # if i==30:
        #     pass
        # else:
        key = str(i) + AA_sort_list[j]

        PSSM_dict[key] = aa_site


def PSSM_code(aa_line, length):
    total_code_list = []
    # print( PSSM_dict)
    for line in aa_line:
        line = list(line.strip())
        line = line[30 - int(length):int(length) + 30 + 5]
        # print(line)
        code_list = []
        for i in range(len(line)):
            if line[i] in AA_sort_list:
                code_list.append(float(PSSM_dict[str(30 - int(length) + i) + line[i]]))
            else:
                code_list.append(float(0))
                # print(code_list)
        total_code_list.append(code_list)
    return np.array(total_code_list)