# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li
# @Email   : 595438103@qq.com
# @File    : PSSMSumoylation.py
# @Software: PyCharm
import numpy as np

PSSM_dict = {}
AA_sort_list = ["G", "A", "L", "I", "V", "P", "F", "M", "W", "S", "Q", "T", "C", "N", "Y", "D", "E", "K", "R", "H", "O"]
# f=open("/root/DeepSUMO/src/libdata/sumoylation_libdata/PSSM_sumoylation_all.txt","r")  # run on server
f = open("libdata/PSSM_sumoylation_all.txt","r")
# f = open("/data/zengyanru/DeepSumo/SourceData/PosAndNegForTrain/SumoDivideJASSA/PSSM_sumoylation_all.txt",
#          "r")  # row 61, col 21

f = f.readlines()

for i in range(len(f)):
    line = f[i]
    line = line.strip("\n").strip("[").strip("]").split(",")
    # print("i")
    # print(line)
    # print(len(line))
    for j in range(len(line)):
        aa_site = line[j]
        if i == 30:
            pass
        else:
            key = str(i) + AA_sort_list[
                j]  # index of an aa in input sequence and the probability of occurance in this site
            PSSM_dict[key] = aa_site  # actually aa site is score!


def PSSM_code(aa_line, length):
    total_code_list = []
    # print( PSSM_dict)
    for line in aa_line:
        line = list(line.strip())
        line = line[30 - int(length):int(length) + 31]  # a clip of seq
        # print(line)
        code_list = []
        for i in range(len(line)):

            if i == int(length):  # didn't take the k into account
                pass
            else:
                if line[i] in AA_sort_list:
                    # if line[i] in AA_sort_list and line[i] != "O":
                    code_list.append(float(PSSM_dict[str(30 - int(length) + i) + line[i]]))
                else:
                    code_list.append(float(0))
                # print(code_list)
        total_code_list.append(code_list)
    return np.array(total_code_list)


# unknown usage
def PSSM_code_randomforest(aa_line, length):
    total_code_list = []
    # print( PSSM_dict)
    for line in aa_line:

        line = list(line.strip())
        line = line[30 - int(length):int(length) + 30]
        # print(line)
        code_list = []
        for i in range(len(line)):
            if i in [3, 15, 20, 22, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36]:
                code_list.append(float(PSSM_dict[str(i) + line[i]]))
        # print(len(code_list))
        # print(code_list)
        total_code_list.append(code_list)

    return np.array(total_code_list)