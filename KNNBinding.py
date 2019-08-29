# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li and ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : KNNBinding.py
# @Software: PyCharm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import numpy as np

blosum62_matrix = {}
# file=open(r"/root/DeepSUMO/src/libdata/blosum62.txt","r")
file = open(r"libdata/blosum62.txt","r")
# file = open("/data/zengyanru/DeepSumo/PretreatData/Result/blosum62.txt", "r")
file = file.readlines()
header_l = file[0].strip().split("\t")
# print(header_l)
for i in range(len(file) - 1):
    i += 1
    li = file[i].strip().split("\t")
    for j in range(len(header_l)):
        key1 = li[0] + header_l[j]
        blosum62_matrix[key1] = li[j + 1]


def blosum62(_list1, _list2):
    score_def = 0
    for i in range(len(_list1)):
        score_def += int(blosum62_matrix[str(_list1[i]) + str(_list2[i])])
    return score_def


def get_rate(_line, _positive_line, _negative_line, avg2):
    positive_score_list = []
    negative_score_list = []
    if _line + "\n" in _positive_line:
        _positive_line.remove(_line + "\n")
    elif _line + "\n" in _negative_line:
        _negative_line.remove(_line + "\n")
        # print(negative_score_list)
    else:
        pass
    for positive_line in _positive_line:
        positive_line = positive_line.strip()[30 - int(avg2):37 + int(avg2)]
        score = blosum62(_line, positive_line)
        score1 = [score, "pos"]
        positive_score_list.append(score1)
        # if str(line) in positive_scores_dict:
        #     positive_scores_dict[str(_line)].update({'positive_score_list':positive_score_list})
        # else:
        #     positive_scores_dict.update({str(line):{'positive_score_list':positive_score_list}})
    for negative_line in _negative_line:
        negative_line = negative_line.strip()[30 - int(avg2):37 + int(avg2)]
        score = blosum62(_line, negative_line)
        score1 = [score, "neg"]
        negative_score_list.append(score1)
        # if str(line) in positive_scores_dict:
        #     positive_scores_dict[str(line)].update({'negative_score_list':negative_score_list})
        # else:
        #     positive_scores_dict.update({str(line):{'negative_score_list':negative_score_list}})

    # 1.1 topk rate
    positive_rate_list = []
    for i in range(30):  #
        p = (i + 1) / 100  # used to be i/100, but it will lead to zero division if the sample size is less than 100!
        np_sum = int(len(_positive_line) * p)
        if np_sum == 0:
            np_sum = 1
        # np_sum = np.ceil(len(_positive_line) * p)  # in case number of positive seq less than 100

        # print("type(positive_score_list)")
        # print(type(positive_score_list))
        # print(positive_score_list)

        positive_score_list.sort(reverse=True)
        negative_score_list.sort(reverse=True)

        two_scores_list = positive_score_list + negative_score_list
        # print("type(positive_score_list)")
        # print(type(positive_score_list))
        # print(list(positive_score_list))
        # print(type(list(positive_score_list)))
        two_scores_list.sort(reverse=True)
        split_score = two_scores_list[np_sum - 1]
        posorneg = split_score[1]
        # print("np_sum")
        # print(np_sum)
        # print("two_scores_list")
        # print(two_scores_list)
        # print("split_score")
        # print(split_score)
        # 求出positive-rate
        if posorneg == "pos":
            positive_num = positive_score_list.index(split_score)
            # print(" positive_num")
            # print( positive_num)
        else:
            positive_num = np_sum - negative_score_list.index(split_score)
            # print(" positive_num")
            # print( positive_num)
        positive_rate = positive_num / np_sum
        positive_rate_list.append(positive_rate)
    # print("positive_rate_list")
    # print(positive_rate_list)
    # positive_scores_dict.update({str(line):{'positive_rate_list':positive_rate_list}})
    return positive_rate_list


def KNN_code(aa_line, _positive_line, _negative_line, avg2):
    pool = Pool(20)

    data_list = []
    X_list = []
    sample = len(_positive_line)
    for line in aa_line:
        line = line.strip()
        line = line[30 - int(avg2):30 + 7 + int(
            avg2)]  # here 7 and the 7 in function "get_rate" should be changed if we change the matching length
        data_list.append(
            pool.apply_async(get_rate, args=(line, _positive_line, _negative_line[0:sample], avg2)))  # [0:sample]

    for result in data_list:
        # print(result.get())
        X_list.append(result.get())
    pool.close()
    pool.join()
    return np.array(X_list)


def get_score_rate(_line, _positive_line):
    # print(len(_positive_line))
    if _line + "\n" in _positive_line:
        _positive_line.remove(_line + "\n")

    # print(len(_positive_line))
    positive_score_list = []
    for positive_line in _positive_line:
        positive_line = positive_line.strip()
        score = blosum62(_line, positive_line)
        positive_score_list.append(score)
    # 1.1 计算topk score
    positive_rate_list = []
    positive_score_sum = 0
    for i in range(30):  # 改
        p = (i + 1) / 100  # 改
        np_sum = int(len(_positive_line) * p)
        # print(positive_score_list)

        # print("pos")

        positive_score_list.sort(reverse=True)
        # print(positive_score_list)
        for i in range(np_sum):
            positive_score_sum += positive_score_list[i]
        positive_rate = positive_score_sum / np_sum
        positive_rate_list.append(positive_rate)
    # print("positive_rate_list")
    # print(positive_rate_list)
    # positive_scores_dict.update({str(line):{'positive_rate_list':positive_rate_list}})
    return positive_rate_list


def KNN_positive_code(aa_line, _positive_line):
    pool = Pool(30)
    data_list = []
    X_list = []
    sample = len(_positive_line)
    for line in aa_line:
        line = line.strip()
        data_list.append(pool.apply_async(get_score_rate, args=(line, _positive_line)))

    for result in data_list:
        # print(result.get())
        X_list.append(result.get())
    return np.array(X_list)