# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li and ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : SumoylationPredict.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from PSSMSumoylation import PSSM_code
from OnehotSumoylation import onehot_x
from KNNSumoylation import KNN_code

import SumoylationModel

def read_fasta(file):
    # file is a fasta file. This function returns a dictionary
    fasta = {}
    seq_name = ''
    sequence = ''
    f = open(file,'r')
    for line in f:
        line = line.strip("\n")
        if not line.strip():
            continue
        if line.startswith(">"):
            seq_name = line[1:] #TODO:the index name should be changed here
            seqlist = []
        else:
            sequence = line
            sequence = sequence.replace("*","O")
            seqlist.append(sequence)
        if seq_name not in fasta:
            fasta[seq_name] = []
            continue
        fasta[seq_name] = "".join(seqlist)
    return fasta


def PreEncodeData(datafile,intfmt="txt",mode="Train"):
    AA_sort_list = ["G", "A", "L", "I", "V", "P", "F", "M", "W", "S", "Q", "T", "C", "N", "Y", "D", "E", "K", "R", "H","O"]
    # fmt can be fasta or txt
    # return a matrix with three kinds of coding method in it
    # onehot, knn and pssm encoded in colrange 27, 11 and 26 respectively
    # pos_train = open("/data/zengyanru/DeepSumo/SourceData/PosAndNegForTrain/cdhit_protein_level/Sumoylation_TrainSet_cdhit_pos.txt")
    # neg_train = open("/data/zengyanru/DeepSumo/SourceData/PosAndNegForTrain/cdhit_protein_level/Sumoylation_TrainSet_cdhit_neg_no_pos_shuf.txt")
    pos_train = open(
        "libdata/SumoylationAllCurrentPos.txt")
    neg_train = open(
        "libdata/SumoylationAllCurrentNeg.txt")
    pos_train = pos_train.readlines()
    neg_train = neg_train.readlines()
    if intfmt == "fasta":
        id_fa_dict = read_fasta(datafile)
    elif intfmt == "txt":
        id_fa_dict = {}
        fa_list = [i.strip("\n") for i in open(datafile,"r").readlines()]
        rg = len(fa_list)  # range
        for i in range(rg):
            id_fa_dict[str(i+1)] = fa_list[i]  # if the input is not in fasta format
    if mode=="Train":  # this will only output those k within index 31
        id_fa_list = []
        for i in id_fa_dict:
            id_fa_list.append([i,id_fa_dict[i]])
    else:  # output all k site because we just need to predict
        id_fa_list = []
        for key_id in id_fa_dict:
            fa = 30 * "O" + id_fa_dict[key_id] + (30) * "O"
            for i in range(len(fa) - 30):
                i = i + 30
                site_fa = fa.strip()[i]
                if site_fa == "K" or site_fa == "k":
                    site_fa_seq = fa.strip()[i - 30:i + 30 + 1]
                    site_fa_seq = site_fa_seq.upper()
                    id_fa_list.append([key_id + ":" + str(i - 30 + 1), site_fa_seq])
    id_fa_list = np.array(id_fa_list)
    all_id = np.array(id_fa_list[:, 0])
    all_aa_fa = id_fa_list[:, 1]
    onehot_length = 7
    predict_set0 = onehot_x(all_aa_fa, onehot_length)
    predict_set1 = KNN_code(all_aa_fa, pos_train, neg_train, onehot_length)
    predict_set2 = PSSM_code(all_aa_fa, onehot_length)
    predict_set01 = predict_set0.reshape([-1,onehot_length*2,21]).astype(float)
    seq_idx = 0
    for seq in predict_set2:  # combine onehot and PSSM
        aa_idx = 0
        for aa_site in seq:
            try:
                rep_site = int(np.where(predict_set01[seq_idx,aa_idx,:]==1)[0])
                predict_set01[seq_idx,aa_idx,rep_site] = aa_site
                aa_idx += 1
            except:
                pass
        seq_idx += 1
    predict_set01 = np.around(predict_set01.reshape(-1,21*2*onehot_length),decimals=4)
    predict_set = np.concatenate((predict_set01, predict_set1), axis=1)
    return predict_set,all_id,all_aa_fa


def sumoylation_predict(cut_off,input_file,result_dir):
    cut_off = cut_off.lower()
    # cut_off_list = {"low": 0.344, "medium": 0.466, "high": 0.6, "all": 0}  # TODO: need to change!
    cut_off_list = {'low':0.046250,'medium':0.064375,'high':0.113125,'all':0}
    input_data,all_id,all_aa_fa = PreEncodeData(input_file,intfmt="fasta",mode="test")
    m_dir = "libdata/SumoylationModel/"  # model directory

    cnn = SumoylationModel.CNN_SUMO(
        model_path=m_dir,
        model_name="model", keep_prob=1, train_length=7)
    pred = cnn.predict(other_data=input_data,
                       model_path=m_dir,
                       modelname="model")
    predict_score = [i[0] for i in pred]
    # construct results
    result = ""
    for i in range(len(all_id)):
        if float(predict_score[i]) >= float(cut_off_list[cut_off]):
            write_str_seq = all_aa_fa[i][23:38].replace('O', '*')
            # separate write str seq into three parts
            write_str_seq = write_str_seq[0:7] + " " + write_str_seq[7] + " " + write_str_seq[8:]
            result += str(all_id[i].split(":")[0]) + "\t" + str(all_id[i].split(":")[1]) + "\t" + str(
                write_str_seq) + "\t" + str(predict_score[i]) + "\t" + str(
                cut_off_list[cut_off]) + "\t" + "SUMOylation" + "\n"
    return result