# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li and ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : SumobindingPredict.py
# @Software: PyCharm
import numpy as np
from PSSMBinding import PSSM_code
from OnehotSumobinding import onehot_x
from KNNBinding import KNN_code

from SumobindingModel import CNN_SUMO
import tensorflow as tf
import sys


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
            sequence = line.replace("*","O")
            seqlist.append(sequence)
        if seq_name not in fasta:
            fasta[seq_name] = []
            continue
        fasta[seq_name] = "".join(seqlist)
    return fasta


def pre_encode_data(datafile,intfmt="txt",mode="Train",onehot_length=6):
    pos_train = open(
        r"libdata/SumobindingAllCurrentPos.txt")
    neg_train = open(
        r"libdata/SumobindingAllCurrentNeg.txt")
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
    id_fa_list = []
    if mode=="Train":  # for training, that is, we simply use site at the middle of the seqs input
        id_fa_list = []
        for i in id_fa_dict:
            id_fa_list.append([i,id_fa_dict[i]])
    else:
        for key_id in id_fa_dict:
            start_loc = int(0)
            fa = 30 * "O" + id_fa_dict[key_id] + (30 + 6) * "O"  # even longer than the sequences that we input
            for i in range(len(fa) - 36):
                i = i + 30  # actually its index starts from where it is not "O"
                site_fa = fa.strip()[i:i + 5]  # see if they are potential binding motif on every five aas
                site_fa = site_fa.upper()
                if site_fa.startswith("I") or site_fa.startswith("V") or site_fa.startswith("L"):
                    num = site_fa.count("I") + site_fa.count("V") + site_fa.count("L")
                    if num >= 3 and num <= 5:  # motif pattern matches
                        if start_loc == 0:
                            site_fa_seq = fa.strip()[i - 30:i + 30 + 5]  # the index has to be either I V L,and length is 65,IVL is in site 31
                            site_fa_seq = site_fa_seq.upper()
                            id_fa_list.append([key_id + ":" + str(i - 30 + 1), site_fa_seq])  # i-30+1 will be a 1-index site
                            start_loc = int(i - 30 + 1)
                        else:
                            if int(i - 30 + 1) <= start_loc:  # jump every (n-1) amino acid to select another motif or seq,used to be +4
                                pass
                            else:
                                site_fa_seq = fa.strip()[i - 30:i + 30 + 5]
                                site_fa_seq = site_fa_seq.upper()
                                id_fa_list.append([key_id + ":" + str(i - 30 + 1), site_fa_seq])
                                start_loc = int(i - 30 + 1)
    id_fa_list = np.array(id_fa_list)
    all_id = id_fa_list[:, 0]
    all_aa_fa = id_fa_list[:, 1]
    # encoding
    predict_set0 = onehot_x(all_aa_fa, onehot_length)
    predict_set1 = KNN_code(all_aa_fa, pos_train, neg_train, onehot_length)
    predict_set2 = PSSM_code(all_aa_fa, onehot_length)
    # predict_set3 = PFREncode(all_aa_fa,onehot_length,int_type="bind")
    predict_set01 = predict_set0.reshape([-1, onehot_length * 2 + 5, 21]).astype(float)
    seq_idx = 0
    for seq in predict_set2:
        aa_idx = 0
        for aa_site in seq:
            try:
                rep_site = int(np.where(predict_set01[seq_idx, aa_idx, :] == 1)[0])
                predict_set01[seq_idx, aa_idx, rep_site] = aa_site
                aa_idx += 1
            except:
                pass
        seq_idx += 1
    predict_set01 = np.around(predict_set01.reshape(-1, 21 * (onehot_length * 2 + 5)), decimals=4)
    predict_set = np.concatenate((predict_set01, predict_set1), axis=1)
    return predict_set,all_id,all_aa_fa



def sim_predict(cut_off, input_file, result_dir):
    np.random.seed(89757)
    cut_off = cut_off.lower()
    # cut_off_list = {"low":0.349,"medium":0.537,"high":0.592,"all":0}  # TODO: need to change here
    cut_off_list = {'low':0.083750,'medium':0.145625,'high':0.268125,'all':0}
    # get input data, read as fasta format
    predict_set, all_id, all_aa_fa = pre_encode_data(input_file,intfmt="fasta",mode="Test",onehot_length=6)
    # print(predict_set.shape)
    tf.reset_default_graph()
    # prediction
    cnn = CNN_SUMO(model_path="libdata/SumobindingModel/",
                   model_name="model", epochs=50, batch_size=30, display_step=1, learning_rate=0.0001,
                   keep_prob=1, train_length=6)
    predict_score = cnn.predict(other_data=predict_set,
                                model_path="libdata/SumobindingModel/",
                                modelname="model")  # the modelname used to be SumoBinding, I moved it into dir:obsolete
    # construct output info
    result = ""
    for i in range(len(all_id)):
        if float(predict_score[i][0]) >= float(cut_off_list[cut_off]):
            # if float(predict_score[i]) >=0:
            write_str_seq = all_aa_fa[i][23:38 + 4].replace('O', '*')  # left 7 and right 7 + 5(5 is the length of motif)
            write_str_seq = write_str_seq[0:7] + " " + write_str_seq[7:12] + " " + write_str_seq[12:]
            result += str(all_id[i].split(":")[0]) + "\t" + str(all_id[i].split(":")[1]) + "-" + str(
                int(all_id[i].split(":")[1]) + 4) + "\t" + str(write_str_seq) + "\t" + str(
                predict_score[i][0]) + "\t" + str(cut_off_list[cut_off]) + "\t" + "SUMO-interaction" + "\n"
    return result