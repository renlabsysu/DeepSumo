# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li and ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : predict_main.py
# @Software: PyCharm
from SumobindingPredict import sim_predict
from SumoylationPredict import sumoylation_predict
import sys, getopt
import os


def get_highest(sim_res_list):
    # sim_rest_list is the list contains output into
    # result_header = "ID" + "\t" + "Position" + "\t" + "Sequence" + "\t" + "Score" + "\t" + "Cutoff" + "\t" + "Cluster" + "\n"
    while "" in sim_res_list:
        sim_res_list.remove("")
    to_drop = []
    if len(sim_res_list) > 1:
        sim_res_list = sim_res_list[1:]
        for res1 in sim_res_list:
            for res2 in sim_res_list:
                cur = res1.split("\t")
                compare = res2.split("\t")
                cur_name, cur_score, cur_site = cur[0], float(cur[3]), int(cur[1].split("-")[0])
                cp_name, cp_score, cp_site = compare[0], float(compare[3]), int(compare[1].split("-")[0])
                if cur_name==cp_name:
                    judge1 = cur_site < cp_site and (cur_site + 5) > cp_site
                    judge2 = cur_site > cp_site and cur_site < (cp_site + 5)
                    if (judge1 or judge2)==True:
                        if cur_score > cp_score:
                            to_drop.append(res2)
        to_drop = list(set(to_drop))
        for td in to_drop:
            sim_res_list.remove(td)
        # sim_res_list = sorted(list(set(sim_res_list)))
    # result_new = result_header + "\n".join(sim_res_list)
    return sim_res_list

def predict_main(sumylation_cut_off, sumobinding_cut_off, input_fasta, result_dir):
    result_header = "ID" + "\t" + "Position" + "\t" + "Sequence" + "\t" + "Score" + "\t" + "Cutoff" + "\t" + "Cluster" + "\n"
    if sumylation_cut_off == "None" and sumobinding_cut_off != "None":
        result_new = result_header
        result_new += sim_predict(sumobinding_cut_off, input_fasta, result_dir)
        ###############
        # if there are overlap sites, we just save the one which get the highest prediction score
        sim_res_list = result_new.split("\n")
        sim_res_list = get_highest(sim_res_list)
        result_new = result_header + "\n".join(sim_res_list)
        #################
        result_log = "success"
    elif sumylation_cut_off != "None" and sumobinding_cut_off == "None":
        result_new = result_header
        result_new += sumoylation_predict(sumylation_cut_off, input_fasta, result_dir)
        result_log = "success"
    elif sumylation_cut_off != "None" and sumobinding_cut_off != "None":

        # this may just indicates that we want the same protein to show in the same/near position.
        all_id_list = []
        sumylation_dict = {}

        result = result_header
        result += sumoylation_predict(sumylation_cut_off, input_fasta, result_dir)
        sumoylation_result = sumoylation_predict(sumylation_cut_off, input_fasta, result_dir)  # TODO: drop one for faster running

        sumylation_res_list = sumoylation_result.split("\n")

        for res in sumylation_res_list:
            if len(res) >= 2:
                res_list = res.split("\t")

                id1 = res_list[0]
                all_id_list.append(id1)
                pos = res_list[1]
                if id1 + "_pos" in sumylation_dict:
                    sumylation_dict[id1 + "_pos"].append([int(pos), "sumoylation"])
                else:
                    sumylation_dict[id1 + "_pos"] = [[int(pos), "sumoylation"]]

                if id1 in sumylation_dict:
                    sumylation_dict[id1][pos] = res
                else:
                    sumylation_dict.update({id1: {pos: res}})

        result += sim_predict(sumobinding_cut_off, input_fasta, result_dir)
        sim_dict = {}

        sim_result = sim_predict(sumobinding_cut_off, input_fasta, result_dir)
        sim_res_list = sim_result.split("\n")
        ####### if two motif overlap, just save the motif with higher score ########
        sim_res_list = get_highest(sim_res_list)
        #########################################################
        for res in sim_res_list:
            if len(res) >= 2:
                res_list = res.split("\t")
                id1 = res_list[0]
                all_id_list.append(id1)
                pos = res_list[1]
                if id1 + "_pos" in sim_dict:
                    sim_dict[id1 + "_pos"].append([int(pos.split("-")[0]), "sim"])
                else:
                    sim_dict[id1 + "_pos"] = [[int(pos.split("-")[0]), "sim"]]

                if id1 in sim_dict:
                    sim_dict[id1][pos] = res
                else:
                    sim_dict.update({id1: {pos: res}})

        all_id_list = list(set(all_id_list))

        result_new = result_header
        for id2 in all_id_list:
            sorted_list = []
            # print("id2")
            # print(id2)
            if id2 in sumylation_dict:
                sorted_list += sumylation_dict[id2 + "_pos"]
            if id2 in sim_dict:
                sorted_list += sim_dict[id2 + "_pos"]

            sorted_list.sort(reverse=False)

            for ptm_pos in sorted_list:

                if ptm_pos[1] == "sumoylation":
                    result_new += sumylation_dict[id2][str(ptm_pos[0])] + "\n"
                else:
                    result_new += sim_dict[id2][str(ptm_pos[0]) + "-" + str(ptm_pos[0] + 4)] + "\n"
        ####################################################
        result_log = "success"
    elif sumylation_cut_off == "None" and sumobinding_cut_off == "None":
        result_new = "Please select at least one modification type."
        result_log = "Please select at least one modification type."
    else:
        result_new = "error"
        result_log = "error"
    # output results

    # server version
    # result_f = open("/root/DeepSUMO/data/" + result_dir + "result_output.txt", "w")
    # result_f.write(result_new)
    # result_f_log = open("/root/DeepSUMO/data/" + result_dir + "result_output.log", "w")
    # result_f_log.write(result_log)
    # print(result_log)
    # result_f.close()
    # result_f_log.close()

    # local version
    if result_dir.endswith(os.sep)==False:
        result_dir = result_dir + os.sep
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_f = open(result_dir + "result_output.txt", "w")
    result_f.write(result_new)
    result_f_log = open(result_dir + "result_output.log", "w")
    result_f_log.write(result_log)
    print(result_log)
    result_f.close()
    result_f_log.close()




# def main(cut_off1, cut_off2, input_file, result_dir, log_file):
#     predict_main(cut_off1, cut_off2, input_file, result_dir, log_file)

def main(cut_off1, cut_off2, input_file, result_dir):
    predict_main(cut_off1, cut_off2, input_file, result_dir)


if __name__ == '__main__':

    cut_off1 = ""
    cut_off2 = ""
    input_file = ""
    result_dir = ""
    log_file = ""
    # opts, args = getopt.getopt(sys.argv[1:], 'i:o:', ["t1=", "t2=", "log="])
    opts, args = getopt.getopt(sys.argv[1:], 'i:o:', ["t1=", "t2="])
    for op, value in opts:
        if op == "--t1":  # sumoylation cutoff
            cut_off1 = value
        if op == "--t2":  # sumobinding cutoff
            cut_off2 = value
        if op == "-i":  # input file (fasta file)
            input_file = value
        if op == "-o":  # output file
            result_dir = value
        # if op == "--log":
        #     log_file = value
    # print(cut_off2)
    # main(cut_off1, cut_off2, input_file, result_dir, log_file)
    main(cut_off1, cut_off2, input_file, result_dir)
