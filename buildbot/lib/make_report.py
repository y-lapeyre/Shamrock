from enum import Enum
import json
import math


Tex_template = r"""

\documentclass{article}

\begin{document}
%%content%%
\end{document}
"""


class ReportFormat(Enum):
    Tex = 1
    HTML = 1
    Txt = 1

def load_test_report(file):
    out_file = open(file, "r")
    lst_ln = out_file.readlines()
    out_file.close()

    cur_test = ""
    cur_world_sz = -1
    cur_world_rk = -1


    cur_assert = ""
    cur_assert_log = ""
    cur_assert_result = -1

    cur_assert_log_state = False

    


    dic_loaded = {}


    for l in (lst_ln):
        if l.startswith(r"%test_name = "):

            test_name = l[l.find("\"")+1:l.find("\"",l.find("\"")+1)]
            cur_test = test_name
            #print(" -> starting_test", test_name)

            if not (cur_test in dic_loaded.keys()):
                dic_loaded[cur_test] = {}

        elif l.startswith(r"%end_test"):
            #print(" -> end_test", test_name)

            cur_test = ""
            cur_world_sz = -1
            cur_world_rk = -1

        # elif l.startswith(r"%world_size = "):
        #     cur_world_sz = int(l[len("%world_size = "):])
        #     print("     -> world size",cur_world_sz)

        #     dic_loaded[cur_test]["world_size"] = cur_world_sz

        elif l.startswith(r"%world_rank = "):
            cur_world_rk = int(l[len("%world_rank = "):])
            #print("     -> world rank",cur_world_rk)

            dic_loaded[cur_test][cur_world_rk] = []


        elif l.startswith(r"%start_assert"):
            assert_name = l[l.find("\"")+1:l.find("\"",l.find("\"")+1)]
            cur_assert = assert_name
            #print("         -> start_assert",assert_name)


        elif l.startswith(r"%end_assert"):
            #print("         -> end_assert",cur_assert)

            dic_loaded[cur_test][cur_world_rk].append({"name" : cur_assert, "log" : cur_assert_log, "result" : cur_assert_result})

            cur_assert = ""
            cur_assert_log = ""
            cur_assert_result = -1



        elif l.startswith(r"%result = "):
            cur_assert_result = int(l[len("%result = "):])
            #print("             -> assert result",assert_name,cur_assert_result)

        elif l.startswith(r"%startlog"):
            cur_assert_log_state = True
            #print("             -> start log")
        elif l.startswith(r"%endlog"):
            cur_assert_log_state = False

            #print("             -> end log content : \n",cur_assert_log)

        elif cur_assert_log_state:
            cur_assert_log += l + "\n"
        

    return (dic_loaded)


def get_succes_count_data(dt):
    out_dic = {}
    for k_cur_test in  dt.keys():

        tmp = {}

        sum_cnt_assert = 0
        sum_cnt_succes = 0

        for k_cur_wrk in  dt[k_cur_test].keys():

            cnt_assert = 0
            cnt_succes = 0

            for asserts in dt[k_cur_test][k_cur_wrk]:
                cnt_assert += 1
                cnt_succes += asserts["result"]

            sum_cnt_assert += cnt_assert
            sum_cnt_succes += cnt_succes

            #print("test ",k_cur_test, "world size =",k_cur_wrk,"| succes rate =",cnt_succes,"/",len(dt[k_cur_test][k_cur_wrk]))
            tmp[k_cur_wrk] = {"suc_cnt":cnt_succes,"assert_cnt":cnt_assert}

            
        
        tmp["suc_cnt"] = sum_cnt_succes
        tmp["assert_cnt"] = sum_cnt_assert

        out_dic[k_cur_test] = tmp
    return out_dic







def make_tex_repport(dat):



    str_file = ""


    dic_int = {}

    for config_k in dat.keys():
        conf_dic = dat[config_k]

        for k in conf_dic.keys():
            if k.startswith("world_size="):
                wsz = int(k[len("world_size="):])

                dic_int["world size = " + str(wsz)] = {}

    


    for config_k in dat.keys():
        conf_dic = dat[config_k]

        for k in conf_dic.keys():
            if k.startswith("world_size="):
                wsz = int(k[len("world_size="):])

                dic_res = load_test_report(dat[config_k][k])
                dic_suc_cnt = get_succes_count_data(dic_res)



                dic_int["world size = " + str(wsz)][ dat[config_k]["description"]] = {
                    "results" : dic_res,
                    "succes_cnt" : dic_suc_cnt
                }

    

    out_file = open("tmp.json", "w")
    json.dump(dic_int, out_file, indent = 6)
    out_file.close()


    #     str_file += r"""
    # \section{""" + dat[config_k]["description"] + r"""}
    #     """




    print(Tex_template.replace(r"%%content%%",str_file))











def make_report(format, out_res_map_file):

    out_file = open(out_res_map_file, "r")
    data = json.load(out_file)
    out_file.close()

    #print(data)

    if format == ReportFormat.Tex:
        make_tex_repport(data)





if __name__ == '__main__':
    make_report(ReportFormat.Tex, "../../test_pipeline/test_result_list.json")

    # dat_ld = load_test_report("/home/tim/Documents/these/codes/sycl_workspace/shamrock/test_pipeline/build_ss/test_res_2.sutest")
    # print(dat_ld)
    # print(get_succes_count_dat(dat_ld))