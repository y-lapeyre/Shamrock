import sys
import os
import matplotlib.pyplot as plt


#to use do : python ..this file.. shamrock_benchmark_report

plt.style.use('custom_style.mplstyle')

f = open(sys.argv[1],'r')

lns = f.readlines()

f.close()

try : 
    os.mkdir("figures")
except : 
    a = 0


def plot_tree_field_compute_perf():


    dic_test = {}

    current_reduc_lev = 0
    current_queue = ""

    def get_name():
        return current_queue + " reduction = " + current_reduc_lev


    X = []
    Y = []

    is_on = False
    for l in lns:

        if "treefieldcomputeperf" in l:
            is_on = True
        if "%end_bench" in l:
            is_on = False

        if "%compute_queue_name = " in l:
            current_queue = l[len("%compute_queue_name = "):-1]

            if len(X) > 0:
                dic_test[get_name()] = {"X" : X, "Y" : Y, "label" : get_name()}
                X = []
                Y = []

        if "%tree_reduc = " in l:
            current_reduc_lev = l[len("%tree_reduc = "):-1]

            if len(X) > 0:
                dic_test[get_name()] = {"X" : X, "Y" : Y, "label" : get_name()}
                X = []
                Y = []

        if is_on:
            if "%result" in l:
                splt = l.split()[-1].split(",")
                X.append(float(splt[0]))
                Y.append(float(splt[1])/1e9)

    for k in dic_test:

        curve = dic_test[k]

        plt.plot(curve["X"],curve["Y"],'-',label = curve["label"])


    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r"$N_{\rm part}$")
    plt.ylabel(r"$t (s)$")

    plt.title("Tree compute field time ($op = \max$)")

    plt.legend()
    
    plt.savefig("figures/tree_compute_field_time.pdf")

plot_tree_field_compute_perf()