import sys
import os
import matplotlib.pyplot as plt
import json
import numpy as np


import report_modules.monoreport.to_tex

#to use do : python ..this file.. shamrock_benchmark_report

plt.style.use('custom_style.mplstyle')

if __name__ == '__main__':

    list_jsons = sys.argv[1:]

    jsons = []

    for f in list_jsons:
        print("reading :",f)
        jsons.append(json.load(open(f,'r')))

    buf = ""

    #buf += make_unittest_report(jsons)
    #buf += r"\section{FMM analysis}"
    #buf += make_fmm_prec_plot(jsons)
    #buf += r"\section{Sycl algs perf}"
    #buf += make_sort_perf_plot(jsons)
    #buf += make_reduc_perf_plot(jsons)

    is_multiple_jsons = len(jsons) > 1

    cnt = 0


    os.system("mkdir figures")

    for j in jsons:
        if(is_multiple_jsons):
            buf += r"\chapter{Configuration "+str(cnt)+" }" "\n"
            
        buf += report_modules.monoreport.to_tex.convert(str(cnt),j)

        cnt += 1

    repport_template = ""

    if is_multiple_jsons:
        f = open("template_multirep.tex",'r')
        repport_template = f.read()
        f.close()
    else:
        f = open("template_1rep.tex",'r')
        repport_template = f.read()
        f.close()

    fout = repport_template.replace("%%%%%%%%%%%%%%%%%data", buf)

    f = open("report.tex",'w')
    f.write(fout)
    f.close()