
import matplotlib.pyplot as plt 
import numpy as np

from matplotlib.ticker import MultipleLocator
def bar_plot_custom(list_print,max_ax = 2.7,min_ax = 0):
    

    base_width = 0.7
    margin_width = 0.8

    margin_text = 0.05


    cases_n = list_print.keys()
    ncase = len(cases_n)
    width = base_width/ncase


    fig, ax = plt.subplots()

    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))


    cnt = 0
    for case_k in cases_n :

        keys = list_print[case_k].keys()
        vals = [list_print[case_k][k] for k in keys]

        x = np.arange(len(keys))

        xpos = x + width*(cnt - (ncase-1)/2.)

        rect = ax.bar(xpos,vals,width*margin_width,label=case_k,edgecolor = 'black',linewidth=1.5)

        # labels = []

        # for v in vals:
        #     if v > max_ax :
        #         label.append("")
        #     else:
        #         label.append("x {}".format(v))

        #ax.bar_label(rect, padding=3)
        ax.set_xticks(x, keys)

        for (x,v) in zip(xpos,vals) :
            if v > max_ax : 
                plt.text(x,max_ax*(1 + margin_text),"$\cdots$ $\,$x$\,${:.3f}".format(v),horizontalalignment='center',rotation = "vertical")
            else : 
                plt.text(x,v + max_ax*margin_text,"$\,$x$\,${:.3f}".format(v),horizontalalignment='center',rotation = "vertical")


        cnt += 1


    ax.set_ylim(min_ax,max_ax)

    plt.legend()
    plt.tight_layout()
    plt.ylabel("Relative time (lower is better)")
    plt.xlabel("Test name")



def get_test_dataset(test_result, dataset_name, table_name):
    test_data = test_result["test_data"]

    for d in test_data:
        if(d["dataset_name"] == dataset_name):
            for t in d["dataset"]:
                if(t["name"] == table_name):
                    return t["data"]

    return None


def reduction_pref_plot(fileprefix : str, report):
    reduc_perf = []
    
    for r in report["results"]:
        if r["type"] == "Benchmark" and r["name"] == "core/utils/sycl_algs:reduction":
            reduc_perf.append(r)

    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

    for s in reduc_perf:

        for dataset in s["test_data"]:

            n = dataset["dataset_name"]

            vec_N = get_test_dataset(s,n,"Nobj");
            vec_T = get_test_dataset(s,n,"t_sort");

            plt.plot(np.array(vec_N),np.abs(vec_T), label = n)

    axs.set_title('Reduction perf')

    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.set_xlabel(r"$N$")

    axs.set_ylabel(r"$t_{\rm sort} (s)$")

    axs.legend(bbox_to_anchor=(1,1), loc="upper left")
    axs.grid()

    plt.tight_layout()

    plt.savefig("figures/"+fileprefix+"reduc_perf.pdf")







    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

    for s in reduc_perf:

        for dataset in s["test_data"]:

            n = dataset["dataset_name"]

            vec_N = get_test_dataset(s,n,"Nobj")
            vec_T = get_test_dataset(s,n,"t_sort")

            plt.plot(np.array(vec_N),vec_N/np.abs(vec_T), label = n)

    axs.set_title('Reduction perf')

    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.set_xlabel(r"$N$")

    axs.set_ylabel(r"key per s")

    axs.legend(bbox_to_anchor=(1,1), loc="upper left")
    axs.grid()

    plt.tight_layout()

    plt.savefig("figures/"+fileprefix+"reduc_perf_comp.pdf")







    list_print = {}

    div = 1

    for s in reduc_perf:
        for dataset in s["test_data"]:
            type_test = dataset["dataset_name"].split()[0]
            if not type_test in list_print.keys():
                list_print[type_test] = {}

            if dataset["dataset_name"] == "f32 manual : wg=2":
                div = get_test_dataset(s,dataset["dataset_name"],"t_sort")[-1]

    for s in reduc_perf:
        for dataset in s["test_data"]:
            type_test = dataset["dataset_name"].split()[0]

            case = dataset["dataset_name"][len(type_test):]
            list_print[type_test][case] = get_test_dataset(s,dataset["dataset_name"],"t_sort")[-1]/div


    bar_plot_custom(list_print,max_ax=1.2,min_ax=0.6)

    plt.xticks(rotation=20)
    plt.savefig("figures/"+fileprefix+"max_perf_reduc.pdf")







def make_sort_perf_plot(fileprefix, report) -> str:





    sort_perf = []
    
    for r in report["results"]:
        if r["type"] == "Benchmark" and r["name"] == "core/tree/kernels/key_pair_sort (benchmark)":
            sort_perf.append(r)


    if len(sort_perf) == 0:
        return ""

    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

    for s in sort_perf:

        for dataset in s["test_data"]:

            n = dataset["dataset_name"]

            vec_N = get_test_dataset(s,n,"Nobj");
            vec_T = get_test_dataset(s,n,"t_sort");

            plt.plot(np.array(vec_N),np.abs(vec_T), label = n)

    axs.set_title('Bitonic sort perf')

    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.set_xlabel(r"$N$")

    axs.set_ylabel(r"$t_{\rm sort} (s)$")

    axs.legend()
    axs.grid()

    plt.tight_layout()

    plt.savefig("figures/"+fileprefix+"sort_perf.pdf")



    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

    for s in sort_perf:

        for dataset in s["test_data"]:

            n = dataset["dataset_name"]

            vec_N = np.array(get_test_dataset(s,n,"Nobj"));
            vec_T = np.array(get_test_dataset(s,n,"t_sort"));

            plt.plot(np.array(vec_N),vec_N/(np.abs(vec_T)), label = n)

    axs.set_title('Bitonic sort perf')

    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.set_xlabel(r"$N$")

    axs.set_ylabel(r"key per s")

    axs.legend()
    axs.grid()

    plt.tight_layout()

    plt.savefig("figures/"+fileprefix+"sort_perf_comp.pdf")

    return r"""

    \subsection{Bitonic performance}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+ "figures/"+fileprefix+"sort_perf.pdf" + r"""}
    \caption{TODO}
    \label{fig:fmm_prec}
    \end{figure}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+ "figures/"+fileprefix+"sort_perf_comp.pdf" + r"""}
    \caption{TODO}
    \label{fig:fmm_prec}
    \end{figure}


    """




















def make_reduc_perf_plot(fileprefix : str, report) -> str:

    cnt = 0
    for r in report["results"]:
        if r["type"] == "Benchmark" and r["name"] == "core/utils/sycl_algs:reduction":
            cnt += 1

    if cnt == 0:
        return ""
    


    reduction_pref_plot(fileprefix, report)

    return r"""

    \subsection{Reduction performance}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+ "figures/"+fileprefix+"reduc_perf.pdf" + r"""}
    \caption{TODO}
    \label{fig:reduc_perf}
    \end{figure}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+ "figures/"+fileprefix+"reduc_perf_comp.pdf" + r"""}
    \caption{TODO}
    \label{fig:reduc_perf_comp}
    \end{figure}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+ "figures/"+fileprefix+"max_perf_reduc.pdf" + r"""}
    \caption{TODO}
    \label{fig:comp_reduc_perf}
    \end{figure}

    """

def make_syclalgs_report(fileprefix : str, report) -> str:
    buf = ""

    buf += make_reduc_perf_plot(fileprefix,report) 
    buf += make_sort_perf_plot(fileprefix,report)


    return buf