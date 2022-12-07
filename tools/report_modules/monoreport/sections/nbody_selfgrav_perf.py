import matplotlib.pyplot as plt 
import numpy as np







def get_test_dataset(test_result, dataset_name, table_name):
    test_data = test_result["test_data"]

    for d in test_data:
        if(d["dataset_name"] == dataset_name):
            for t in d["dataset"]:
                if(t["name"] == table_name):
                    return t["data"]

    return None

def make_bench_nbody_selfgrav_plot(fileprefix,report) -> str:

    tests = []
    
    for r in report["results"]:
        if r["type"] == "Benchmark" and r["name"] == "benchmark selfgrav nbody":
            tests.append(r)

    if len(tests) == 0:
        return ""
    
    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

    for t in tests:
        fig.suptitle(t["compute_queue"])

        for dat in t["test_data"]:
            name = dat["dataset_name"]
            X = np.array(get_test_dataset(t, name,"Npart"))
            Y = np.array(get_test_dataset(t, name,"times"))

            axs[0].plot(X,Y,label = name)
            axs[1].plot(X,X/Y,label = name)




    axs[0].legend()
    axs[0].grid()
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')



    axs[1].legend()
    axs[1].grid()
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')



    axs[0].set_xlabel(r"$N_{\rm part}$")
    axs[1].set_xlabel(r"$N_{\rm part}$")

    axs[0].set_ylabel(r"$t_{\rm iter}$")
    axs[1].set_ylabel(r"$N_{\rm part} / t_{\rm iter}$")


    plt.tight_layout()

    plt.savefig("figures/"+fileprefix+"bench_nbody_selfgrav.pdf")


    return r"""

    \subsection{Nbody Selfgrav perf}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+"figures/"+fileprefix+"bench_nbody_selfgrav.pdf"+r"""}
    \caption{TODO}
    \end{figure}

    """




def make_bench_nbody_selfgrav_report(fileprefix : str, report) -> str:
    buf = ""

    buf += make_bench_nbody_selfgrav_plot(fileprefix,report)


    return buf