

import numpy as np

def get_test_dataset(test_result, dataset_name, table_name):
    test_data = test_result["test_data"]

    for d in test_data:
        if(d["dataset_name"] == dataset_name):
            for t in d["dataset"]:
                if(t["name"] == table_name):
                    return t["data"]

    return None

def standalone(json_lst : list) -> str:


    buf = r"\section{Shamalgs key pair sort}" + "\n\n"

    

    i = 0
    for report in json_lst:

        sort_perf = []

        fileprefix = str(i)

        for r in report["results"]:
            if r["type"] == "Benchmark" and r["name"] == "core/tree/kernels/key_pair_sort (benchmark)":
                sort_perf.append(r)


        if len(sort_perf) == 0:
            return ""

        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

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



        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

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

        

        buf += get_str_foreach(i,report)
        buf +=  r"""


        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.8\textwidth]{"""+ "figures/"+fileprefix+"sort_perf.pdf" + r"""}
        \caption{TODO}
        \label{fig:fmm_prec}
        \end{figure}

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.8\textwidth]{"""+ "figures/"+fileprefix+"sort_perf_comp.pdf" + r"""}
        \caption{TODO}
        \label{fig:fmm_prec}
        \end{figure}

        """
        i += 1
    return buf