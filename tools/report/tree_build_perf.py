from _testlib import *





import matplotlib.pyplot as plt
plt.style.use('custom_style.mplstyle')

import numpy as np

def standalone(json_lst : list, figure_folder : str) -> str:


    buf = r"\section{Tree build}" + "\n\n"

    i = 0
    for report in json_lst:

        res = TestResults(i,report)

        build_perf = res.get_test_instances("Benchmark","shamrock/tree/RadixTree:build:benchmark")

        fileprefix = str(i)

        if len(build_perf) == 0:
            return ""

        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

        for s in build_perf:

            for dataset in s.test_data:

                n = dataset["dataset_name"]

                vec_N = s.get_test_dataset(n,"Npart");
                vec_T = s.get_test_dataset(n,"times");

                plt.plot(np.array(vec_N),np.abs(vec_T), label = n)

        axs.set_title('Tree build performance')
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_xlabel(r"$N_{\rm part}$")
        axs.set_ylabel(r"$t_{\rm build} (s)$")
        axs.legend()
        axs.grid()
        plt.tight_layout()
        plt.savefig(figure_folder+fileprefix+"build_tree_perf.pdf")


        buf += res.get_config_str()
        buf +=  r"""


        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.8\textwidth]{"""+ "figures/"+fileprefix+"build_tree_perf.pdf" + r"""}
        \caption{TODO}
        \label{fig:fmm_prec}
        \end{figure}


        """
        i += 1

    return buf


def stacked(json_lst : list, figure_folder : str):

    print(json_in)


def compared(json_lst : list, figure_folder : str):

    print(json_in)



from _test_reader import *
run(standalone, stacked,compared)