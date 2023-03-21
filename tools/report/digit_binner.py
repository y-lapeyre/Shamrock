from _testlib import *





import matplotlib.pyplot as plt
plt.style.use('custom_style.mplstyle')

import numpy as np


def standalone(json_lst : list, figure_folder : str) -> str:


    buf = r"\section{Shamalgs digit binner}" + "\n\n"

    i = 0
    for report in json_lst:

        res = TestResults(i,report)

        sort_perf = res.get_test_instances("Benchmark","shamalgs/algorithm/details/DigitBinner:benchmark")

        fileprefix = str(i)

        if len(sort_perf) == 1:


            fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

            for s in sort_perf:

                for dataset in s.test_data:

                    n = dataset["dataset_name"]

                    vec_N = np.array(s.get_test_dataset(n,"Nobj"));
                    vec_T = np.array(s.get_test_dataset(n,"time"));

                    plt.plot(np.array(vec_N),(np.abs(vec_T)/vec_N), label = n)

            axs.set_title('Shamalgs digit binning')

            axs.set_xscale('log')
            axs.set_yscale('log')

            axs.set_xlabel(r"$N$")

            axs.set_ylabel(r"s per key")

            axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs.grid()

            plt.tight_layout()

            plt.savefig(figure_folder+fileprefix+"digit_binner.pdf")

            

            buf += res.get_config_str()
            buf +=  r"""

            \begin{figure}[ht!]
            \center
            \includegraphics[width=0.9\textwidth]{"""+ "figures/"+fileprefix+"digit_binner.pdf" + r"""}
            \caption{TODO}
            \label{fig:fmm_prec}
            \end{figure}

            """
            i += 1

    if i == 0:
        return ""

    return buf


def stacked(json_lst : list, figure_folder : str):

    print(json_in)


def compared(json_lst : list, figure_folder : str):

    print(json_in)



from _test_reader import *
run(standalone, stacked,compared)