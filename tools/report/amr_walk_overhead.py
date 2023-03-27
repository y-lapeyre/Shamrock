from _testlib import *





import matplotlib
import matplotlib.pyplot as plt
plt.style.use('custom_short_cycler.mplstyle')

import numpy as np


def standalone(json_lst : list, figure_folder : str) -> str:


    buf = r"\section{Tree walk amr overhead}" + "\n\n"

    i = 0
    for report in json_lst:

        res = TestResults(i,report)

        sort_perf = res.get_test_instances("Benchmark","shamrock_article1:amr_walk_perf")

        

        if len(sort_perf) == 1:

            for s in sort_perf:

                for dataset in s.test_data:
                    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

                    fileprefix = str(i)

                    n = dataset["dataset_name"]

                    vec_N = np.array(s.get_test_dataset(n,"Ncell"));
                    vec_T = np.array(s.get_test_dataset(n,"time_walk"));
                    vec_Navg = np.array(s.get_test_dataset(n,"avg_neigh"));
                    vec_Nvar = np.array(s.get_test_dataset(n,"var_neigh"));

                    vec_stddev = np.sqrt(vec_Nvar)
                    vec_relative_stddev = vec_stddev/vec_Navg


                    vec_rpart = np.array(s.get_test_dataset(n,"rpart"));

                    plt.scatter(np.array(vec_N),(np.abs(vec_T)/vec_N),c=vec_Navg,norm=matplotlib.colors.LogNorm())

                    #axs.set_title('Bitonic sort perf')
                    plt.colorbar(label='neighbours count')
                    axs.set_xscale('log')
                    axs.set_yscale('log')

                    axs.set_ylim(1e-9,5e-6)

                    axs.set_xlabel(r"$N$")

                    axs.set_ylabel(r"$t_{\rm walk}/N$ (s)")

                    #axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    axs.grid()

                    plt.tight_layout()

                    plt.savefig(figure_folder+fileprefix+"walk_amr_overhead.pdf")
                    #plt.show()
                    

                    buf += res.get_config_str()
                    buf +=  r"""

                    \begin{figure}[ht!]
                    \center
                    \includegraphics[width=0.9\textwidth]{"""+ "figures/"+fileprefix+"walk_amr_overhead.pdf" + r"""}
                    \caption{amr walk : """+ n.replace("_"," ") + r"""}
                    \label{fig:fmm_prec}
                    \end{figure}

                    """
                    i += 1


        if len(sort_perf) == 1:

            for s in sort_perf:

                for dataset in s.test_data:
                    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

                    fileprefix = str(i)

                    n = dataset["dataset_name"]

                    vec_N = np.array(s.get_test_dataset(n,"Ncell"));
                    vec_T = np.array(s.get_test_dataset(n,"time_refine"));
                    vec_Navg = np.array(s.get_test_dataset(n,"avg_neigh"));
                    vec_Nvar = np.array(s.get_test_dataset(n,"var_neigh"));
                    vec_lambda = np.array(s.get_test_dataset(n,"lambda_tilde"));

                    vec_stddev = np.sqrt(vec_Nvar)
                    vec_relative_stddev = vec_stddev/vec_Navg


                    vec_rpart = np.array(s.get_test_dataset(n,"rpart"));

                    plt.scatter(np.array(vec_N),(np.abs(vec_T)/vec_N),c=vec_Navg,norm=matplotlib.colors.LogNorm())

                    #axs.set_title('Bitonic sort perf')
                    plt.colorbar(label='neighbours count')
                    axs.set_xscale('log')
                    axs.set_yscale('log')

                    axs.set_ylim(1e-9,5e-6)

                    axs.set_xlabel(r"$N$")

                    axs.set_ylabel(r"$t_{\rm walk}/N$ (s)")

                    #axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    axs.grid()

                    plt.tight_layout()

                    plt.savefig(figure_folder+fileprefix+"refine_amr_overhead.pdf")
                    #plt.show()
                    

                    buf += res.get_config_str()
                    buf +=  r"""

                    \begin{figure}[ht!]
                    \center
                    \includegraphics[width=0.9\textwidth]{"""+ "figures/"+fileprefix+"refine_amr_overhead.pdf" + r"""}
                    \caption{amr refine : """+ n.replace("_"," ") + r"""}
                    \label{fig:fmm_prec}
                    \end{figure}

                    """
                    i += 1

        
        if len(sort_perf) == 1:

            for s in sort_perf:

                for dataset in s.test_data:
                    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

                    fileprefix = str(i)

                    n = dataset["dataset_name"]

                    vec_N = np.array(s.get_test_dataset(n,"Ncell"));
                    vec_T = np.array(s.get_test_dataset(n,"time_tree"));
                    vec_Navg = np.array(s.get_test_dataset(n,"avg_neigh"));
                    vec_Nvar = np.array(s.get_test_dataset(n,"var_neigh"));
                    vec_lambda = np.array(s.get_test_dataset(n,"lambda_tilde"));

                    vec_stddev = np.sqrt(vec_Nvar)
                    vec_relative_stddev = vec_stddev/vec_Navg


                    vec_rpart = np.array(s.get_test_dataset(n,"rpart"));

                    plt.scatter(np.array(vec_N),(np.abs(vec_T)/vec_N),c=vec_Navg,norm=matplotlib.colors.LogNorm())

                    #axs.set_title('Bitonic sort perf')
                    plt.colorbar(label='neighbours count')
                    axs.set_xscale('log')
                    axs.set_yscale('log')

                    axs.set_ylim(1e-9,5e-6)

                    axs.set_xlabel(r"$N$")

                    axs.set_ylabel(r"$t_{\rm walk}/N$ (s)")

                    #axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    axs.grid()

                    plt.tight_layout()

                    plt.savefig(figure_folder+fileprefix+"tree_build_amr_overhead.pdf")
                    #plt.show()
                    

                    buf += res.get_config_str()
                    buf +=  r"""

                    \begin{figure}[ht!]
                    \center
                    \includegraphics[width=0.9\textwidth]{"""+ "figures/"+fileprefix+"tree_build_amr_overhead.pdf" + r"""}
                    \caption{amr tree build : """+ n.replace("_"," ") + r"""}
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