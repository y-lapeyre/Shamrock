from _testlib import *





import matplotlib.pyplot as plt
plt.style.use('custom_short_cycler.mplstyle')

import numpy as np




def standalone(json_lst : list, figure_folder : str) -> str:


    buf = r"\section{Shamrock article 1 plots}" + "\n\n"

    reports_tree = []
    reports_treedetail = []

    i = 0
    for report in json_lst:

        
        res = TestResults(i,report)

        build_perf = res.get_test_instances("Benchmark","shamrock_article1:tree_build_perf")

        fileprefix = str(i)

        if len(build_perf) == 1:

            Nparts = []
            results = {}

            fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(7,6))  

            for s in build_perf:

                for dataset in s.test_data:

                    n = dataset["dataset_name"]    

                    if n == "morton = u32, field type = f32":

                        vecN = np.array(s.get_test_dataset(n,"Npart")[1::])
                        
                        #the first point is wrong because a buffer of size 1e8 is moved
                        plt.plot(vecN, np.array(s.get_test_dataset(n,"times_full_tree"          )[1::])/vecN, label = "total build time",color='black',linewidth = 2)
                        #plt.plot(vecN, np.array(s.get_test_dataset(n,"times_morton"             )[1::])/vecN, label = "morton list build",color='black',linestyle="--")
                        plt.plot(vecN, np.array(s.get_test_dataset(n,"times_morton_build"       )[1::])/vecN, label = "compute morton")
                        plt.plot(vecN, np.array(s.get_test_dataset(n,"times_morton_sort"        )[1::])/vecN, label = "bitonic sort",color='black',linestyle="--")
                        plt.plot(vecN, np.array(s.get_test_dataset(n,"times_reduc"              )[1::])/vecN, label = "remove duplicates")
                        plt.plot(vecN, np.array(s.get_test_dataset(n,"times_karras"             )[1::])/vecN, label = "T. Karras 2012")
                        plt.plot(vecN, np.array(s.get_test_dataset(n,"times_compute_int_range"  )[1::])/vecN, label = "compute ranges")
                        plt.plot(vecN, np.array(s.get_test_dataset(n,"times_compute_coord_range")[1::])/vecN, label = "convert ranges")
                        #plt.plot(vecN, np.array(s.get_test_dataset(n,"times_trailling_fill"     )[1::])/vecN, label = "tralling index fill")
                        #plt.plot(vecN, np.array(s.get_test_dataset(n,"times_index_gen"          )[1::])/vecN, label = "index table gen")


            axs.set_xscale('log')
            axs.set_yscale('log')
            axs.set_xlabel(r"$N$")
            axs.set_ylabel(r"$t_{\rm build} /N (s)$")
            axs.legend()
            axs.grid()
            plt.tight_layout()
            plt.savefig(figure_folder+fileprefix+"article1_shamrock_build_perf_tree_random.pdf")



            buf += res.get_config_str()
            buf +=  r"""

            \begin{figure}[ht!]
            \center
            \includegraphics[width=0.8\textwidth]{"""+ "figures/"+fileprefix+"article1_shamrock_build_perf_tree_random.pdf" + r"""}
            \caption{TODO}
            \label{fig:fmm_prec}
            \end{figure}


            """




            vec_Npart = 0
            dic_ = {}

            for s in build_perf:

                for dataset in s.test_data:
                    n = dataset["dataset_name"]

                    vec_Npart = np.array(s.get_test_dataset(n,"Npart")[1::])

                    dic_[n] = np.array(s.get_test_dataset(n,"times_full_tree")[1::])


            fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(7,6))  

            ref = dic_["morton = u32, field type = f32"]

            for k in dic_.keys():
                if (not (k == "morton = u32, field type = f32")):
                    if (k.startswith("morton = u64")):
                        plt.plot(vec_Npart, dic_[k]/ref,label = k,linewidth = 2)
                    else:
                        plt.plot(vec_Npart, dic_[k]/ref,label = k,linestyle="--")
                    

            axs.set_xscale('log')
            axs.set_xlabel(r"$N$")
            axs.set_ylabel(r"$t_{\rm build} /t_{\text{ build (f32/u32)}} $")
            axs.legend()
            axs.grid()
            plt.tight_layout()
            plt.savefig(figure_folder+fileprefix+"article1_shamrock_build_perf_tree_random_comp_var.pdf")

            buf +=  r"""

            \begin{figure}[ht!]
            \center
            \includegraphics[width=0.8\textwidth]{"""+ "figures/"+fileprefix+"article1_shamrock_build_perf_tree_random_comp_var.pdf" + r"""}
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