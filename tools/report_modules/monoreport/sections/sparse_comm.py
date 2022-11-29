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



def make_sparse_comm_plot(fileprefix,report) -> str:
    sparse_comm_result = []
    
    for r in report["results"]:
        if r["type"] == "Benchmark" and r["name"] == "core/comm/sparse_communicator_patchdata_field:":
            sparse_comm_result.append(r)

    if len(sparse_comm_result) == 0:
        return ""

    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(15,6))

    vec_nobj            = []
    vec_fetch_time      = []
    vec_comm_time       = []
    vec_comm_bandwith   = []

    for spres in sparse_comm_result:
        vec_nobj           += get_test_dataset(spres,"bandwith","nb_obj")
        vec_fetch_time     += get_test_dataset(spres,"bandwith","fetch_time")
        vec_comm_time      += get_test_dataset(spres,"bandwith","comm_time")
        vec_comm_bandwith  += get_test_dataset(spres,"bandwith","comm_bandwith")

    vec_nobj          = np.array(vec_nobj         )
    vec_fetch_time    = np.array(vec_fetch_time   )
    vec_comm_time     = np.array(vec_comm_time    )
    vec_comm_bandwith = np.array(vec_comm_bandwith)

    inds = vec_nobj.argsort()
    vec_nobj          = vec_nobj         [inds]
    vec_fetch_time    = vec_fetch_time   [inds]
    vec_comm_time     = vec_comm_time    [inds]
    vec_comm_bandwith = vec_comm_bandwith[inds]

    axs[0].plot(vec_nobj,np.abs(vec_fetch_time  ))
    axs[1].plot(vec_nobj,np.abs(vec_comm_time  ))
    axs[2].plot(vec_nobj,np.abs(vec_comm_bandwith  )/1e9)

    axs[0].set_title('fetch time')
    axs[1].set_title('comm time')
    axs[2].set_title('comm bandwith')

    axs[0].set_xscale('log')
    axs[1].set_yscale('log')
    axs[2].set_xscale('log')

    axs[0].set_xscale('log')
    axs[1].set_yscale('log')
    #axs[2].set_yscale('log')

    axs[0].set_xlabel(r"$n_{\rm obj}$")
    axs[1].set_xlabel(r"$n_{\rm obj}$")
    axs[2].set_xlabel(r"$n_{\rm obj}$")

    axs[0].set_ylabel(r"$(s)$")
    axs[1].set_ylabel(r"$(s)$")
    axs[2].set_ylabel(r"$(Gb . s^{-1})$")

    #axs[0].legend()
    axs[0].grid()
    #axs[1].legend()
    axs[1].grid()
    #axs[2].legend()
    axs[2].grid()



    plt.tight_layout()

    plt.savefig("figures/"+fileprefix+"sparse_comm_perf.pdf")

    return r"""

    \subsection{Sparse communication}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+"figures/"+fileprefix+"sparse_comm_perf.pdf"+r"""}
    \caption{TODO}
    \end{figure}


    """


def make_sparse_comm_report(fileprefix : str, report) -> str:
    buf = ""

    buf += make_sparse_comm_plot(fileprefix,report)


    return buf