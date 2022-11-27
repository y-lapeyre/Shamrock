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



def make_fmm_prec_plot(fileprefix,report) -> str:
    fmm_prec_test_res = []
    
    for r in report["results"]:
        if r["type"] == "Analysis" and r["name"] == "models/generic/fmm/precision":
            fmm_prec_test_res.append(r)

    if len(fmm_prec_test_res) == 0:
        return ""

    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

    def plot_curve(ax,X,Y,lab):

        ratio = 40

        cnt = len(X)//ratio 

        X_m = X.reshape((cnt,ratio))
        X_m = np.max(X_m,axis=1)

        Y_m = Y.reshape((cnt,ratio))
        Y_m = np.max(Y_m,axis=1)


        ax.plot(X_m,Y_m, label = lab)


    for fmmmmmmmmm in fmm_prec_test_res:
        vec_angle          = get_test_dataset(fmmmmmmmmm,"fmm_precision","angle");
        vec_result_pot_5   = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_pot_5");
        vec_result_pot_4   = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_pot_4");
        vec_result_pot_3   = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_pot_3");
        vec_result_pot_2   = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_pot_2");
        vec_result_pot_1   = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_pot_1");
        vec_result_pot_0   = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_pot_0");
        vec_result_force_5 = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_force_5");
        vec_result_force_4 = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_force_4");
        vec_result_force_3 = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_force_3");
        vec_result_force_2 = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_force_2");
        vec_result_force_1 = get_test_dataset(fmmmmmmmmm,"fmm_precision","result_force_1"); 

        plot_curve(axs[0],np.array(vec_angle),np.abs(vec_result_pot_5  ), "fmm order = 5")
        plot_curve(axs[0],np.array(vec_angle),np.abs(vec_result_pot_4  ), "fmm order = 4")
        plot_curve(axs[0],np.array(vec_angle),np.abs(vec_result_pot_3  ), "fmm order = 3")
        plot_curve(axs[0],np.array(vec_angle),np.abs(vec_result_pot_2  ), "fmm order = 2")
        plot_curve(axs[0],np.array(vec_angle),np.abs(vec_result_pot_1  ), "fmm order = 1")
        plot_curve(axs[0],np.array(vec_angle),np.abs(vec_result_pot_0  ), "fmm order = 0")
        plot_curve(axs[1],np.array(vec_angle),np.abs(vec_result_force_5), "fmm order = 5")
        plot_curve(axs[1],np.array(vec_angle),np.abs(vec_result_force_4), "fmm order = 4")
        plot_curve(axs[1],np.array(vec_angle),np.abs(vec_result_force_3), "fmm order = 3")
        plot_curve(axs[1],np.array(vec_angle),np.abs(vec_result_force_2), "fmm order = 2")
        plot_curve(axs[1],np.array(vec_angle),np.abs(vec_result_force_1), "fmm order = 1")

    axs[0].set_title('Gravitational potential ($\Phi$)')
    axs[1].set_title('Gravitational force ($\mathbf{f}$)')

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    axs[0].set_ylim(1e-16,1)
    axs[1].set_ylim(1e-16,1)


    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    axs[0].set_xlabel(r"$\theta$")
    axs[1].set_xlabel(r"$\theta$")

    axs[0].set_ylabel(r"$\vert \Phi_{\rm fmm} - \Phi_{\rm th} \vert /\vert \Phi_{\rm th}\vert$")
    axs[1].set_ylabel(r"$\vert \mathbf{f}_{\rm fmm} - \mathbf{f}_{\rm th} \vert /\vert \mathbf{f}_{\rm th}\vert$")

    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()



    plt.tight_layout()

    plt.savefig("figures/"+fileprefix+"fmm_precision.pdf")

    return r"""

    \subsection{Precision on particle pairs}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{"""+"figures/"+fileprefix+"fmm_precision.pdf"+r"""}
    \caption{Precision of the fmm using multiple parameters. 
    On the left we compare fmm precision of the potential versus theory, the y axis is limited to $10^{-16}$ as we hit normally the limit of double precision floating point numbers. On the right the same comparaison can be seen for the corresponding force.}
    \label{fig:fmm_prec}
    \end{figure}

    \textbf{To check} : In Fig.\ref{fig:fmm_prec} We should see the potential hitting the limit of double precision normally.

    """


def make_fmm_report(fileprefix : str, report) -> str:
    buf = ""

    buf += make_fmm_prec_plot(fileprefix,report)


    return buf