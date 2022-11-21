import sys
import os
import matplotlib.pyplot as plt
import json
import numpy as np



repport_template = r"""


\documentclass{article}

\usepackage[a4paper,total={170mm,260mm},left=20mm,top=20mm,]{geometry}

\usepackage{fancyhdr} % entêtes et pieds de pages personnalisés

\pagestyle{fancy}
\fancyhead[L]{\scriptsize \textsc{Soundwave test}} % À changer
\fancyhead[R]{\scriptsize \textsc{\textsc{SHAMROCK}}} % À changer
\fancyfoot[C]{ \thepage}

\usepackage{graphicx}

\usepackage{titling}

\setlength{\droptitle}{-4\baselineskip} % Move the title up

\pretitle{\begin{center}\Huge\bfseries} % Article title formatting
\posttitle{\end{center}} % Article title closing formatting
\title{\textsc{SHAMROCK} Test Suite} % Article title
\author{%
\textsc{Timothée David--Cléris}\thanks{timothee.david--cleris@ens-lyon.fr} \\[1ex] % Your name
\normalsize CRAL ENS de Lyon \\ % Your institution
}
\date{\today}

\usepackage{xcolor}
\definecolor{linkcolor}{rgb}{0,0,0.6}


\usepackage[ pdftex,colorlinks=true,
pdfstartview=ajustementV,
linkcolor= linkcolor,
citecolor= linkcolor,
urlcolor= linkcolor,
hyperindex=true,
hyperfigures=false]
{hyperref}



\usepackage{color}


\definecolor{GREEN}{rgb}{0,.7,0}
\definecolor{RED}{rgb}{.8,0,0}

\def\OK{\textcolor{GREEN}{OK}}
\def\FAIL{\textcolor{RED}{FAIL}}

\begin{document}
\maketitle


%%%%%%%%%%%%%%%%%data


\end{document}

"""









def get_test_dataset(test_result, dataset_name, table_name):
    test_data = test_result["test_data"]

    for d in test_data:
        if(d["dataset_name"] == dataset_name):
            for t in d["dataset"]:
                if(t["name"] == table_name):
                    return t["data"]

    return None





















def make_unittest_report(reports) -> str:
    buf = ""


    buf += r"\section{UnitTests results}"

    for rep in reports:


        buf += r"""
            \begin{center}
                \begin{tabular}{|c|c|c|}
                \hline
                Test name & Status & Succesfull asserts / total number of asserts \\  \hline \hline
            """

        has_fail = False
        for r in rep["results"]:
            if r["type"] == "Unittest":
                

                asserts = 0
                asserts_ok = 0

                for a in r["asserts"]:
                    asserts += 1
                    asserts_ok += a["value"]

                buf += r"\verb|"+ r["name"] + "| & "

                if asserts_ok < asserts:
                    buf += "\FAIL & "
                    has_fail = True
                else:
                    buf += "\OK & "

                buf += "$" + str(asserts_ok) + "/" + str(asserts) +r"$\\ \hline"+"\n"

        buf += r"""
                \end{tabular}\end{center}
            """



        buf += r"""
            \subsection{Failed test log :}
        """

        for r in rep["results"]:
            if r["type"] == "Unittest":


                asserts = 0
                asserts_ok = 0

                for a in r["asserts"]:
                    asserts += 1
                    asserts_ok += a["value"]

                if asserts_ok < asserts:
                    buf += r"\verb|"+ r["name"] + r"| : \\ asserts :" + " \n"

                    buf += r"\begin{itemize}"

                    for a in r["asserts"]:
                        if a["value"] == 0:
                            buf += r"\item "
                            buf += r"assert : \verb|"+ a["name"] + r"| : \FAIL ."
                            if "comment" in a.keys():
                                buf += r" Log : " + r"\verb|"+ a["comment"] + r"|\\" + " \n"

                    buf += r"\end{itemize}"

        buf += r"""
            \subsection{Passed test log :}
        """

        for r in rep["results"]:
            if r["type"] == "Unittest":


                asserts = 0
                asserts_ok = 0
                comment_count = 0

                for a in r["asserts"]:
                    asserts += 1
                    asserts_ok += a["value"]
                    if "comment" in a.keys():
                        comment_count += 1

                if asserts_ok == asserts:
                    if comment_count > 0:
                        buf += r"\verb|"+ r["name"] + r"| : \\ asserts :" + " \n"

                        buf += r"\begin{itemize}"

                        for a in r["asserts"]:
                            if a["value"] == 1:
                                if "comment" in a.keys():
                                    buf += r"\item "
                                    buf += r"assert : \verb|"+ a["name"] + r"| : \OK ."
                                    buf += r" Log : " + r"\verb|"+ a["comment"] + r"|\\" + " \n"

                        buf += r"\end{itemize}"





    return buf

def make_fmm_prec_plot(reports) -> str:
    fmm_prec_test_res = []
    
    for rep in reports:
        for r in rep["results"]:
            if r["type"] == "Analysis" and r["name"] == "models/generic/fmm/precision":
                fmm_prec_test_res.append(r)


    vec_angle          = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","angle");
    vec_result_pot_5   = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_pot_5");
    vec_result_pot_4   = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_pot_4");
    vec_result_pot_3   = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_pot_3");
    vec_result_pot_2   = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_pot_2");
    vec_result_pot_1   = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_pot_1");
    vec_result_pot_0   = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_pot_0");
    vec_result_force_5 = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_force_5");
    vec_result_force_4 = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_force_4");
    vec_result_force_3 = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_force_3");
    vec_result_force_2 = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_force_2");
    vec_result_force_1 = get_test_dataset(fmm_prec_test_res[0],"fmm_precision","result_force_1"); 

    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,6))


    def plot_curve(ax,X,Y,lab):

        ratio = 40

        cnt = len(X)//ratio 

        X_m = X.reshape((cnt,ratio))
        X_m = np.max(X_m,axis=1)

        Y_m = Y.reshape((cnt,ratio))
        Y_m = np.max(Y_m,axis=1)


        ax.plot(X_m,Y_m, label = lab)

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

    fig.set

    plt.tight_layout()

    plt.savefig("figures/fmm_precision.pdf")

    return r"""

    \subsection{Precision on particle pairs}

    \begin{figure}[ht!]
    \center
    \includegraphics[width=1\textwidth]{figures/fmm_precision.pdf}
    \caption{Precision of the fmm using multiple parameters. 
    On the left we compare fmm precision of the potential versus theory, the y axis is limited to $10^{-16}$ as we hit normally the limit of double precision floating point numbers. On the right the same comparaison can be seen for the corresponding force.}
    \label{fig:fmm_prec}
    \end{figure}

    \textbf{To check} : In Fig.\ref{fig:fmm_prec} We should see the potential hitting the limit of double precision normally.

    """



#to use do : python ..this file.. shamrock_benchmark_report

plt.style.use('custom_style.mplstyle')

if __name__ == '__main__':

    list_jsons = sys.argv[1:]

    jsons = []

    for f in list_jsons:
        print("reading :",f)
        jsons.append(json.load(open(f,'r')))

    buf = ""
    buf += make_unittest_report(jsons)
    buf += r"\section{FMM analysis}"
    buf += make_fmm_prec_plot(jsons)

    fout = repport_template.replace("%%%%%%%%%%%%%%%%%data", buf)

    f = open("report.tex",'w')
    f.write(fout)
    f.close()