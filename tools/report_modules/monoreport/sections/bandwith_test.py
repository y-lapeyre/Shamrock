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



def make_bandwith_matrix(fileprefix,report) -> str:
    bw_tests = []

    world_size = int(report["world_size"])
    
    for r in report["results"]:
        if r["type"] == "Benchmark" and r["name"] == "bandwith-tests/mpi-pair-comm/bw-matrix":

            bw_tests.append(r["test_data"])

    entries = {}

    for bwt in bw_tests:
        for e in bwt:
            entries[e["dataset_name"]] = np.zeros((world_size,world_size),dtype=float)



    for bwt in bw_tests:
        for e in bwt:

            dname = e["dataset_name"]

            rank_send = []
            rank_recv = []
            bandwith = []

            for d in e["dataset"]:
                if d["name"] == "rank_send":
                    rank_send = d["data"]
                if d["name"] == "rank_recv":
                    rank_recv = d["data"]
                if d["name"] == "bandwith (GB.s^-1)":
                    bandwith = d["data"]

            
            for rs,rr,bw in zip(rank_send,rank_recv,bandwith):
                #print(rs,rr,bw)
                entries[dname][int(rs),int(rr)] += bw
                #print(entries)

    for k in entries.keys():
        entries[k] /= world_size

    cnt = 0
    for k in entries.keys():
        plt.figure()
        plt.title(k.replace("->"," to "))
        plt.imshow(entries[k])
        plt.xlabel("sender")
        plt.ylabel("receiver")
        cbar = plt.colorbar()
        cbar.set_label("Bandwith GB.s-1")
        plt.tight_layout()

        plt.savefig("figures/"+fileprefix+"bandwith_test"+str(cnt)+".pdf")
        cnt += 1
        
    buf = r"""
    \section{Bandwith Test}"

    """

    cnt = 0
    for k in entries.keys():
        buf+= r"""
        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.5\textwidth]{"""+"figures/"+fileprefix+"bandwith_test"+str(cnt)+".pdf"+r"""}
        \caption{TODO}
        \end{figure}

        """

        cnt += 1




    return buf


def make_bandwith_matrix_report(fileprefix : str, report) -> str:
    buf = ""

    buf += make_bandwith_matrix(fileprefix,report)


    return buf