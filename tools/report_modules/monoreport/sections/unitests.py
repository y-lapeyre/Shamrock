

def make_unittest_report(report) -> str:

    nb_unittest = 0
    for r in report["results"]:
        if r["type"] == "Unittest":
            nb_unittest+=1

    if nb_unittest == 0:
        return r"\section{UnitTests results} ""\n No Unittest in this log"


    buf = ""


    buf += r"\section{UnitTests results}"



    buf += r"""
        \begin{center}
            \begin{tabular}{|c|c|c|}
            \hline
            Test name & Status & Succesfull asserts / total number of asserts \\  \hline \hline
        """

    has_fail = False
    for r in report["results"]:
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

    for r in report["results"]:
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

    for r in report["results"]:
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