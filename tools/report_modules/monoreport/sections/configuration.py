
def make_configuration_report(report) -> str:

    buf = r"\section{Configuration}" + "\n" + "\n"

    buf += r"\begin{itemize}" + "\n"
    buf += r"\item " "commit hash : " + report["commit_hash"] + "\n"
    buf += r"\item " "compiler : "    + report["compiler"   ] + "\n"
    #buf += r"\item " "compilation args : "   +r"\verb|" + report["comp_args"  ]+ "|" + "\n"
    buf += r"\item " "world size : "  +report["world_size"  ]  + "\n"
    buf += r"\end{itemize}" + "\n"

    return buf