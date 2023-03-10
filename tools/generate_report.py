import glob
import dill
import os
import json



import argparse

parser = argparse.ArgumentParser(description='Generate shamrock report & plots')

parser.add_argument('--standalone', action='store_true')
parser.add_argument('--stack', action='store_true')
parser.add_argument('--compare', action='store_true')
parser.add_argument("--inputs", action="extend", nargs="+", type=str)

args = parser.parse_args()


print("Loading jsons")
results = []
for fname in args.inputs :
    print(" -",fname)
    with open(fname,'r') as f:
        results += (json.load(f))

try:
    os.mkdir("report_sections_dill")
except:
    a=0

globbed = glob.glob("report_sections/*")

print("\nRunning scripts")

tex_buf = ""


def get_str_foreach(index, result) -> str:
    buf = ""
    buf += r"\subsection{"
    buf += f"Config {index}"
    buf += r"}" + "\n\n" + r"\begin{itemize}"
    
    buf += f"""
    \\item Commit Hash {result["commit_hash"]}
    \\item World size {result["world_size"]}
    \\item Compiler {result["compiler"]}
    """

    buf += r"\end{itemize}"

    return buf






import matplotlib.pyplot as plt
plt.style.use('custom_style.mplstyle')

    
for fname in globbed:
    print(" -",fname)
    
    #dill_name = fname.replace("report_sections/", "report_sections_dill/").replace(".py", "")
    #os.system(f"python3 dillify.py {fname} {dill_name}")
    #fname_pick = (dill_name + ".standalone.dill")
    #
    #print("   dill file :",fname_pick)
    #with open(fname_pick,'rb') as f:
    #    func = dill.load(f)
    #    tex_buf += func(result)

    with open(fname,'r') as fpy:
        exec(fpy.read())

        tex_buf += standalone(results)

    


else:
    print("unknown report mode")

f = open("template_1rep.tex",'r')
repport_template = f.read()
f.close()

fout = repport_template.replace("%%%%%%%%%%%%%%%%%data", tex_buf)

f = open("report.tex",'w')
f.write(fout)
f.close()