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


def try_mkdir(name):
    try:
        os.mkdir(name)
    except:
        pass

try_mkdir("_build")
try_mkdir("_build/figures")

fname_merge_json = "_build/merged_json.json"

with open(fname_merge_json,'w') as fjson:
    json.dump(results, fjson) 

globbed = glob.glob("report/*.py")

print("\nRunning scripts")

fexclude_list = ["report/_testlib.py","report/_test_reader.py"]


tex_buf = ""


if args.standalone:
    for fname in globbed:
        if(not fname in fexclude_list):
            print(" -",fname)

            outtex_file = fname.replace("report/", "_build/")  + ".tex"

            cmd = "python3 "+fname
            cmd += " --standalone --input "+fname_merge_json
            cmd += " --outtex " + outtex_file
            cmd += " --outfigfolder _build/figures"

            os.system(cmd)

            with open(outtex_file,'r') as ftexin:
                tex_buf += ftexin.read()

    


else:
    print("unknown report mode")

f = open("template_1rep.tex",'r')
repport_template = f.read()
f.close()

fout = repport_template.replace("%%%%%%%%%%%%%%%%%data", tex_buf)

f = open("_build/report.tex",'w')
f.write(fout)
f.close()