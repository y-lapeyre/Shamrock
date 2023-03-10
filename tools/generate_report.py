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

globbed = glob.glob("report/*")

print("\nRunning scripts")

tex_buf = ""
















class TestInstance:
    def __init__(self, result):
        self.type = result["type"]
        self.name = result["name"]
        self.compute_queue = result["compute_queue"]
        self.alt_queue = result["alt_queue"]
        self.world_rank = result["world_rank"]
        self.asserts = result["asserts"]
        self.test_data = result["test_data"]

    def get_test_dataset(self,dataset_name, table_name):
        for d in self.test_data:
            if(d["dataset_name"] == dataset_name):
                for t in d["dataset"]:
                    if(t["name"] == table_name):
                        return t["data"]

        return None

    

class TestResults:

    def __init__(self, index, result):
        self.commit_hash = result["commit_hash"]
        self.world_size = result["world_size"]
        self.compiler = result["compiler"]
        self.comp_args = result["comp_args"]
        self.index = index
        self.results = result["results"]

    def get_config_str(self) -> str:
        buf = ""
        buf += r"\subsection{"
        buf += f"Config {self.index}"
        buf += r"}" + "\n\n" + r"\begin{itemize}"
        
        buf += f"""
        \\item Commit Hash : {self.commit_hash}
        \\item World size : {self.world_size}
        \\item Compiler : {self.compiler}
        """

        buf += r"\end{itemize}"

        return buf

    
    def get_test_instances(self, type, name):
        instances = []
        for r in self.results:
            if r["type"] == type and r["name"] == name:
                instances.append(TestInstance(r))

        return instances















import matplotlib.pyplot as plt
plt.style.use('custom_style.mplstyle')

if args.standalone:
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