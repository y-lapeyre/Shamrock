import glob
from lib.buildbot import * 

print_buildbot_info("make status file table")

print("Item | Price |")
print("---|---|")


names = []
for f in glob.glob(abs_src_dir + "/**",recursive=True):
    if(f.endswith(".cpp") or f.endswith(".hpp")):

        ln = "src"+ f.split(abs_src_dir)[1] 

        fname = ln.split("/")[-1]

        ln = ln.replace(fname, "**"+fname+"**")
        names.append(ln)

names.sort()

for ln in names:
    print("|   |",ln,"|")