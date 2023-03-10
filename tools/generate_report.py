import glob
import dill
import os


try:
    os.mkdir("report_sections_dill")
except:
    a=0

globbed = glob.glob("report_sections/*")

print(globbed)

for fname in globbed:
    print("Running :",fname)
    
    dill_name = fname.replace("report_sections/", "report_sections_dill/").replace(".py", "")
    os.system(f"python3 dillify.py {fname} {dill_name}")
    fname_pick = (dill_name + ".standalone.dill")

    print("Running dill :",fname_pick)
    with open(fname_pick,'rb') as f:
        func = dill.load(f)
        print(func(0))

