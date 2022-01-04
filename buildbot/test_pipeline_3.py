import os
import argparse
import re

import colorama
from colorama import Fore
from colorama import Style
import pathlib
from pathlib import Path




colorama.init()
print("\n"+Fore.BLUE + Style.BRIGHT + "   >>> Test Pipeline SPHIVE <<<   "+ Style.RESET_ALL + "\n")



test_suite_dir = "_test_suite"


parser = argparse.ArgumentParser(description='Configure utility for the code')

parser.add_argument("--ninja", action='store_true', help="use NINJA build system instead of Make")
parser.add_argument("--cuda", action='store_true', help="use CUDA instead of OPENCL")
parser.add_argument("llvm_root",help="llvm location", type=str)

args = parser.parse_args()









abs_proj_dir = os.path.abspath(os.path.join(__file__, "../.."))
abs_src_dir = os.path.join(abs_proj_dir,"src")
abs_build_dir_src = os.path.join(abs_proj_dir,"build_pipe")

abs_llvm_dir = os.path.abspath(os.path.join(os.getcwd(),args.llvm_root))

sycl_comp = "DPC++"


print(Fore.BLUE + Style.BRIGHT + "Project directory : "+ Style.RESET_ALL + abs_proj_dir)
print(Fore.BLUE + Style.BRIGHT + " Source directory : "+ Style.RESET_ALL + abs_src_dir)
print(Fore.BLUE + Style.BRIGHT + " Build  dir pref  : "+ Style.RESET_ALL + abs_build_dir_src)
print()
print(Fore.BLUE + Style.BRIGHT + "LLVM directory    : "+ Style.RESET_ALL + abs_llvm_dir)
print(Fore.BLUE + Style.BRIGHT + "SyCL compiler     : "+ Style.RESET_ALL + sycl_comp)


print()
print(" - configuring :")

abs_test_suite_dir = os.path.join(abs_proj_dir,test_suite_dir)
print(Fore.BLUE + Style.BRIGHT + " test suite working dir : "+ Style.RESET_ALL + abs_test_suite_dir)



# regex to fin all new 
# /\s+([^ ]+)\s*(=\s*new [^;]+;)/g
# replace by 
#  $1 $2 \n      log_new($1,log_alloc_ln);\n

#regex to find all delete []
# /delete\s*\[\]\s*([^ ]+)\s*;/g
# replace by 
# delete [] $1; log_delete($1);

#regex to find all delete
# /delete\s*([^ ]+)\s*;/g
# replace by 
# delete $1; log_delete($1);

def patch_file(file):
    lines_in = ""
    with open(file, "r") as f_in:
            lines_in = f_in.read()
            
    lines_in = re.sub(r"delete\s*([^ ]+)\s*;", "delete \g<1>; log_delete(\g<1>);", lines_in)

    lines_in = re.sub(r"delete\s*\[\]\s*([^ ]+)\s*;", "delete [] \g<1>; log_delete(\g<1>);", lines_in)
    
    lines_in = re.sub(r"\s+([^ ]+)\s*(=\s*new [^;]+;)", r" \g<1> \g<2>       log_new(\g<1>,log_alloc_ln);\n", lines_in)

    with open(file, "w") as f_out:
            f_out.write(lines_in)
            

os.system("rm -r "+ abs_test_suite_dir)

try:
    os.mkdir(abs_test_suite_dir)
except:
    ...

os.system("cp -r " +abs_src_dir+ " "+ abs_test_suite_dir)

for path in Path(abs_test_suite_dir).rglob('*.cpp'):
    abs_path_in = str(path.absolute())
    print("patching : " + abs_path_in)
    patch_file(abs_path_in)