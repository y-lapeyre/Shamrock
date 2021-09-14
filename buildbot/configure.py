import os
import argparse

import colorama
from colorama import Fore
from colorama import Style
import pathlib



colorama.init()
print("\n"+Fore.BLUE + Style.BRIGHT + "   >>> Configuration utility for SPHIVE <<<   "+ Style.RESET_ALL + "\n")





parser = argparse.ArgumentParser(description='Configure utility for the code')

parser.add_argument("--ninja", action='store_true', help="use NINJA build system instead of Make")
parser.add_argument("--cuda", action='store_true', help="use CUDA instead of OPENCL")

parser.add_argument("--test", action='store_true', help="add test target to build configuration")
parser.add_argument("--sph", action='store_true', help="add sph target to build configuration")
parser.add_argument("--amr", action='store_true', help="add amr target to build configuration")
parser.add_argument("--visu", action='store_true', help="add visualisation target to build configuration")


parser.add_argument("llvm_root",help="llvm location", type=str)

args = parser.parse_args()









abs_proj_dir = os.path.abspath(os.path.join(__file__, "../.."))
abs_src_dir = os.path.join(abs_proj_dir,"src")
abs_build_dir = os.path.join(abs_proj_dir,"build")

abs_llvm_dir = os.path.abspath(os.path.join(os.getcwd(),args.llvm_root))

sycl_comp = "DPC++"


print(Fore.BLUE + Style.BRIGHT + "Project directory : "+ Style.RESET_ALL + abs_proj_dir)
print(Fore.BLUE + Style.BRIGHT + " Source directory : "+ Style.RESET_ALL + abs_src_dir)
print(Fore.BLUE + Style.BRIGHT + " Build  directory : "+ Style.RESET_ALL + abs_build_dir)
print()
print(Fore.BLUE + Style.BRIGHT + "LLVM directory    : "+ Style.RESET_ALL + abs_llvm_dir)



print(Fore.BLUE + Style.BRIGHT + "SyCL compiler     : "+ Style.RESET_ALL + sycl_comp)


cmake_cmd = "cmake"
cmake_cmd += " -S " + abs_src_dir
cmake_cmd += " -B " + abs_build_dir

if args.ninja:
    cmake_cmd += " -G Ninja"

if sycl_comp == "DPC++":
    cmake_cmd += " -DSyCL_Compiler=DPC++"
    cmake_cmd += " -DDPCPP_INSTALL_LOC=" + os.path.join(abs_llvm_dir,"build")
    if args.cuda:
        cmake_cmd += " -DSyCL_Compiler_BE=CUDA"

if args.test:
    cmake_cmd += " -DBUILD_TEST=true"

if args.sph:
    cmake_cmd += " -DBUILD_SPH=true"

if args.amr:
    cmake_cmd += " -DBUILD_AMR=true"


if args.visu:
    cmake_cmd += " -DBUILD_VISU=true"

print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_cmd + "\n")

os.system(cmake_cmd)
