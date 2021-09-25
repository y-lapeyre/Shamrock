import os
import argparse

import colorama
from colorama import Fore
from colorama import Style
import pathlib



colorama.init()
print("\n"+Fore.BLUE + Style.BRIGHT + "   >>> Test Pipeline SPHIVE <<<   "+ Style.RESET_ALL + "\n")






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
print(Fore.BLUE + Style.BRIGHT + " Build  directory prefix : "+ Style.RESET_ALL + abs_build_dir_src)
print()
print(Fore.BLUE + Style.BRIGHT + "LLVM directory    : "+ Style.RESET_ALL + abs_llvm_dir)
print(Fore.BLUE + Style.BRIGHT + "SyCL compiler     : "+ Style.RESET_ALL + sycl_comp)


def generate_cmake_conf(build_dir):
    cmake_conf_cmd = "cmake"
    cmake_conf_cmd += " -S " + abs_src_dir
    cmake_conf_cmd += " -B " + build_dir

    if args.ninja:
        cmake_conf_cmd += " -G Ninja"

    if sycl_comp == "DPC++":
        cmake_conf_cmd += " -DSyCL_Compiler=DPC++"
        cmake_conf_cmd += " -DDPCPP_INSTALL_LOC=" + os.path.join(abs_llvm_dir,"build")
        if args.cuda:
            cmake_conf_cmd += " -DSyCL_Compiler_BE=CUDA"

    cmake_conf_cmd += " -DBUILD_TEST=true"

    return cmake_conf_cmd

def generate_cmake_comp(build_dir):
    cmake_comp_cmd = "cmake"
    cmake_comp_cmd += " --build"
    cmake_comp_cmd += " "+build_dir

    return cmake_comp_cmd


test_exe_filename = "shamrock_test"


config_names = [
        "ss",
        "ds",
        "sm",
        "dm",
        "sd",
        "dd"]
        
flags_list = [
     "-DMorton_precision=single -DPhysics_precision=single"
    ,"-DMorton_precision=double -DPhysics_precision=single"
    ,"-DMorton_precision=single -DPhysics_precision=mixed"
    ,"-DMorton_precision=double -DPhysics_precision=mixed"
    ,"-DMorton_precision=single -DPhysics_precision=double"
    ,"-DMorton_precision=double -DPhysics_precision=double"]

process_cnt = [4 for i in range(len(config_names))]

#configure step
for cid in range(len(config_names)):
    cmake_cmd_cov =  generate_cmake_conf(abs_build_dir_src + "/conf_" + config_names[cid])
    cmake_cmd_cov += " -DCMAKE_BUILD_TYPE=COVERAGE_MAP " + flags_list[cid]

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_cmd_cov + "\n")
    os.system(cmake_cmd_cov)


#compilation step
for cid in range(len(config_names)):
    cmake_comp_cmd =  generate_cmake_comp(abs_build_dir_src + "/conf_" + config_names[cid])

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_comp_cmd + "\n")
    os.system(cmake_comp_cmd)


#running step
for cid in range(len(config_names)):

    abs_build_dir = abs_build_dir_src + "/conf_" + config_names[cid]

    runcmd = "mpirun --show-progress "

    for i in range(process_cnt[cid]):
        runcmd += '-n 1 -x LLVM_PROFILE_FILE="'+abs_build_dir+'/program'+str(i)+'.profraw" '+abs_build_dir+"/"+test_exe_filename

        if i < process_cnt[cid]-1:
            runcmd += " : "
    
    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + runcmd + "\n")
    os.system(runcmd)

#merge profiling data

for cid in range(len(config_names)):

    llvm_profdata_cmd = abs_llvm_dir + "/build/bin/llvm-profdata merge "


    abs_build_dir = abs_build_dir_src + "/conf_" + config_names[cid]

    for i in range(process_cnt[cid]):
        llvm_profdata_cmd += abs_build_dir+"/program"+str(i)+".profraw "

    llvm_profdata_cmd += "-o "+abs_build_dir+"/program.profdata"

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + llvm_profdata_cmd + "\n")
    os.system(llvm_profdata_cmd)


for cid in range(len(config_names)):

    llvmcovshow = abs_llvm_dir+"/build/bin/llvm-cov show "


    abs_build_dir = abs_build_dir_src + "/conf_" + config_names[cid] 

    llvmcovshow += " " + abs_build_dir+"/"+test_exe_filename + " "

    llvmcovshow += " -instr-profile=" + abs_build_dir+"/program.profdata -use-color --format html -output-dir=coverage_src_"+config_names[cid] 

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + llvmcovshow + "\n")
    os.system(llvmcovshow)

exit()
