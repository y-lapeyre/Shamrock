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
abs_build_dir = os.path.join(abs_proj_dir,"build_pipe")

abs_llvm_dir = os.path.abspath(os.path.join(os.getcwd(),args.llvm_root))

sycl_comp = "DPC++"


print(Fore.BLUE + Style.BRIGHT + "Project directory : "+ Style.RESET_ALL + abs_proj_dir)
print(Fore.BLUE + Style.BRIGHT + " Source directory : "+ Style.RESET_ALL + abs_src_dir)
print(Fore.BLUE + Style.BRIGHT + " Build  directory : "+ Style.RESET_ALL + abs_build_dir)
print()
print(Fore.BLUE + Style.BRIGHT + "LLVM directory    : "+ Style.RESET_ALL + abs_llvm_dir)
print(Fore.BLUE + Style.BRIGHT + "SyCL compiler     : "+ Style.RESET_ALL + sycl_comp)



cmake_conf_cmd = "cmake"
cmake_conf_cmd += " -S " + abs_src_dir
cmake_conf_cmd += " -B " + abs_build_dir

if args.ninja:
    cmake_conf_cmd += " -G Ninja"

if sycl_comp == "DPC++":
    cmake_conf_cmd += " -DSyCL_Compiler=DPC++"
    cmake_conf_cmd += " -DDPCPP_INSTALL_LOC=" + os.path.join(abs_llvm_dir,"build")
    if args.cuda:
        cmake_conf_cmd += " -DSyCL_Compiler_BE=CUDA"

cmake_conf_cmd += " -DBUILD_TEST=true"

        
cmake_comp_cmd = "cmake"
cmake_comp_cmd += " --build"
cmake_comp_cmd += " "+abs_build_dir


test_exe_filename = "shamrock_test"


def generate_gcov():
    cmake_cmd_gcov =  cmake_conf_cmd
    cmake_cmd_gcov += " -DCMAKE_BUILD_TYPE=GCOV"
    
    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_cmd_gcov + "\n")
    os.system(cmake_cmd_gcov)
    
    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_comp_cmd + "\n")
    os.system(cmake_comp_cmd)
    
    os.system("mpirun "+abs_build_dir+"/"+test_exe_filename)

    os.system("rm log_full_*")
    
    os.system('find '+abs_build_dir+' -iname "*.gcda" -exec '+abs_llvm_dir+'/build/bin/llvm-cov gcov -m -n -f {} \; > gcov_report.txt')
    
    
def generate_clang_covmap_legacy(process_cnt):

    cmake_cmd_cov =  cmake_conf_cmd
    cmake_cmd_cov += " -DCMAKE_BUILD_TYPE=COVERAGE_MAP "

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_cmd_cov + "\n")
    os.system(cmake_cmd_cov)
    
    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_comp_cmd + "\n")
    os.system(cmake_comp_cmd)

    runcmd = "mpirun --show-progress "

    for i in range(process_cnt):
        runcmd += '-n 1 -x LLVM_PROFILE_FILE="program'+str(i)+'.profraw" '+abs_build_dir+"/"+test_exe_filename

        if i < process_cnt-1:
            runcmd += " : "


    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + runcmd + "\n")
    os.system(runcmd)

    llvm_profdata_cmd = abs_llvm_dir + "/build/bin/llvm-profdata merge "
    for i in range(process_cnt):
        llvm_profdata_cmd += "program"+str(i)+".profraw "
    llvm_profdata_cmd += "-o program.profdata"

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + llvm_profdata_cmd + "\n")
    os.system(llvm_profdata_cmd)

    os.system(abs_llvm_dir+"/build/bin/llvm-cov show "+abs_build_dir+"/"+test_exe_filename+" -instr-profile=program.profdata -use-color --format html -output-dir=coverage")

    os.system("rm log_full_*")
    
    os.system('find '+abs_build_dir+' -iname "*.gcda" -exec '+abs_llvm_dir+'/build/bin/llvm-cov gcov -m -f {} \; > gcov_report.txt')
    
    os.system("mv ./*.gcov coverage")

    os.system("mv gcov_report.txt coverage")

    os.system("mv *.profdata *.profraw coverage")
    

def generate_clang_covmap(process_cnt,config_name,flags):

    cmake_cmd_cov =  cmake_conf_cmd
    cmake_cmd_cov += " -DCMAKE_BUILD_TYPE=COVERAGE_MAP "

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_cmd_cov + "\n")
    os.system(cmake_cmd_cov)
    
    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_comp_cmd + "\n")
    os.system(cmake_comp_cmd)

    runcmd = "mpirun --show-progress "

    for i in range(process_cnt):
        runcmd += '-n 1 -x LLVM_PROFILE_FILE="program'+str(i)+'.profraw" '+abs_build_dir+"/"+test_exe_filename

        if i < process_cnt-1:
            runcmd += " : "


    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + runcmd + "\n")
    os.system(runcmd)

    llvm_profdata_cmd = abs_llvm_dir + "/build/bin/llvm-profdata merge "
    for i in range(process_cnt):
        llvm_profdata_cmd += "program"+str(i)+".profraw "
    llvm_profdata_cmd += "-o program.profdata"

    print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + llvm_profdata_cmd + "\n")
    os.system(llvm_profdata_cmd)

    os.system(abs_llvm_dir+"/build/bin/llvm-cov show "+abs_build_dir+"/"+test_exe_filename+" -instr-profile=program.profdata -use-color --format html -output-dir=coverage")

    os.system("rm log_full_*")
    
    os.system('find '+abs_build_dir+' -iname "*.gcda" -exec '+abs_llvm_dir+'/build/bin/llvm-cov gcov -m -f {} \; > gcov_report.txt')
    
    os.system("mv ./*.gcov coverage")

    os.system("mv gcov_report.txt coverage")

    os.system("mv *.profdata *.profraw coverage")
#def run_test():


#generate_gcov()
generate_clang_covmap_legacy(4)
exit()
generate_clang_covmap(4,[
        "ss",
        "ds",
        "sm",
        "dm",
        "sd",
        "dd"],[
     "-DMorton_precision=single -DPhysics_precision=single"
    ,"-DMorton_precision=double -DPhysics_precision=single"
    ,"-DMorton_precision=single -DPhysics_precision=mixed"
    ,"-DMorton_precision=double -DPhysics_precision=mixed"
    ,"-DMorton_precision=single -DPhysics_precision=double"
    ,"-DMorton_precision=double -DPhysics_precision=double"])