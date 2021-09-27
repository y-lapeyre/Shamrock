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




#generate_report

folder_report = "coverage_report"

from lxml import etree



print("making " + folder_report)

try:
    os.mkdir(folder_report)
except :
    os.system("rm -r " + folder_report)
    os.mkdir(folder_report)

os.system("cp " + "coverage_src_"+config_names[0] +"/style.css " + folder_report)





report_html_main = ""
head = ""
header_info_clang = ""
version_clang_info = ""
html_tables = []

for cid in range(len(config_names)):

    f_in_html = open("coverage_src_"+config_names[cid] + "/index.html" ,'r')


    str_in_html = f_in_html.read()

    html_in = etree.HTML(str_in_html)

    head = (etree.tostring(html_in[0], pretty_print=True, method="html")).decode('ASCII')
    #print(head)

    body_result = html_in[1]

    header_info_clang =  (etree.tostring(body_result[0], pretty_print=True, method="html")).decode('ASCII')
    header_info_clang += (etree.tostring(body_result[1], pretty_print=True, method="html")).decode('ASCII')
    header_info_clang += (etree.tostring(body_result[2], pretty_print=True, method="html")).decode('ASCII')

    html_tables.append( 
        ("<h4>"+ flags_list[cid] +"</h4>\n" + (etree.tostring(body_result[3], pretty_print=True, method="html")).decode('ASCII'))
            .replace(str(abs_src_dir), "_"+config_names[cid])
    )

    version_clang_info = (etree.tostring(body_result[4], pretty_print=True, method="html")).decode('ASCII')
    
    result = etree.tostring(html_in, pretty_print=True, method="html")
    #print(result)

    f_in_html.close()

f_out_html = open(folder_report + "/index.html" ,'w')


final_str_html = ""

final_str_html += ("<!doctype html><html>\n" + head + "<body>" + header_info_clang)

for sttr in html_tables:
    final_str_html += (sttr)

final_str_html += (version_clang_info +"</body>" +"\n</html>")

#print(str(abs_src_dir))

final_str_html = final_str_html.replace(str(abs_src_dir), "")

for cid in range(len(config_names)):
    os.system("mv "+"coverage_src_"+config_names[cid]+"/coverage"+abs_src_dir +" " + folder_report + "/coverage_"+config_names[cid])


f_out_html.write(final_str_html)





#fix paths from coverage subfolders
import glob
import re

for root, dirs, files in os.walk(folder_report):
    for file in files:
        if file.endswith('.html'):
            if not "index.html" in file:

                str_file = open(root + "/" + str(file) ,'r').read()

                relat_path_stylecss = str(os.path.relpath(folder_report,root + "/" + str(file)))

                str_out = re.sub("href='?(.*?)/style.css'", "href='"+relat_path_stylecss+"/"+folder_report+"/style.css'", str_file,  flags=re.DOTALL)
                str_out = str_out.replace(abs_src_dir,"src")

                str_file = open(root + "/" + str(file) ,'w').write(str_out)
    

exit()
#python test_pipeline.py --ninja --cuda ../../llvm