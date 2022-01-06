from lib.buildbot import * 


import argparse

parser = argparse.ArgumentParser(description='Test pipeline utility for the code')
parser.add_argument("llvm_root",help="llvm location", type=str)
args = parser.parse_args()

abs_llvm_dir = os.path.abspath(os.path.join(os.getcwd(),args.llvm_root))



print_buildbot_info("Test pipeline")

abs_test_pipeline_dir = os.path.join(abs_proj_dir,"test_pipeline")

run_cmd("rm -r "+abs_test_pipeline_dir)
run_cmd("mkdir "+abs_test_pipeline_dir)

abs_test_pipeline_dir_src = os.path.join(abs_test_pipeline_dir,"src_patched")

gen_mem_patched_dir(abs_src_dir,abs_test_pipeline_dir_src)



configure_dpcpp(abs_test_pipeline_dir_src, abs_test_pipeline_dir + "/build_ss",abs_llvm_dir,BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Single,PrecisionMode.Single)
configure_dpcpp(abs_test_pipeline_dir_src, abs_test_pipeline_dir + "/build_sm",abs_llvm_dir,BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Single,PrecisionMode.Mixed)
configure_dpcpp(abs_test_pipeline_dir_src, abs_test_pipeline_dir + "/build_sd",abs_llvm_dir,BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Single,PrecisionMode.Double)
configure_dpcpp(abs_test_pipeline_dir_src, abs_test_pipeline_dir + "/build_ds",abs_llvm_dir,BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Double,PrecisionMode.Single)
configure_dpcpp(abs_test_pipeline_dir_src, abs_test_pipeline_dir + "/build_dm",abs_llvm_dir,BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Double,PrecisionMode.Mixed)
configure_dpcpp(abs_test_pipeline_dir_src, abs_test_pipeline_dir + "/build_dd",abs_llvm_dir,BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Double,PrecisionMode.Double)

conf_dir_lst = [
    abs_test_pipeline_dir + "/build_ss",
    abs_test_pipeline_dir + "/build_sm",
    abs_test_pipeline_dir + "/build_sd",
    abs_test_pipeline_dir + "/build_ds",
    abs_test_pipeline_dir + "/build_dm",
    abs_test_pipeline_dir + "/build_dd",
]

for dir_ in conf_dir_lst:
    compile_prog(dir_)

current_wdir = os.getcwd()

for dir_ in conf_dir_lst:

    os.chdir(dir_)

    run_cmd("./shamrock_test -o test_res_1")
    run_cmd("mpirun -n 2 ./shamrock_test -o test_res_2")
    run_cmd("mpirun -n 3 ./shamrock_test -o test_res_3")
    run_cmd("mpirun -n 4 ./shamrock_test -o test_res_4")

