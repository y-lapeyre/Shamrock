import os
import argparse

from lib.buildbot import * 


parser = argparse.ArgumentParser(description='Configure utility for the code')

parser.add_argument("--ninja", action='store_true', help="use NINJA build system instead of Make")
parser.add_argument('--buildmode',action='store', type=str, default="None", help='target build mode')

parser.add_argument("--cuda", action='store_true', help="use CUDA instead of OPENCL")
parser.add_argument("--test", action='store_true', help="add test target to build configuration")
parser.add_argument("--sph", action='store_true', help="add sph target to build configuration")
parser.add_argument("--amr", action='store_true', help="add amr target to build configuration")
parser.add_argument("--visu", action='store_true', help="add visualisation target to build configuration")
#parser.add_argument("--xray", action='store_true', help="add xray instrumentation to all targets")
parser.add_argument('--morton',   action='store', type=str, default="single", help='precision for morton codes')
parser.add_argument('--phyprec',     action='store', type=str, default="single", help='precision mode for physical variables')
parser.add_argument('--interactive',     action='store_true', help='enables interactive configuration')

parser.add_argument("llvm_root",help="llvm location", type=str)

args = parser.parse_args()


print_buildbot_info("configure tool")

abs_build_dir = os.path.join(abs_proj_dir,"build")
abs_llvm_dir = os.path.abspath(os.path.join(os.getcwd(),args.llvm_root))

print("\033[1;34mLLVM  directory \033[0;0m: "+ abs_llvm_dir)
print("\033[1;34mBuild directory \033[0;0m: "+ abs_build_dir)
print()



if args.interactive:
    while 1:
        print("\033[1;34mInteractive configuration \033[0;0m:")

        args.ninja = input("    do you want to use ninja instead of make (y/n)") == "y"


        print("    compile mode available :")
        print("           0: Normal  (no special flags)")
        print("           1: Release (optimisation flags)")
        print("           2: Debug   (debug flags)")
        tmp_release = int(input("    with which mode do you want to compile :"))

        if tmp_release == 0:
            args.buildmode = "Normal"
        elif tmp_release == 1:
            args.buildmode = "Release"
        elif tmp_release == 2:
            args.buildmode = "Debug"

        args.sph = input("    do you want to compile the sph mode (y/n)") == "y"
        args.amr = input("    do you want to compile the amr mode (y/n)") == "y"
        args.test = input("    do you want to compile the test mode (y/n)") == "y"
        args.visu = input("    do you want to compile the visualisation mode (y/n)") == "y"

        args.cuda = input("    do you want to use cuda as sycl backend (y/n)") == "y"

        args.morton  = input("    which precision do you want for morton codes (single/double)") 
        args.phyprec  = input("    which precision do you want for physics computations (single/mixed/double)") 

        print("\033[1;34mOptions summary \033[0;0m: ")

        print("    ninja      =",args.ninja)
        print("    build mode =",args.buildmode)
        print("    cuda       =",args.cuda)
        print("    sph        =",args.sph)
        print("    amr        =",args.amr)
        print("    test       =",args.test)
        print("    visu       =",args.visu)
        print("    morton     =",args.morton)
        print("    phyprec    =",args.phyprec)
        print()
        if input("confirm choices (y/N)") == "y":
            break
            print()
    print()


build_sys = BuildSystem.Makefiles
if args.ninja:
    build_sys = BuildSystem.Ninja


target_buildmode = BuildMode.Normal
if args.buildmode == "Normal":
    target_buildmode = BuildMode.Normal
elif args.buildmode == "Release":
    target_buildmode = BuildMode.Release
elif args.buildmode == "Debug":
    target_buildmode = BuildMode.Debug



target_lst = []
if args.sph:
    target_lst.append(Targets.SPH)
if args.amr:
    target_lst.append(Targets.AMR)
if args.test:
    target_lst.append(Targets.Test)
if args.visu:
    target_lst.append(Targets.Visu)


sycl_be = -1
if args.cuda:
    sycl_be = SyCLBE.CUDA

prec_mort = -1
prec_phys = -1

if args.morton == ("single"):
    prec_mort = PrecisionMode.Single
elif args.morton == ("double"):
    prec_mort = PrecisionMode.Double


if args.phyprec == ("single"):
    prec_phys = PrecisionMode.Single
elif args.phyprec == ("mixed"):
    prec_phys = PrecisionMode.Mixed
elif args.phyprec == ("double"):
    prec_phys = PrecisionMode.Double






configure_dpcpp(
    abs_src_dir, 
    abs_build_dir,
    abs_llvm_dir,
    target_buildmode,
    build_sys,
    sycl_be,
    target_lst,
    prec_mort,
    prec_phys)
