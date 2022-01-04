import subprocess
import os 
from enum import Enum



class BuildSystem(Enum):
    Makefiles = 1
    Ninja = 2

class SyCLBE(Enum):
    CUDA = 1

class PrecisionMode(Enum):
    Single = 1
    Mixed = 2
    Double = 3

class Targets(Enum):
    SPH = 1
    AMR = 2
    Test = 3
    Visu = 4





title_wide = """
  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░ 
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███   
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████    
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███   
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███  
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░ 
"""

title_normal = """
███████╗██╗  ██╗ █████╗ ███╗   ███╗██████╗  ██████╗  ██████╗██╗  ██╗
██╔════╝██║  ██║██╔══██╗████╗ ████║██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
███████╗███████║███████║██╔████╔██║██████╔╝██║   ██║██║     █████╔╝ 
╚════██║██╔══██║██╔══██║██║╚██╔╝██║██╔══██╗██║   ██║██║     ██╔═██╗ 
███████║██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝╚██████╗██║  ██╗
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝
"""

title_small = """
╔═╗╦ ╦╔═╗╔╦╗╦═╗╔═╗╔═╗╦╔═
╚═╗╠═╣╠═╣║║║╠╦╝║ ║║  ╠╩╗
╚═╝╩ ╩╩ ╩╩ ╩╩╚═╚═╝╚═╝╩ ╩
"""



abs_proj_dir = os.path.abspath(os.path.join(__file__, "../../.."))
abs_src_dir = os.path.join(abs_proj_dir,"src")









def print_buildbot_info(utility_name):
    
    col_cnt = os.get_terminal_size().columns

    if(col_cnt > 112):
        print(title_wide)
    elif(col_cnt > 69): #nice
        print(title_normal)
    else:
        print(title_small)

    print("\033[1;90m"+"-"*col_cnt+"\033[0;0m\n")

    print("\033[1;34mCurrent tool      \033[0;0m: ", utility_name)
    print("\033[1;34mProject directory \033[0;0m: ", abs_proj_dir)
    print("\033[1;34mSource  directory \033[0;0m: ", abs_src_dir)
    
    print()

    str_git = os.popen("git log -n 1 --decorate=full").read()

    git_hash = str_git.split()[1]
    git_head = str_git[str_git.find("HEAD -> ")+8:str_git.find(")")]


    git_head = git_head.split(",")

    if len(git_head) == 1:
        git_head = "\033[1;92m" + git_head[0] + "\033[0;0m"
    else:
        git_head = "\033[1;92m" + git_head[0] + "\033[0;0m , \033[1;91m" + git_head[0] + "\033[0;0m"


    print("\033[1;34mGit status \033[0;0m: ")
    print("     \033[1;93mcommit \033[0;0m: ",git_hash)
    print("     \033[1;36mHEAD   \033[0;0m: ",git_head)
    print("     \033[1;31mmodified files\033[0;0m (since last commit):")
    print(os.popen("git diff-index --name-only HEAD -- | sed \"s/^/        /g\"").read())
    print("\033[1;90m"+"-"*col_cnt+"\033[0;0m\n")



def run_cmd(str):

    col_cnt = os.get_terminal_size().columns

    print("\033[1;34mRunning \033[0;0m: " + str)
    print("\033[1;90mcmd out "+"-"*(col_cnt-8)+"\033[0;0m")
    os.system(str)
    print("\033[1;90m"+"-"*col_cnt+"\033[0;0m")



def chdir(path):
    print("\033[1;34mcd \033[0;0m: " + path)
    os.chdir(path)






def is_ninja_available():

    try:
        # pipe output to /dev/null for silence
        null = open("/dev/null", "w")
        subprocess.Popen(['ninja', '--version'], stdout=null, stderr=null)
        null.close()
        return True
    except OSError:
        return False



def compile_prog(abs_build_dir):
    cmake_cmd = "cmake"
    cmake_cmd += " --build"
    cmake_cmd += " "+abs_build_dir

    run_cmd(cmake_cmd)


def get_current_buildsystem(abs_build_dir):

    if os.path.isfile(abs_build_dir + "/build.ninja"):
        return BuildSystem.Ninja 
    if os.path.isfile(abs_build_dir + "/Makefile"):
        return BuildSystem.Makefiles

    raise "buildsystem not recognized" 
    


def clean_build_dir(abs_build_dir):
    current_build_sys = get_current_buildsystem(abs_build_dir)
    current_dir = os.getcwd()
    chdir(abs_build_dir)

    if current_build_sys == BuildSystem.Ninja:
        run_cmd("ninja clean")

    if current_build_sys == BuildSystem.Makefiles:
        run_cmd("make clean")

    chdir(current_dir)



#configure_dpcpp("../src", "../build2","../../llvm",BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Single,PrecisionMode.Single)
def configure_dpcpp(src_dir, build_dir,llvm_root,build_sys,sycl_BE,target_lst,morton_prec,phy_prec):

    print("\033[1;34mConfiguring SHAMROCK with DPC++\033[0;0m")

    enabled_targets_str = ""
    if Targets.Test in target_lst:
        enabled_targets_str += "test "

    if Targets.SPH in target_lst:
        enabled_targets_str += "sph "

    if Targets.AMR in target_lst:
        enabled_targets_str += "amr "

    if Targets.Visu in target_lst:
        enabled_targets_str += "visu"

    print("  -> \033[1;34mEnabled targets   : \033[0;0m" +enabled_targets_str )


    cmake_cmd = "cmake"
    cmake_cmd += " -S " + src_dir
    cmake_cmd += " -B " + build_dir


    if build_sys == BuildSystem.Ninja:
        cmake_cmd += " -G Ninja"

    if build_sys == BuildSystem.Makefiles:
        cmake_cmd += ' -G "Unix Makefiles"'


    cmake_cmd += " -DSyCL_Compiler=DPC++"
    cmake_cmd += " -DDPCPP_INSTALL_LOC=" + os.path.join(llvm_root,"build")

    if sycl_BE == SyCLBE.CUDA:
        cmake_cmd += " -DSyCL_Compiler_BE=CUDA"


    if Targets.Test in target_lst:
        cmake_cmd += " -DBUILD_TEST=true"

    if Targets.SPH in target_lst:
        cmake_cmd += " -DBUILD_SPH=true"

    if Targets.AMR in target_lst:
        cmake_cmd += " -DBUILD_AMR=true"

    if Targets.Visu in target_lst:
        cmake_cmd += " -DBUILD_VISU=true"

    


    if morton_prec == PrecisionMode.Single:
        print("  -> \033[1;34mMorton mode       : \033[0;0msingle")
        cmake_cmd += " -DMorton_precision=single"
    elif morton_prec == PrecisionMode.Double:
        print("  -> \033[1;34mMorton mode       : \033[0;0mdouble")
        cmake_cmd += " -DMorton_precision=double"
    else:
        raise "unknown morton precision mode : " + str(morton_prec)


    if phy_prec == PrecisionMode.Single:
        print("  -> \033[1;34mPrecision mode    : \033[0;0msingle")
        cmake_cmd += " -DPhysics_precision=single"
    elif phy_prec == PrecisionMode.Mixed:
        print("  -> \033[1;34mPrecision mode    : \033[0;0mmixed")
        cmake_cmd += " -DPhysics_precision=mixed"
    elif phy_prec == PrecisionMode.Double:
        print("  -> \033[1;34mPrecision mode    : \033[0;0mdouble")
        cmake_cmd += " -DPhysics_precision=double"
    else:
        raise "unknown phy precision mode : " + str(phy_prec)


    #if args.xray:
    #    cmake_cmd += " -DXRAY_INSTRUMENTATION=true"

    run_cmd(cmake_cmd)