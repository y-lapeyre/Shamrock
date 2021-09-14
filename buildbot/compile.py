import os
import argparse

import colorama
from colorama import Fore
from colorama import Style
import pathlib



colorama.init()
print("\n"+Fore.BLUE + Style.BRIGHT + "   >>> Compilation utility for SPHIVE <<<   "+ Style.RESET_ALL + "\n")


abs_proj_dir = os.path.abspath(os.path.join(__file__, "../.."))
abs_src_dir = os.path.join(abs_proj_dir,"src")
abs_build_dir = os.path.join(abs_proj_dir,"build")


print(Fore.BLUE + Style.BRIGHT + "Project directory : "+ Style.RESET_ALL + abs_proj_dir)
print(Fore.BLUE + Style.BRIGHT + " Source directory : "+ Style.RESET_ALL + abs_src_dir)
print(Fore.BLUE + Style.BRIGHT + " Build  directory : "+ Style.RESET_ALL + abs_build_dir)
print()


cmake_cmd = "cmake"
cmake_cmd += " --build"
cmake_cmd += " "+abs_build_dir


print("\n"+ Fore.BLUE + Style.BRIGHT + "Running : "+ Style.RESET_ALL + cmake_cmd + "\n")

os.system(cmake_cmd)
