from lib.buildbot import * 
import glob
import sys

print_buildbot_info("include guard check tool")

check_line = R'''#pragma once'''

file_list = glob.glob(str(abs_src_dir)+"/**",recursive=True)

file_list.sort()

pragma_once_missing = []

for fname in file_list:

    if (not fname.endswith(".hpp")):
        continue

    if fname.endswith("version.cpp"):
        continue

    f = open(fname,'r')
    res = f.readlines()
    
    has_line_before_guard = False
    for l in res:
        is_pragma = l.startswith(check_line)
        if is_pragma:
            break
        else:
            if not (l.startswith(r"//") or l.startswith(r"/*") or l.startswith(r"/*")):

                print(l)
                has_line_before_guard = True


    f.close()

    if has_line_before_guard : 
        pragma_once_missing.append(fname)


if len(pragma_once_missing) > 0:
    print(" => \033[1;34m#pragma once missing in \033[0;0m: ")

    for i in pragma_once_missing:
        print(" -",i.split(abs_proj_dir)[-1])

    sys.exit("Missing include guard for some source files")
else : 
    print(" => \033[1;34mInclude guard status \033[0;0m: OK !")