from lib.buildbot import * 
import glob
import sys

print_buildbot_info("licence check tool")

licence = R'''// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//'''

file_list = glob.glob(str(abs_src_dir)+"/**",recursive=True)

file_list.sort()

missing_licence = []

for fname in file_list:

    if (not fname.endswith(".cpp")) and (not fname.endswith(".hpp")):
        continue

    if fname.endswith("version.cpp"):
        continue

    f = open(fname,'r')
    res = f.read().startswith(licence)
    f.close()

    if not res : 
        missing_licence.append(fname)


if len(missing_licence) > 0:
    print(" => \033[1;34mlicence missing in \033[0;0m: ")

    for i in missing_licence:
        print(" -",i.split(abs_proj_dir)[-1])

    sys.exit("Missing liscence for some source files")