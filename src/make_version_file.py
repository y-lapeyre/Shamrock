import os
import sys

strfile = ""

try:
    fvers = open(sys.argv[1],'r')
    strfile = fvers.read()
    fvers.close()
except :
    None



str_git = os.popen("git log -n 1 --decorate=full").read()
git_hash = str_git.split()[1]
git_head = str_git[str_git.find("HEAD -> ")+8:str_git.find(")")]



str_ = '''
#include "aliases.hpp"

std::string git_info_str = R"%%(

'''

str_ += "     commit : "+git_hash+"\n"
str_ += "     HEAD   : "+git_head+"\n"
str_ += "     modified files (since last commit):"+"\n"
str_ += os.popen("git diff-index --name-only HEAD -- | sed \"s/^/        /g\"").read()
str_ += r')%%";'




if not (strfile == str_):
    f = open(sys.argv[1],'w')
    f.write(str_)
    f.close()

# else:
#     print("no change in version.cpp")