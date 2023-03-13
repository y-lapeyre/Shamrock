import re
import os

base_toc = r"""

# Table of contents
# Learn more at https://jupyterbook.org/customize/toc.html

format: jb-book
root: src/intro
parts:
"""


summary = ""

with open("./shamrock-doc/src/SUMMARY.md",'r') as f:
    summary = f.readlines()

summary = summary[3:]

parenthesis_extract = re.compile(r"\((.*?)\)")

last_level = 0
for i in range(len(summary)):

    if (summary[i].startswith("#")):
        summary[i] = summary[i].replace("#"," - caption: ") + "   chapters:"
        last_level=0
    elif (summary[i].startswith("- [")):
        summary[i] = "    - file: " +parenthesis_extract.findall(summary[i])[0].replace("./","src/").replace(".md","") + "\n"
        last_level=1
    elif (summary[i].startswith("    - [")):

        pre = ""
        if(last_level == 1):
            pre = "      sections:\n"

        summary[i] = pre+"      - file: " +parenthesis_extract.findall(summary[i])[0].replace("./","src/").replace(".md","") + "\n"
        last_level=2

for l in summary:
    print(l,end='')

with open("./jbook/_toc.yml",'w') as f:
    f.write(base_toc)

    for l in summary:
        f.write(l)


os.chdir("jbook")
os.system("sh copy_src.sh")
