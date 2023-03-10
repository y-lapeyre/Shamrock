import json
import sys

lst = []
for i in range(1,len(sys.argv)):
    f = open(sys.argv[i])
    data = json.load(f)
    lst.append(data)

    f.close()

with open("merged_out.json",'w') as f:
    json.dump(lst, f)
