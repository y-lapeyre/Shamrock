from pathlib import Path
import os

import subprocess


flags = "-DPRECISION_MORTON_DOUBLE -DPRECISION_FULL_DOUBLE"

for path in Path('../src').rglob('*.cpp'):
    
    break

    print(path.name)

    os.system("../../llvm/build/bin/clang++ -fsycl "+flags+" -S -emit-llvm "+str(path)+" -o "+path.name+".llvm")

    os.system('../../llvm/build/bin/clang-offload-bundler --inputs ' + path.name+'.llvm'+' --type ll --outputs '+path.name+'.llvm.off'+' --targets "host-x86_64-unknown-linux-gnu" --unbundle')
    os.system('../../llvm/build/bin/clang-offload-bundler --inputs ' + path.name+'.llvm'+' --type ll --outputs '+path.name+'.llvm.offspr'+' --targets "sycl-spir64-unknown-unknown" --unbundle')

    p1 = subprocess.Popen(["cat", path.name+'.llvm.off'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["../../llvm/build/bin/opt", "--sycl-opt", "-analyze", "-dot-callgraph"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    output,err = p2.communicate()

    with open(path.name+".callgraph.dot", 'w') as fout:
        p1 = subprocess.Popen(["cat", "<stdin>.callgraph.dot"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["c++filt"], stdin=p1.stdout, stdout=subprocess.PIPE)
        p3 = subprocess.Popen(["sed",'s,>,\\>,g; s,-\\>,->,g; s,<,\\<,g'], stdin=p2.stdout, stdout=fout)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        p2.stdout.close()
        output,err = p3.communicate()

    os.system("rm \<stdin\>.callgraph.dot")





    p1 = subprocess.Popen(["cat", path.name+'.llvm.offspr'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["../../llvm/build/bin/opt", "--sycl-opt", "-analyze", "-dot-callgraph"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    output,err = p2.communicate()

    with open(path.name+".kern.callgraph.dot", 'w') as fout:
        p1 = subprocess.Popen(["cat", "<stdin>.callgraph.dot"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["c++filt"], stdin=p1.stdout, stdout=subprocess.PIPE)
        p3 = subprocess.Popen(["sed",'s,>,\\>,g; s,-\\>,->,g; s,<,\\<,g'], stdin=p2.stdout, stdout=fout)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        p2.stdout.close()
        output,err = p3.communicate()

    os.system("rm \<stdin\>.callgraph.dot")

    


import graphviz


excludelist = [
    "std::",
    "cl::",
    "{__",
    "void __",
    "llvm.",
    "nlohmann::"
]



#####################################################
#remove excluded Nodes
#####################################################

def is_excluded(l):
    for i in excludelist:
        if i in l:
            return True

    return False



removed_nodes = []
line_to_rm = []

_gbody = []
for path in Path('.').rglob('*.callgraph.dot'):

    gfile = graphviz.Source.from_file(path.name).source.split('\n')[3:-2]

    for i in range(len(gfile)):
        if "main" in gfile[i]:
            gfile[i] = gfile[i].replace("main",path.name.split(".")[0])

            print(gfile[i])

    
    _gbody += gfile


for l in _gbody:
    if is_excluded(l):

        nd = l[l.find("Node"):l.find("Node")+18]
        removed_nodes.append(nd)
        line_to_rm.append(l)
        print(nd+"|")

for l in line_to_rm:
    _gbody.remove(l)

print("----------------------------------")
print("----------------------------------")
print("----------------------------------")

for l in _gbody:
    if "label" in l:
        print(is_excluded(l),l)


print("----------------------------------")
print("----------------------------------")
print("----------------------------------")


def is_con_excluded(l):
    for i in removed_nodes:
        if i in l:
            return True

    return False

line_to_rm = []
for l in _gbody:
    if is_con_excluded(l):
        #print(l)
        line_to_rm.append(l)

for l in line_to_rm:
    _gbody.remove(l)

print("----------------------------------")
print("----------------------------------")
print("----------------------------------")





#####################################################
#merge nodes
#####################################################
affect_dic = {}
inverse_affect_dic = {}
replace_dic = {}
line_to_rm = []


def get_label(l):
    startidx = l.find(r'label="{')+8
    endidx = l[startidx:].find(r'}"')
    return l[startidx:startidx+endidx]


for l in _gbody:
    if "label=" in l:

        lbl = get_label(l)

        if(lbl == "main"):
            continue

        nd = l[l.find("Node"):l.find("Node")+18]

        if lbl in affect_dic.keys():
            replace_dic[nd] = affect_dic[lbl]
            line_to_rm.append(l)
        else:
            affect_dic[lbl] = nd
            inverse_affect_dic[nd] = lbl

print(affect_dic)

print(replace_dic)

print(inverse_affect_dic.keys())


for l in line_to_rm:

    nd = l[l.find("Node"):l.find("Node")+18]
    print(nd,nd in inverse_affect_dic.keys())


    _gbody.remove(l)


for key in replace_dic.keys():

    new_nd = replace_dic[key]
    old_nd = key

    print(new_nd, old_nd)

    for i in range(len(_gbody)):
        

        _gbody[i] = _gbody[i].replace(old_nd, new_nd)




#####################################################
# restrict to used nodes
#####################################################

nodes = {}
links = {}

for l in _gbody:
    if "label=" in l:
        nd = l[l.find("Node"):l.find("Node")+18]
        nodes[nd] = l[l.find(" [")+1:l.find(";")]
    elif " -> " in l:
        tmp = l.split(" -> ")

        nd1 = tmp[0][tmp[0].find("Node"):tmp[0].find("Node")+18]
        nd2 = tmp[1][tmp[1].find("Node"):tmp[1].find("Node")+18]

        if nd1 in links.keys():
            links[nd1] += [nd2]
        else:
            links[nd1] = [nd2]
    else:
        print(l)



print(nodes)
print(links)

#get_starting_node

def get_starting_node(node_label):
    start_node_id = ""
    for k in nodes.keys():
        if node_label == get_label(nodes[k]):
            start_node_id = k
    return start_node_id


_gbody = []
#remake body dot graph

used_nodes = {}
used_links = {}
def add_childs(node_id,depth = 0):

    if depth > 10 : return

    if depth < 3: print(nodes[node_id])
    
    used_nodes[node_id] = True

    if node_id in links.keys():
        for child_node_id in links[node_id]:
            used_links[node_id + " -> " + child_node_id] = True

            add_childs(child_node_id,depth+1)

print('add_childs(get_starting_node("main_test"))')
add_childs(get_starting_node("main_test"))

print('add_childs(get_starting_node("main_amr"))')
add_childs(get_starting_node("main_amr"))

print('add_childs(get_starting_node("main_sph"))')
add_childs(get_starting_node("main_sph"))

print('add_childs(get_starting_node("main_visu"))')
add_childs(get_starting_node("main_visu"))


# for n in nodes:
#     print(nodes[n])

#     if get_label(nodes[n]).startswith("typeinfo name for "):
#         nodes[n] = nodes[n].replace("typeinfo name for ","")

#         print('add_childs('+n+')')
#         add_childs(n)



#####################################################
#rebuild graph
#####################################################



list_std = [
    "printf",
    "vfprintf",
    "vsnprintf",
    "vsprintf","fclose",
    "exit",
    "system",
    "operator new(unsigned long)",
    "operator delete(void*)",
    "operator new[](unsigned long)",
    "operator delete[](void*)",
    "labs"]

def is_std_group(node_label):
    if node_label in list_std:
        return True
    
    if node_label.startswith("std::"):
        return True

    return False


def is_run_tests_group(node_label):
    if node_label.startswith("run_tests"):
        return True

    return False


def is_unittest_group(node_label):
    if node_label.startswith("unit_test::") :
        return True

    return False



for nid in used_nodes.keys():
    if get_label(nodes[nid]).startswith("MPI_"):
        print(nid)
    elif is_std_group(get_label(nodes[nid])):
        print(nid)
    elif is_run_tests_group(get_label(nodes[nid])):
        print(nid)
    elif is_unittest_group(get_label(nodes[nid])):
        print(nid)
    else:
        _gbody.append(nid + " " + nodes[nid] + ";\n")

#mpi cluster
_gbody += ['subgraph clustermpi_subgraph { style=filled; rank=max;label = "MPI instructions";  ']
for nid in used_nodes.keys():
    if get_label(nodes[nid]).startswith("MPI_"):
        _gbody.append(nid + ' [style=filled;color=cyan1;label="' + get_label(nodes[nid]) + '"];\n')
_gbody += ["}"]


#std group cluster
_gbody += ['subgraph clusterstdgroup_subgraph { style=filled; rank=max;label = "std instructions";  ']
for nid in used_nodes.keys():
    if is_std_group(get_label(nodes[nid])):
        if get_label(nodes[nid]) == "exit":
            _gbody.append(nid + ' [style=filled;color=red;label="' + get_label(nodes[nid]) + '"];\n')
        else:
            _gbody.append(nid + ' [style=filled;color=bisque;label="' + get_label(nodes[nid]) + '"];\n')
_gbody += ["}"]


#std run_tests cluster
_gbody += ['subgraph clusterrun_testsgroup_subgraph { style=filled;label = "run_tests";  ']
for nid in used_nodes.keys():
    if is_run_tests_group(get_label(nodes[nid])):
            _gbody.append(nid + ' [style=filled;color=white;label="' + get_label(nodes[nid]) + '"];\n')
_gbody += ["}"]



#std unit_tests cluster
_gbody += ['subgraph clusterunit_testsgroup_subgraph { style=filled;label = "unit_test::";  ']
for nid in used_nodes.keys():
    if is_unittest_group(get_label(nodes[nid])):
            _gbody.append(nid + ' [style=filled;color=white;label="' + get_label(nodes[nid]) + '"];\n')
_gbody += ["}"]


for lk in used_links.keys():
    _gbody.append(lk + ";\n")









#####################################################
#modify look mains
#####################################################

rep_stmap = {
    "main_sph" : 'fillcolor=green , style=filled ',
    "main_amr" : 'fillcolor=green , style=filled ',
    "main_test" : 'fillcolor=green , style=filled ',
    "main_visu" : 'fillcolor=green , style=filled '
}


move_map = {
    "main_sph" : 'rank=min; ',
    "main_amr" : 'rank=min; ',
    "main_test" : 'rank=min; ',
    "main_visu" : 'rank=min; '
}


for kk in rep_stmap.keys():
    for i in range(len(_gbody)):
        if get_label(_gbody[i]) == kk:

            _gbody[i] = _gbody[i].replace("shape=record", rep_stmap[kk])

            print(_gbody[i])


for kk in move_map.keys():
    for i in range(len(_gbody)):
        if get_label(_gbody[i]) == kk:

            _gbody[i] = "{\n    " + move_map[kk] + _gbody[i].replace("{" +kk+ "}",kk) + "}\n"

            print(_gbody[i])












#####################################################
#output & rendering
#####################################################
g = graphviz.Digraph('callgraph', filename='callgraph.dot',engine='dot',body=['rankdir="LR";newrank=true;'] + _gbody)

#print(g)

g.render()
