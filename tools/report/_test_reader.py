import json
import argparse

parser = argparse.ArgumentParser(description='Generate a report & plots')

parser.add_argument('--standalone', action='store_true')
parser.add_argument('--stack', action='store_true')
parser.add_argument('--compare', action='store_true')
parser.add_argument("--input", type=str)
parser.add_argument("--outtex", type=str)
parser.add_argument("--outfigfolder", type=str)

args = parser.parse_args()

if(args.input == None):
    exit("you should give a json input")

if(args.outtex == None):
    exit("you should give a output folder")

if(args.outfigfolder == None):
    exit("you should give a output folder")

def run(standalone, stacked, compared):
    with open(args.input,'r') as f:
        results = json.load(f)

        with open(args.outtex, 'w') as fouttex:

            out_tex = ""

            out_tex = standalone(results,args.outfigfolder + "/")

            if out_tex == None:
                exit("the output cannot be none")

            fouttex.write(out_tex)
