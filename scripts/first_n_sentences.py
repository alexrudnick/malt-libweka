#!/usr/bin/env python3

"""
$ python3 first_n_sentences.py treebank.conll 1000 langname

Spits the first 1000 sentences to langname_1000.conll.
"""

import sys
import os

def main():
    infn = sys.argv[1]
    N = int(sys.argv[2])
    langname = sys.argv[3]
    sentences = 0

    outfn = "training/{0}_{1}.conll".format(langname, N)
    if os.path.exists(outfn):
        print("careful, this path exists already:", outfn)
        return

    with open(infn) as infile:
        with open(outfn, "w") as outfile:
            while sentences < N:
                line = infile.readline()
                if line == "": break
                print(line, end="", file=outfile)
                if line.strip() == "":
                    sentences += 1
    print("created {0} with {1} sentences.".format(outfn, sentences))

if __name__ == "__main__": main()
