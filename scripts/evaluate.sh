#!/bin/bash

## usage like this:
## scripts/evaluate.sh outfile.conll ~/corpora/conll2006/data/danish/ddt/test/danish_ddt_test.conll

perl scripts/eval.pl -q -s $1 -g $2 
