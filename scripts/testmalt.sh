#!/bin/bash

## usage like this:
## scripts/testmalt.sh danish_1000 ~/corpora/conll2006/data/danish/ddt/test/danish_ddt_test.conll

THEJAR=maltparser-1.7/dist/maltparser-1.7/maltparser-1.7.jar

java -jar $THEJAR -c $1 -i $2 -m parse -o outfile.conll
