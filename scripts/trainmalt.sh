#!/bin/bash

## usage: scripts/trainmalt.sh danish_1000

THEJAR=maltparser-1.7/dist/maltparser-1.7/maltparser-1.7.jar

java -jar $THEJAR -l libsvm -c $1 -i training/$1.conll -m learn
