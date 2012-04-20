#!/bin/bash

## script to do the whole train/test/evaluate pipeline.

## Parameters are:
## $1: language_size (eg, danish_1000)
## $2: learner: libweka? libsvm?
## $3: "options for the library"
## $4: test set (eg /path/to/danish_ddt_test.conll)


EXPECTED_ARGS=4
E_BADARGS=65
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: scripts/`basename $0` danish_1000 libweka j48 testsets/danish_test.conll"
  exit $E_BADARGS
fi

THEJAR=maltparser-1.7/dist/maltparser-1.7/maltparser-1.7.jar
LANG_SIZE=$1
LEARNER=$2
WEKACLASSIFIER=$3
TESTSET=$4

EXPNAME=$LANG_SIZE"_"$LEARNER"_"$WEKACLASSIFIER
MODELNAME=$EXPNAME
OUTFILE=$EXPNAME"_outfile.conll"
RESULTS="results/"$EXPNAME".results"

## training part
java -jar $THEJAR -l $LEARNER -weka $WEKACLASSIFIER -c $MODELNAME -i training/$LANG_SIZE.conll -m learn -f /home/alex/svn/parsing-qual/options.xml > trainingoutput

## testing part
echo "OK NOW TESTING."
java -jar $THEJAR -c $MODELNAME -i $TESTSET -m parse -o $OUTFILE -f /home/alex/svn/parsing-qual/options.xml > testingoutput

## evaluate.
echo "OK NOW EVALUATING."
perl scripts/eval.pl -q -s $OUTFILE -g $TESTSET > $RESULTS
cat $RESULTS
