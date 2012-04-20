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

LANG_SIZE=$1
LEARNER=$2
WEKACLASSIFIER=$3
TESTSET=$4

EXPNAME=$LANG_SIZE"_"$LEARNER"_"$WEKACLASSIFIER
MODELNAME=$EXPNAME
OUTFILE=$EXPNAME"_outfile.conll"
RESULTS="results/"$EXPNAME".results"

CLASSPATH=lib/libsvm.jar:lib/log4j.jar:lib/maltparser-1.7.1.jar:lib/weka.jar:dist/lib/maltlibweka.jar

## training part
java -cp $CLASSPATH org.maltparser.Malt \
  -l $LEARNER \
  -weka $WEKACLASSIFIER \
  -c $MODELNAME \
  -i training/$LANG_SIZE.conll \
  -m learn \
  > trainingoutput

## testing part
echo "OK NOW TESTING."
java -cp $CLASSPATH org.maltparser.Malt \
  -c $MODELNAME \
  -i $TESTSET \
  -m parse \
  -o $OUTFILE \
  > testingoutput

## evaluate.
echo "OK NOW EVALUATING."
perl scripts/eval.pl -q -s $OUTFILE -g $TESTSET > $RESULTS
cat $RESULTS
