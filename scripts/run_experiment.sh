#!/bin/bash

## script to do the whole train/test/evaluate pipeline.

## Parameters are:
## $1: language_size (eg, danish_1000)
## $2: learner: libweka? libsvm?
## $3: "options for the library"
## $4: test set (eg /path/to/danish_ddt_test.conll)

EXPECTED_ARGS_NONWEKA=3
EXPECTED_ARGS_WEKA_NOOPTION=4
EXPECTED_ARGS_WEKA_OPTION=5
E_BADARGS=65

HANDLED="nothandled"

if [ $# -eq $EXPECTED_ARGS_NONWEKA ]; then
  LANG_SIZE=$1
  LEARNER=$2
  WEKACLASSIFIER=$LEARNER
  TESTSET=$3
  EXPNAME=$LANG_SIZE"_"$LEARNER"_"$WEKACLASSIFIER
  HANDLED="handled"
fi

if [ $# -eq $EXPECTED_ARGS_WEKA_NOOPTION ]; then
  LANG_SIZE=$1
  LEARNER=$2
  WEKACLASSIFIER=$3
  TESTSET=$4
  EXPNAME=$LANG_SIZE"_"$LEARNER"_"$WEKACLASSIFIER
  HANDLED="handled"
fi

if [ $# -eq $EXPECTED_ARGS_WEKA_OPTION ]; then
  LANG_SIZE=$1
  LEARNER=$2
  WEKACLASSIFIER=$3
  WEKAOPTS=$4
  TESTSET=$5
  EXPNAME=$LANG_SIZE"_"$LEARNER"_"$WEKACLASSIFIER
  HANDLED="handled"
fi

if [ "$HANDLED" != "handled" ]; then
  echo "Usage: scripts/`basename $0` danish_1000 libweka j48 [\"weka options\"] testsets/danish_test.conll"
  exit $E_BADARGS
fi

MODELNAME=$EXPNAME
OUTFILE=$EXPNAME"_outfile.conll"
RESULTS="results/"$EXPNAME".results"

CLASSPATH=lib/libsvm.jar:lib/log4j.jar:lib/maltparser-1.7.1.jar:lib/weka-3.6.6.jar:dist/lib/maltlibweka.jar:lib/commons-net-3.1.jar

## training part
if [ $# -eq $EXPECTED_ARGS_NONWEKA ]; then
  java -Xmx8192m \
    -cp $CLASSPATH org.maltparser.Malt \
    -l $LEARNER \
    -c $MODELNAME \
    -i training/$LANG_SIZE.conll \
    -m learn \
    > trainingoutput
fi

if [ $# -eq $EXPECTED_ARGS_WEKA_NOOPTION ]; then
  java -Xmx8192m \
    -cp $CLASSPATH org.maltparser.Malt \
    -l $LEARNER \
    -weka $WEKACLASSIFIER \
    -c $MODELNAME \
    -i training/$LANG_SIZE.conll \
    -m learn \
    > trainingoutput
fi

if [ $# -eq $EXPECTED_ARGS_WEKA_OPTION ]; then
  java -Xmx8192m \
    -cp $CLASSPATH org.maltparser.Malt \
    -l $LEARNER \
    -weka $WEKACLASSIFIER \
    -wekaopts "$WEKAOPTS" \
    -c $MODELNAME \
    -i training/$LANG_SIZE.conll \
    -m learn \
    > trainingoutput
fi

## testing part
echo "OK NOW TESTING."
java -Xmx8192m \
  -cp $CLASSPATH org.maltparser.Malt \
  -c $MODELNAME \
  -i $TESTSET \
  -m parse \
  -o $OUTFILE \
  > testingoutput

## evaluate.
echo "OK NOW EVALUATING."
perl scripts/eval.pl -q -s $OUTFILE -g $TESTSET > $RESULTS
cat $RESULTS
