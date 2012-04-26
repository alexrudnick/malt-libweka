#!/bin/bash

## script to do the whole train/test/evaluate pipeline for a given language,
## and learner, varying the training size.

## Parameters are:
## $1: language (eg, danish)
## $2: learner: libweka? libsvm?
## $3: "options for the library"

EXPECTED_ARGS=3
E_BADARGS=65
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: scripts/`basename $0` danish libweka j48"
  exit $E_BADARGS
fi

LANG=$1
LEARNER=$2
WEKACLASSIFIER=$3

for n in `seq 1 11`; do
  scripts/run_experiment.sh "$LANG"_"$n"000 $LEARNER $WEKACLASSIFIER testsets/"$LANG"_test.conll
done
