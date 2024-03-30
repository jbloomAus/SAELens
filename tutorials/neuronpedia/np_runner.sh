# script for working around memory issues - this ensures every batch gets the whole system available memory
# better fix is to investigate and fix the memory issues

#!/bin/bash
LAYER=$1
TYPE=$2
SOURCE_AUTHOR_SUFFIX=$3
FEATURES_AT_A_TIME=$4
START_BATCH_INCLUSIVE=$5
END_BATCH_INCLUSIVE=$6
for j in $(seq $5 $6)
    do
    echo "Iteration: $j"
    python np_runner_batch.py $1 $2 $3 $4 $j $j 
done